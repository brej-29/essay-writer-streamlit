# core/graph.py
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from typing import Generator

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langsmith import trace, Client

from core.config import get_secret
from core.prompts import (
    PLAN_PROMPT,
    RESEARCH_PLAN_PROMPT,
    WRITER_PROMPT,
    REFLECTION_PROMPT,
    RESEARCH_CRITIQUE_PROMPT,
    build_length_instruction,
)
from core.research import run_tavily_search
from core.schemas import AgentState, EssayRunConfig, Queries

logger = logging.getLogger("essay_writer.graph")


@dataclass
class EssayRunResult:
    final_state: Dict[str, Any]
    snapshots: List[Dict[str, Any]]
    drafts: List[str]
    critiques: List[str]
    plan: str
    content_notes: List[str]
    trace_id: Optional[str] = None


def _build_llm(cfg: EssayRunConfig) -> ChatOpenAI:
    # ChatOpenAI supports passing api_key directly (otherwise it reads from env var). :contentReference[oaicite:1]{index=1}
    openai_key = get_secret("OPENAI_API_KEY", required=True)
    return ChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        api_key=openai_key,
        max_retries=2,
        timeout=60,
    )


def _build_graph(llm: ChatOpenAI) -> Any:
    """
    Build the LangGraph graph (compiled). Mirrors the notebook architecture:
    planner -> research_plan -> generate -> (END or reflect) -> research_critique -> generate ...
    """
    def plan_node(state: AgentState) -> Dict[str, Any]:
        cfg_dict = state.get("config", {})
        tone = cfg_dict.get("tone", "Academic")
        audience = cfg_dict.get("audience", "General")
        paragraph_count = cfg_dict.get("paragraph_count", 5)
        length_mode = cfg_dict.get("length_mode", "Medium")
        target_words = cfg_dict.get("target_words")

        length_instruction = build_length_instruction(length_mode, target_words)

        messages = [
            SystemMessage(
                content=PLAN_PROMPT.format(
                    tone=tone,
                    audience=audience,
                    paragraph_count=paragraph_count,
                    length_instruction=length_instruction,
                )
            ),
            HumanMessage(content=state["task"]),
        ]
        resp = llm.invoke(messages)
        return {"plan": resp.content}

    def research_plan_node(state: AgentState) -> Dict[str, Any]:
        cfg_dict = state.get("config", {})
        use_research = bool(cfg_dict.get("use_research", True))
        max_results = int(cfg_dict.get("max_results", 2))

        content = list(state.get("content", []))

        if not use_research:
            return {"content": content}

        # Generate queries in structured format
        queries = llm.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_PLAN_PROMPT.format(max_queries=3)),
                HumanMessage(content=state["task"]),
            ]
        )

        for q in queries.queries:
            notes = run_tavily_search(q, max_results=max_results)
            content.extend(notes)

        return {"content": content}

    def generation_node(state: AgentState) -> Dict[str, Any]:
        cfg_dict = state.get("config", {})
        tone = cfg_dict.get("tone", "Academic")
        audience = cfg_dict.get("audience", "General")
        paragraph_count = cfg_dict.get("paragraph_count", 5)
        length_mode = cfg_dict.get("length_mode", "Medium")
        target_words = cfg_dict.get("target_words")

        content_text = "\n\n".join(state.get("content", []))
        critique = state.get("critique", "")

        length_instruction = build_length_instruction(length_mode, target_words)

        user_prompt = (
            f"{state['task']}\n\n"
            f"Constraints:\n"
            f"- Tone: {tone}\n"
            f"- Audience: {audience}\n"
            f"- Paragraphs: {paragraph_count}\n"
            f"- Length: {length_instruction}\n\n"
            f"Here is my plan:\n\n{state.get('plan','')}\n"
        )
        if critique:
            user_prompt += f"\nCritique to address:\n{critique}\n"

        messages = [
            SystemMessage(content=WRITER_PROMPT.format(tone=tone, audience=audience, content=content_text)),
            HumanMessage(content=user_prompt),
        ]
        resp = llm.invoke(messages)

        return {
            "draft": resp.content,
            "revision_number": int(state.get("revision_number", 1)) + 1,
        }

    def reflection_node(state: AgentState) -> Dict[str, Any]:
        messages = [
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=state.get("draft", "")),
        ]
        resp = llm.invoke(messages)
        return {"critique": resp.content}

    def research_critique_node(state: AgentState) -> Dict[str, Any]:
        cfg_dict = state.get("config", {})
        use_research = bool(cfg_dict.get("use_research", True))
        max_results = int(cfg_dict.get("max_results", 2))

        content = list(state.get("content", []))

        if not use_research:
            return {"content": content}

        critique = state.get("critique", "")
        if not critique.strip():
            return {"content": content}

        queries = llm.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_CRITIQUE_PROMPT.format(max_queries=3)),
                HumanMessage(content=critique),
            ]
        )

        for q in queries.queries:
            notes = run_tavily_search(q, max_results=max_results)
            content.extend(notes)

        return {"content": content}

    def should_continue(state: AgentState) -> str:
        if int(state.get("revision_number", 1)) > int(state.get("max_revisions", 2)):
            return END
        return "reflect"

    builder = StateGraph(AgentState)

    builder.add_node("planner", plan_node)
    builder.add_node("research_plan", research_plan_node)
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.add_node("research_critique", research_critique_node)

    builder.set_entry_point("planner")

    builder.add_edge("planner", "research_plan")
    builder.add_edge("research_plan", "generate")

    builder.add_conditional_edges("generate", should_continue, {END: END, "reflect": "reflect"})
    builder.add_edge("reflect", "research_critique")
    builder.add_edge("research_critique", "generate")

    # In-memory checkpointing (simple + works well for Streamlit sessions). :contentReference[oaicite:2]{index=2}
    graph = builder.compile(checkpointer=MemorySaver())
    return graph


@st.cache_resource(show_spinner=False)
def get_compiled_graph(model: str, temperature: float) -> Any:
    """
    Cache the compiled graph per (model, temperature) to avoid recompiling on each run.
    """
    cfg = EssayRunConfig(
        model=model,
        temperature=temperature,
        # dummy required field
        task="__dummy__",
    )
    llm = _build_llm(cfg)
    return _build_graph(llm)


def run_essay(cfg_dict: Dict[str, Any]) -> EssayRunResult:
    """
    Runs the graph and returns outputs + history.
    Also creates a root LangSmith trace (to capture trace_id for later feedback). :contentReference[oaicite:3]{index=3}
    """
    cfg = EssayRunConfig(**cfg_dict)

    graph = get_compiled_graph(cfg.model, cfg.temperature)

    # Thread id is required to keep checkpoint continuity per session/thread. :contentReference[oaicite:4]{index=4}
    thread_id = cfg_dict.get("thread_id") or str(uuid.uuid4())
    runnable_config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "task": cfg.task,
        "max_revisions": cfg.max_revisions,
        "revision_number": 1,
        "content": [],
        "config": cfg.model_dump(),
    }

    snapshots: List[Dict[str, Any]] = []
    final_state: Dict[str, Any] = {}
    trace_id: Optional[str] = None

    # Create a root trace so we can attach user feedback later. :contentReference[oaicite:5]{index=5}
    inputs = {"task": cfg.task, "config": cfg.model_dump()}
    with trace(name="essay_writer_run", inputs=inputs) as root_run:
        for chunk in graph.stream(initial_state, runnable_config):
            # chunk is typically a dict like {"node_name": {...updated state...}} OR values depending on mode.
            snapshots.append(chunk)
            final_state = chunk

        root_run.outputs = {"final": final_state}
        trace_id = str(root_run.id)

    # Extract plan/drafts/critiques/content from snapshots robustly
    plan = ""
    drafts: List[str] = []
    critiques: List[str] = []
    content_notes: List[str] = []

    def _pull_state(obj: Any) -> Dict[str, Any]:
        # Try to normalize both "values" and "updates" style chunks
        if isinstance(obj, dict):
            # If it's {"node": {...}} use the inner dict
            if len(obj) == 1 and isinstance(next(iter(obj.values())), dict):
                return next(iter(obj.values()))
            return obj
        return {}

    last_draft = None
    last_critique = None

    for s in snapshots:
        stt = _pull_state(s)
        if isinstance(stt.get("plan"), str) and stt["plan"].strip():
            plan = stt["plan"]
        if isinstance(stt.get("content"), list):
            content_notes = stt["content"]
        if isinstance(stt.get("draft"), str) and stt["draft"].strip():
            if stt["draft"] != last_draft:
                drafts.append(stt["draft"])
                last_draft = stt["draft"]
        if isinstance(stt.get("critique"), str) and stt["critique"].strip():
            if stt["critique"] != last_critique:
                critiques.append(stt["critique"])
                last_critique = stt["critique"]

    # final_state may still be chunk-wrapped; normalize it
    final_state_norm = _pull_state(final_state)

    return EssayRunResult(
        final_state=final_state_norm,
        snapshots=snapshots,
        drafts=drafts,
        critiques=critiques,
        plan=plan,
        content_notes=content_notes,
        trace_id=trace_id,
    )

def run_essay_stream(cfg_dict: Dict[str, Any]) -> Generator[Dict[str, Any], None, EssayRunResult]:
    """
    Stream node updates from LangGraph and yield events for the UI.
    Uses LangGraph stream_mode="updates" so each chunk is {node_name: update_dict}. :contentReference[oaicite:2]{index=2}
    At the end, returns EssayRunResult (via generator return).
    """
    cfg = EssayRunConfig(**cfg_dict)
    graph = get_compiled_graph(cfg.model, cfg.temperature)

    thread_id = cfg_dict.get("thread_id") or str(uuid.uuid4())
    runnable_config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "task": cfg.task,
        "max_revisions": cfg.max_revisions,
        "revision_number": 1,
        "content": [],
        "config": cfg.model_dump(),
    }

    snapshots: List[Dict[str, Any]] = []
    plan = ""
    drafts: List[str] = []
    critiques: List[str] = []
    content_notes: List[str] = []
    last_draft = None
    last_critique = None
    trace_id: Optional[str] = None

    inputs = {"task": cfg.task, "config": cfg.model_dump()}

    with trace(name="essay_writer_run", inputs=inputs) as root_run:
        for chunk in graph.stream(
            initial_state,
            runnable_config,
            stream_mode="updates",  # yields {node_name: update_dict} :contentReference[oaicite:3]{index=3}
        ):
            snapshots.append(chunk)

            # Parse {node_name: update_dict}
            node_name = next(iter(chunk.keys()))
            update = chunk[node_name] if isinstance(chunk[node_name], dict) else {}

            # Keep latest fields
            if isinstance(update.get("plan"), str) and update["plan"].strip():
                plan = update["plan"]

            if isinstance(update.get("content"), list):
                content_notes = update["content"]

            if isinstance(update.get("draft"), str) and update["draft"].strip():
                if update["draft"] != last_draft:
                    drafts.append(update["draft"])
                    last_draft = update["draft"]

            if isinstance(update.get("critique"), str) and update["critique"].strip():
                if update["critique"] != last_critique:
                    critiques.append(update["critique"])
                    last_critique = update["critique"]

            # Yield a UI-friendly event
            yield {
                "type": "node_update",
                "node": node_name,
                "update": update,
                "thread_id": thread_id,
            }

        # finalize trace
        root_run.outputs = {"plan": plan, "drafts": len(drafts), "critiques": len(critiques)}
        trace_id = str(root_run.id)

    final_state = {
        "task": cfg.task,
        "plan": plan,
        "content": content_notes,
        "draft": drafts[-1] if drafts else "",
        "critique": critiques[-1] if critiques else "",
        "revision_number": cfg.max_revisions + 1,
        "max_revisions": cfg.max_revisions,
        "config": cfg.model_dump(),
        "thread_id": thread_id,
    }

    return EssayRunResult(
        final_state=final_state,
        snapshots=snapshots,
        drafts=drafts,
        critiques=critiques,
        plan=plan,
        content_notes=content_notes,
        trace_id=trace_id,
    )
