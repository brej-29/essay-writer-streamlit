# app.py
from __future__ import annotations

import logging
import streamlit as st

from core.config import get_secret, MissingSecretError
from core.telemetry import configure_langsmith_from_secrets
from core.schemas import EssayRunConfig
from core.research import run_tavily_search, ResearchError
from core.graph import run_essay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("essay_writer")

st.set_page_config(page_title="Essay Writer", page_icon="📝", layout="wide")

# --- Configure tracing (LangSmith) from secrets (optional at this stage)
configure_langsmith_from_secrets()

st.title("📝 Essay Writer (LangGraph + Research + Revisions)")
st.caption("Step 1: UI shell + secrets setup. Pipeline wiring comes in Step 4.")

# --- Sidebar controls
with st.sidebar:
    st.header("Settings")

    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-5-nano", "gpt-4.1-nano", "gpt-4.1-mini"],
        index=0,
        help="Default is cost-effective. You can switch to higher quality models if needed."
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.divider()
    st.subheader("Essay")
    tone = st.selectbox("Tone", ["Academic", "Formal", "Conversational", "Persuasive"], index=0)
    audience = st.selectbox("Audience", ["General", "Undergraduate", "Executive", "Technical"], index=0)
    paragraph_count = st.number_input("Paragraphs", min_value=3, max_value=12, value=5, step=1)

    length_mode = st.selectbox("Length", ["Short", "Medium", "Long", "Custom word count"], index=1)
    target_words = None
    if length_mode == "Custom word count":
        target_words = st.number_input("Target words", min_value=150, max_value=3000, value=800, step=50)

    st.divider()
    st.subheader("Research + Revisions")
    use_research = st.toggle("Use web research (Tavily)", value=True)
    max_results = st.slider("Max results per query", 1, 5, 2)
    max_revisions = st.slider("Max revisions", 0, 5, 2)

    st.divider()
    st.subheader("Debug")
    show_intermediates = st.toggle("Show intermediate outputs", value=True)

# --- Main input form (batched submit)
with st.form("essay_form"):
    task = st.text_area(
        "Essay topic / task",
        placeholder="Example: Explain the impact of electric vehicles on urban air quality in India...",
        height=140,
    )
    submitted = st.form_submit_button("Generate Essay")

# --- Validate secrets minimally (so users don’t discover later)
def validate_secrets() -> list[str]:
    missing = []
    for k in ["OPENAI_API_KEY"]:
        try:
            _ = get_secret(k, required=True)
        except MissingSecretError:
            missing.append(k)

    if use_research:
        try:
            _ = get_secret("TAVILY_API_KEY", required=True)
        except MissingSecretError:
            missing.append("TAVILY_API_KEY")

    return missing

if submitted:
    if not task.strip():
        st.error("Please enter an essay topic / task.")
        st.stop()

    missing = validate_secrets()
    if missing:
        st.error(
            "Missing required secrets: " + ", ".join(missing) +
            ". Add them to `.streamlit/secrets.toml` (local) or Streamlit Cloud secrets."
        )
        st.stop()

    # Store config in session state (we’ll use it in Step 4 when wiring LangGraph)
    raw_config = {
        "model": model,
        "temperature": temperature,
        "tone": tone,
        "audience": audience,
        "paragraph_count": paragraph_count,
        "length_mode": length_mode,
        "target_words": target_words,
        "use_research": use_research,
        "max_results": max_results,
        "max_revisions": max_revisions,
        "show_intermediates": show_intermediates,
        "task": task,
    }

    try:
        validated = EssayRunConfig(**raw_config)
    except Exception as e:
        st.error(f"Config validation failed: {e}")
        st.stop()

    st.session_state["run_config"] = validated.model_dump()

    st.info("Running the essay pipeline...")

    try:
        result = run_essay(st.session_state["run_config"])
    except Exception as e:
        st.error(f"Essay generation failed: {e}")
        st.stop()

    # Store for later (including LangSmith trace id)
    st.session_state["essay_result"] = {
        "plan": result.plan,
        "content_notes": result.content_notes,
        "drafts": result.drafts,
        "critiques": result.critiques,
        "final": (result.drafts[-1] if result.drafts else ""),
        "trace_id": result.trace_id,
    }

    st.success("Done ✅")

if "essay_result" in st.session_state:
    data = st.session_state["essay_result"]

    tabs = st.tabs(["Outline", "Research notes", "Drafts", "Critiques", "Final + Download", "Debug"])
    with tabs[0]:
        st.subheader("Outline / Plan")
        st.write(data.get("plan", ""))

    with tabs[1]:
        st.subheader("Research notes")
        notes = data.get("content_notes", [])
        if not notes:
            st.info("No research notes (research disabled or none returned).")
        else:
            for n in notes:
                st.markdown(n)

    with tabs[2]:
        st.subheader("Draft versions")
        drafts = data.get("drafts", [])
        if not drafts:
            st.info("No drafts produced.")
        else:
            for i, d in enumerate(drafts, start=1):
                st.markdown(f"### Draft v{i}")
                st.write(d)

    with tabs[3]:
        st.subheader("Critiques")
        critiques = data.get("critiques", [])
        if not critiques:
            st.info("No critiques produced (max revisions may be 0).")
        else:
            for i, c in enumerate(critiques, start=1):
                st.markdown(f"### Critique v{i}")
                st.write(c)

    with tabs[4]:
        st.subheader("Final Essay")
        final = data.get("final", "")
        st.write(final)

        st.download_button(
            "Download as Markdown",
            data=final,
            file_name="essay.md",
            mime="text/markdown",
        )

        trace_id = data.get("trace_id")
        if trace_id:
            st.caption(f"LangSmith trace_id captured: {trace_id}")

    with tabs[5]:
        st.subheader("Debug")
        st.json(data)
