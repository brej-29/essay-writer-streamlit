# app.py
from __future__ import annotations

import logging
import streamlit as st
import uuid
from datetime import datetime

from core.config import get_secret, MissingSecretError
from core.telemetry import configure_langsmith_from_secrets
from core.schemas import EssayRunConfig
from core.graph import run_essay_stream
from core.feedback import submit_langsmith_feedback, FeedbackError
from core.exporters import build_export_bundle, ExportError
from core.bundle_zip import build_run_bundle_zip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("essay_writer")

st.set_page_config(page_title="Essay Writer", page_icon="📝", layout="wide")
# --- Session init (persists across reruns for this user session)
if "run_history" not in st.session_state:
    st.session_state["run_history"] = []  # list[dict]

if "selected_run_id" not in st.session_state:
    st.session_state["selected_run_id"] = None

if "live_ui_state" not in st.session_state:
    st.session_state["live_ui_state"] = {}
# --- Configure tracing (LangSmith) from secrets (optional at this stage)
configure_langsmith_from_secrets()

st.title("📝 Essay Writer (LangGraph + Research + Revisions)")
st.caption("Research-assisted essay generator with revision loop + LangSmith tracing & feedback.")

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
    show_intermediates = st.toggle("Show Complete process JSON", value=True)
    st.divider()
    st.header("Run history")

    history = st.session_state.get("run_history", [])
    if not history:
        st.caption("No runs yet.")
    else:
        # Newest first
        options = [r["run_id"] for r in history]

        def _label(run_id: str) -> str:
            r = next((x for x in history if x["run_id"] == run_id), None)
            if not r:
                return run_id
            t = r.get("ts", "")
            task = (r.get("task", "") or "").strip().replace("\n", " ")
            return f"{t} — {task[:50]}{'…' if len(task) > 50 else ''}"

        selected = st.selectbox(
            "Select a previous run",
            options=options,
            index=0,
            format_func=_label,
            key="history_select",
        )

        cols = st.columns(2)
        with cols[0]:
            if st.button("Load run", key="load_run_btn"):
                chosen = next(r for r in history if r["run_id"] == selected)
                st.session_state["essay_result"] = chosen["essay_result"]
                st.session_state["run_config"] = chosen.get("config", st.session_state.get("run_config", {}))
                st.session_state["selected_run_id"] = chosen["run_id"]
                st.success("Loaded ✅")

        with cols[1]:
            if st.button("Clear history", key="clear_history_btn"):
                st.session_state["run_history"] = []
                st.session_state["selected_run_id"] = None
                st.session_state.pop("essay_result", None)
                st.success("Cleared ✅")

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
    
    status = st.status("🚀 Running essay pipeline...", expanded=False)

    # Unique run id for widget keys + history
    ui_run_id = str(uuid.uuid4())

    # Delta tracking to prevent flooding and show only new notes
    st.session_state["live_ui_state"][ui_run_id] = {
        "notes_idx": 0,        # how many notes we have already rendered
        "notes_rendered": [],  # rendered notes (capped)
        "drafts_seen": [],     # drafts (capped)
        "critiques_seen": [],  # critiques (capped)
    }

    live_plan = st.expander("🧠 Plan (live)", expanded=True)
    live_research = st.expander("🔎 Research notes (live)", expanded=True)
    live_draft = st.expander("✍️ Draft (live)", expanded=True)
    live_critique = st.expander("🧑‍🏫 Critique (live)", expanded=True)

    with live_plan:
        plan_ph = st.empty()

    with live_research:
        research_meta_ph = st.empty()
        research_ph = st.empty()

    with live_draft:
        draft_meta_ph = st.empty()
        draft_ph = st.empty()

    with live_critique:
        critique_meta_ph = st.empty()
        critique_ph = st.empty()

    node_to_label = {
        "planner": "🧠 Planning outline...",
        "research_plan": "🔎 Researching for plan...",
        "generate": "✍️ Writing / revising draft...",
        "reflect": "🧑‍🏫 Critiquing draft...",
        "research_critique": "🔎 Researching to address critique...",
    }

    try:
        gen = run_essay_stream(st.session_state["run_config"])
        result = None
        while True:
            try:
                event = next(gen)  # advance generator manually
            except StopIteration as si:
                result = si.value  # <-- EssayRunResult is here
                break

            if event.get("type") != "node_update":
                continue

            node = event.get("node", "unknown")
            update = event.get("update", {}) or {}

            status.update(label=node_to_label.get(node, f"Running: {node}"), state="running")

            if "plan" in update and isinstance(update["plan"], str):
                plan_ph.subheader("Outline (live)")
                plan_ph.write(update["plan"])

            if "content" in update and isinstance(update["content"], list):
                ui = st.session_state["live_ui_state"][ui_run_id]
                all_notes = update["content"]

                start = ui["notes_idx"]
                new_notes = all_notes[start:] if start < len(all_notes) else []

                if new_notes:
                    ui["notes_idx"] = len(all_notes)

                    # Append and cap to last 30 to avoid UI flooding
                    ui["notes_rendered"].extend(new_notes)
                    ui["notes_rendered"] = ui["notes_rendered"][-30:]

                    research_meta_ph.caption(
                        f"Total notes: {len(all_notes)} | Showing last {len(ui['notes_rendered'])} | Newly added: {len(new_notes)}"
                    )

                    # Replace one placeholder (no duplicates)
                    research_ph.markdown("\n\n---\n\n".join(ui["notes_rendered"]))

            if "draft" in update and isinstance(update["draft"], str):
                ui = st.session_state["live_ui_state"][ui_run_id]
                d = update["draft"].strip()
                if d:
                    if not ui["drafts_seen"] or ui["drafts_seen"][-1] != d:
                        ui["drafts_seen"].append(d)
                        ui["drafts_seen"] = ui["drafts_seen"][-5:]  # keep last 5 versions

                    draft_meta_ph.caption(f"Draft versions captured (live): {len(ui['drafts_seen'])}")
                    draft_ph.write(d[:2500] + ("..." if len(d) > 2500 else ""))

            if "critique" in update and isinstance(update["critique"], str):
                ui = st.session_state["live_ui_state"][ui_run_id]
                c = update["critique"].strip()
                if c:
                    if not ui["critiques_seen"] or ui["critiques_seen"][-1] != c:
                        ui["critiques_seen"].append(c)
                        ui["critiques_seen"] = ui["critiques_seen"][-5:]

                    critique_meta_ph.caption(f"Critiques captured (live): {len(ui['critiques_seen'])}")
                    critique_ph.write(c[:2500] + ("..." if len(c) > 2500 else ""))

        if result is None:
            raise RuntimeError("run_essay_stream finished but returned no result (unexpected).")

    except Exception as e:
        status.update(label="❌ Failed", state="error", expanded=True)
        st.error(f"Essay generation failed: {e}")
        st.stop()

    status.update(label="✅ Completed", state="complete", expanded=False)

    st.session_state["essay_result"] = {
        "plan": result.plan,
        "content_notes": result.content_notes,
        "drafts": result.drafts,
        "critiques": result.critiques,
        "final": (result.drafts[-1] if result.drafts else ""),
        "trace_id": result.trace_id,
    }

    try:
        title = "Essay"
        final_text = st.session_state["essay_result"].get("final", "")
        bundle = build_export_bundle(title=title, essay_text=final_text)

        st.session_state["essay_result"]["exports"] = {
            "essay.md": bundle.md,
            "essay.txt": bundle.txt,
            "essay.docx": bundle.docx,
            "essay.pdf": bundle.pdf,
        }
    except ExportError as e:
        st.session_state["essay_result"]["exports_error"] = str(e)

    essay_result_slim = dict(st.session_state["essay_result"])
    essay_result_slim.pop("exports", None)
    essay_result_slim.pop("exports_error", None)

    run_record = {
        "run_id": essay_result_slim.get("trace_id") or ui_run_id,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": st.session_state["run_config"]["task"],
        "config": st.session_state["run_config"],
        "essay_result": essay_result_slim,
    }

    st.session_state["run_history"].insert(0, run_record)
    st.session_state["run_history"] = st.session_state["run_history"][:15]  # keep last 15 runs
    st.session_state["selected_run_id"] = run_record["run_id"]


    st.success("Done ✅")

if "essay_result" in st.session_state:
    data = st.session_state["essay_result"]

    cfg = st.session_state.get("run_config", {})
    show_debug = bool(cfg.get("show_intermediates", True))

    tab_labels = ["Outline", "Research notes", "Drafts", "Critiques", "Final + Download"]
    if show_debug:
        tab_labels.append("Debug")

    tabs = st.tabs(tab_labels)

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
                with st.expander(f"Draft v{i}", expanded=(i == len(drafts))):
                    st.write(d)

    with tabs[3]:
        st.subheader("Critiques")
        critiques = data.get("critiques", [])
        if not critiques:
            st.info("No critiques produced (max revisions may be 0).")
        else:
            for i, c in enumerate(critiques, start=1):
                with st.expander(f"Critique v{i}", expanded=(i == len(critiques))):
                    st.write(c)

    with tabs[4]:
        st.subheader("Final Essay")
        final = data.get("final", "")
        st.write(final)

        exports_error = data.get("exports_error")
        exports = data.get("exports", {})

        if exports_error:
            st.warning(f"Export generation issue: {exports_error}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.download_button(
                "Download .md",
                data=exports.get("essay.md", final.encode("utf-8")),
                file_name="essay.md",
                mime="text/markdown",
            )

        with col2:
            st.download_button(
                "Download .txt",
                data=exports.get("essay.txt", final.encode("utf-8")),
                file_name="essay.txt",
                mime="text/plain",
            )

        with col3:
            st.download_button(
                "Download .docx",
                data=exports.get("essay.docx", b""),
                file_name="essay.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                disabled=("essay.docx" not in exports),
            )

        with col4:
            st.download_button(
                "Download .pdf",
                data=exports.get("essay.pdf", b""),
                file_name="essay.pdf",
                mime="application/pdf",
                disabled=("essay.pdf" not in exports),
            )
        
        st.divider()
        st.subheader("Run bundle (ZIP)")

        # Build once per displayed run (keep in memory; Streamlit stores download data in-memory) :contentReference[oaicite:2]{index=2}
        if "bundle_zip" not in data:
            data["bundle_zip"] = build_run_bundle_zip(
                essay_result=data,
                run_config=st.session_state.get("run_config", {}),
            )

        zip_bytes = data["bundle_zip"]
        zip_name = "essay_run_bundle.zip"

        st.download_button(
            "Download full run bundle (.zip)",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
        )
        st.caption("Includes plan + research notes + drafts + critiques + final (+ config/metadata).")


        st.divider()
        st.subheader("Feedback")

        trace_id = data.get("trace_id")
        if not trace_id:
            st.info("No LangSmith run id available for feedback.")
        else:
            rating = st.radio("Was this essay helpful?", ["👍 Yes", "👎 No"], horizontal=True)
            comment = st.text_area("Optional feedback (what to improve?)", height=100)

            if st.button("Submit feedback to LangSmith"):
                try:
                    score = 1 if rating.startswith("👍") else -1
                    submit_langsmith_feedback(run_id=trace_id, score=score, comment=comment)
                    st.success("Feedback submitted ✅")
                except FeedbackError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Failed to submit feedback: {e}")

    if show_debug:
        with tabs[-1]:
            st.subheader("Debug")
            debug = dict(data)
            if "exports" in debug:
                debug["exports"] = {k: f"{len(v)} bytes" for k, v in (debug.get("exports") or {}).items()}
            if "bundle_zip" in debug:
                debug["bundle_zip"] = f"{len(debug['bundle_zip'])} bytes"
            st.json(debug)
