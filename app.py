# app.py
from __future__ import annotations

import logging
import streamlit as st

from core.config import get_secret, MissingSecretError
from core.telemetry import configure_langsmith_from_secrets
from core.schemas import EssayRunConfig
from core.research import run_tavily_search, ResearchError

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

    st.success("Config captured ✅ (Next steps will wire the LangGraph pipeline.)")
    st.json(st.session_state["run_config"])
    
st.divider()
st.subheader("🔎 Research Smoke Test (Step 3)")

if use_research:
    test_query = st.text_input(
        "Test search query",
        value="Latest trends in electric vehicles and urban air quality in India",
        help="This is only for validating Tavily connectivity and formatting."
    )

    if st.button("Run Tavily test search"):
        try:
            notes = run_tavily_search(test_query, max_results=max_results)
            if not notes:
                st.warning("No notes returned.")
            else:
                st.success(f"Got {len(notes)} note blocks.")
                for n in notes:
                    st.markdown(n)
        except ResearchError as e:
            st.error(str(e))
else:
    st.info("Enable 'Use web research (Tavily)' in the sidebar to test.")
