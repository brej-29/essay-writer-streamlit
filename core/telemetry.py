# core/telemetry.py
from __future__ import annotations

import os
import streamlit as st


def configure_langsmith_from_secrets() -> None:
    """
    Configure LangSmith tracing via environment variables.
    LangChain reads these env vars to enable tracing.
    """
    # Only set if present; don’t crash the UI shell
    for k in ("LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT"):
        if k in st.secrets and str(st.secrets.get(k, "")).strip():
            os.environ[k] = str(st.secrets[k])
