# core/config.py
from __future__ import annotations

import streamlit as st


class MissingSecretError(RuntimeError):
    pass


def get_secret(key: str, *, required: bool = True, default: str | None = None) -> str | None:
    """
    Read secrets from Streamlit native secrets manager (st.secrets).
    Raises a clean error if missing and required=True.
    """
    try:
        value = st.secrets.get(key, default)
    except Exception as e:
        # If secrets are misconfigured, st.secrets access can fail
        raise MissingSecretError(
            f"Unable to read Streamlit secrets. Ensure .streamlit/secrets.toml exists locally "
            f"or secrets are set in deployment. Missing: {key}"
        ) from e

    if required and (value is None or str(value).strip() == ""):
        raise MissingSecretError(
            f"Missing secret: {key}. Add it to .streamlit/secrets.toml (local) or Streamlit Cloud secrets."
        )
    return value
