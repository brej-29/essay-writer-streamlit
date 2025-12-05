# core/feedback.py
from __future__ import annotations

from langsmith import Client

from core.config import get_secret, MissingSecretError


class FeedbackError(RuntimeError):
    pass


def submit_langsmith_feedback(
    *,
    run_id: str,
    score: int,
    comment: str | None,
    key: str = "user_helpfulness",
) -> None:
    """
    Logs feedback for a LangSmith run using Client.create_feedback(). :contentReference[oaicite:4]{index=4}
    """
    try:
        api_key = get_secret("LANGCHAIN_API_KEY", required=True)
    except MissingSecretError as e:
        raise FeedbackError(str(e)) from e

    client = Client(api_key=api_key)

    # Use both run_id and trace_id as run_id for the root run (safe + recommended pattern). :contentReference[oaicite:5]{index=5}
    client.create_feedback(
        run_id=run_id,
        trace_id=run_id,
        key=key,
        score=float(score),
        comment=(comment or "").strip() or None,
    )
