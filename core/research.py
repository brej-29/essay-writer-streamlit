# core/research.py
from __future__ import annotations

import logging
from typing import List, Dict, Any

import streamlit as st
from tavily import TavilyClient

from core.config import get_secret, MissingSecretError

logger = logging.getLogger("essay_writer.research")


class ResearchError(RuntimeError):
    """Raised for Tavily-related failures we want to surface nicely in the UI."""


def _format_tavily_response_to_notes(resp: Dict[str, Any]) -> List[str]:
    """
    Convert Tavily response dict into a list of readable research notes.
    We keep it simple: include the answer if present + top results with title/url/content.
    """
    notes: List[str] = []

    # Optional: "answer" is a short synthesized response if include_answer=True
    answer = resp.get("answer")
    if isinstance(answer, str) and answer.strip():
        notes.append(f"**Summary answer:** {answer.strip()}")

    results = resp.get("results", [])
    if isinstance(results, list):
        for i, r in enumerate(results, start=1):
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            content = (r.get("content") or "").strip()

            chunk = f"**Result {i}: {title or 'Untitled'}**\n"
            if url:
                chunk += f"- URL: {url}\n"
            if content:
                chunk += f"- Notes: {content}\n"
            notes.append(chunk.strip())

    return notes


@st.cache_data(ttl=60 * 60, show_spinner=False)
def tavily_search_cached(query: str, max_results: int) -> Dict[str, Any]:
    """
    Cached Tavily search call.
    Streamlit caches the returned data object to speed up reruns and avoid duplicate API calls.
    """
    api_key = get_secret("TAVILY_API_KEY", required=True)
    client = TavilyClient(api_key=api_key)

    # Tavily docs emphasize setting max_results manually; include_answer adds a concise summary.
    resp = client.search(
        query=query,
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
    )
    if not isinstance(resp, dict):
        raise ResearchError("Unexpected Tavily response type (expected dict).")
    return resp


def run_tavily_search(query: str, max_results: int = 2) -> List[str]:
    """
    Safe wrapper: validates inputs + catches common failures to show clean UI errors.
    Returns formatted notes list.
    """
    q = (query or "").strip()
    if not q:
        return []

    if not isinstance(max_results, int) or max_results < 1 or max_results > 10:
        raise ValueError("max_results must be an integer between 1 and 10.")

    try:
        resp = tavily_search_cached(q, max_results)
        notes = _format_tavily_response_to_notes(resp)
        return notes

    except MissingSecretError as e:
        # Missing Tavily key (local secrets/cloud secrets not set)
        raise ResearchError(str(e)) from e

    except Exception as e:
        # Common Tavily failures: Unauthorized, rate limits, network issues, etc.
        logger.exception("Tavily search failed")
        raise ResearchError(f"Tavily search failed: {e}") from e
