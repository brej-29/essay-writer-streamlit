# core/schemas.py
from __future__ import annotations

from typing import List, TypedDict, Optional, Literal
from typing import NotRequired  # Python 3.11+
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    # Core fields used by the LangGraph pipeline
    task: str
    plan: NotRequired[str]
    draft: NotRequired[str]
    critique: NotRequired[str]
    content: NotRequired[List[str]]
    revision_number: NotRequired[int]
    max_revisions: NotRequired[int]

    # Optional: store config/metadata if needed later
    config: NotRequired[dict]


class Queries(BaseModel):
    queries: List[str] = Field(default_factory=list)


LengthMode = Literal["Short", "Medium", "Long", "Custom word count"]


class EssayRunConfig(BaseModel):
    # UI / LLM settings
    model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # Essay settings
    tone: str = "Academic"
    audience: str = "General"
    paragraph_count: int = 5
    length_mode: LengthMode = "Medium"
    target_words: Optional[int] = None

    # Research + revisions
    use_research: bool = True
    max_results: int = 2
    max_revisions: int = 2

    # Debug/UI
    show_intermediates: bool = True

    # Required user input
    task: str
