# core/bundle_zip.py
from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict, List
import zipfile


def _norm(s: str) -> str:
    return (s or "").replace("\r\n", "\n").strip()


def build_run_bundle_zip(
    *,
    essay_result: Dict[str, Any],
    run_config: Dict[str, Any] | None = None,
) -> bytes:
    """
    Create a ZIP bundle bytes containing:
      - plan.md
      - research_notes.md
      - drafts/draft_v1.md ...
      - critiques/critique_v1.md ...
      - final.md
      - config.json (optional)
      - metadata.json (trace_id, etc)
    """
    plan = _norm(essay_result.get("plan", ""))
    notes: List[str] = essay_result.get("content_notes", []) or []
    drafts: List[str] = essay_result.get("drafts", []) or []
    critiques: List[str] = essay_result.get("critiques", []) or []
    final = _norm(essay_result.get("final", "")) or (_norm(drafts[-1]) if drafts else "")

    trace_id = essay_result.get("trace_id")

    # Build markdown content
    notes_md = "\n\n---\n\n".join([_norm(n) for n in notes if _norm(n)])

    metadata = {
        "trace_id": trace_id,
        "drafts_count": len(drafts),
        "critiques_count": len(critiques),
        "notes_count": len(notes),
    }

    # In-memory ZIP
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("plan.md", plan or "# Plan\n\n(No plan captured)\n")
        zf.writestr("research_notes.md", notes_md or "# Research notes\n\n(No research notes)\n")
        zf.writestr("final.md", final or "# Final\n\n(No final essay)\n")

        # Drafts
        if drafts:
            for i, d in enumerate(drafts, start=1):
                zf.writestr(f"drafts/draft_v{i}.md", _norm(d) or "(empty)")

        # Critiques
        if critiques:
            for i, c in enumerate(critiques, start=1):
                zf.writestr(f"critiques/critique_v{i}.md", _norm(c) or "(empty)")

        # Optional config + metadata
        if run_config is not None:
            zf.writestr("config.json", json.dumps(run_config, indent=2, ensure_ascii=False))
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))

        # Small readme
        zf.writestr(
            "README.txt",
            "Essay Writer Run Bundle\n"
            "- plan.md: outline\n"
            "- research_notes.md: Tavily notes (if enabled)\n"
            "- drafts/: versioned drafts\n"
            "- critiques/: versioned critiques\n"
            "- final.md: final essay\n"
            "- config.json: run configuration (if available)\n"
            "- metadata.json: trace_id and counts\n"
        )

    return buf.getvalue()
