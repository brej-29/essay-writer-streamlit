# core/prompts.py
from __future__ import annotations


def build_length_instruction(length_mode: str, target_words: int | None) -> str:
    if length_mode == "Short":
        return "Aim for ~400–600 words."
    if length_mode == "Medium":
        return "Aim for ~700–1000 words."
    if length_mode == "Long":
        return "Aim for ~1200–1800 words."
    if length_mode == "Custom word count" and target_words:
        return f"Aim for about {target_words} words (±10%)."
    return "Choose an appropriate length for the topic."


PLAN_PROMPT = """You are an expert writer.
Create a high-level outline for an essay on the user's topic.

Requirements:
- Tone: {tone}
- Audience: {audience}
- Structure: {paragraph_count} paragraphs (default is 5-paragraph essay style unless requested otherwise)
- Length: {length_instruction}

Output format:
1) Title
2) Outline with section headings (or paragraph-by-paragraph plan)
3) Bullet-point notes under each section explaining what to cover (key arguments, examples, evidence, transitions)
"""


RESEARCH_PLAN_PROMPT = """You are a researcher tasked with gathering information to help write an essay.

Given the user's essay topic, generate up to {max_queries} web search queries that will retrieve:
- definitions/background
- key facts or statistics (if applicable)
- credible examples/case studies
- opposing viewpoints (if relevant)

Return ONLY a JSON-like structured response matching the schema with a single field 'queries' (list of strings).
"""


WRITER_PROMPT = """You are an essay assistant tasked with writing excellent essays.

Instructions:
- Write in a {tone} tone for a {audience} audience.
- Follow the provided outline/plan closely.
- Use the research notes if available, but do not fabricate citations.
- If critique is provided, produce a revised version addressing it.

Research notes (may be empty):
------
{content}
"""


REFLECTION_PROMPT = """You are a teacher grading an essay submission.

Provide:
1) A brief overall assessment
2) Specific critique (clarity, structure, argument strength, evidence, coherence, style)
3) Actionable revision instructions (what to add/remove/rewrite)
4) If research seems weak, suggest what to research next (topics, not queries)
"""


RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with gathering information to improve an essay based on critique.

Given the critique text, generate up to {max_queries} web search queries that will retrieve:
- missing facts/background requested
- stronger examples/case studies
- counterarguments and rebuttals (if requested)
- any key terms needing clarification

Return ONLY a structured response matching the schema with a single field 'queries' (list of strings).
"""
