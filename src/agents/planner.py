from __future__ import annotations
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.llm import content_to_text, get_llm

MIN_IDEAS = 2
MAX_IDEAS = 5

SYSTEM = """You are the planning editor of a data-driven newsroom.

Given the SCHEMA of one or more datasets (table names, columns, dtypes, row
counts), propose the most NEWSWORTHY investigation questions a data analyst
should run to surface a story from THIS specific data.

Rules:
- Propose between 3 and 5 questions. Each must be answerable using ONLY the
  columns shown. Do NOT invent columns that are not listed.
- Name the EXACT columns each question uses.
- Favor questions that yield rankings, comparisons between groups, correlations
  between named numeric columns, or trends over a time/year column — whichever
  the available columns actually support.
- Be specific to THIS data, not generic. "Rank directors by median revenue
  across their films" beats "find the top movies".
- Each question must be a single self-contained instruction that ends in
  concrete printed numbers (ranked tuples, correlation values, group aggregates).
- Do NOT include a data-cleaning step; that is handled separately.
- Output ONLY a JSON array of strings. No prose, no object keys, no markdown.
"""

USER_TEMPLATE = """SCHEMA:
{schema}

Propose the investigation questions as a JSON array of strings.
"""


def _format_schema(files: list[dict]) -> str:
    blocks = []
    for f in files:
        cols = f.get("analysis_columns") or f.get("columns", [])
        dtypes = f.get("dtypes", {})
        col_lines = ", ".join(f"{c} ({dtypes.get(c, '?')})" for c in cols)
        blocks.append(
            f"- {f.get('name', f.get('path', 'table'))} "
            f"({f.get('n_rows', '?')} rows)\n  columns: {col_lines}"
        )
    return "\n".join(blocks)


def _parse_ideas(text: str) -> list[str]:
    ideas: list[str] = []
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError):
            data = None
        if isinstance(data, list):
            ideas = [str(x).strip() for x in data if str(x).strip()]

    if not ideas:
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if len(cleaned) >= 10:
                ideas.append(cleaned)

    seen: set[str] = set()
    unique: list[str] = []
    for idea in ideas:
        key = idea.lower()
        if key not in seen:
            seen.add(key)
            unique.append(idea)
    return unique[:MAX_IDEAS]


def planner_node(state: AgentState) -> dict:
    schema = state.get("schema") or {}
    files = schema.get("files") or []
    if not files:
        return {}

    fallback_plan = state.get("analysis_plan") or []
    cleaning_step = fallback_plan[0] if fallback_plan else None

    user = USER_TEMPLATE.format(schema=_format_schema(files))
    try:
        llm = get_llm(temperature=0.4)
        response = llm.invoke(
            [SystemMessage(content=SYSTEM), HumanMessage(content=user)]
        )
        ideas = _parse_ideas(content_to_text(response.content))
    except Exception:
        return {}

    if len(ideas) < MIN_IDEAS:
        return {}

    plan = ([cleaning_step] if cleaning_step else []) + ideas
    return {"analysis_plan": plan}
