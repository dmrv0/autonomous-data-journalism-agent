from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.llm import content_to_text, get_llm

SYSTEM = """You are a senior data journalist for a Reuters-style wire service.

Rules:
- Every numerical claim MUST come from the provided analysis stdout. Never invent figures.
- Never introduce proper nouns, names, or attributions (artists, directors, studios, people, places) that do not appear in the analysis stdout — even if you believe you know them. If the output lists only a title, refer only to that title.
- Name each metric by what it ACTUALLY measures in the data. If a ranking is by revenue, say "highest-grossing" or "second-highest by revenue" — never "most-streamed", "most-viewed", or "best-selling" unless the data literally measures streams/views/sales.
- Cite ranks with ordinal phrasing tied to that metric ("the second-highest-grossing film").
- When mentioning correlations, state the direction explicitly ("positive correlation", "negative correlation"). You may round to two decimals (0.73).
- You may format figures for readability ($2.9 billion or $2,923,706,026), but the underlying value must match stdout.
- Tone: factual, concise, news-style. No hype. No editorializing.
- Structure: HEADLINE on the first line, then a lede paragraph, 3-5 body paragraphs, then a closing line.
- Output the article only, no commentary or markdown.
"""

USER_TEMPLATE = """Analysis stdout from the data team:

{analyses}

{revision_notes}

Draft the article.
"""


def journalist_node(state: AgentState) -> dict:
    results = state["analysis_results"]
    if not any(r["stdout"].strip() for r in results):
        return {
            "article_draft": "INSUFFICIENT DATA: the analysis produced no output, "
            "so there are no verified findings to report."
        }
    llm = get_llm(temperature=0.4)
    analyses = "\n\n---\n\n".join(
        f"Script:\n{r['script']}\n\nStdout:\n{r['stdout']}"
        for r in state["analysis_results"]
    )
    revision_notes = ""
    if state.get("evaluation_errors"):
        revision_notes = (
            "PREVIOUS DRAFT HAD ERRORS:\n"
            + "\n".join(f"- {e}" for e in state["evaluation_errors"])
            + "\nRewrite to remove these errors. Use only verified figures."
        )
    user = USER_TEMPLATE.format(analyses=analyses, revision_notes=revision_notes)
    response = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)])
    return {"article_draft": content_to_text(response.content).strip()}
