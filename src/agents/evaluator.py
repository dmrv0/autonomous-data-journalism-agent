from __future__ import annotations
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.llm import content_to_text, get_llm

MAX_REVISIONS = 3

SEMANTIC_SYSTEM = """You are a meticulous fact-checking editor for a wire service.

You are given a draft ARTICLE and the ANALYSIS OUTPUT (raw stdout from a trusted
Python data pipeline). The ANALYSIS OUTPUT is the ONLY source of truth.

Flag a claim ONLY when it clearly contradicts the ANALYSIS OUTPUT or has no
support in it. Be conservative and precise: a false alarm forces a needless,
expensive rewrite, so when in doubt, do NOT flag.

NUMBERS — do NOT flag a figure if it can be obtained from an output value by:
- Rounding to fewer decimals or significant figures (0.73 IS correct for 0.7317;
  0.74 IS correct for 0.7374; 6.4 IS correct for 6.39).
- Standard formatting: currency symbols, thousands separators, or magnitude
  words. "$2.9 billion", "$2,923,706,026", and 2923706026 are the SAME number.
Only flag a number that cannot be produced this way from ANY output value.

DO flag:
- A metric described as something it is not — e.g. calling revenue rankings
  "most-streamed", "most-viewed", or "best-selling" when the output measures revenue.
- Correlation directions stated backwards, or a correlation attributed to the
  wrong pair of columns.
- Ranking/superlative claims the output's ordered lists do not support.
- Trend claims ("steady rise", "consistent upward trend") with no supporting
  series or correlation in the output.
- Attributing a data point to a specific person, artist, company, or place that
  does NOT appear anywhere in the output — e.g. naming the artist of a song when
  the output lists only the song title and a number. (Do NOT flag generic
  descriptive nouns like "the film industry" or "global box office" — only
  specific factual attributions tied to a ranked/listed data point.)
- COMMON-SENSE / PLAUSIBILITY: a finding that is clearly a data-cleaning
  artifact rather than a real insight, EVEN IF the number matches the output.
  Examples: a median or mean rating at or near 0 for a major group (a sign
  unrated titles were not filtered out), a "trend" anchored on an impossible or
  placeholder year (e.g. 1800 — before cinema existed), an average feature-film
  runtime of only a few minutes (shorts/episodes not filtered out), a 100%/0%
  or otherwise extreme per-group rate backed by only a handful of rows (e.g.
  "100% of films from a country with 2 entries"), or an extreme figure driven by
  a single junk row. Flag these as implausible and ask for the metric to be
  recomputed on credible, filtered rows.

Before flagging, re-read the exact column/metric names in the output and identify
the specific output line you are contradicting. Keep each issue atomic, specific,
and independently verifiable.

Respond with ONLY a JSON object, no prose, of the form:
{"supported": true}
or
{"supported": false, "issues": ["<specific problem, quoting the article claim and the exact output line that contradicts it>", ...]}
"""

SEMANTIC_USER = """ARTICLE:
{article}

ANALYSIS OUTPUT (ground truth):
{stdout}
"""


def parse_llm_verdict(text: str) -> list[str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        verdict = json.loads(match.group(0))
    except (ValueError, TypeError):
        return []
    if verdict.get("supported") is True:
        return []
    issues = verdict.get("issues") or []
    return [str(issue) for issue in issues if str(issue).strip()]


def verify_semantic(article: str, stdouts: list[str], llm=None) -> list[str]:
    if not article.strip() or not any(s.strip() for s in stdouts):
        return []
    llm = llm or get_llm(temperature=0.0)
    user = SEMANTIC_USER.format(article=article, stdout="\n".join(stdouts))
    try:
        response = llm.invoke(
            [SystemMessage(content=SEMANTIC_SYSTEM), HumanMessage(content=user)]
        )
    except Exception:
        return []
    return parse_llm_verdict(content_to_text(response.content))

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_POS_CORR_RE = re.compile(
    r"(positive(?:ly)?\s+(?:correlat|associat|relat))",
    re.IGNORECASE,
)
_NEG_CORR_RE = re.compile(
    r"(negative(?:ly)?\s+(?:correlat|associat|relat)|inverse(?:ly)?\s+(?:correlat|relat))",
    re.IGNORECASE,
)
_SUPERLATIVE_RE = re.compile(
    r"(?P<entity>[A-Z][\w\.&' -]{1,40}?)\s+is\s+the\s+most\s+(?P<metric>[a-z]+)",
    re.IGNORECASE,
)


def _normalize_num(token: str) -> str:
    return token.replace(",", "")


def verify_values(article: str, stdouts: list[str]) -> list[str]:
    errors: list[str] = []
    joined = "\n".join(stdouts)
    stdout_nums = {_normalize_num(t) for t in _NUMBER_RE.findall(joined)}
    for token in _NUMBER_RE.findall(article):
        norm = _normalize_num(token)
        if len(norm) < 2:
            continue
        if norm in stdout_nums:
            continue
        try:
            val = float(norm)
        except ValueError:
            continue
        tolerance = max(abs(val) * 0.01, 0.01)
        if not any(
            abs(val - float(s)) <= tolerance
            for s in stdout_nums
            if _is_floatable(s)
        ):
            errors.append(f"Value '{token}' not found in stdout (no match within 1% tolerance)")
    return errors


def _is_floatable(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def verify_signs(article: str, stdouts: list[str]) -> list[str]:
    errors: list[str] = []
    joined = "\n".join(stdouts)
    pos_in_article = bool(_POS_CORR_RE.search(article))
    neg_in_article = bool(_NEG_CORR_RE.search(article))

    corr_matches = re.findall(r"corr[^=]*=\s*(-?\d*\.?\d+)", joined, re.IGNORECASE)
    if not corr_matches:
        return errors
    signs = {("neg" if float(v) < 0 else "pos") for v in corr_matches}

    if pos_in_article and "pos" not in signs:
        errors.append("Article claims positive correlation but stdout shows none.")
    if neg_in_article and "neg" not in signs:
        errors.append("Article claims negative correlation but stdout shows none.")
    return errors


def verify_ranks(article: str, stdouts: list[str]) -> list[str]:
    errors: list[str] = []
    joined = "\n".join(stdouts)
    matches = list(_SUPERLATIVE_RE.finditer(article))
    if not matches:
        return errors
    list_pattern = re.compile(r"\[\s*\(([^)]+)\)")
    first_items = list_pattern.findall(joined)
    if not first_items:
        return errors
    top_names = []
    for item in first_items:
        parts = [p.strip().strip("'\"") for p in item.split(",")]
        if parts:
            top_names.append(parts[0].lower())
    for m in matches:
        entity = m.group("entity").strip().lower()
        if entity not in top_names:
            errors.append(
                f"Article claims '{m.group('entity')}' is the most {m.group('metric')}, "
                f"but stdout's top entries are: {top_names}"
            )
    return errors


def select_blocking_errors(
    value_errors: list[str],
    sign_errors: list[str],
    rank_errors: list[str],
    semantic_errors: list[str],
) -> list[str]:
    blocking: list[str] = []
    for err in [*sign_errors, *rank_errors, *semantic_errors]:
        if err not in blocking:
            blocking.append(err)
    return blocking


def evaluator_node(state: AgentState, llm=None) -> dict:
    stdouts = [r["stdout"] for r in state["analysis_results"]]
    article = state["article_draft"]
    if not any(s.strip() for s in stdouts):
        errors = [
            "The analysis produced no output (empty stdout), so no claim in the "
            "article can be grounded in a computed statistic. Re-run the analysis "
            "so it prints real results before drafting."
        ]
    else:
        errors = select_blocking_errors(
            value_errors=verify_values(article, stdouts),
            sign_errors=verify_signs(article, stdouts),
            rank_errors=verify_ranks(article, stdouts),
            semantic_errors=verify_semantic(article, stdouts, llm=llm),
        )
    revision_count = state.get("revision_count", 0)
    if not errors:
        return {"evaluation_errors": [], "final_article": article}
    if revision_count + 1 >= MAX_REVISIONS:
        tagged = article + "\n\n[unverified] " + " ".join(errors)
        return {
            "evaluation_errors": errors,
            "revision_count": revision_count + 1,
            "final_article": tagged,
        }
    return {
        "evaluation_errors": errors,
        "revision_count": revision_count + 1,
    }


def evaluator_route(state: AgentState) -> str:
    if state.get("final_article"):
        return "end"
    return "analyst"
