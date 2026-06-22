from __future__ import annotations
import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.sandbox import Sandbox
from src.llm import content_to_text, get_llm

SYSTEM = """You are a senior data analyst. You write short, focused Python scripts that print results clearly.

Rules:
- Use only pandas, numpy, scipy.
- MEMORY IS LIMITED. The datasets can have over a million rows. For each dataset, you MUST read it as: pd.read_csv(path, **read_csv_settings). Those settings already pin usecols and dtypes that keep the load within memory — do NOT read extra columns and do NOT override the dtypes. Reading the raw file with all columns will crash the sandbox.
- DATA IS DIRTY. Public datasets are full of junk/placeholder rows (zeros, sentinel values, missing categories, impossible dates). Before ANY ranking, correlation, or group statistic you MUST filter to credible rows, and do the filtering BEFORE heavy groupby/explode so large files stay within memory:
  * Base every filter on the dataset's actual domain and available columns. Never invent or reference release-status, vote-count, runtime, budget, or other fields that are not listed in the schema.
  * Use plausible value ranges for EVERY metric you analyze. For film-like schemas, apply film-specific checks only when the matching columns exist: keep released titles, use sensible release years (cinema starts around 1888; exclude future years), require positive budget/revenue, and use a feature-length runtime range (roughly 40–240 minutes) when reporting feature-film runtime. For music-like schemas, validate streams, duration, and audio features using their real units and ranges; never apply film-runtime or film-release rules to tracks.
  * For a rating metric, require a reasonable minimum vote/rating count only when a corresponding count column exists (for example, vote_average with vote_count >= 50). Do not let unrated or zero-count rows collapse an average toward 0.
  * For any PER-GROUP statistic (a percentage, mean, or ranking by country/language/genre/artist or another category), require a minimum group size appropriate to the dataset (e.g. n >= 30 for a large dataset), ignore undersized groups, and print each group's n. Extreme results computed from only a handful of rows are noise, not findings.
  * Print how many rows survive the filter. Never rank, average, or correlate on the raw, unfiltered data.
  A rating near 0 for a major group, a trend anchored on a placeholder date, an implausible duration for the domain, or an extreme rate from a tiny group all indicate inadequate cleaning — fix the filter instead of reporting the artifact.
- When grouping or ranking by a categorical key (country, language, genre, studio), drop rows whose key is missing/NaN/empty/'nan' BEFORE aggregating, so a "nan" bucket never appears as a ranked group.
- If two datasets share an obvious key column, you may merge them to surface cross-table insights.
- Wrap the body in try/except and print the full traceback on failure, so errors are never silent.
- ALL findings must be emitted via print(). Do NOT use display() or notebook magics.
- Print RAW numbers only: no currency symbols ($), no thousands separators (commas), no magnitude suffixes (K/M/B). For example, do not print things like $2.9B or 2,923,706,000. Consistent raw numbers let the fact-checker match them; the journalist adds formatting.
- When ranking, print tuples that include the entity AND any natural identifying columns for it (artist for a song, director/studio for a film), then the value. This lets the journalist attribute correctly without inventing facts. Keep the ranked entity as the FIRST element.
- When computing correlations, print as: corr(col_a, col_b) = 0.42 (always include both columns and the rounded value).
- Print each statistic on its own line with a clear label that names the metric.
- Output ONE python code block. No prose.
"""

USER_TEMPLATE = """Datasets available (read each from its path):
{datasets}

Analysis tasks:
{tasks}

{revision_notes}

Write a single Python script that runs all the analysis tasks and prints results in the format described.
"""


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _format_datasets(files: list[dict]) -> str:
    blocks = []
    for f in files:
        size_mb = f.get("size_bytes", 0) // (1024 * 1024)
        analysis_cols = f.get("analysis_columns") or f["columns"]
        dtypes = f.get("dtypes", {})
        analysis_dtypes = {c: dtypes.get(c) for c in analysis_cols}
        excluded = f.get("excluded_columns", [])
        block = (
            f"- {f.get('name', f['path'])} "
            f"({f.get('n_rows', '?')} rows, ~{size_mb} MB)\n"
            f"  path: {f['path']}\n"
            f"  analysis columns/dtypes: {analysis_dtypes}\n"
            f"  Read CSV settings — read with pd.read_csv(path, **read_csv_settings):\n"
            f"    read_csv_settings = {f.get('read_csv_kwargs', {})}"
        )
        if excluded:
            block += f"\n  (excluded heavy free-text columns: {excluded})"
        blocks.append(block)
    return "\n".join(blocks)


def _build_revision_notes(state: AgentState) -> str:
    parts = []
    runs = state.get("analysis_results", [])
    last = runs[-1] if runs else None
    if last and last.get("stderr", "").strip():
        parts.append(
            "YOUR PREVIOUS SCRIPT RAISED AN ERROR. Fix it so it runs cleanly:\n"
            + last["stderr"][:2000]
        )
    if last and not last.get("stdout", "").strip():
        parts.append(
            "YOUR PREVIOUS SCRIPT PRINTED NOTHING. A large CSV may have exhausted "
            "memory — read only needed columns / use chunksize, and make sure every "
            "result is printed with print()."
        )
    if state.get("evaluation_errors"):
        parts.append(
            "PREVIOUS ARTICLE FAILED VERIFICATION:\n"
            + "\n".join(f"- {e}" for e in state["evaluation_errors"])
            + "\nProvide additional statistics that resolve these issues."
        )
    return "\n\n".join(parts)


def analyst_node(state: AgentState, sandbox: Sandbox) -> dict:
    llm = get_llm(temperature=0.2)
    revision_notes = _build_revision_notes(state)
    user = USER_TEMPLATE.format(
        datasets=_format_datasets(state["schema"]["files"]),
        tasks="\n".join(f"- {t}" for t in state["analysis_plan"]),
        revision_notes=revision_notes,
    )
    response = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)])
    code = _extract_code(content_to_text(response.content))
    result = sandbox.run(code)
    run = {"script": code, "stdout": result["stdout"], "stderr": result["stderr"]}
    return {"analysis_results": state.get("analysis_results", []) + [run]}
