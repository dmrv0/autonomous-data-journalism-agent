---
title: Autonomous Data Journalism Agent
emoji: 👁
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
python_version: '3.13'
app_file: app.py
pinned: false
license: mpl-2.0
short_description: Autonomous Data Journalism w. Entertainment Industry Datasets
---

# Autonomous Data Journalism Agent

A multi-agent [LangGraph](https://langchain-ai.github.io/langgraph/) pipeline that takes a raw
CSV dataset and produces a wire-service-style news article in which **every numerical claim is
verified against code that actually ran**.

Point it at a CSV (upload, or a Kaggle dataset ref). It profiles the schema, decides what is
newsworthy, writes and executes pandas/numpy/scipy analysis inside an isolated
[E2B](https://e2b.dev) sandbox, drafts an article from the resulting stdout, then fact-checks the
draft back against that stdout and loops until the claims hold up.

> Originally built as a Hugging Face Space:
> [thecloudrazor/autonomous-data-journalism-agent](https://huggingface.co/spaces/thecloudrazor/autonomous-data-journalism-agent)
> (currently not running).

---

## How it works

```
                 ┌─────────┐
  CSV / Kaggle ─▶│ ingest  │  profile schema in sandbox, build a memory-safe read recipe
                 └────┬────┘
                      ▼
                 ┌─────────┐
                 │ planner │  LLM proposes 3–5 newsworthy questions from THIS schema
                 └────┬────┘
                      ▼
                 ┌─────────┐
            ┌───▶│ analyst │  LLM writes one Python script; it runs in the E2B sandbox
            │    └────┬────┘
            │         ▼
            │    ┌───────────┐
            │    │ journalist│  drafts the article using ONLY the script's stdout
            │    └────┬──────┘
            │         ▼
            │    ┌───────────┐
            └────│ evaluator │  deterministic + LLM fact-check; re-runs analysis on failure
     (≤3 revisions)└────┬─────┘
                        ▼
                  final article
```

### The agents

| Node | File | Responsibility |
|---|---|---|
| `ingest` | `src/agents/ingest.py` | Downloads/uploads the data into the sandbox and profiles it: columns, dtypes, exact row count (chunked), sampled `describe()`. Emits a `read_csv_kwargs` recipe (`usecols`, `float32`, `category`) so million-row files load without OOM-ing the sandbox. |
| `planner` | `src/agents/planner.py` | Asks the LLM for 3–5 investigation questions grounded in the actual column list. Falls back to a default plan if the LLM output is unusable. |
| `analyst` | `src/agents/analyst.py` | Generates a single Python script — heavily prompted on data cleaning (sentinel values, minimum group sizes, plausible ranges, NaN category keys) and on printing raw, unformatted numbers so they are machine-checkable. Runs it in the sandbox. |
| `journalist` | `src/agents/journalist.py` | Writes the article from the analysis stdout only. Forbidden from inventing figures, proper nouns, or renaming a metric (a revenue ranking may not become "most-streamed"). |
| `evaluator` | `src/agents/evaluator.py` | Fact-checks and routes: back to `analyst` on failure, or to `END`. |

### Verification

The evaluator runs four checks against the analysis stdout:

- **`verify_values`** — every number in the article must match an output number within 1% tolerance.
- **`verify_signs`** — a claimed positive/negative correlation must match the sign of a printed `corr(...) = ...`.
- **`verify_ranks`** — "X is the most Y" must match the head of a printed ranked list.
- **`verify_semantic`** — an LLM fact-checking editor that catches mislabelled metrics, unsupported
  attributions, backwards correlations, and *plausibility artifacts* (a mean rating near 0, a trend
  anchored on year 1800, a 100% rate computed from 2 rows).

Only sign, rank, and semantic errors are blocking (see `select_blocking_errors`) — value drift alone
is tolerated because journalists legitimately round and reformat. Blocking errors are fed back to the
analyst as revision notes. After `MAX_REVISIONS = 3` the article is emitted with an `[unverified]`
tag listing the outstanding issues rather than silently passing.

### Model routing

`src/llm.py` wraps Gemini as primary with Qwen (DashScope, OpenAI-compatible) as fallback via
`FallbackChatModel`. If the Gemini client cannot even be constructed, it degrades to Qwen alone. Both
failing raises a single error naming both causes.

---

## Setup

Requires Python 3.13.

```bash
git clone <your-fork-url>
cd autonomous-data-journalism-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
```

### Keys

| Variable | Required | What for |
|---|---|---|
| `GEMINI_API_KEY` | yes | Primary LLM ([AI Studio](https://aistudio.google.com/apikey)) |
| `QWEN_API_KEY` | yes | Fallback LLM ([DashScope](https://dashscope-intl.console.aliyun.com/)) |
| `E2B_API_KEY` | yes | Code execution sandbox ([e2b.dev](https://e2b.dev/dashboard)) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | only for `kaggle:` input | Dataset download inside the sandbox |

Optional: `GEMINI_MODEL`, `QWEN_MODEL`, `QWEN_BASE_URL`, `LLM_REQUEST_TIMEOUT_SECONDS`,
`LLM_MAX_RETRIES`.

## Usage

### Gradio UI

```bash
python app.py
```

Upload a CSV **or** enter a Kaggle ref (`owner/dataset-name`). The left pane streams the live agent
log — planned investigations, truncated stdout per run, evaluator errors — and the right pane renders
the final article.

### CLI

```bash
python -m src.graph path/to/data.csv
```

Prints the final article to stdout.

### As a library

```python
from src.graph import run

article = run("kaggle:owner/dataset-name")
```

`run()` handles `load_dotenv()`, sandbox creation, and teardown. Paths beginning with `kaggle:` are
downloaded inside the sandbox (no host round-trip — the recommended route for large datasets);
anything else is uploaded from the local filesystem.

---

## Design notes

**Memory is the real constraint.** The E2B sandbox will die on a naive `pd.read_csv` of a 1M-row
film dataset with free-text overview columns. `ingest` measures average string length per column,
drops free-text columns above 30 characters, downcasts floats to `float32`, and converts
low-cardinality strings to `category` — then hands the analyst a `read_csv_settings` dict it is
instructed not to override.

**Dirty data is the second constraint.** Public datasets are full of zero-revenue placeholder rows,
unrated titles, impossible release years, and 3-minute "feature films". Most of the analyst prompt is
cleaning policy, and the evaluator's plausibility rules exist to catch artifacts that survive it —
the number can be perfectly reproducible from the data and still be journalistic nonsense.

**Number formatting is a verification problem.** The analyst is required to print raw numbers (no
`$`, no thousands separators, no `2.9B`) so `verify_values` can match them; the journalist is
explicitly allowed to reformat, and the semantic checker is told that `$2.9 billion`,
`$2,923,706,026`, and `2923706026` are the same number.

## Limitations

- Sandbox lifetime is 900s, a single analysis run 300s; very large datasets can time out.
- Multi-file datasets are profiled individually, but only the largest CSV becomes the "primary" path.
- The graph is linear with one feedback edge — there is no parallel exploration of competing angles.
- `[unverified]` articles are still returned; always read the tag.

## License

[MPL-2.0](LICENSE).
