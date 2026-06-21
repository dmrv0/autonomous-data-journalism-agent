from __future__ import annotations
import json
from src.state import AgentState
from src.sandbox import Sandbox

KAGGLE_PREFIX = "kaggle:"


# Runs INSIDE the E2B VM. `paths` is injected as a Python literal (see
# ingest_node). Every CSV is profiled with the same sampled-schema strategy so
# the analyst can see (and join across) all tables in a multi-file dataset
# without any raw rows ever reaching the LLM.
INGEST_CODE = """
import pandas as pd
import json
import os
SAMPLE_ROWS = 5000
CSV_ENCODINGS = ["utf-8", "utf-8-sig", "iso-8859-1", "cp1252"]
# Textual columns whose sampled rows average more than this many characters are
# free-text (overview, keywords, ...) — useless for statistics and the main
# memory hog on big files. They are excluded from the analyst's read via usecols.
TEXT_LEN_LIMIT = 30


def _avg_len(col):
    return float(col.dropna().astype(str).str.len().mean() or 0.0)


def profile(path):
    sample = None
    encoding = None
    last_error = None
    for candidate in CSV_ENCODINGS:
        try:
            sample = pd.read_csv(
                path,
                nrows=SAMPLE_ROWS,
                low_memory=False,
                encoding=candidate,
            )
            encoding = candidate
            break
        except UnicodeDecodeError as exc:
            last_error = exc

    if sample is None:
        raise last_error or RuntimeError(
            "Could not read CSV sample with configured encodings"
        )

    columns = sample.columns.tolist()
    sample_describe = sample.describe(include="all").fillna("").to_dict()

    # Build a memory-safe read recipe so the analyst can load the FULL row set
    # of even a 1M+ row file without exhausting the sandbox: keep numeric/bool
    # columns + short textual ones, drop big free-text, store floats as float32
    # and low-cardinality text as category (both lossless for our stats).
    analysis_columns = []
    read_dtype = {}
    for c in columns:
        col = sample[c]
        is_num = pd.api.types.is_numeric_dtype(col)
        is_bool = pd.api.types.is_bool_dtype(col)
        if is_num or is_bool or _avg_len(col) <= TEXT_LEN_LIMIT:
            analysis_columns.append(c)
            if is_num and pd.api.types.is_float_dtype(col):
                read_dtype[c] = "float32"
            elif not is_num and not is_bool and len(sample):
                if col.nunique(dropna=True) / len(sample) < 0.5:
                    read_dtype[c] = "category"

    read_csv_kwargs = {
        "encoding": encoding,
        "low_memory": False,
        "usecols": analysis_columns,
    }
    if read_dtype:
        read_csv_kwargs["dtype"] = read_dtype

    n_rows = 0
    if columns:
        first_col = columns[0]
        for chunk in pd.read_csv(
            path,
            usecols=[first_col],
            chunksize=100000,
            encoding=encoding,
            low_memory=False,
        ):
            n_rows += len(chunk)

    return {
        "path": path,
        "name": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "columns": columns,
        "dtypes": {c: str(sample[c].dtype) for c in columns},
        "analysis_columns": analysis_columns,
        "excluded_columns": [c for c in columns if c not in analysis_columns],
        "n_rows": n_rows,
        "describe": sample_describe,
        "encoding": encoding,
        "read_csv_kwargs": read_csv_kwargs,
        "schema_sample_rows": len(sample),
    }


files = [profile(p) for p in paths]
print(json.dumps({"files": files, "ingest_mode": "sampled_schema"}, default=str))
"""


def ingest_node(state: AgentState, sandbox: Sandbox) -> dict:
    dataset_path = state["dataset_path"]
    if dataset_path.startswith(KAGGLE_PREFIX):
        # Big-dataset path: download inside the sandbox and profile every CSV.
        dataset_ref = dataset_path[len(KAGGLE_PREFIX):]
        csv_files = sandbox.download_kaggle(dataset_ref)
        paths = [f["path"] for f in csv_files]
    elif dataset_path.startswith("/home/user/"):
        paths = [dataset_path]
    else:
        paths = [sandbox.upload(dataset_path)]

    code = "paths = " + repr(paths) + "\n" + INGEST_CODE
    result = sandbox.run(code)
    if result["stderr"]:
        raise RuntimeError(f"Ingest failed: {result['stderr']}")
    schema = json.loads(result["stdout"].strip().splitlines()[-1])

    primary = _primary_path(schema["files"])
    schema["primary_path"] = primary
    # Point the sandbox's convenience handle at the largest table.
    sandbox.set_dataset_path(primary)
    return {"schema": schema, "analysis_plan": _initial_plan(schema)}


def _primary_path(files: list[dict]) -> str:
    """The 'main' table — largest by byte size. Used as the sandbox's default
    handle; the analyst still receives every file and may join across them."""
    return max(files, key=lambda f: f.get("size_bytes", 0))["path"]


def _initial_plan(schema: dict) -> list[str]:
    files = schema["files"]
    all_cols = sorted({c for f in files for c in f["columns"]})
    table_summary = ", ".join(f"{f['name']} ({len(f['columns'])} cols)" for f in files)
    return [
        "Clean first: public datasets contain junk/placeholder rows. Before any "
        "ranking or correlation, drop rows with missing, zero, or implausible "
        "sentinel values in the metric being analyzed; where a vote/popularity "
        "count exists, restrict to sufficiently-rated entries; prefer released "
        "titles. Print how many rows remain after filtering.",
        f"Across the available tables [{table_summary}], identify the top 10 "
        f"entries ranked by the most prominent count/streams/revenue column "
        f"among {all_cols} — using the cleaned data.",
        "Compute pairwise correlations between numeric columns and surface the "
        "3 strongest (by |r|). If two tables share a key column, join them first "
        "to surface cross-table relationships.",
        "Identify any temporal column (year, release_date) and describe how key "
        "metrics shift across it.",
    ]
