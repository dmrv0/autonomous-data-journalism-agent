from __future__ import annotations
import json
import os
from e2b_code_interpreter import Sandbox as E2BSandbox

# The sandbox must outlive a multi-iteration revision loop (download + several
# analyst/journalist/evaluator passes), and a single analysis over 1M+ rows can
# take a while — give both generous headroom so executions aren't cut off
# ("UnexpectedEndOfExecution: Connection to the execution was closed").
SANDBOX_TIMEOUT_SECONDS = 900
RUN_TIMEOUT_SECONDS = 300


# Runs INSIDE the E2B VM. `ref` is injected as a Python literal (see
# download_kaggle) so we avoid str.format brace-escaping headaches. The big
# dataset is downloaded over E2B's datacenter link and never touches the host;
# only this small JSON manifest of discovered CSVs travels back.
KAGGLE_DOWNLOAD_CODE = """
import json, os, glob
import kagglehub

download_dir = kagglehub.dataset_download(ref)
csv_files = []
for match in glob.glob(os.path.join(download_dir, "**", "*.csv"), recursive=True):
    csv_files.append({"path": match, "size_bytes": os.path.getsize(match)})
csv_files.sort(key=lambda f: f["path"])
print(json.dumps({"download_dir": download_dir, "csv_files": csv_files}))
"""


def _parse_kaggle_manifest(
    stdout: str, stderr: str, error, dataset_ref: str
) -> list[dict]:
    """Turn a sandbox download execution into the CSV manifest.

    `error` is E2B's structured execution error (an actual Python exception) or
    None. stderr is NOT treated as failure: kagglehub/tqdm write progress bars
    and warnings there on a perfectly successful download.
    """
    if error is not None:
        detail = f"{error.name}: {error.value}"
        if stderr.strip():
            detail += f"\n{stderr.strip()}"
        raise RuntimeError(f"Kaggle download failed: {detail}")

    # Scan from the end for the first line that parses as our manifest, so any
    # stray progress text that leaked onto stdout cannot break ingestion.
    payload = None
    for line in reversed([ln for ln in stdout.splitlines() if ln.strip()]):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and "csv_files" in candidate:
            payload = candidate
            break

    if payload is None:
        raise RuntimeError(
            f"Kaggle download for '{dataset_ref}' produced no manifest. "
            f"stdout tail: {stdout[-400:]!r} | stderr tail: {stderr[-400:]!r}"
        )
    csv_files = payload["csv_files"]
    if not csv_files:
        raise RuntimeError(f"No CSV files found in Kaggle dataset '{dataset_ref}'")
    return csv_files


class Sandbox:
    def __init__(self) -> None:
        # E2BSandbox() constructor is deprecated in e2b-code-interpreter >=1.0;
        # use the Sandbox.create() classmethod instead.
        self._sbx = E2BSandbox.create(timeout=SANDBOX_TIMEOUT_SECONDS)
        self._dataset_remote_path: str | None = None

    def upload(self, local_path: str, remote_name: str = "data.csv") -> str:
        with open(local_path, "rb") as f:
            self._sbx.files.write(f"/home/user/{remote_name}", f.read())
        self._dataset_remote_path = f"/home/user/{remote_name}"
        return self._dataset_remote_path

    def set_dataset_path(self, remote_path: str) -> str:
        self._dataset_remote_path = remote_path
        return self._dataset_remote_path

    def download_kaggle(self, dataset_ref: str) -> list[dict]:
        """Download a Kaggle dataset directly inside the sandbox.

        Returns a manifest of every CSV found: [{"path", "size_bytes"}, ...].
        The ingest node profiles all of them and points the sandbox's primary
        handle at the largest (see ingest._primary_path).
        """
        envs = {
            name: value
            for name in ("KAGGLE_USERNAME", "KAGGLE_KEY")
            if (value := os.environ.get(name))
        }
        # kagglehub is not in the default E2B template; install it in the VM.
        self._sbx.commands.run("pip install -q kagglehub", timeout=300, envs=envs)

        code = "ref = " + repr(dataset_ref) + "\n" + KAGGLE_DOWNLOAD_CODE
        exec_result = self._sbx.run_code(code, envs=envs)
        return _parse_kaggle_manifest(
            stdout="\n".join(exec_result.logs.stdout),
            stderr="\n".join(exec_result.logs.stderr),
            error=exec_result.error,
            dataset_ref=dataset_ref,
        )

    def run(self, code: str) -> dict:
        exec_result = self._sbx.run_code(code, timeout=RUN_TIMEOUT_SECONDS)
        stdout = "\n".join(exec_result.logs.stdout)
        stderr = "\n".join(exec_result.logs.stderr)
        if exec_result.error:
            stderr += f"\n{exec_result.error.name}: {exec_result.error.value}"
        return {"stdout": stdout, "stderr": stderr}

    @property
    def dataset_path(self) -> str:
        if not self._dataset_remote_path:
            raise RuntimeError("No dataset uploaded")
        return self._dataset_remote_path

    def close(self) -> None:
        self._sbx.kill()
