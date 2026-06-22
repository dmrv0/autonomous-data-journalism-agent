from __future__ import annotations
import json
import os
from e2b_code_interpreter import Sandbox as E2BSandbox

SANDBOX_TIMEOUT_SECONDS = 900
RUN_TIMEOUT_SECONDS = 300


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
    if error is not None:
        detail = f"{error.name}: {error.value}"
        if stderr.strip():
            detail += f"\n{stderr.strip()}"
        raise RuntimeError(f"Kaggle download failed: {detail}")

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
        self._sbx = E2BSandbox.create(timeout=SANDBOX_TIMEOUT_SECONDS)
        self._dataset_remote_path: str | None = None
        self._context = None

    def upload(self, local_path: str, remote_name: str = "data.csv") -> str:
        with open(local_path, "rb") as f:
            self._sbx.files.write(f"/home/user/{remote_name}", f.read())
        self._dataset_remote_path = f"/home/user/{remote_name}"
        return self._dataset_remote_path

    def set_dataset_path(self, remote_path: str) -> str:
        self._dataset_remote_path = remote_path
        return self._dataset_remote_path

    def download_kaggle(self, dataset_ref: str) -> list[dict]:
        envs = {
            name: value
            for name in ("KAGGLE_USERNAME", "KAGGLE_KEY")
            if (value := os.environ.get(name))
        }
        self._sbx.commands.run("pip install -q kagglehub", timeout=300, envs=envs)

        code = "ref = " + repr(dataset_ref) + "\n" + KAGGLE_DOWNLOAD_CODE
        exec_result = self._sbx.run_code(code, envs=envs)
        return _parse_kaggle_manifest(
            stdout="\n".join(exec_result.logs.stdout),
            stderr="\n".join(exec_result.logs.stderr),
            error=exec_result.error,
            dataset_ref=dataset_ref,
        )

    def _fresh_context(self):
        if self._context is None:
            self._context = self._sbx.create_code_context()
        else:
            self._sbx.restart_code_context(self._context)
        return self._context

    def run(self, code: str) -> dict:
        context = self._fresh_context()
        exec_result = self._sbx.run_code(
            code, context=context, timeout=RUN_TIMEOUT_SECONDS
        )
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
