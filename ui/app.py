from __future__ import annotations
import os
import time
import gradio as gr
from dotenv import load_dotenv
from src.sandbox import Sandbox
from src.graph import build_graph, initial_state

load_dotenv()


def run_pipeline(csv_file, kaggle_ref):
    kaggle_ref = (kaggle_ref or "").strip()
    if csv_file is None and not kaggle_ref:
        yield "Please upload a CSV or enter a Kaggle dataset ref first.", ""
        return
    log = "Starting pipeline...\n"
    article = ""
    file_size_mb = 0.0
    if kaggle_ref:
        log += f"Kaggle dataset: {kaggle_ref} (downloaded inside the sandbox)\n"
    else:
        file_size_mb = os.path.getsize(csv_file.name) / (1024 * 1024)
        log += f"CSV size: {file_size_mb:.1f} MB\n"
        if file_size_mb > 100:
            log += "Large CSV detected; a Kaggle ref avoids the host upload entirely.\n"
    yield log, article

    sandbox = None
    started_at = time.monotonic()
    try:
        log += "Creating E2B sandbox...\n"
        yield log, article
        sandbox = Sandbox()

        if kaggle_ref:
            log += "E2B sandbox ready. Will download the dataset inside the sandbox...\n"
            yield log, article
            graph = build_graph(sandbox)
            initial = initial_state(f"kaggle:{kaggle_ref}")
        else:
            log += f"E2B sandbox ready. Uploading CSV to E2B ({file_size_mb:.1f} MB)...\n"
            yield log, article
            remote_path = sandbox.upload(csv_file.name)
            log += "Upload complete. Building LangGraph workflow...\n"
            yield log, article
            graph = build_graph(sandbox)
            initial = initial_state(remote_path)

        log += "Running graph with dataset...\n"
        yield log, article
        for event in graph.stream(initial, config={"recursion_limit": 25}):
            elapsed = time.monotonic() - started_at
            for node, update in event.items():
                log += f"\n▶ {node} completed at +{elapsed:.1f}s\n"
                if node == "planner":
                    plan = update.get("analysis_plan")
                    if plan:
                        log += "  planned investigations (chosen from the schema):\n"
                        for i, task in enumerate(plan, 1):
                            log += f"    {i}. {task[:160]}\n"
                    else:
                        log += "  (kept the default investigation plan)\n"
                if "analysis_results" in update and update["analysis_results"]:
                    last = update["analysis_results"][-1]
                    log += f"  stdout (truncated):\n  {last['stdout'][:500]}\n"
                    if last.get("stderr", "").strip():
                        log += f"  ⚠ stderr (truncated):\n  {last['stderr'][:500]}\n"
                if "evaluation_errors" in update and update["evaluation_errors"]:
                    log += f"  ⚠ errors: {update['evaluation_errors']}\n"
                if "final_article" in update and update["final_article"]:
                    article = update["final_article"]
                yield log, article
    except Exception as exc:
        log += f"\n✖ Pipeline failed: {type(exc).__name__}: {exc}\n"
        yield log, article
    finally:
        if sandbox is not None:
            sandbox.close()


with gr.Blocks(title="Autonomous Data Journalism Agent") as demo:
    gr.Markdown("# Autonomous Data Journalism Agent")
    gr.Markdown(
        "Upload a CSV of entertainment industry data, "
        "**or** give a Kaggle dataset ref to download it straight into the sandbox. "
        "The agent performs EDA, drafts an article, and verifies every numerical claim."
    )
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="CSV dataset", file_types=[".csv"])
            kaggle_input = gr.Textbox(
                label="…or Kaggle dataset ref (owner/dataset-name)",
                placeholder="owner/dataset-name",
            )
            run_btn = gr.Button("Run pipeline", variant="primary")
            log_output = gr.Textbox(label="Live agent log", lines=20, max_lines=40)
        with gr.Column():
            article_output = gr.Markdown(label="Final article")
    run_btn.click(
        run_pipeline,
        inputs=[file_input, kaggle_input],
        outputs=[log_output, article_output],
    )


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
