from typing import TypedDict


class AnalysisRun(TypedDict):
    script: str
    stdout: str
    stderr: str


class AgentState(TypedDict):
    dataset_path: str
    schema: dict
    analysis_plan: list[str]
    analysis_results: list[AnalysisRun]
    article_draft: str
    evaluation_errors: list[str]
    revision_count: int
    final_article: str
