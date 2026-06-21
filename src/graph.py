from __future__ import annotations
import sys
from functools import partial
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.sandbox import Sandbox
from src.agents.ingest import ingest_node
from src.agents.analyst import analyst_node
from src.agents.journalist import journalist_node
from src.agents.evaluator import evaluator_node, evaluator_route


def build_graph(sandbox: Sandbox):
    workflow = StateGraph(AgentState)
    workflow.add_node("ingest", partial(ingest_node, sandbox=sandbox))
    workflow.add_node("analyst", partial(analyst_node, sandbox=sandbox))
    workflow.add_node("journalist", journalist_node)
    workflow.add_node("evaluator", evaluator_node)

    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "analyst")
    workflow.add_edge("analyst", "journalist")
    workflow.add_edge("journalist", "evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        evaluator_route,
        {"analyst": "analyst", "end": END},
    )
    return workflow.compile()


def initial_state(dataset_path: str) -> AgentState:
    return {
        "dataset_path": dataset_path,
        "schema": {},
        "analysis_plan": [],
        "analysis_results": [],
        "article_draft": "",
        "evaluation_errors": [],
        "revision_count": 0,
        "final_article": "",
    }


def run(dataset_path: str) -> str:
    load_dotenv()
    sandbox = Sandbox()
    try:
        graph = build_graph(sandbox)
        final = graph.invoke(initial_state(dataset_path), config={"recursion_limit": 25})
        return final["final_article"]
    finally:
        sandbox.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.graph <path-to-csv>")
        sys.exit(1)
    article = run(sys.argv[1])
    print("\n=== FINAL ARTICLE ===\n")
    print(article)
