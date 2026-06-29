"""
Graph builder — assembles and compiles the AutoPlanner LangGraph workflow.

Topology:
  START → analyser → clarifier → planner → reviewer
                                               ├─(approved)──► writer → END
                                               └─(rejected)──► planner  (retry)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from autoplanner.nodes import (
    analyser_node,
    clarifier_node,
    planner_node,
    reviewer_node,
    should_retry,
    writer_node,
)
from autoplanner.state import PlannerState


def build_graph():
    """Build and compile the AutoPlanner StateGraph."""

    builder = StateGraph(PlannerState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("analyser", analyser_node)
    builder.add_node("clarifier", clarifier_node)
    builder.add_node("planner", planner_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("writer", writer_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    builder.add_edge(START, "analyser")
    builder.add_edge("analyser", "clarifier")
    builder.add_edge("clarifier", "planner")
    builder.add_edge("planner", "reviewer")

    # Conditional: reviewer → (writer | planner)
    builder.add_conditional_edges(
        "reviewer",
        should_retry,
        {
            "writer": "writer",
            "planner": "planner",
        },
    )

    builder.add_edge("writer", END)

    return builder.compile()


# Singleton compiled graph
graph = build_graph()
