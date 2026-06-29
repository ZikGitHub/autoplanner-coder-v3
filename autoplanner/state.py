"""
State schema for the AutoPlanner LangGraph workflow.

Every node reads from and writes back to this TypedDict.
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class PlannerState(TypedDict):
    """Shared state that flows through all nodes of the planner graph."""

    # ── Inputs ────────────────────────────────────────────────────────────────
    user_request: str
    """Natural-language description of what the user wants to build / change."""

    codebase_path: str
    """Absolute or relative path to the target codebase (may be empty for new projects)."""

    # ── Intermediate artefacts ────────────────────────────────────────────────
    codebase_summary: str
    """High-level description of the existing codebase produced by the analyser node."""

    clarified_request: str
    """Optionally refined request after the clarifier node runs."""

    plan: dict[str, Any]
    """Structured plan produced by the planner node (tasks, files, dependencies, …)."""

    review_feedback: str
    """Feedback / approval produced by the reviewer node."""

    approved: bool
    """True once the reviewer has approved the plan."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Full conversation history (for LLM context)."""

    # ── Outputs ───────────────────────────────────────────────────────────────
    output_path: str
    """Path where the final plan artefacts were written."""

    error: str
    """Non-empty if any node encountered an unrecoverable error."""
