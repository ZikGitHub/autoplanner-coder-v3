"""
Individual node functions for the AutoPlanner LangGraph workflow.

Each node takes the current PlannerState, performs its work, and returns
a partial dict of state updates.

Graph topology:
  [START]
    └─► analyser
          └─► clarifier
                └─► planner
                      └─► reviewer
                            ├─(approved)──► writer ──► [END]
                            └─(rejected)──► planner   (retry loop)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from autoplanner.llm import get_llm
from autoplanner.state import PlannerState
from autoplanner.utils import (
    extract_json,
    make_output_dir,
    read_file_snippet,
    scan_codebase,
    write_plan_artefacts,
)

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _llm_call(system: str, human: str) -> str:
    """Simple single-turn LLM call, returns the string response."""
    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return response.content  # type: ignore[return-value]


# ── Node 1 — Analyser ─────────────────────────────────────────────────────────

def analyser_node(state: PlannerState) -> dict[str, Any]:
    """
    Scan the codebase (if one exists) and produce a high-level summary.
    For a new project the summary is 'empty codebase – new project'.
    """
    logger.info("[analyser] Scanning codebase: %s", state["codebase_path"])

    codebase_path = state.get("codebase_path", "").strip()
    scan = scan_codebase(codebase_path) if codebase_path else {}

    if not scan or scan["total_files"] == 0:
        summary = (
            "This is a **new / empty project**. No existing source files were found. "
            "The plan should cover project scaffolding and initial implementation."
        )
        return {"codebase_summary": summary, "error": ""}

    # Pick a few representative files to show the LLM
    sample_files = scan["structure"][:8]
    snippets: list[str] = []
    for f in sample_files:
        full_path = Path(codebase_path) / f["path"]
        snippet = read_file_snippet(str(full_path), max_lines=40)
        snippets.append(f"### {f['path']}\n```\n{snippet}\n```")

    file_tree = "\n".join(f"  - {f['path']}" for f in scan["structure"][:40])
    lang_info = ", ".join(f"{k}: {v}" for k, v in scan["languages"].items())

    prompt = f"""You are a senior software architect. Analyse the following codebase and produce a concise technical summary.

**Codebase path:** {codebase_path}
**Languages detected:** {lang_info}
**Total files:** {scan['total_files']}{'  *(truncated)*' if scan['truncated'] else ''}

**File structure (first 40):**
{file_tree}

**File snippets (first 8 files):**
{chr(10).join(snippets)}

Write a 200-400 word structured summary covering:
1. Purpose / domain of the codebase
2. Architecture pattern (MVC, microservices, monolith, …)
3. Key modules and their responsibilities
4. Technologies / frameworks in use
5. Any obvious technical debt or areas to be careful with
"""

    summary = _llm_call(
        system="You are a senior software architect. Be concise and technical.",
        human=prompt,
    )
    logger.info("[analyser] Summary produced (%d chars).", len(summary))
    return {"codebase_summary": summary, "error": ""}


# ── Node 2 — Clarifier ────────────────────────────────────────────────────────

def clarifier_node(state: PlannerState) -> dict[str, Any]:
    """
    Re-state the user request in precise, unambiguous technical language,
    incorporating context from the codebase summary.
    """
    logger.info("[clarifier] Clarifying request.")

    prompt = f"""You are a technical lead. Based on the codebase analysis and the user's original request, write a **clarified, precise technical specification** of what needs to be done.

**Codebase summary:**
{state.get('codebase_summary', 'New project – no existing code.')}

**User's original request:**
{state['user_request']}

Output a single, self-contained paragraph (100-250 words) that a developer could use directly to start planning. Remove any ambiguity. Specify which modules/files are likely involved, what the expected inputs/outputs are, and any constraints.
"""

    clarified = _llm_call(
        system="You are a technical lead clarifying requirements. Be concise and precise.",
        human=prompt,
    )
    logger.info("[clarifier] Clarified request produced.")
    return {"clarified_request": clarified}


# ── Node 3 — Planner ──────────────────────────────────────────────────────────

_PLAN_SCHEMA = """{
  "title": "string — short plan title",
  "summary": "string — 2-3 sentence executive summary",
  "tasks": [
    {
      "id": 1,
      "title": "string",
      "description": "string",
      "priority": "high | medium | low",
      "type": "feature | bugfix | refactor | test | docs | infra",
      "files": ["list of file paths to create or modify"],
      "dependencies": [list of task IDs this task depends on]
    }
  ],
  "implementation_order": ["ordered list of task titles"],
  "risks": [
    {"risk": "string", "mitigation": "string"}
  ],
  "estimated_effort": "string e.g. 3-5 developer days"
}"""


def planner_node(state: PlannerState) -> dict[str, Any]:
    """
    Generate a structured development plan (JSON) from the clarified request.
    """
    logger.info("[planner] Generating plan.")

    review_context = ""
    if state.get("review_feedback") and not state.get("approved", False):
        review_context = f"""
⚠️  **Previous plan was rejected by the reviewer. Feedback:**
{state['review_feedback']}

Please address the above issues in the new plan.
"""

    prompt = f"""You are a senior software engineer. Create a **detailed, actionable development plan** for the task below.

**Codebase summary:**
{state.get('codebase_summary', 'New project – no existing code.')}

**Task specification:**
{state.get('clarified_request') or state['user_request']}
{review_context}

Output ONLY valid JSON matching this schema (no extra text before or after):
{_PLAN_SCHEMA}
"""

    raw = _llm_call(
        system="You are a senior software engineer who creates detailed, structured development plans. Output only valid JSON.",
        human=prompt,
    )

    plan = extract_json(raw)
    if not plan:
        # Fallback: wrap raw text as a minimal plan
        plan = {
            "title": "Generated Plan",
            "summary": raw[:500],
            "tasks": [],
            "implementation_order": [],
            "risks": [],
            "estimated_effort": "Unknown",
        }

    logger.info("[planner] Plan generated with %d tasks.", len(plan.get("tasks", [])))
    return {"plan": plan}


# ── Node 4 — Reviewer ─────────────────────────────────────────────────────────

def reviewer_node(state: PlannerState) -> dict[str, Any]:
    """
    Review the plan for completeness, feasibility and alignment with the request.
    Returns approved=True if the plan passes, False otherwise (triggers a retry).
    """
    logger.info("[reviewer] Reviewing plan.")

    plan_json = json.dumps(state.get("plan", {}), indent=2)

    prompt = f"""You are a critical technical reviewer. Evaluate the development plan below against the task specification.

**Task specification:**
{state.get('clarified_request') or state['user_request']}

**Plan:**
```json
{plan_json}
```

Evaluate:
1. Does it fully address the specification?
2. Are all tasks clearly described with concrete file targets?
3. Is the implementation order correct (dependencies satisfied)?
4. Are risks identified and mitigated?
5. Is the effort estimate realistic?

Respond with:
- **APPROVED** — if the plan is ready to use (followed by a brief rationale).
- **REJECTED** — if improvements are needed (followed by a specific, actionable list of changes required).

Start your response with exactly the word APPROVED or REJECTED.
"""

    feedback = _llm_call(
        system="You are a thorough technical reviewer. Be critical but constructive.",
        human=prompt,
    )

    approved = feedback.strip().upper().startswith("APPROVED")
    logger.info("[reviewer] Decision: %s", "APPROVED" if approved else "REJECTED")
    return {"review_feedback": feedback, "approved": approved}


# ── Node 5 — Writer ───────────────────────────────────────────────────────────

def writer_node(state: PlannerState) -> dict[str, Any]:
    """
    Persist all plan artefacts to an output directory and return its path.
    """
    logger.info("[writer] Writing artefacts.")

    out_dir = make_output_dir(base="output")
    write_plan_artefacts(out_dir, state)

    logger.info("[writer] Artefacts written to: %s", out_dir)
    return {"output_path": str(out_dir)}


# ── Conditional edge — should_retry ──────────────────────────────────────────

MAX_RETRIES = 2
_retry_count: dict[str, int] = {}  # keyed by graph run-id placeholder


def should_retry(state: PlannerState) -> str:
    """
    Routing function after the reviewer node.
    Returns 'writer' if approved, 'planner' if not (up to MAX_RETRIES times).
    """
    if state.get("approved", False):
        return "writer"

    # Count retries using a simple module-level counter (thread-safe enough for
    # single-user CLI usage; replace with state field for concurrent setups).
    key = id(state)  # unique per invocation
    _retry_count[key] = _retry_count.get(key, 0) + 1
    if _retry_count[key] >= MAX_RETRIES:
        logger.warning("[router] Max retries reached — writing plan as-is.")
        return "writer"

    logger.info("[router] Plan rejected — retrying planner (attempt %d).", _retry_count[key])
    return "planner"
