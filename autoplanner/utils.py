"""
Utility helpers shared across nodes.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path


# ── Codebase scanner ──────────────────────────────────────────────────────────

_SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".eggs", "*.egg-info",
}

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".scala", ".html", ".css", ".scss", ".json", ".yaml", ".yml",
    ".toml", ".md", ".txt", ".sh", ".bash", ".dockerfile",
}


def scan_codebase(root: str, max_files: int = 200) -> dict:
    """
    Walk *root* and return a lightweight summary dict.

    Returns:
        {
          "root": str,
          "total_files": int,
          "structure": [{"path": str, "size_bytes": int}, …],
          "languages": {"Python": 12, "JavaScript": 4, …},
          "truncated": bool,
        }
    """
    root_path = Path(root)
    if not root_path.exists():
        return {"root": root, "total_files": 0, "structure": [], "languages": {}, "truncated": False}

    files: list[dict] = []
    lang_map: dict[str, int] = {}
    ext_to_lang = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".jsx": "JavaScript", ".tsx": "TypeScript", ".java": "Java",
        ".go": "Go", ".rs": "Rust", ".cpp": "C++", ".c": "C",
        ".h": "C/C++ Header", ".cs": "C#", ".rb": "Ruby",
        ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin",
        ".scala": "Scala", ".html": "HTML", ".css": "CSS",
        ".scss": "SCSS", ".json": "JSON", ".yaml": "YAML",
        ".yml": "YAML", ".toml": "TOML", ".md": "Markdown",
        ".sh": "Shell", ".bash": "Shell",
    }

    truncated = False
    for path in sorted(root_path.rglob("*")):
        # Skip hidden dirs / common noise dirs
        if any(part.startswith(".") or part in _SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in _CODE_EXTENSIONS:
            rel = str(path.relative_to(root_path)).replace("\\", "/")
            size = path.stat().st_size
            files.append({"path": rel, "size_bytes": size})
            lang = ext_to_lang.get(path.suffix, "Other")
            lang_map[lang] = lang_map.get(lang, 0) + 1
            if len(files) >= max_files:
                truncated = True
                break

    return {
        "root": str(root_path),
        "total_files": len(files),
        "structure": files,
        "languages": lang_map,
        "truncated": truncated,
    }


def read_file_snippet(path: str, max_lines: int = 60) -> str:
    """Return the first *max_lines* of a file as a string."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = []
            for i, line in enumerate(fh):
                if i >= max_lines:
                    lines.append(f"... (truncated at {max_lines} lines)")
                    break
                lines.append(line.rstrip())
            return "\n".join(lines)
    except Exception as exc:
        return f"[Could not read file: {exc}]"


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """
    Extract the first JSON object from *text* (LLM response).
    Falls back to an empty dict on failure.
    """
    # Try to find ```json … ``` fences first
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # Raw JSON object anywhere in the text
    raw = re.search(r"\{.*\}", text, re.DOTALL)
    if raw:
        try:
            return json.loads(raw.group(0))
        except json.JSONDecodeError:
            pass

    return {}


# ── Output helpers ────────────────────────────────────────────────────────────

def make_output_dir(base: str = "output") -> Path:
    """
    Create a timestamped output directory under *base* and return its Path.
    e.g.  output/plan_20260629_201400/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / f"plan_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_plan_artefacts(output_dir: Path, state: dict) -> None:
    """
    Persist plan-related artefacts into *output_dir*.

    Files written:
      plan.json          — structured plan
      plan_summary.md    — human-readable plan summary
      codebase_summary.md — codebase analysis
      review.md          — reviewer feedback
    """
    # --- plan.json ---
    plan = state.get("plan", {})
    if plan:
        plan_json = output_dir / "plan.json"
        plan_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- plan_summary.md ---
    md_lines: list[str] = [
        "# AutoPlanner — Codebase Activity Plan",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## User Request",
        "",
        state.get("clarified_request") or state.get("user_request", ""),
        "",
    ]

    if plan:
        md_lines += ["## Tasks", ""]
        for i, task in enumerate(plan.get("tasks", []), 1):
            md_lines.append(f"### Task {i}: {task.get('title', 'Untitled')}")
            md_lines.append(f"**Description:** {task.get('description', '')}")
            md_lines.append(f"**Priority:** {task.get('priority', 'medium')}")
            files = task.get("files", [])
            if files:
                md_lines.append(f"**Files affected:** `{'`, `'.join(files)}`")
            deps = task.get("dependencies", [])
            if deps:
                md_lines.append(f"**Depends on:** Task(s) {', '.join(str(d) for d in deps)}")
            md_lines.append("")

        if plan.get("implementation_order"):
            md_lines += ["## Implementation Order", ""]
            for step in plan["implementation_order"]:
                md_lines.append(f"- {step}")
            md_lines.append("")

        if plan.get("risks"):
            md_lines += ["## Risks & Mitigations", ""]
            for r in plan["risks"]:
                md_lines.append(f"- **{r.get('risk', '')}**: {r.get('mitigation', '')}")
            md_lines.append("")

        if plan.get("estimated_effort"):
            md_lines += [
                "## Estimated Effort",
                "",
                f"{plan['estimated_effort']}",
                "",
            ]

    (output_dir / "plan_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    # --- codebase_summary.md ---
    cb_summary = state.get("codebase_summary", "")
    if cb_summary:
        (output_dir / "codebase_summary.md").write_text(
            f"# Codebase Analysis\n\n{cb_summary}\n", encoding="utf-8"
        )

    # --- review.md ---
    review = state.get("review_feedback", "")
    if review:
        (output_dir / "review.md").write_text(
            f"# Plan Review\n\n{review}\n", encoding="utf-8"
        )
