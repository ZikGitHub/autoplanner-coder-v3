#!/usr/bin/env python
"""
AutoPlanner CLI — interactive entry point.

Usage:
    python main.py
    python main.py --request "Add JWT authentication" --codebase ./myproject
    python main.py --request "Build a REST API" --codebase ""  (new project)
    python main.py --help
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

# ── Env / logging setup ───────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("autoplanner")

# ── Suppress noisy 3rd-party loggers ─────────────────────────────────────────
for _noisy in ("httpx", "httpcore", "langchain", "langgraph"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║           AutoPlanner Coder v3 — LangGraph Workflow      ║
║     Plans codebase activities from a natural-language    ║
║     request using a multi-agent LangGraph pipeline.      ║
╚══════════════════════════════════════════════════════════╝
"""

# ── Ollama health-check ───────────────────────────────────────────────────────

def check_ollama(base_url: str, model: str) -> None:
    """
    Verify the Ollama server is reachable and that *model* is available.
    Prints a status line and raises SystemExit with a helpful message on failure.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            import json
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError) as exc:
        print(
            f"\n❌  Cannot reach Ollama at {base_url}\n"
            f"   Error  : {exc}\n"
            f"   Fix    : make sure `ollama serve` is running.\n"
            f"   Docs   : https://ollama.com/download"
        )
        sys.exit(1)

    available = [m["name"] for m in data.get("models", [])]
    # Normalise: ollama may store "llama3.2:1b" or just "llama3.2"
    short_model = model.split(":")[0]
    found = any(
        m == model or m.startswith(short_model)
        for m in available
    )

    if not found:
        print(
            f"\n⚠️   Model '{model}' not found on Ollama.\n"
            f"   Available models : {available or '(none pulled yet)'}\n"
            f"   Fix              : run  ollama pull {model}"
        )
        sys.exit(1)

    print(f"  🟢  Ollama OK — using model '{model}' at {base_url}")


WORKFLOW_DESCRIPTION = """
Workflow stages:
  1. Analyser   — scans the codebase and produces a technical summary
  2. Clarifier  — sharpens the user request into a precise specification
  3. Planner    — generates a structured JSON development plan
  4. Reviewer   — reviews the plan (retries up to 2x if rejected)
  5. Writer     — saves all artefacts to the output/ directory
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoPlanner Coder v3 — LangGraph codebase activity planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=WORKFLOW_DESCRIPTION,
    )
    parser.add_argument(
        "--request", "-r",
        type=str,
        default=None,
        help="User request (if omitted you will be prompted interactively)",
    )
    parser.add_argument(
        "--codebase", "-c",
        type=str,
        default="",
        help="Path to the codebase to analyse (leave empty for a new project)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def run_planner(user_request: str, codebase_path: str) -> None:
    """Invoke the LangGraph workflow and print a summary."""

    # Late import so dotenv is loaded first
    from autoplanner.graph import graph

    initial_state = {
        "user_request": user_request,
        "codebase_path": codebase_path,
        "codebase_summary": "",
        "clarified_request": "",
        "plan": {},
        "review_feedback": "",
        "approved": False,
        "messages": [],
        "output_path": "",
        "error": "",
    }

    # ── Ollama pre-flight check ───────────────────────────────────────────────
    provider = os.getenv("PLANNER_PROVIDER", "").lower()
    # Default to ollama if no cloud keys are set
    if not provider:
        if not any(os.getenv(k) for k in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")):
            provider = "ollama"

    if provider == "ollama" or os.getenv("OLLAMA_BASE_URL"):
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("PLANNER_MODEL", "llama3.2:1b")
        check_ollama(ollama_url, ollama_model)

    print("\n🚀  Starting AutoPlanner workflow…\n")

    # Stream graph events for live feedback
    node_order = []
    final_state = None

    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, updates in event.items():
            node_order.append(node_name)
            _print_node_event(node_name, updates)
        # Keep updating final_state
        if isinstance(event, dict):
            if final_state is None:
                final_state = dict(initial_state)
            for updates in event.values():
                if isinstance(updates, dict):
                    final_state.update(updates)

    # Print summary
    print("\n" + "═" * 62)
    print("✅  AutoPlanner workflow complete!")
    if final_state:
        out_path = final_state.get("output_path", "")
        if out_path:
            print(f"📁  Output artefacts → {Path(out_path).resolve()}")
            _list_artefacts(out_path)
        approved = final_state.get("approved", False)
        plan = final_state.get("plan", {})
        tasks = plan.get("tasks", [])
        print(f"\n📋  Plan title  : {plan.get('title', 'N/A')}")
        print(f"📝  Tasks       : {len(tasks)}")
        print(f"⏱️   Effort      : {plan.get('estimated_effort', 'N/A')}")
        print(f"🔍  Reviewer    : {'✅ Approved' if approved else '⚠️  Auto-approved after max retries'}")
    print("═" * 62 + "\n")


def _print_node_event(node_name: str, updates: dict) -> None:
    """Pretty-print a node completion event."""
    icons = {
        "analyser": "🔍",
        "clarifier": "✏️ ",
        "planner": "🗂️ ",
        "reviewer": "🔎",
        "writer": "💾",
    }
    icon = icons.get(node_name, "⚙️ ")
    print(f"  {icon}  [{node_name.upper()}] done", end="")
    if node_name == "planner" and isinstance(updates, dict):
        plan = updates.get("plan", {})
        n_tasks = len(plan.get("tasks", [])) if plan else 0
        if n_tasks:
            print(f" — {n_tasks} task(s) planned", end="")
    if node_name == "reviewer" and isinstance(updates, dict):
        verdict = "✅ approved" if updates.get("approved") else "❌ rejected"
        print(f" — {verdict}", end="")
    print()


def _list_artefacts(out_path: str) -> None:
    p = Path(out_path)
    if p.exists():
        files = list(p.iterdir())
        if files:
            print("\n  Files written:")
            for f in sorted(files):
                size_kb = f.stat().st_size / 1024
                print(f"    📄  {f.name}  ({size_kb:.1f} KB)")


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print(BANNER)

    # Resolve request
    user_request = args.request
    if not user_request:
        print("Enter your codebase activity request (press Enter twice when done):")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        user_request = "\n".join(lines).strip()

    if not user_request:
        print("❌  No request provided. Exiting.")
        sys.exit(1)

    # Resolve codebase path
    codebase_path = args.codebase.strip()
    if not codebase_path:
        codebase_path = input(
            "\nPath to existing codebase (leave blank for a new project): "
        ).strip()

    run_planner(user_request=user_request, codebase_path=codebase_path)


if __name__ == "__main__":
    main()
