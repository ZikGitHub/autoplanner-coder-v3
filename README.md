# AutoPlanner Coder v3

A **LangGraph**-powered multi-agent workflow that generates structured development plans for any codebase — existing or new — from a single natural-language request.

---

## Architecture

```
[START]
   │
   ▼
┌──────────┐     scans files, detects languages,
│ Analyser │ ──► produces codebase technical summary
└──────────┘
   │
   ▼
┌───────────┐    sharpens the user request into a
│ Clarifier │ ──► precise, unambiguous specification
└───────────┘
   │
   ▼
┌─────────┐     generates a structured JSON plan
│ Planner │ ──► (tasks, files, priorities, effort)
└─────────┘
   │
   ▼
┌──────────┐    reviews plan for completeness &
│ Reviewer │ ──► feasibility — approves or rejects
└──────────┘
   │            (up to 2 retry loops back to Planner)
   ▼
┌────────┐      writes artefacts to output/
│ Writer │ ──► plan.json, plan_summary.md, review.md
└────────┘
   │
 [END]
```

---

## Output Artefacts

Each run creates a timestamped directory under `output/`:

```
output/
└── plan_20260629_201400/
    ├── plan.json           ← structured plan (tasks, deps, risks)
    ├── plan_summary.md     ← human-readable Markdown plan
    ├── codebase_summary.md ← technical summary of the analysed code
    └── review.md           ← reviewer verdict & feedback
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your LLM API key

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY)
```

### 3. Run

**Interactive mode** (prompts you for request & path):
```bash
python main.py
```

**CLI flags:**
```bash
python main.py --request "Add JWT authentication to all API endpoints" \
               --codebase ./my-fastapi-app
```

**New project** (no existing codebase):
```bash
python main.py --request "Build a REST API for a task management app using FastAPI"
```

---

## Supported LLM Providers

| Provider | Key env var | Default model |
|---|---|---|
| Google Gemini ⭐ | `GOOGLE_API_KEY` | `gemini-2.0-flash` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20241022` |

Force a specific provider/model:
```bash
PLANNER_PROVIDER=openai PLANNER_MODEL=gpt-4o-mini python main.py
```

---

## Project Structure

```
autoplanner-coder-v3/
├── autoplanner/
│   ├── __init__.py     ← package marker
│   ├── state.py        ← PlannerState TypedDict (shared graph state)
│   ├── llm.py          ← LLM factory (auto-detects provider)
│   ├── utils.py        ← codebase scanner, JSON extractor, artefact writer
│   ├── nodes.py        ← all five node functions + routing logic
│   └── graph.py        ← StateGraph assembly & compilation
├── main.py             ← CLI entry point (streaming output)
├── output/             ← generated plan artefacts (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```
