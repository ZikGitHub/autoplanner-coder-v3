"""
LLM factory — returns a configured chat model based on the environment.

Priority (auto-detection order):
  1. Ollama        (PLANNER_PROVIDER=ollama  OR  OLLAMA_BASE_URL is set)
  2. Google Gemini (GOOGLE_API_KEY)
  3. OpenAI        (OPENAI_API_KEY)
  4. Anthropic     (ANTHROPIC_API_KEY)

Override via environment variables:
  PLANNER_PROVIDER  = "ollama" | "google" | "openai" | "anthropic"
  PLANNER_MODEL     = any model name understood by the chosen provider
  OLLAMA_BASE_URL   = base URL of the Ollama server (default: http://localhost:11434)
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.language_models import BaseChatModel


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.2) -> BaseChatModel:
    """Return a cached LLM instance."""
    provider = os.getenv("PLANNER_PROVIDER", "").lower()

    # ── Auto-detect provider from environment ─────────────────────────────────
    if not provider:
        if os.getenv("OLLAMA_BASE_URL"):
            provider = "ollama"
        elif os.getenv("GOOGLE_API_KEY"):
            provider = "google"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            # Default to local Ollama — no API key needed
            provider = "ollama"

    # ── Ollama (local) ────────────────────────────────────────────────────────
    if provider == "ollama":
        from langchain_ollama import ChatOllama  # type: ignore[import]

        model = os.getenv("PLANNER_MODEL", "llama3.2:1b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)

    # ── Google Gemini ─────────────────────────────────────────────────────────
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]

        model = os.getenv("PLANNER_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    # ── OpenAI ────────────────────────────────────────────────────────────────
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore[import]

        model = os.getenv("PLANNER_MODEL", "gpt-4o")
        return ChatOpenAI(model=model, temperature=temperature)

    # ── Anthropic ─────────────────────────────────────────────────────────────
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore[import]

        model = os.getenv("PLANNER_MODEL", "claude-3-5-sonnet-20241022")
        return ChatAnthropic(model=model, temperature=temperature)  # type: ignore[call-arg]

    raise ValueError(f"Unknown PLANNER_PROVIDER: {provider!r}")
