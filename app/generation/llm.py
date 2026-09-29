"""
LLM client abstraction.

Supports Anthropic and OpenAI behind one interface so the rest of the app
(query rewriting, answer generation, groundedness checking) never depends on
a specific vendor SDK. Provider + model are chosen via config/.env.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Protocol

from app.config import get_settings


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...


class AnthropicClient:
    def __init__(self, api_key: str, model: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIClient:
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


@lru_cache
def get_llm_client() -> LLMClient:
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIClient(settings.openai_api_key, settings.openai_model)
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider}")


def safe_json_parse(text: str) -> dict:
    """LLMs sometimes wrap JSON in prose or code fences — strip that defensively."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM output: {text[:200]}")
    return json.loads(text[start:end + 1])
