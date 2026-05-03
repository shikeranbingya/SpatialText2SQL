"""LLM-backed SQL generators."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


@dataclass
class SQLGeneratorResponse:
    text: str
    raw_response: Any = None
    usage: dict[str, Any] | None = None
    attempts: int = 1


class SQLGenerator(Protocol):
    def generate(self, prompt: str) -> SQLGeneratorResponse:
        ...


class OpenAICompatibleSQLGenerator:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        max_retries: int,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = self._create_client()

    def _resolve_api_key(self) -> str:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {self.api_key_env}")
        return api_key

    def _create_client(self):
        if OpenAI is None:
            raise ImportError("openai is not installed. Install it before running SQL synthesis.")
        return OpenAI(
            api_key=self._resolve_api_key(),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    def generate(self, prompt: str) -> SQLGeneratorResponse:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                choice = response.choices[0]
                content = getattr(choice.message, "content", "") or ""
                usage = getattr(response, "usage", None)
                usage_dict = None
                if usage is not None:
                    usage_dict = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                return SQLGeneratorResponse(
                    text=content,
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
                    usage=usage_dict,
                    attempts=attempt,
                )
            except Exception as exc:  # pragma: no cover - real API errors mocked in tests
                last_error = exc
                if attempt > self.max_retries:
                    break
                time.sleep(min(2** (attempt - 1), 4))
        raise RuntimeError(f"SQL generation failed after retries: {last_error}") from last_error


class OllamaSQLGenerator(OpenAICompatibleSQLGenerator):
    def _resolve_api_key(self) -> str:
        api_key = os.environ.get(self.api_key_env, "")
        return api_key or "ollama"


def build_sql_generator(
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key_env: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> SQLGenerator:
    normalized = str(provider or "").strip().lower().replace("-", "_")
    if normalized in {"openai_compatible", "openai"}:
        return OpenAICompatibleSQLGenerator(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    if normalized == "ollama":
        return OllamaSQLGenerator(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    raise ValueError(
        f"Unsupported SQL synthesis provider: {provider!r}. Supported values: openai_compatible, ollama."
    )


class MockSQLGenerator:
    def __init__(self, responses: list[Any] | None = None, callback: Callable[[str], Any] | None = None):
        self.responses = list(responses or [])
        self.callback = callback
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> SQLGeneratorResponse:
        self.prompts.append(prompt)
        if self.callback is not None:
            result = self.callback(prompt)
        else:
            if not self.responses:
                raise RuntimeError("MockSQLGenerator has no more queued responses.")
            result = self.responses.pop(0)
        if isinstance(result, SQLGeneratorResponse):
            return result
        return SQLGeneratorResponse(text=str(result), raw_response=result)
