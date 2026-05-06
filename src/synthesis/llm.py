"""Shared LLM configuration and OpenAI-compatible client wrappers for synthesis workflows."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, TypeVar

from openai import OpenAI


@dataclass(frozen=True)
class SynthesisLLMConfig:
    provider: str = "openai_compatible"
    model: str = "gpt-4o-mini"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout: int = 120
    max_retries: int = 2


@dataclass
class LLMGenerationResponse:
    text: str
    raw_response: Any = None
    usage: dict[str, Any] | None = None
    attempts: int = 1


class LLMClient(Protocol):
    def generate(self, prompt: str) -> LLMGenerationResponse:
        ...


class OpenAICompatibleLLMClient:
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
        run_label: str = "LLM request",
        missing_dependency_label: str = "this synthesis workflow",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.run_label = run_label
        self.missing_dependency_label = missing_dependency_label
        self.client = self._create_client()

    def _resolve_api_key(self) -> str:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {self.api_key_env}")
        return api_key

    def _create_client(self):
        return OpenAI(
            api_key=self._resolve_api_key(),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    def generate(self, prompt: str) -> LLMGenerationResponse:
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
                return LLMGenerationResponse(
                    text=content,
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
                    usage=usage_dict,
                    attempts=attempt,
                )
            except Exception as exc:  # pragma: no cover - real API failures are mocked in tests
                last_error = exc
                if attempt > self.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"{self.run_label} failed after retries: {last_error}") from last_error


class OllamaLLMClient(OpenAICompatibleLLMClient):
    def _resolve_api_key(self) -> str:
        api_key = os.environ.get(self.api_key_env, "")
        return api_key or "ollama"


class MockLLMClient:
    def __init__(self, responses: list[Any] | None = None, callback: Callable[[str], Any] | None = None):
        self.responses = list(responses or [])
        self.callback = callback
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> LLMGenerationResponse:
        self.prompts.append(prompt)
        if self.callback is not None:
            result = self.callback(prompt)
        else:
            if not self.responses:
                raise RuntimeError(f"{self.__class__.__name__} has no more queued responses.")
            result = self.responses.pop(0)
        if isinstance(result, LLMGenerationResponse):
            return result
        return LLMGenerationResponse(text=str(result), raw_response=result)


def normalize_llm_provider(provider: str) -> str:
    return str(provider or "").strip().lower().replace("-", "_")


def build_llm_client(
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key_env: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    openai_client_cls: type[OpenAICompatibleLLMClient] = OpenAICompatibleLLMClient,
    ollama_client_cls: type[OllamaLLMClient] = OllamaLLMClient,
    mock_client_cls: type[MockLLMClient] = MockLLMClient,
    run_label: str = "LLM request",
    missing_dependency_label: str = "this synthesis workflow",
) -> LLMClient:
    normalized = normalize_llm_provider(provider)
    common_kwargs = {
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": max_retries,
        "run_label": run_label,
        "missing_dependency_label": missing_dependency_label,
    }
    if normalized in {"openai_compatible", "openai"}:
        return openai_client_cls(**common_kwargs)
    if normalized == "ollama":
        return ollama_client_cls(**common_kwargs)
    if normalized == "mock":
        return mock_client_cls()
    raise ValueError(
        f"Unsupported provider: {provider!r}. Supported values: openai_compatible, openai, ollama, mock."
    )


TLLMConfig = TypeVar("TLLMConfig", bound=SynthesisLLMConfig)


def build_llm_config_from_section(
    llm_section: Mapping[str, Any] | None,
    default_config: TLLMConfig,
    *,
    as_text: Callable[[Any, str], str],
    as_float: Callable[[Any, float], float],
    as_positive_int: Callable[[Any, int], int],
    as_non_negative_int: Callable[[Any, int], int],
) -> TLLMConfig:
    payload = llm_section or {}
    config_cls = type(default_config)
    return config_cls(
        provider=as_text(payload.get("provider"), default_config.provider),
        model=as_text(payload.get("model"), default_config.model),
        base_url=as_text(payload.get("base_url"), default_config.base_url),
        api_key_env=as_text(payload.get("api_key_env"), default_config.api_key_env),
        temperature=as_float(payload.get("temperature"), default_config.temperature),
        max_tokens=as_positive_int(payload.get("max_tokens"), default_config.max_tokens),
        timeout=as_positive_int(payload.get("timeout"), default_config.timeout),
        max_retries=as_non_negative_int(payload.get("max_retries"), default_config.max_retries),
    )
