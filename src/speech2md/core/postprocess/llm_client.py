from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from speech2md.core.models import LLMBackend

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _post_with_retry(
    url: str,
    *,
    json: dict[str, Any],
    timeout: int,
    retries: int,
    backoff: float,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            resp = httpx.post(url, json=json, headers=headers, timeout=timeout)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
        else:
            if resp.status_code not in RETRYABLE_STATUS:
                return resp
            last_exc = httpx.HTTPStatusError(
                f"HTTP {resp.status_code} from {url}: {resp.text[:200]}",
                request=resp.request,
                response=resp,
            )

        if attempt < retries:
            delay = backoff * (2**attempt)
            logger.warning(
                "LLM request failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                retries + 1,
                last_exc,
                delay,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        ...


class OllamaClient(LLMClient):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma4",
        timeout: int = 120,
        max_tokens: int = 0,
        retries: int = 2,
        retry_backoff: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_backoff = retry_backoff

    def complete(self, prompt: str) -> str:
        params: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        if self.max_tokens:
            params["options"] = {"num_predict": self.max_tokens}
        resp = _post_with_retry(
            f"{self.base_url}/api/generate",
            json=params,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.retry_backoff,
        )
        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama: модель '{self.model}' не найдена. "
                f"Выполните: ollama pull {self.model}\nОтвет сервера: {resp.text}"
            )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return str(data["response"]).strip()


class OpenAIClient(LLMClient):
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        timeout: int = 120,
        max_tokens: int = 0,
        retries: int = 2,
        retry_backoff: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_backoff = retry_backoff

    def complete(self, prompt: str) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        resp = _post_with_retry(
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.retry_backoff,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return str(data["choices"][0]["message"]["content"]).strip()


def create_client(
    backend: LLMBackend | str,
    url: str = "",
    model: str = "",
    api_key: str = "",
    timeout: int = 120,
    max_tokens: int = 0,
    retries: int = 2,
    retry_backoff: float = 2.0,
) -> LLMClient:
    backend = LLMBackend(backend) if isinstance(backend, str) else backend
    if backend == LLMBackend.ollama:
        return OllamaClient(
            base_url=url,
            model=model or "gemma4",
            timeout=timeout,
            max_tokens=max_tokens,
            retries=retries,
            retry_backoff=retry_backoff,
        )
    return OpenAIClient(
        base_url=url,
        model=model or "gpt-4o-mini",
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        retries=retries,
        retry_backoff=retry_backoff,
    )
