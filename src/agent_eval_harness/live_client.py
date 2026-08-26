"""Minimal Responses API client with bounded retries and no provider dependency."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIResponseError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class ResponseCall:
    payload: dict[str, Any]
    latency_ms: float


def load_api_key(env_path: Path | None = None) -> str:
    existing = os.environ.get("OPENAI_API_KEY", "").strip()
    if existing:
        return existing
    path = env_path or Path(".env.local")
    if not path.exists():
        raise OpenAIResponseError("OPENAI_API_KEY is not configured")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("OPENAI_API_KEY="):
            continue
        value = line.split("=", 1)[1].strip().strip('"').strip("'")
        if value:
            return value
    raise OpenAIResponseError("OPENAI_API_KEY is not configured")


class ResponsesClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 90.0,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @staticmethod
    def _safe_error(payload: dict[str, Any], fallback: str) -> tuple[str, str | None]:
        error = payload.get("error") or {}
        message = str(error.get("message") or fallback)
        code = error.get("code") or error.get("type")
        return message, str(code) if code is not None else None

    def create(self, payload: dict[str, Any]) -> ResponseCall:
        encoded = json.dumps(payload).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            request = Request(
                RESPONSES_URL,
                data=encoded,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "agent-evaluation-harness/0.4",
                },
                method="POST",
            )
            started = time.perf_counter()
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                decoded = json.loads(body)
                latency_ms = (time.perf_counter() - started) * 1_000
                if decoded.get("error"):
                    message, code = self._safe_error(decoded, "OpenAI API error")
                    raise OpenAIResponseError(message, code=code)
                return ResponseCall(decoded, latency_ms)
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                try:
                    decoded = json.loads(body)
                except json.JSONDecodeError:
                    decoded = {}
                message, code = self._safe_error(decoded, f"OpenAI API HTTP {exc.code}")
                retryable = exc.code in {408, 409, 429} or exc.code >= 500
                if not retryable or attempt >= self.max_retries:
                    raise OpenAIResponseError(message, status=exc.code, code=code) from exc
            except (TimeoutError, URLError) as exc:
                if attempt >= self.max_retries:
                    raise OpenAIResponseError("OpenAI API network request failed") from exc
            delay = min(8.0, (2**attempt) + random.random() * 0.25)
            time.sleep(delay)
        raise AssertionError("unreachable")
