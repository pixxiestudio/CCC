"""Anthropic API provider — transparent httpx passthrough."""
from __future__ import annotations

import json
from typing import AsyncGenerator, NoReturn

import httpx

from ..config_loader import Deployment
from ..exceptions import (
    AuthenticationError,
    ContentPolicyError,
    ContextWindowError,
    InternalServerError,
    RateLimitError,
    TimeoutError,
)
from .base import BaseProvider

_ANTHROPIC_API_BASE = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"

# Keywords used to classify HTTP 400 errors
_CONTEXT_KEYWORDS = ("context_length_exceeded", "too_long", "context window",
                     "prompt is too long", "maximum context length")
_POLICY_KEYWORDS = ("content_policy", "content policy", "safety", "harmful")


def _classify_400(body: bytes) -> Exception:
    """Classify an HTTP 400 response body into ContextWindowError or ContentPolicyError."""
    try:
        text = body.decode(errors="replace").lower()
    except Exception:
        text = ""
    if any(kw in text for kw in _CONTEXT_KEYWORDS):
        return ContextWindowError(text)
    if any(kw in text for kw in _POLICY_KEYWORDS):
        return ContentPolicyError(text)
    return InternalServerError(f"HTTP 400: {text[:200]}")


class AnthropicProvider(BaseProvider):
    def __init__(self, deployment: Deployment, timeout: int = 60,
                 forward_client_headers: bool = False) -> None:
        super().__init__(deployment, timeout)
        self.forward_client_headers = forward_client_headers
        self._base = (deployment.api_base or _ANTHROPIC_API_BASE).rstrip("/")

    def _rewrite_model(self, body: bytes) -> bytes:
        """Replace the 'model' field with the real upstream model ID."""
        payload = json.loads(body)
        payload["model"] = self.deployment.model.removeprefix("anthropic/")
        return json.dumps(payload).encode()

    async def complete(self, body: bytes, client_headers: dict[str, str]) -> httpx.Response:
        headers = self._build_headers(client_headers)
        body = self._rewrite_model(body)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self._base}/v1/messages",
                    content=body,
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            raise TimeoutError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise InternalServerError(str(exc)) from exc

        if response.status_code == 200:
            return response

        self._raise_for_status(response)
        raise AssertionError("unreachable")  # keeps mypy happy

    async def stream(
        self, body: bytes, client_headers: dict[str, str]
    ) -> AsyncGenerator[bytes, None]:
        """True streaming — keeps the httpx connection open and yields chunks."""
        headers = self._build_headers(client_headers)
        body = self._rewrite_model(body)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._base}/v1/messages",
                    content=body,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        content = await response.aread()
                        self._raise_for_status(
                            httpx.Response(response.status_code, content=content)
                        )
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except httpx.TimeoutException as exc:
            raise TimeoutError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise InternalServerError(str(exc)) from exc

    def _build_headers(self, client_headers: dict[str, str]) -> dict[str, str]:
        headers: dict[str, str] = {
            "content-type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

        # Inject configured api_key unless we're forwarding the client's own token
        if self.forward_client_headers:
            # Forward whatever auth the client sent
            for h in ("x-api-key", "authorization"):
                if h in client_headers:
                    headers[h] = client_headers[h]
        elif self.deployment.api_key:
            headers["x-api-key"] = self.deployment.api_key

        # Forward useful passthrough headers from client
        for h in ("anthropic-beta", "anthropic-version"):
            if h in client_headers:
                headers[h] = client_headers[h]

        return headers

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> NoReturn:
        body = response.content
        status = response.status_code
        text = body[:200].decode(errors="replace")

        if status == 429:
            raise RateLimitError(f"HTTP 429: {text}")
        if status == 401:
            raise AuthenticationError(f"HTTP 401: {text}")
        if status in (408, 504):
            raise TimeoutError(f"HTTP {status}: {text}")
        if status == 400:
            raise _classify_400(body)
        raise InternalServerError(f"HTTP {status}: {text}")
