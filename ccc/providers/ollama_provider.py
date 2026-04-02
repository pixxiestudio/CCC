"""Ollama provider — translates Anthropic ↔ OpenAI chat format."""
from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx

from ..config_loader import Deployment
from ..exceptions import InternalServerError, TimeoutError
from .base import BaseProvider

_DEFAULT_OLLAMA_BASE = "http://host.docker.internal:11434"


class OllamaProvider(BaseProvider):
    """
    Translates an Anthropic /v1/messages request body to OpenAI
    /v1/chat/completions, calls Ollama, and translates the response back to
    Anthropic format.  Used as the last-resort local fallback.
    """

    def __init__(self, deployment: Deployment, timeout: int = 120) -> None:
        super().__init__(deployment, timeout)
        base = (deployment.api_base or _DEFAULT_OLLAMA_BASE).rstrip("/")
        self._url = f"{base}/v1/chat/completions"
        # Strip "ollama/" prefix if present
        self._model = deployment.model.removeprefix("ollama/")

    async def complete(self, body: bytes, client_headers: dict[str, str]) -> httpx.Response:
        anthropic_req = json.loads(body)
        openai_body = self._to_openai(anthropic_req)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                oai_resp = await client.post(
                    self._url,
                    json=openai_body,
                    headers={"content-type": "application/json"},
                )
        except httpx.TimeoutException as exc:
            raise TimeoutError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise InternalServerError(str(exc)) from exc

        if oai_resp.status_code != 200:
            raise InternalServerError(
                f"Ollama HTTP {oai_resp.status_code}: {oai_resp.text[:200]}"
            )

        anthropic_body = self._to_anthropic(oai_resp.json(), anthropic_req)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=json.dumps(anthropic_body).encode(),
        )

    def _to_openai(self, req: dict) -> dict:
        messages = []
        if req.get("system"):
            messages.append({"role": "system", "content": req["system"]})
        for msg in req.get("messages", []):
            content = msg["content"]
            if isinstance(content, list):
                # Extract text from content blocks
                content = " ".join(
                    blk["text"] for blk in content
                    if isinstance(blk, dict) and blk.get("type") == "text"
                )
            messages.append({"role": msg["role"], "content": content})

        body: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": req.get("max_tokens", 1024),
            "stream": False,
        }
        if req.get("temperature") is not None:
            body["temperature"] = req["temperature"]
        return body

    async def stream(
        self, body: bytes, client_headers: dict[str, str]
    ) -> AsyncGenerator[bytes, None]:
        """Ollama is non-streaming; yield the full buffered response as one chunk."""
        response = await self.complete(body, client_headers)
        yield response.content

    @staticmethod
    def _to_anthropic(oai: dict, original_req: dict) -> dict:
        choice = oai["choices"][0]
        text = choice["message"]["content"] or ""
        usage = oai.get("usage", {})
        return {
            "id": oai.get("id", "msg_ollama"),
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": original_req.get("model", oai.get("model", "ollama")),
            "stop_reason": "end_turn" if choice.get("finish_reason") == "stop" else "max_tokens",
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }
