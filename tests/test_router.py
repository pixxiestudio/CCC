"""Unit tests for FallbackRouter using mock providers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ccc.config_loader import Config, Deployment
from ccc.exceptions import (
    AllProvidersFailedError,
    AuthenticationError,
    ContextWindowError,
    InternalServerError,
    ModelNotFoundError,
    RateLimitError,
)
from ccc.router import FallbackRouter


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_deployment(model_name: str, order: int, api_key: str = "tok") -> Deployment:
    return Deployment(
        id=f"{model_name}:{order}",
        model_name=model_name,
        model="anthropic/claude-opus-4-6",
        api_key=api_key,
        api_base=None,
        order=order,
    )


def _ok_response(text: str = "hello") -> httpx.Response:
    body = json.dumps({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-opus-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }).encode()
    return httpx.Response(200, content=body,
                          headers={"content-type": "application/json"})


def _make_config(
    groups: dict[str, list[Deployment]],
    fallbacks: dict | None = None,
    context_window_fallbacks: dict | None = None,
    retry_policy: dict | None = None,
    allowed_fails_policy: dict | None = None,
) -> Config:
    return Config(
        model_groups=groups,
        fallbacks=fallbacks or {},
        context_window_fallbacks=context_window_fallbacks or {},
        retry_policy=retry_policy or {},
        allowed_fails_policy=allowed_fails_policy or {},
        num_retries=2,
        cooldown_time=5,
    )


REQUEST_BODY = json.dumps({
    "model": "it-opus",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 10,
}).encode()

CLIENT_HEADERS: dict[str, str] = {}


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_success_on_first_deployment():
    dep = _make_deployment("it-opus", 1)
    cfg = _make_config({"it-opus": [dep]})
    r = FallbackRouter(cfg)

    with patch.object(r._providers[dep.id], "complete", new=AsyncMock(return_value=_ok_response())):
        resp = await r.complete("it-opus", REQUEST_BODY, CLIENT_HEADERS)

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_falls_back_to_second_deployment_on_error():
    dep1 = _make_deployment("it-opus", 1)
    dep2 = _make_deployment("it-opus", 2, api_key="tok-b")
    cfg = _make_config({"it-opus": [dep1, dep2]}, retry_policy={"RateLimitError": 0})
    r = FallbackRouter(cfg)

    with (
        patch.object(r._providers[dep1.id], "complete",
                     new=AsyncMock(side_effect=RateLimitError("429"))),
        patch.object(r._providers[dep2.id], "complete",
                     new=AsyncMock(return_value=_ok_response())),
    ):
        resp = await r.complete("it-opus", REQUEST_BODY, CLIENT_HEADERS)

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_falls_back_to_fallback_group():
    dep_primary = _make_deployment("it-opus", 1)
    dep_fallback = _make_deployment("api-fallback", 1, api_key="sk-paid")
    cfg = _make_config(
        {"it-opus": [dep_primary], "api-fallback": [dep_fallback]},
        fallbacks={"it-opus": ["api-fallback"]},
        retry_policy={"RateLimitError": 0},
    )
    r = FallbackRouter(cfg)

    with (
        patch.object(r._providers[dep_primary.id], "complete",
                     new=AsyncMock(side_effect=RateLimitError("429"))),
        patch.object(r._providers[dep_fallback.id], "complete",
                     new=AsyncMock(return_value=_ok_response("fallback ok"))),
    ):
        resp = await r.complete("it-opus", REQUEST_BODY, CLIENT_HEADERS)

    assert resp.status_code == 200
    assert b"fallback ok" in resp.content


@pytest.mark.asyncio
async def test_context_window_uses_cw_fallback():
    dep_primary = _make_deployment("it-opus", 1)
    dep_cw = _make_deployment("api-fallback", 1, api_key="sk-paid")
    cfg = _make_config(
        {"it-opus": [dep_primary], "api-fallback": [dep_cw]},
        fallbacks={"it-opus": ["local-fallback"]},          # generic → local
        context_window_fallbacks={"it-opus": ["api-fallback"]},  # cw → api
        retry_policy={"ContextWindowError": 0},
    )
    r = FallbackRouter(cfg)

    with (
        patch.object(r._providers[dep_primary.id], "complete",
                     new=AsyncMock(side_effect=ContextWindowError("too long"))),
        patch.object(r._providers[dep_cw.id], "complete",
                     new=AsyncMock(return_value=_ok_response("cw fallback"))),
    ):
        resp = await r.complete("it-opus", REQUEST_BODY, CLIENT_HEADERS)

    assert b"cw fallback" in resp.content


@pytest.mark.asyncio
async def test_all_providers_fail_raises():
    dep1 = _make_deployment("it-opus", 1)
    dep2 = _make_deployment("api-fallback", 1, api_key="sk-paid")
    cfg = _make_config(
        {"it-opus": [dep1], "api-fallback": [dep2]},
        fallbacks={"it-opus": ["api-fallback"]},
        retry_policy={"InternalServerError": 0},
    )
    r = FallbackRouter(cfg)

    with (
        patch.object(r._providers[dep1.id], "complete",
                     new=AsyncMock(side_effect=InternalServerError("500"))),
        patch.object(r._providers[dep2.id], "complete",
                     new=AsyncMock(side_effect=InternalServerError("500"))),
    ):
        with pytest.raises(AllProvidersFailedError) as exc_info:
            await r.complete("it-opus", REQUEST_BODY, CLIENT_HEADERS)

    assert len(exc_info.value.errors) >= 2


@pytest.mark.asyncio
async def test_unknown_model_group_raises():
    cfg = _make_config({"it-opus": [_make_deployment("it-opus", 1)]})
    r = FallbackRouter(cfg)

    with pytest.raises(ModelNotFoundError):
        await r.complete("unknown-model", REQUEST_BODY, CLIENT_HEADERS)


@pytest.mark.asyncio
async def test_retries_before_fallback():
    dep = _make_deployment("it-opus", 1)
    cfg = _make_config(
        {"it-opus": [dep]},
        retry_policy={"InternalServerError": 2},  # retry twice
    )
    r = FallbackRouter(cfg)

    call_count = 0

    async def flaky(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise InternalServerError("500")
        return _ok_response()

    with patch.object(r._providers[dep.id], "complete", new=AsyncMock(side_effect=flaky)):
        resp = await r.complete("it-opus", REQUEST_BODY, CLIENT_HEADERS)

    assert resp.status_code == 200
    assert call_count == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_rate_limit_zero_retries():
    """RateLimitErrorRetries: 0 means no retry — go straight to next deployment."""
    dep1 = _make_deployment("it-opus", 1)
    dep2 = _make_deployment("it-opus", 2, api_key="tok-b")
    cfg = _make_config(
        {"it-opus": [dep1, dep2]},
        retry_policy={"RateLimitError": 0},
    )
    r = FallbackRouter(cfg)

    call_count_dep1 = 0

    async def rate_limited(*args, **kwargs):
        nonlocal call_count_dep1
        call_count_dep1 += 1
        raise RateLimitError("429")

    with (
        patch.object(r._providers[dep1.id], "complete",
                     new=AsyncMock(side_effect=rate_limited)),
        patch.object(r._providers[dep2.id], "complete",
                     new=AsyncMock(return_value=_ok_response())),
    ):
        resp = await r.complete("it-opus", REQUEST_BODY, CLIENT_HEADERS)

    assert resp.status_code == 200
    assert call_count_dep1 == 1  # tried only once (0 retries)
