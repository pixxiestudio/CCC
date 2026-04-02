"""Unit tests for AnthropicProvider."""
from __future__ import annotations

import json

import pytest

from ccc.config_loader import Deployment
from ccc.providers.anthropic_provider import AnthropicProvider


def _make_provider(model: str = "anthropic/claude-sonnet-4-6") -> AnthropicProvider:
    dep = Deployment(
        id="dep:1",
        model_name="dev-sonnet",
        model=model,
        api_key="sk-test",
        api_base=None,
        order=1,
    )
    return AnthropicProvider(dep)


def test_rewrite_model_strips_anthropic_prefix():
    provider = _make_provider("anthropic/claude-sonnet-4-6")
    body = json.dumps({"model": "dev-sonnet", "messages": []}).encode()
    result = json.loads(provider._rewrite_model(body))
    assert result["model"] == "claude-sonnet-4-6"


def test_rewrite_model_no_prefix():
    """Model IDs without 'anthropic/' prefix are used as-is."""
    provider = _make_provider("claude-opus-4-6")
    body = json.dumps({"model": "it-opus", "messages": []}).encode()
    result = json.loads(provider._rewrite_model(body))
    assert result["model"] == "claude-opus-4-6"


def test_rewrite_model_preserves_other_fields():
    provider = _make_provider("anthropic/claude-haiku-4-5-20251001")
    body = json.dumps({
        "model": "dev-haiku",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 128,
        "stream": True,
    }).encode()
    result = json.loads(provider._rewrite_model(body))
    assert result["model"] == "claude-haiku-4-5-20251001"
    assert result["max_tokens"] == 128
    assert result["stream"] is True
    assert result["messages"] == [{"role": "user", "content": "hi"}]
