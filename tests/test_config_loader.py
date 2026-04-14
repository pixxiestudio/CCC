"""Tests for config_loader — YAML parsing + env var resolution."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from ccc.config_loader import load_config


SAMPLE_CONFIG = textwrap.dedent("""\
    model_list:
      - model_name: it-opus
        litellm_params:
          model: anthropic/claude-opus-4-6
          api_base: https://api.anthropic.com
          api_key: os.environ/SUB_A_OAUTH_TOKEN
        model_info:
          order: 1

      - model_name: it-opus
        litellm_params:
          model: anthropic/claude-opus-4-6
          api_base: https://api.anthropic.com
          api_key: os.environ/SUB_B_OAUTH_TOKEN
        model_info:
          order: 2

      - model_name: api-fallback
        litellm_params:
          model: anthropic/claude-sonnet-4-6
          api_key: os.environ/ANTHROPIC_API_KEY

      - model_name: local-fallback
        litellm_params:
          model: ollama/llama3.2
          api_base: http://host.docker.internal:11434

    router_settings:
      num_retries: 2
      cooldown_time: 60
      fallbacks:
        - it-opus: ["api-fallback", "local-fallback"]
      context_window_fallbacks:
        - it-opus: ["api-fallback"]

    litellm_settings:
      retry_policy:
        RateLimitErrorRetries: 0
        AuthenticationErrorRetries: 0
        TimeoutErrorRetries: 2
        InternalServerErrorRetries: 3
      allowed_fails_policy:
        RateLimitErrorAllowedFails: 1
        TimeoutErrorAllowedFails: 3
        InternalServerErrorAllowedFails: 5

    general_settings:
      disable_key_check: true
      master_key: os.environ/LITELLM_MASTER_KEY
""")


@pytest.fixture()
def config_file(tmp_path: Path):
    f = tmp_path / "config.yaml"
    f.write_text(SAMPLE_CONFIG)
    return f


def test_model_groups_created(config_file, monkeypatch):
    monkeypatch.setenv("SUB_A_OAUTH_TOKEN", "tok-a")
    monkeypatch.setenv("SUB_B_OAUTH_TOKEN", "tok-b")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-paid")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master")

    cfg = load_config(config_file, env_path="/dev/null")

    assert "it-opus" in cfg.model_groups
    assert "api-fallback" in cfg.model_groups
    assert "local-fallback" in cfg.model_groups


def test_deployments_sorted_by_order(config_file, monkeypatch):
    monkeypatch.setenv("SUB_A_OAUTH_TOKEN", "tok-a")
    monkeypatch.setenv("SUB_B_OAUTH_TOKEN", "tok-b")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-paid")

    cfg = load_config(config_file, env_path="/dev/null")

    group = cfg.model_groups["it-opus"]
    assert len(group) == 2
    assert group[0].order == 1
    assert group[1].order == 2


def test_env_var_resolution(config_file, monkeypatch):
    monkeypatch.setenv("SUB_A_OAUTH_TOKEN", "tok-a")
    monkeypatch.setenv("SUB_B_OAUTH_TOKEN", "tok-b")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-paid")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master")

    cfg = load_config(config_file, env_path="/dev/null")

    assert cfg.model_groups["it-opus"][0].api_key == "tok-a"
    assert cfg.model_groups["it-opus"][1].api_key == "tok-b"
    assert cfg.model_groups["api-fallback"][0].api_key == "sk-paid"
    assert cfg.master_key == "sk-master"


def test_env_var_missing_returns_none(config_file, monkeypatch):
    monkeypatch.delenv("SUB_A_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("SUB_B_OAUTH_TOKEN", "tok-b")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-paid")

    cfg = load_config(config_file, env_path="/dev/null")
    assert cfg.model_groups["it-opus"][0].api_key is None


def test_fallback_chains_parsed(config_file, monkeypatch):
    cfg = load_config(config_file, env_path="/dev/null")

    assert cfg.fallbacks["it-opus"] == ["api-fallback", "local-fallback"]
    assert cfg.context_window_fallbacks["it-opus"] == ["api-fallback"]


def test_retry_policy_parsed(config_file, monkeypatch):
    cfg = load_config(config_file, env_path="/dev/null")

    assert cfg.retry_policy["RateLimitError"] == 0
    assert cfg.retry_policy["TimeoutError"] == 2
    assert cfg.retry_policy["InternalServerError"] == 3


def test_allowed_fails_policy_parsed(config_file, monkeypatch):
    cfg = load_config(config_file, env_path="/dev/null")

    assert cfg.allowed_fails_policy["RateLimitError"] == 1
    assert cfg.allowed_fails_policy["TimeoutError"] == 3


def test_ollama_deployment_has_no_api_key(config_file):
    cfg = load_config(config_file, env_path="/dev/null")

    ollama = cfg.model_groups["local-fallback"][0]
    assert ollama.model == "ollama/llama3.2"
    assert ollama.api_key is None
