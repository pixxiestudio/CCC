"""Parse config.yaml and resolve os.environ/VAR_NAME references."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Deployment:
    """One entry in model_list — a specific model + API key combination."""
    id: str           # "{model_name}:{order}"
    model_name: str   # logical group name, e.g. "it-opus"
    model: str        # actual model string, e.g. "anthropic/claude-opus-4-6"
    api_key: str | None
    api_base: str | None
    order: int


@dataclass
class Config:
    # model_name → list of Deployments sorted by order
    model_groups: dict[str, list[Deployment]] = field(default_factory=dict)

    # model_name → ordered list of fallback group names
    fallbacks: dict[str, list[str]] = field(default_factory=dict)
    context_window_fallbacks: dict[str, list[str]] = field(default_factory=dict)

    # error class name (without "Retries"/"AllowedFails") → count
    retry_policy: dict[str, int] = field(default_factory=dict)
    allowed_fails_policy: dict[str, int] = field(default_factory=dict)

    num_retries: int = 2
    cooldown_time: int = 60
    timeout: int = 60

    master_key: str | None = None
    disable_key_check: bool = False
    forward_client_headers: bool = False

    redis_host: str | None = None
    redis_port: int = 6379
    redis_password: str | None = None


# ── Env-var resolution ────────────────────────────────────────────────────────

def _resolve(value: str | None) -> str | None:
    """Resolve 'os.environ/VAR_NAME' to the actual env var value."""
    if value is None:
        return None
    if isinstance(value, str) and value.startswith("os.environ/"):
        var = value[len("os.environ/"):]
        return os.environ.get(var)
    return value


# ── Fallback dict parsing ─────────────────────────────────────────────────────

def _parse_fallback_list(raw: list[dict] | None) -> dict[str, list[str]]:
    """[{model: [fb1, fb2]}, ...] → {model: [fb1, fb2]}"""
    result: dict[str, list[str]] = {}
    if not raw:
        return result
    for entry in raw:
        for model, targets in entry.items():
            result[model] = list(targets)
    return result


# ── Policy parsing ────────────────────────────────────────────────────────────

def _parse_retry_policy(raw: dict | None) -> dict[str, int]:
    """RateLimitErrorRetries: 0 → {"RateLimitError": 0}"""
    result: dict[str, int] = {}
    if not raw:
        return result
    for key, val in raw.items():
        error_name = key.replace("Retries", "")
        result[error_name] = int(val)
    return result


def _parse_allowed_fails(raw: dict | None) -> dict[str, int]:
    """RateLimitErrorAllowedFails: 1 → {"RateLimitError": 1}"""
    result: dict[str, int] = {}
    if not raw:
        return result
    for key, val in raw.items():
        error_name = key.replace("AllowedFails", "")
        result[error_name] = int(val)
    return result


# ── Main loader ───────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config.yaml",
                env_path: str | Path = ".env") -> Config:
    load_dotenv(env_path, override=False)

    raw = yaml.safe_load(Path(config_path).read_text())

    cfg = Config()

    # ── master_key ────────────────────────────────────────────────────────────
    general = raw.get("general_settings", {})
    cfg.master_key = _resolve(general.get("master_key")) or os.environ.get("LITELLM_MASTER_KEY")
    cfg.disable_key_check = general.get("disable_key_check", False)
    cfg.forward_client_headers = general.get("forward_llm_provider_auth_headers", False)

    # ── model list → groups ───────────────────────────────────────────────────
    groups: dict[str, list[Deployment]] = {}
    for i, entry in enumerate(raw.get("model_list", [])):
        name: str = entry["model_name"]
        params: dict = entry.get("litellm_params", {})
        info: dict = entry.get("model_info", {})

        order = int(info.get("order", i + 1))
        deployment = Deployment(
            id=f"{name}:{order}",
            model_name=name,
            model=params.get("model", ""),
            api_key=_resolve(params.get("api_key")),
            api_base=_resolve(params.get("api_base")),
            order=order,
        )
        groups.setdefault(name, []).append(deployment)

    # sort each group by order
    for name in groups:
        groups[name].sort(key=lambda d: d.order)
    cfg.model_groups = groups

    # ── router settings ───────────────────────────────────────────────────────
    router = raw.get("router_settings", {})
    cfg.fallbacks = _parse_fallback_list(router.get("fallbacks"))
    cfg.context_window_fallbacks = _parse_fallback_list(router.get("context_window_fallbacks"))
    cfg.num_retries = int(router.get("num_retries", 2))
    cfg.cooldown_time = int(router.get("cooldown_time", 60))
    cfg.timeout = int(router.get("timeout", 60))

    if router.get("redis_host"):
        cfg.redis_host = _resolve(router["redis_host"])
        cfg.redis_port = int(router.get("redis_port", 6379))
        cfg.redis_password = _resolve(router.get("redis_password"))

    # ── litellm settings (retry / allowed_fails) ──────────────────────────────
    litellm = raw.get("litellm_settings", {})
    cfg.retry_policy = _parse_retry_policy(litellm.get("retry_policy"))
    cfg.allowed_fails_policy = _parse_allowed_fails(litellm.get("allowed_fails_policy"))
    # litellm_settings.num_retries overrides router_settings if present
    if "num_retries" in litellm:
        cfg.num_retries = int(litellm["num_retries"])

    return cfg
