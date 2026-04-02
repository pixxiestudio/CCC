"""Typed error hierarchy that maps to Anthropic HTTP error codes."""
from __future__ import annotations


class CCCError(Exception):
    """Base class for all CCC errors."""
    http_status: int = 500


class RateLimitError(CCCError):
    """HTTP 429 — provider rate limit hit."""
    http_status = 429


class AuthenticationError(CCCError):
    """HTTP 401 — invalid or expired API key."""
    http_status = 401


class TimeoutError(CCCError):
    """HTTP 408/504 — request timed out."""
    http_status = 408


class InternalServerError(CCCError):
    """HTTP 500 — provider internal error."""
    http_status = 500


class ContextWindowError(CCCError):
    """HTTP 400 — prompt exceeds model context window."""
    http_status = 400


class ContentPolicyError(CCCError):
    """HTTP 400 — request blocked by content policy."""
    http_status = 400


class ModelNotFoundError(CCCError):
    """HTTP 404 — unknown model group name."""
    http_status = 404


class AllProvidersFailedError(CCCError):
    """All deployments and fallbacks exhausted."""
    http_status = 503

    def __init__(self, errors: list[tuple[str, Exception]]) -> None:
        self.errors = errors
        parts = "; ".join(f"{name}: {err}" for name, err in errors)
        super().__init__(f"All providers failed — {parts}")
