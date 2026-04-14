"""FallbackRouter — retry-then-fallback core logic."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncGenerator

import httpx

from .config_loader import Config, Deployment
from .exceptions import (
    AllProvidersFailedError,
    CCCError,
    ContentPolicyError,
    ContextWindowError,
    ModelNotFoundError,
)
from .health import HealthTracker
from .providers.anthropic_provider import AnthropicProvider
from .providers.ollama_provider import OllamaProvider

if TYPE_CHECKING:
    from .providers.base import BaseProvider

logger = logging.getLogger(__name__)

_DEFAULT_RETRIES = 2


class FallbackRouter:
    """
    Routes a request to the correct model group, applies per-error retry
    policies, and falls back through configured fallback groups on failure.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.health = HealthTracker(
            allowed_fails_policy=config.allowed_fails_policy,
            cooldown_time=config.cooldown_time,
            redis_host=config.redis_host,
            redis_port=config.redis_port,
            redis_password=config.redis_password,
        )
        self._providers: dict[str, BaseProvider] = self._build_providers()

    # ── Public ─────────────────────────────────────────────────────────────────

    async def complete(
        self,
        model_group: str,
        body: bytes,
        client_headers: dict[str, str],
    ) -> httpx.Response:
        """
        Attempt the request against `model_group`, applying retries and
        fallbacks as configured.  Returns the first successful httpx.Response.
        """
        if model_group not in self.config.model_groups:
            raise ModelNotFoundError(f"Unknown model group: '{model_group}'")

        errors: list[tuple[str, Exception]] = []

        # Try primary group deployments
        response = await self._try_group(model_group, body, client_headers, errors)
        if response is not None:
            return response

        # Determine which fallback list to use based on last error type
        last_error = errors[-1][1] if errors else None
        fallback_names = self._get_fallback_names(model_group, last_error)

        for fb_group in fallback_names:
            if fb_group not in self.config.model_groups:
                logger.warning("Fallback group '%s' not found in model_list — skipping", fb_group)
                continue
            response = await self._try_group(fb_group, body, client_headers, errors)
            if response is not None:
                return response

        raise AllProvidersFailedError(errors)

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _try_group(
        self,
        group_name: str,
        body: bytes,
        client_headers: dict[str, str],
        errors: list[tuple[str, Exception]],
    ) -> httpx.Response | None:
        """Try every deployment in a group. Return response or None if all fail."""
        for deployment in self.config.model_groups[group_name]:
            if not self.health.is_available(deployment.id):
                logger.debug("Deployment %s is in cooldown — skipping", deployment.id)
                continue

            response = await self._try_deployment(deployment, body, client_headers, errors)
            if response is not None:
                return response

        return None

    async def _try_deployment(
        self,
        deployment: Deployment,
        body: bytes,
        client_headers: dict[str, str],
        errors: list[tuple[str, Exception]],
    ) -> httpx.Response | None:
        """Attempt a single deployment with retries. Return response or None."""
        provider = self._providers[deployment.id]
        last_error: CCCError | None = None
        attempt = 0

        while True:
            try:
                response = await provider.complete(body, client_headers)
                self.health.record_success(deployment.id)
                logger.info(
                    "Success: deployment=%s attempt=%d", deployment.id, attempt + 1
                )
                return response

            except CCCError as exc:
                last_error = exc
                error_type = type(exc).__name__

                # How many retries does the policy allow for this error type?
                policy_retries = self.config.retry_policy.get(error_type, _DEFAULT_RETRIES)

                logger.warning(
                    "Failure: deployment=%s attempt=%d error=%s: %s",
                    deployment.id, attempt + 1, error_type, str(exc)[:120],
                )

                if attempt >= policy_retries:
                    # Exhausted retries for this error type → record and give up on deployment
                    self.health.record_failure(deployment.id, exc)
                    errors.append((deployment.id, exc))
                    return None

                attempt += 1

        # Should not be reached, but guard anyway
        if last_error:
            self.health.record_failure(deployment.id, last_error)
            errors.append((deployment.id, last_error))
        return None

    def _get_fallback_names(
        self, model_group: str, last_error: Exception | None
    ) -> list[str]:
        """Return the appropriate fallback group list based on error type."""
        if isinstance(last_error, ContextWindowError):
            names = self.config.context_window_fallbacks.get(model_group)
            if names:
                return names
        # Default: generic fallbacks
        return self.config.fallbacks.get(model_group, [])

    async def stream(
        self,
        model_group: str,
        body: bytes,
        client_headers: dict[str, str],
    ) -> AsyncGenerator[bytes, None]:
        """
        Streaming variant of complete(): yields raw bytes from the first
        successful provider, applying the same group → fallback chain logic.
        Raises AllProvidersFailedError if every provider fails before yielding.
        """
        if model_group not in self.config.model_groups:
            raise ModelNotFoundError(f"Unknown model group: '{model_group}'")

        errors: list[tuple[str, Exception]] = []

        # Build ordered list of (group_name, deployment) to try
        candidates = list(self.config.model_groups[model_group])
        tried_primary = False

        for deployment in candidates:
            if not self.health.is_available(deployment.id):
                logger.debug("Deployment %s is in cooldown — skipping", deployment.id)
                continue

            provider = self._providers[deployment.id]
            gen = provider.stream(body, client_headers)

            # Peek at first chunk: errors surface here before we're committed
            try:
                first_chunk = await gen.__anext__()
            except StopAsyncIteration:
                self.health.record_success(deployment.id)
                return  # empty but successful response
            except CCCError as exc:
                logger.warning("Stream failure: deployment=%s error=%s", deployment.id, exc)
                self.health.record_failure(deployment.id, exc)
                errors.append((deployment.id, exc))
                if not tried_primary:
                    tried_primary = True
                continue

            # First chunk received — committed to this provider
            self.health.record_success(deployment.id)
            logger.info("Stream start: deployment=%s", deployment.id)
            yield first_chunk
            async for chunk in gen:
                yield chunk
            return

            tried_primary = True  # noqa: unreachable — marks primary group done

        # All primary deployments failed — try fallback groups
        last_error = errors[-1][1] if errors else None
        for fb_group in self._get_fallback_names(model_group, last_error):
            if fb_group not in self.config.model_groups:
                logger.warning("Fallback group '%s' not found — skipping", fb_group)
                continue

            for deployment in self.config.model_groups[fb_group]:
                if not self.health.is_available(deployment.id):
                    continue

                provider = self._providers[deployment.id]
                gen = provider.stream(body, client_headers)

                try:
                    first_chunk = await gen.__anext__()
                except StopAsyncIteration:
                    self.health.record_success(deployment.id)
                    return
                except CCCError as exc:
                    logger.warning("Stream fallback failure: deployment=%s error=%s",
                                   deployment.id, exc)
                    self.health.record_failure(deployment.id, exc)
                    errors.append((deployment.id, exc))
                    continue

                self.health.record_success(deployment.id)
                logger.info("Stream fallback success: deployment=%s", deployment.id)
                yield first_chunk
                async for chunk in gen:
                    yield chunk
                return

        raise AllProvidersFailedError(errors)

    def _build_providers(self) -> dict[str, BaseProvider]:
        """Instantiate one provider per deployment."""
        providers: dict[str, BaseProvider] = {}
        for deployments in self.config.model_groups.values():
            for dep in deployments:
                model_lower = dep.model.lower()
                if model_lower.startswith("ollama/"):
                    providers[dep.id] = OllamaProvider(dep, timeout=self.config.timeout)
                else:
                    providers[dep.id] = AnthropicProvider(
                        dep,
                        timeout=self.config.timeout,
                        forward_client_headers=self.config.forward_client_headers,
                    )
        return providers
