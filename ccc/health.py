"""Deployment health tracking — failure counts and cooldown."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .exceptions import CCCError


class HealthTracker:
    """
    Tracks per-deployment failure counts broken down by error type.
    When failures of a given type exceed the allowed_fails threshold,
    the deployment enters cooldown for `cooldown_time` seconds.

    Falls back to in-memory storage when Redis is not configured.
    Uses Redis for shared state across multiple proxy instances when available.
    """

    def __init__(
        self,
        allowed_fails_policy: dict[str, int],
        cooldown_time: int = 60,
        redis_host: str | None = None,
        redis_port: int = 6379,
        redis_password: str | None = None,
    ) -> None:
        self.allowed_fails_policy = allowed_fails_policy
        self.cooldown_time = cooldown_time
        self._redis = self._connect_redis(redis_host, redis_port, redis_password)

        # In-memory fallback: deployment_id → {error_type: count}
        self._fail_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # deployment_id → cooldown_until timestamp
        self._cooldowns: dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_available(self, deployment_id: str) -> bool:
        """Return True if the deployment is not in cooldown."""
        if self._redis:
            return not self._redis_is_cooled_down(deployment_id)
        until = self._cooldowns.get(deployment_id, 0.0)
        return time.time() >= until

    def record_failure(self, deployment_id: str, error: "CCCError") -> None:
        """Record a failure; enter cooldown if threshold exceeded."""
        error_type = type(error).__name__
        if self._redis:
            self._redis_record_failure(deployment_id, error_type)
        else:
            self._mem_record_failure(deployment_id, error_type)

    def record_success(self, deployment_id: str) -> None:
        """Reset failure counts on success."""
        if self._redis:
            self._redis_reset(deployment_id)
        else:
            self._fail_counts.pop(deployment_id, None)
            self._cooldowns.pop(deployment_id, None)

    # ── In-memory implementation ───────────────────────────────────────────────

    def _mem_record_failure(self, deployment_id: str, error_type: str) -> None:
        self._fail_counts[deployment_id][error_type] += 1
        threshold = self.allowed_fails_policy.get(error_type, 3)
        if self._fail_counts[deployment_id][error_type] >= threshold:
            self._cooldowns[deployment_id] = time.time() + self.cooldown_time

    # ── Redis implementation ───────────────────────────────────────────────────

    @staticmethod
    def _connect_redis(host: str | None, port: int, password: str | None):
        if not host:
            return None
        try:
            import redis
            r = redis.Redis(host=host, port=port, password=password,
                            decode_responses=True, socket_connect_timeout=2)
            r.ping()
            return r
        except Exception:
            return None  # degrade gracefully to in-memory

    def _redis_record_failure(self, deployment_id: str, error_type: str) -> None:
        key = f"ccc:fails:{deployment_id}:{error_type}"
        count = self._redis.incr(key)
        self._redis.expire(key, self.cooldown_time * 10)
        threshold = self.allowed_fails_policy.get(error_type, 3)
        if count >= threshold:
            cooldown_key = f"ccc:cooldown:{deployment_id}"
            self._redis.setex(cooldown_key, self.cooldown_time, "1")

    def _redis_is_cooled_down(self, deployment_id: str) -> bool:
        return bool(self._redis.exists(f"ccc:cooldown:{deployment_id}"))

    def _redis_reset(self, deployment_id: str) -> None:
        keys = self._redis.keys(f"ccc:fails:{deployment_id}:*")
        if keys:
            self._redis.delete(*keys)
        self._redis.delete(f"ccc:cooldown:{deployment_id}")
