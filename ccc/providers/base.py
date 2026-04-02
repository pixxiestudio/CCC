"""Abstract provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from ..config_loader import Deployment


class BaseProvider(ABC):
    """
    A provider sends a raw Anthropic /v1/messages request body to an upstream
    endpoint and returns the raw httpx.Response (for transparent passthrough).

    Implementations must classify upstream HTTP errors into the typed exceptions
    defined in ccc.exceptions before raising them.
    """

    def __init__(self, deployment: Deployment, timeout: int = 60) -> None:
        self.deployment = deployment
        self.timeout = timeout

    @abstractmethod
    async def complete(self, body: bytes, client_headers: dict[str, str]) -> httpx.Response:
        """
        Send request; return raw httpx.Response on success.
        Raise a CCCError subclass on failure.
        """
        ...
