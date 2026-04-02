"""Abstract provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

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
        Send request (non-streaming); return raw httpx.Response on success.
        Raise a CCCError subclass on failure.
        """
        ...

    @abstractmethod
    async def stream(
        self, body: bytes, client_headers: dict[str, str]
    ) -> AsyncGenerator[bytes, None]:
        """
        Send request and yield raw response bytes as they arrive (true streaming).
        Raise a CCCError subclass on failure before yielding starts.
        """
        ...
