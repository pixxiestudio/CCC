"""FastAPI proxy server — /v1/messages with auth + streaming passthrough."""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config_loader import load_config
from .exceptions import (
    AllProvidersFailedError,
    AuthenticationError,
    CCCError,
    ModelNotFoundError,
)
from .router import FallbackRouter

logger = logging.getLogger(__name__)

# ── App state ──────────────────────────────────────────────────────────────────

router: FallbackRouter | None = None
master_key: str | None = None
disable_key_check: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router, master_key, disable_key_check

    config_path = os.environ.get("CCC_CONFIG", "config.yaml")
    env_path = os.environ.get("CCC_ENV", ".env")

    cfg = load_config(config_path, env_path)
    router = FallbackRouter(cfg)
    master_key = cfg.master_key
    disable_key_check = cfg.disable_key_check

    logger.info("CCC proxy ready — %d model groups loaded", len(cfg.model_groups))
    yield


app = FastAPI(title="CCC Proxy", version="0.1.0", lifespan=lifespan)

# ── Auth helper ────────────────────────────────────────────────────────────────

def _check_auth(request: Request) -> bool:
    if disable_key_check or not master_key:
        return True
    provided = (
        request.headers.get("x-api-key")
        or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    )
    return provided == master_key


def _client_headers(request: Request) -> dict[str, str]:
    """Extract relevant passthrough headers (lowercase keys)."""
    passthrough = ("x-api-key", "authorization", "anthropic-beta", "anthropic-version")
    return {k.lower(): v for k, v in request.headers.items() if k.lower() in passthrough}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/messages")
async def messages(request: Request):
    if not _check_auth(request):
        return JSONResponse(
            status_code=401,
            content={"type": "error", "error": {"type": "authentication_error",
                                                  "message": "Invalid API key"}},
        )

    body = await request.body()

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400,
                            content={"type": "error",
                                     "error": {"type": "invalid_request_error",
                                               "message": "Invalid JSON body"}})

    model_group: str = payload.get("model", "")
    is_stream: bool = payload.get("stream", False)

    try:
        upstream = await router.complete(model_group, body, _client_headers(request))
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404,
                            content=_error_body("not_found_error", str(exc)))
    except AllProvidersFailedError as exc:
        logger.error("All providers failed for '%s': %s", model_group, exc)
        return JSONResponse(status_code=503,
                            content=_error_body("overloaded_error", str(exc)))
    except CCCError as exc:
        return JSONResponse(status_code=exc.http_status,
                            content=_error_body("api_error", str(exc)))

    # Stream passthrough
    if is_stream:
        return StreamingResponse(
            upstream.aiter_bytes(),
            status_code=upstream.status_code,
            media_type="text/event-stream",
            headers=_safe_headers(upstream.headers),
        )

    # Non-streaming
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type="application/json",
        headers=_safe_headers(upstream.headers),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _error_body(error_type: str, message: str) -> dict:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _safe_headers(upstream_headers) -> dict[str, str]:
    """Forward a safe subset of upstream response headers."""
    forward = ("content-type", "x-request-id", "anthropic-version",
               "request-id", "cf-ray")
    return {k: v for k, v in upstream_headers.items() if k.lower() in forward}


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    uvicorn.run(
        "ccc.server:app",
        host=os.environ.get("CCC_HOST", "0.0.0.0"),
        port=int(os.environ.get("CCC_PORT", "4000")),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
