"""Pydantic models for the Anthropic Messages API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class ContentBlockText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"] = "image"
    source: dict[str, Any]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ContentBlockText | ContentBlockImage]


class MessagesRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 1024
    system: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    metadata: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


# ── Response ──────────────────────────────────────────────────────────────────

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ResponseContentBlock(BaseModel):
    type: str
    text: str | None = None

    model_config = {"extra": "allow"}


class MessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ResponseContentBlock]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: Usage

    model_config = {"extra": "allow"}


# ── Error response (Anthropic format) ────────────────────────────────────────

class ErrorDetail(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    error: ErrorDetail = Field(...)
