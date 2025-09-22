# app/schemas.py
from __future__ import annotations

import base64
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator


class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ContextItem(BaseModel):
    kind: str = Field(..., description="topic|task|question|decision|blocker|deadline")
    text: str = Field(..., description="short distilled fact or need")
    who: Optional[str] = None
    when: Optional[str] = None  # ISO8601 or relative
    weight: float = 1.0         # importance 0..1


class ExtractionResult(BaseModel):
    items: List[ContextItem]
    summary: str
    confidence: float = 0.85


class IngestPayload(BaseModel):
    doc_id: str
    text: str
    metadata: dict | None = None


class ExtractPayload(BaseModel):
    dialog: List[Message]
    k: int = 4


class UploadChatPayload(BaseModel):
    user_id: str = Field(..., description="User identifier (UUID)")
    text: Optional[str] = Field(default=None, description="Chat text payload")
    file_b64: Optional[str] = Field(default=None, description="Base64 encoded chat text")
    email: Optional[str] = Field(default=None)
    external_id: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_payload(cls, values: "UploadChatPayload") -> "UploadChatPayload":  # type: ignore[override]
        if not values.text and not values.file_b64:
            raise ValueError("Either 'text' or 'file_b64' must be provided")
        return values

    def resolve_text(self) -> str:
        if self.text is not None:
            return self.text
        assert self.file_b64 is not None
        try:
            decoded = base64.b64decode(self.file_b64.encode("utf-8"), validate=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("file_b64 must be base64-encoded UTF-8 text") from exc
        return decoded.decode("utf-8", errors="ignore")


class UploadChatResponse(BaseModel):
    chat_id: str
    status: str = "queued"


class ProfileResponse(BaseModel):
    user_id: str
    profile: dict[str, Any] = Field(default_factory=dict)
    version: int
    updated_at: Optional[datetime] = None


class ContextEventResponse(BaseModel):
    id: str
    user_id: str
    chat_id: str
    excerpt: str
    derived_traits: List[str]
    timestamp: datetime
