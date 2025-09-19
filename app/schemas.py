# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

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
