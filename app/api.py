# app/api.py
from __future__ import annotations

from typing import List
from uuid import UUID

import orjson
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .deps import get_session, get_settings
from .llm import LLMClient
from .profile_service import process_chat
from .rag import RAG
from .schemas import (
    ContextEventResponse,
    ExtractPayload,
    ExtractionResult,
    IngestPayload,
    Message,
    ProfileResponse,
    UploadChatPayload,
    UploadChatResponse,
)
from .sgrops import SYSTEM_SGR, USER_TEMPLATE, schema_brief
from .db import Chat, ContextEvent, Profile, User, init_db

app = FastAPI(default_response_class=ORJSONResponse)
rag = RAG(settings.chroma_host, settings.chroma_port, settings.collection)
llm = LLMClient()


@app.on_event("startup")
async def _startup() -> None:
    await init_db()


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/ingest")
async def ingest(p: IngestPayload):
    rag.ingest(p.doc_id, p.text, p.metadata)
    return {"status": "ok"}


def _truncate_dialog(dialog: List[Message], max_chars: int = 4000) -> List[Message]:
    acc, out = 0, []
    for m in reversed(dialog):
        if acc + len(m.content) <= max_chars:
            out.append(m)
            acc += len(m.content)
        else:
            break
    return list(reversed(out))


@app.post("/extract", response_model=ExtractionResult)
async def extract(p: ExtractPayload, cfg=Depends(get_settings)):
    dialog = _truncate_dialog(p.dialog)
    # RAG: берем top-k сниппетов по последнему юзер-сообщению
    q = next((m.content for m in reversed(dialog) if m.role == "user"), dialog[-1].content if dialog else "")
    snippets = rag.query(q, k=p.k)
    user = USER_TEMPLATE.format(
        dialog=orjson.dumps([m.model_dump() for m in dialog]).decode(),
        snippets=orjson.dumps(snippets).decode(),
        schema_brief=schema_brief()
    )
    raw = await llm.chat(SYSTEM_SGR, user, timeout_ms=cfg.timeout_ms)
    # Жесткая очистка — только JSON
    json_start = raw.find("{")
    json_end = raw.rfind("}")
    cleaned = raw[json_start:json_end+1] if json_start != -1 else '{"items":[],"summary":"","confidence":0.0}'
    return ExtractionResult.model_validate_json(cleaned)


def _normalize_user_id(user_id: str) -> str:
    try:
        return str(UUID(user_id))
    except Exception:
        return user_id


async def _ensure_user(session: AsyncSession, user_id: str, email: str | None, external_id: str | None) -> User:
    normalized_id = _normalize_user_id(user_id)
    user = await session.get(User, normalized_id)
    if user is None:
        user = User(id=normalized_id, email=email, external_id=external_id)
        session.add(user)
        await session.flush()
        return user

    updated = False
    if email and email != user.email:
        user.email = email
        updated = True
    if external_id and external_id != user.external_id:
        user.external_id = external_id
        updated = True
    if updated:
        await session.flush()
    return user


@app.post("/upload_chat", response_model=UploadChatResponse)
async def upload_chat(
    payload: UploadChatPayload,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> UploadChatResponse:
    try:
        chat_text = payload.resolve_text()
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized_text = chat_text.strip()
    if not normalized_text:
        raise HTTPException(status_code=400, detail="Chat text cannot be empty")

    user = await _ensure_user(session, payload.user_id, payload.email, payload.external_id)
    chat = Chat(user_id=user.id, raw_text=normalized_text)
    session.add(chat)
    await session.flush()
    chat_id = chat.id
    await session.commit()

    background.add_task(process_chat, chat_id)
    return UploadChatResponse(chat_id=chat_id, status="queued")


@app.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: str, session: AsyncSession = Depends(get_session)) -> ProfileResponse:
    normalized_id = _normalize_user_id(user_id)
    profile = await session.get(Profile, normalized_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="profile not found")

    return ProfileResponse(
        user_id=profile.user_id,
        profile=profile.profile_json or {},
        version=profile.version,
        updated_at=profile.updated_at,
    )


@app.get("/profile/{user_id}/contexts", response_model=List[ContextEventResponse])
async def list_contexts(user_id: str, session: AsyncSession = Depends(get_session)) -> List[ContextEventResponse]:
    normalized_id = _normalize_user_id(user_id)
    query = (
        select(ContextEvent)
        .where(ContextEvent.user_id == normalized_id)
        .order_by(ContextEvent.timestamp.desc())
    )
    result = await session.execute(query)
    events = result.scalars().all()
    return [
        ContextEventResponse(
            id=event.id,
            user_id=event.user_id,
            chat_id=event.chat_id,
            excerpt=event.excerpt,
            derived_traits=event.derived_traits,
            timestamp=event.timestamp,
        )
        for event in events
    ]
