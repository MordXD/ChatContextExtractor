# app/api.py
from fastapi import FastAPI, Depends
from fastapi.responses import ORJSONResponse
from .schemas import IngestPayload, ExtractPayload, ExtractionResult, Message
from .config import settings
from .deps import get_settings
from .rag import RAG
from .llm import LLMClient
from .sgrops import SYSTEM_SGR, USER_TEMPLATE, schema_brief
import orjson
from typing import List

app = FastAPI(default_response_class=ORJSONResponse)
rag = RAG(settings.chroma_host, settings.chroma_port, settings.collection)
llm = LLMClient()

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
