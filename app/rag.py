from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
except Exception:  # pragma: no cover - fallback when chromadb unavailable
    chromadb = None
    Settings = None


class _InMemoryCollection:
    """Minimal in-memory fallback for tests when ChromaDB is unavailable."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}
        self._order: Deque[str] = deque()
        self._max_documents = 128

    def upsert(self, documents: List[str], ids: List[str], metadatas: Optional[List[dict]] = None) -> None:
        for doc_id, text in zip(ids, documents, strict=False):
            self._store[doc_id] = text
            self._order.append(doc_id)
            if len(self._order) > self._max_documents:
                oldest = self._order.popleft()
                self._store.pop(oldest, None)

    def query(self, query_texts: List[str], n_results: int = 4) -> dict:
        # naive fallback: return most recent documents containing query terms
        query = query_texts[0] if query_texts else ""
        if not query:
            items = list(self._store.values())[-n_results:]
            return {"documents": [items]}
        lowered = query.lower()
        matches = [text for text in reversed(list(self._store.values())) if lowered in text.lower()]
        return {"documents": [matches[:n_results]]}


class RAG:
    def __init__(self, host: str, port: int, collection: str):
        if chromadb is None:
            self.client = None
            self.col = _InMemoryCollection()
        else:  # pragma: no cover - requires chromadb runtime
            self.client = chromadb.Client(Settings(chroma_server_host=host, chroma_server_http_port=port))
            self.col = self.client.get_or_create_collection(collection)

    def ingest(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        self.col.upsert(documents=[text], ids=[doc_id], metadatas=[metadata or {}])

    def query(self, q: str, k: int = 4) -> List[str]:
        res = self.col.query(query_texts=[q], n_results=k)
        return res.get("documents", [[]])[0]
