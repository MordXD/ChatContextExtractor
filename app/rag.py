import chromadb
from chromadb.config import Settings
from typing import List

class RAG:
    def __init__(self, host: str, port: int, collection: str):
        self.client = chromadb.Client(Settings(chroma_server_host=host, chroma_server_http_port=port))
        self.col = self.client.get_or_create_collection(collection)

    def ingest(self, doc_id: str, text: str, metadata: dict|None=None):
        self.col.upsert(documents=[text], ids=[doc_id], metadatas=[metadata or {}])

    def query(self, q: str, k: int=4) -> List[str]:
        res = self.col.query(query_texts=[q], n_results=k)
        return res.get("documents", [[]])[0]
