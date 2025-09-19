
import httpx, orjson
from .config import settings

class LLMClient:
    def __init__(self):
        self.base = settings.llm_base_url
        self.model = settings.llm_model
        self.backend = settings.llm_backend

    async def chat(self, system: str, user: str, timeout_ms: int = None) -> str:
        timeout_ms = timeout_ms or settings.timeout_ms
        async with httpx.AsyncClient(timeout=timeout_ms/1000) as client:
            if self.backend == "vllm":
                # OpenAI-compatible /v1/chat/completions
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role":"system","content":system},
                        {"role":"user","content":user}
                    ],
                    "max_tokens": 384,
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                }
                r = await client.post(f"{self.base}/v1/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            else:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role":"system","content":system},
                        {"role":"user","content":user}
                    ],
                    "options": {"num_predict": 384, "temperature": settings.temperature, "top_p": settings.top_p}
                }
                r = await client.post(f"{self.base}/api/chat", json=payload)
                r.raise_for_status()
                return r.json()["message"]["content"]
