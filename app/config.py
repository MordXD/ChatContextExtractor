from pydantic import BaseModel
import os


class Settings(BaseModel):
    llm_backend: str = os.getenv("LLM_BACKEND", "vllm")  # vllm | ollama
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://vllm:8000")
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct")
    chroma_host: str = os.getenv("CHROMA_HOST", "chroma")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    collection: str = os.getenv("CHROMA_COLLECTION", "dialogs")
    max_ctx: int = int(os.getenv("MAX_CTX", "2048"))
    timeout_ms: int = int(os.getenv("TIMEOUT_MS", "800"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    database_url: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")
    analysis_max_traits: int = int(os.getenv("ANALYSIS_MAX_TRAITS", "5"))
    profile_min_confidence: float = float(os.getenv("PROFILE_MIN_CONFIDENCE", "0.35"))


settings = Settings()
