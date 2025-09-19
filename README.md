# ChatContextExtractor (FastAPI + Qwen3-4B + ChromaDB + SGR)

Микросервис для извлечения контекста из диалогов с жёстко управляемым **SGR**-циклом и строгим **Structured Output** (Pydantic). Поддерживает RAG на **ChromaDB**, абстракцию под **vLLM** (OpenAI-совместимый API) и **Ollama**. Заточен под низкую задержку и стабильность вывода (валидный JSON).

## Стек
- **FastAPI** + `orjson` (быстрые ответы)
- **SGR** (фиксированный reasoning-loop + schema guard)
- **Qwen3-4B-Instruct** через **vLLM** (или `qwen2.5` в Ollama как fallback)
- **ChromaDB** для простого RAG
- **Pydantic v2** схемы
- Docker/Compose для быстрого старта

