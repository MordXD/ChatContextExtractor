# tests/test_extract.py
import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_extract.db")

import pytest

from app.schemas import ExtractionResult, Message
from app.api import _truncate_dialog


def test_truncate():
    dialog = [Message(role="user", content="a"*1000) for _ in range(10)]
    out = _truncate_dialog(dialog, max_chars=2000)
    assert len(out) < len(dialog)


def test_result_schema():
    # Проверяем, что схема адекватна при валидации
    payload = {
        "items":[{"kind":"task","text":"Implement /extract","who":None,"when":"2025-09-19","weight":0.9}],
        "summary":"Key tasks extracted",
        "confidence":0.9
    }
    assert ExtractionResult.model_validate(payload).summary.startswith("Key")
