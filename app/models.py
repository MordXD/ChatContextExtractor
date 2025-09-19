from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Tuple
from collections import OrderedDict
import hashlib
import time

@dataclass(frozen=True)
class DialogTurn:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class Dialog:
    turns: List[DialogTurn] = field(default_factory=list)

    @classmethod
    def from_pairs(cls, pairs: Iterable[Tuple[str, str]]) -> "Dialog":
        return cls([DialogTurn(role=r, content=c) for r, c in pairs])

    def tail(self, max_chars: int = 4000) -> "Dialog":
        """Возвращает усечённый хвост диалога по символам (с конца)."""
        acc, out = 0, []
        for t in reversed(self.turns):
            n = len(t.content)
            if acc + n <= max_chars:
                out.append(t)
                acc += n
            else:
                break
        out.reverse()
        return Dialog(out)

    def as_prompt(self) -> str:
        """Плоское представление для хэширования/кэша (минимально детерминированно)."""
        return "\n".join(f"[{t.role}] {t.content}" for t in self.turns)


@dataclass(frozen=True)
class ContextItemModel:
    kind: str                       # topic|task|question|decision|blocker|deadline
    text: str                       # короткий пункт (≤160 символов)
    who: Optional[str] = None
    when: Optional[str] = None      # ISO8601 или относительное
    weight: float = 1.0             # важность 0..1


@dataclass
class ExtractionResultModel:
    items: List[ContextItemModel] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0

class LRUCacheTTL:
    """
    Неблокирующий in-memory LRU+TTL.
    Используется для memo извлечений по ключу (диалог-хвост + версия промпта/схемы).
    """

    def __init__(self, capacity: int = 512, ttl_seconds: int = 30):
        self.capacity = max(8, capacity)
        self.ttl = max(1, ttl_seconds)
        self._od: OrderedDict[str, tuple[float, ExtractionResultModel]] = OrderedDict()

    def _evict(self):
        while len(self._od) > self.capacity:
            self._od.popitem(last=False)

    def get(self, key: str) -> Optional[ExtractionResultModel]:
        now = time.time()
        if key in self._od:
            ts, val = self._od[key]
            if now - ts <= self.ttl:
                # move to end (recently used)
                self._od.move_to_end(key, last=True)
                return val
            else:
                del self._od[key]
        return None

    def set(self, key: str, value: ExtractionResultModel):
        self._od[key] = (time.time(), value)
        self._od.move_to_end(key, last=True)
        self._evict()


def make_cache_key(dialog: Dialog, schema_version: str = "v1", prompt_version: str = "v1") -> str:
    """
    Строгий ключ кэша: SHA1( tail_prompt + schema_version + prompt_version )
    """
    blob = f"{dialog.as_prompt()}||schema:{schema_version}||prompt:{prompt_version}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


# Глобальный кэш можно импортировать в app.api и использовать опционально
global_extract_cache = LRUCacheTTL(capacity=1024, ttl_seconds=20)
