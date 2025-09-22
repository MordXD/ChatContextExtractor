from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

from sqlalchemy import select

from .config import settings
from .db import Chat, ContextEvent, Profile, get_sessionmaker


@dataclass
class TraitObservation:
    name: str
    confidence: float
    excerpt: str


@dataclass
class ContextPayload:
    excerpt: str
    traits: List[str]


_TRAIT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "optimistic": ("optimistic", "excited", "glad", "positive", "hopeful"),
    "stressed": ("stressed", "worried", "anxious", "overwhelmed", "burned out"),
    "decisive": ("decision", "decide", "commit", "finalize", "locked in"),
    "collaborative": ("together", "team", "collaborate", "align", "sync"),
    "detail_oriented": ("detail", "thorough", "careful", "checklist", "review"),
    "leader": ("lead", "leadership", "drive", "ownership", "guide"),
}


def _make_excerpt(text: str, keyword: str, window: int = 120) -> str:
    lowered = text.lower()
    idx = lowered.find(keyword)
    if idx == -1:
        snippet = text[:window]
    else:
        start = max(0, idx - window // 2)
        end = min(len(text), idx + len(keyword) + window // 2)
        snippet = text[start:end]
    normalized = " ".join(snippet.split())
    return normalized[: window * 2].strip()


def analyze_chat(text: str, max_traits: int | None = None) -> List[TraitObservation]:
    """Heuristic extraction of personality traits from a chat body."""

    max_traits = max_traits or settings.analysis_max_traits
    lowered = text.lower()
    observations: List[TraitObservation] = []
    for trait, keywords in _TRAIT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                freq = lowered.count(keyword)
                confidence = min(1.0, 0.55 + 0.1 * freq)
                observations.append(
                    TraitObservation(
                        name=trait,
                        confidence=round(confidence, 3),
                        excerpt=_make_excerpt(text, keyword),
                    )
                )
                break
    if not observations and text.strip():
        excerpt = " ".join(text.strip().split())[:200]
        observations.append(TraitObservation(name="neutral", confidence=0.4, excerpt=excerpt))

    filtered = [obs for obs in observations if obs.confidence >= settings.profile_min_confidence]
    filtered.sort(key=lambda obs: obs.confidence, reverse=True)
    return filtered[:max_traits]


def merge_profiles(
    existing: dict | None,
    observations: Iterable[TraitObservation],
    chat_id: str,
    timestamp: datetime,
) -> tuple[dict, List[ContextPayload]]:
    """Merge observed traits into aggregated profile structure."""

    profile = {"traits": {}}
    if existing:
        profile = {"traits": dict(existing.get("traits", {}))}
        for trait_name, trait_payload in profile["traits"].items():
            trait_payload.setdefault("confidence", 0.0)
            trait_payload.setdefault("count", 1)
            trait_payload.setdefault("evidence", [])
            trait_payload.setdefault("first_seen", trait_payload.get("first_seen", timestamp.isoformat()))
            trait_payload.setdefault("last_seen", trait_payload.get("last_seen", timestamp.isoformat()))

    contexts_map: Dict[str, ContextPayload] = {}
    for obs in observations:
        trait = profile["traits"].get(obs.name)
        if trait:
            count = int(trait.get("count", 1))
            running = float(trait.get("confidence", 0.0)) * count
            new_conf = round(min(1.0, (running + obs.confidence) / (count + 1)), 3)
            trait["confidence"] = new_conf
            trait["count"] = count + 1
            trait["last_seen"] = timestamp.isoformat()
        else:
            trait = {
                "confidence": obs.confidence,
                "count": 1,
                "first_seen": timestamp.isoformat(),
                "last_seen": timestamp.isoformat(),
                "evidence": [],
            }
            profile["traits"][obs.name] = trait
        evidence = trait.setdefault("evidence", [])
        evidence.append(
            {
                "chat_id": chat_id,
                "excerpt": obs.excerpt,
                "confidence": obs.confidence,
                "timestamp": timestamp.isoformat(),
            }
        )
        contexts_map.setdefault(obs.excerpt, ContextPayload(excerpt=obs.excerpt, traits=[])).traits.append(obs.name)

    contexts = list(contexts_map.values())
    for ctx in contexts:
        ctx.traits = sorted(set(ctx.traits))
    return profile, contexts


async def process_chat(chat_id: str) -> None:
    """Process chat text, update user profile and store context events."""

    session_factory = get_sessionmaker()
    async with session_factory() as session:
        chat = await session.get(Chat, chat_id)
        if chat is None:
            return

        observations = analyze_chat(chat.raw_text, settings.analysis_max_traits)
        if not observations:
            return

        timestamp = datetime.now(timezone.utc)
        profile = await session.get(Profile, chat.user_id)
        if profile is None:
            profile = Profile(user_id=chat.user_id, profile_json={"traits": {}}, version=0, updated_at=timestamp)
            session.add(profile)
            await session.flush()

        merged_profile, contexts = merge_profiles(profile.profile_json, observations, chat.id, timestamp)
        profile.profile_json = merged_profile
        profile.version = (profile.version or 0) + 1
        profile.updated_at = timestamp

        if contexts:
            existing = await session.execute(
                select(ContextEvent.chat_id, ContextEvent.excerpt, ContextEvent.derived_traits).where(
                    ContextEvent.chat_id == chat.id
                )
            )
            seen = {
                (row.chat_id, row.excerpt, tuple(row.derived_traits))
                for row in existing
            }
            for ctx in contexts:
                key = (chat.id, ctx.excerpt, tuple(ctx.traits))
                if key in seen:
                    continue
                session.add(
                    ContextEvent(
                        user_id=chat.user_id,
                        chat_id=chat.id,
                        excerpt=ctx.excerpt,
                        derived_traits=ctx.traits,
                        timestamp=timestamp,
                    )
                )

        await session.commit()
