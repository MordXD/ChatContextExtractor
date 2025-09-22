import asyncio
from datetime import datetime, timezone
from typing import Set

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app.profile_service import TraitObservation, analyze_chat, merge_profiles, process_chat
from app.api import app


def test_analyze_chat_detects_traits():
    text = "We should collaborate as a team and stay optimistic about the launch."
    observations = analyze_chat(text, max_traits=5)
    names: Set[str] = {obs.name for obs in observations}
    assert "collaborative" in names
    assert "optimistic" in names


def test_merge_profiles_confidence_growth():
    timestamp = datetime.now(timezone.utc)
    base = {
        "traits": {
            "leader": {
                "confidence": 0.6,
                "count": 1,
                "evidence": [],
                "first_seen": timestamp.isoformat(),
                "last_seen": timestamp.isoformat(),
            }
        }
    }
    observations = [TraitObservation(name="leader", confidence=0.9, excerpt="I will lead this project decisively")]
    merged, contexts = merge_profiles(base, observations, "chat-1", timestamp)
    assert merged["traits"]["leader"]["confidence"] > 0.6
    assert contexts and contexts[0].traits == ["leader"]


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "profiles.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    db_module.configure_engine(db_url)
    asyncio.run(db_module.init_db(drop_existing=True))
    with TestClient(app) as test_client:
        yield test_client


def test_upload_and_fetch_profile_flow(client):
    user_id = "11111111-1111-1111-1111-111111111111"
    payload = {
        "user_id": user_id,
        "text": "I will lead the launch and collaborate with the team to stay optimistic.",
        "email": "user@example.com",
    }
    response = client.post("/upload_chat", json=payload)
    assert response.status_code == 200
    chat_id = response.json()["chat_id"]

    profile_resp = client.get(f"/profile/{user_id}")
    if profile_resp.status_code == 404:
        asyncio.run(process_chat(chat_id))
        profile_resp = client.get(f"/profile/{user_id}")
    assert profile_resp.status_code == 200
    profile_data = profile_resp.json()
    assert profile_data["version"] == 1
    assert "traits" in profile_data["profile"]
    assert "leader" in profile_data["profile"]["traits"]

    contexts_resp = client.get(f"/profile/{user_id}/contexts")
    assert contexts_resp.status_code == 200
    contexts = contexts_resp.json()
    assert contexts
    assert any("leader" in ctx["derived_traits"] for ctx in contexts)
