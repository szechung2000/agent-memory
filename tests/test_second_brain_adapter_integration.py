"""Feature 001 HTTP contract test for the agent-memory API."""

from fastapi.testclient import TestClient

from agent_memory.api.main import app


def test_keyed_upsert_accepts_a_slash_key_through_fastapi(tmp_path, monkeypatch):
    monkeypatch.setenv("AM_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from agent_memory.core.config import get_settings

    get_settings.cache_clear()
    with TestClient(app) as client:
        response = client.put(
            "/v1/memories/obsidian:Atomic/claim.md",
            json={
                "content": "verified claim with source provenance",
                "namespace": "brain",
                "metadata": {"provenance": "source"},
            },
        )
        assert response.status_code == 200
        memory_id = response.json()["id"]

        hits_response = client.post(
            "/recall", json={"query": "verified claim", "namespace": "brain", "k": 1}
        )

    assert hits_response.status_code == 200
    hits = hits_response.json()
    assert len(hits) == 1
    assert hits[0]["id"] == memory_id
    assert hits[0]["content"] == "verified claim with source provenance"
    assert hits[0]["namespace"] == "brain"
    assert hits[0]["metadata"] == {"provenance": "source"}
