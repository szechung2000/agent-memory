"""Real HTTP -> SQLAlchemy -> PostgreSQL/pgvector tests, no external model calls."""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from agent_memory.api.main import app, get_answer_generator
from agent_memory.core.config import get_settings

SOURCE = "Synthetic CMBS Loan A was 30 days delinquent in September 2026."


class CitedGenerator:
    def generate(self, question, evidence):
        return json.dumps(
            {
                "insufficient_evidence": False,
                "claims": [
                    {
                        "text": "Loan A had a 30-day delinquency.",
                        "citations": [{"source_id": evidence[0].ref, "quote": SOURCE}],
                    }
                ],
            }
        )


@pytest.fixture()
def pg_client(isolated_pg_uri, monkeypatch):
    monkeypatch.setenv("AM_DATABASE_URL", isolated_pg_uri)
    monkeypatch.setenv("AM_OPENAI_API_KEY", "")
    get_settings.cache_clear()
    app.dependency_overrides[get_answer_generator] = lambda: CitedGenerator()
    try:
        with TestClient(app) as client:
            yield client, isolated_pg_uri
    finally:
        app.dependency_overrides.pop(get_answer_generator, None)
        get_settings.cache_clear()


def test_postgres_api_lifecycle_and_pgvector_schema(pg_client):
    client, uri = pg_client
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}

    payload = {
        "content": SOURCE,
        "namespace": "cmbs",
        "metadata": {"source": "synthetic/sep-remittance.txt", "page": 4},
    }
    first = client.put("/v1/memories/loan-A/2026-09", json=payload)
    assert first.status_code == 200
    second = client.put("/v1/memories/loan-A/2026-09", json=payload)
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]

    recalls = client.post(
        "/recall", json={"query": "Loan A delinquent September", "namespace": "cmbs"}
    )
    assert recalls.status_code == 200
    assert len(recalls.json()) == 1
    assert recalls.json()[0]["metadata"]["page"] == 4

    context = client.get("/context", params={"query": "Loan A September"})
    assert context.status_code == 200
    assert context.json()["semantic"][0]["source"] == "synthetic/sep-remittance.txt"

    answer = client.post(
        "/answer", json={"question": "Was Loan A delinquent?", "namespace": "cmbs"}
    )
    assert answer.status_code == 200
    assert answer.json()["status"] == "answered"
    assert answer.json()["citations"][0]["quote"] == SOURCE
    assert answer.json()["citations"][0]["metadata"]["page"] == 4

    with create_engine(uri).connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM memories")).scalar_one() == 1
        stored_type = conn.execute(text("SELECT pg_typeof(embedding)::text FROM memories"))
        assert stored_type.scalar_one() == "vector"
        indexes = conn.execute(
            text("SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()")
        ).scalars().all()
        assert "ix_memories_embedding_hnsw" in indexes


def test_postgres_api_no_evidence_abstains(pg_client):
    client, _ = pg_client
    response = client.post(
        "/answer", json={"question": "What is the cap rate of Loan Z?", "namespace": "cmbs"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_evidence"
    assert data["citations"] == []
