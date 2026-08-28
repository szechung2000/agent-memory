"""Tests for the memory service (SQLite backend, local hash embedder)."""

from fastapi.testclient import TestClient

from agent_memory.api.main import app


def make_client(tmp_path, monkeypatch):
    """Client with lifespan run (context manager) and isolated DB per test."""
    monkeypatch.setenv("AM_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from agent_memory.core.config import get_settings

    get_settings.cache_clear()
    client = TestClient(app)
    client.__enter__()
    return client


def test_healthz(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    assert client.get("/healthz").json() == {"status": "ok"}


def test_remember_and_recall(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    docs = {
        "doc1": "pgvector is a postgres extension for vector similarity search",
        "doc2": "CS-6515 covers algorithms including dynamic programming and NP completeness",
        "doc3": "the trading bot uses a momentum strategy on TSLA with a stop loss",
    }
    ids = {}
    for key, content in docs.items():
        r = client.post("/remember", json={"content": content, "title": key})
        assert r.status_code == 200
        ids[key] = r.json()["id"]
    assert len(set(ids.values())) == 3

    r = client.post("/recall", json={"query": "vector similarity search in postgres", "k": 3})
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) == 3
    assert hits[0]["id"] == ids["doc1"]

    r2 = client.post("/recall", json={"query": "TSLA stop loss strategy", "k": 1})
    assert r2.json()[0]["content"] == docs["doc3"]


def test_kind_and_namespace_filtering(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    client.post("/remember", json={"content": "episodic event alpha", "kind": "episodic"})
    client.post("/remember", json={"content": "semantic fact beta"})

    r = client.post("/recall", json={"query": "alpha", "kind": "episodic"})
    contents = [h["content"] for h in r.json()]
    assert contents == ["episodic event alpha"]


def test_context_endpoint(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    client.post("/remember", json={"content": "momentum trading strategy notes"})
    r = client.get("/context", params={"query": "trading strategy", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["semantic"]) >= 1


def test_put_memory_is_keyed_and_idempotent(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    body = {
        "content": "PDF evidence from page one",
        "namespace": "paper-corpus",
        "metadata": {"document_id": "sha256:paper-a", "page_start": 1},
    }

    first = client.put("/v1/memories/sha256:paper-a:page:1:chunk:0", json=body)
    second = client.put(
        "/v1/memories/sha256:paper-a:page:1:chunk:0",
        json={**body, "metadata": {"citation": {"locator": "p. 1"}}},
    )

    assert first.status_code == second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    hit = client.post(
        "/recall", json={"query": "PDF evidence", "namespace": "paper-corpus"}
    ).json()[0]
    assert hit["namespace"] == "paper-corpus"
    assert hit["metadata"] == {
        "document_id": "sha256:paper-a",
        "page_start": 1,
        "citation": {"locator": "p. 1"},
    }


def test_namespace_scopes_recall_and_keyed_upserts_independently(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    external_id = "obsidian:Atomic/claim.md"

    brain = client.put(
        f"/v1/memories/{external_id}",
        json={"content": "brain-only verified claim", "namespace": "brain"},
    )
    corpus = client.put(
        f"/v1/memories/{external_id}",
        json={"content": "corpus-only source evidence", "namespace": "paper-corpus"},
    )

    assert brain.status_code == corpus.status_code == 200
    assert brain.json()["id"] != corpus.json()["id"]
    brain_hits = client.post(
        "/recall", json={"query": "claim evidence", "namespace": "brain", "k": 5}
    ).json()
    corpus_hits = client.post(
        "/recall", json={"query": "claim evidence", "namespace": "paper-corpus", "k": 5}
    ).json()
    assert [hit["namespace"] for hit in brain_hits] == ["brain"]
    assert [hit["content"] for hit in brain_hits] == ["brain-only verified claim"]
    assert [hit["namespace"] for hit in corpus_hits] == ["paper-corpus"]
    assert [hit["content"] for hit in corpus_hits] == ["corpus-only source evidence"]
