"""HTTP checks for the optional cited-answer endpoint."""

import json

from fastapi.testclient import TestClient

from agent_memory.api.main import app, get_answer_generator
from agent_memory.core.config import get_settings

SOURCE = "Synthetic CMBS Loan A was 30 days delinquent in September 2026."


class CitedGenerator:
    def __init__(self, bad=False):
        self.bad = bad

    def generate(self, question, evidence):
        return json.dumps(
            {
                "insufficient_evidence": False,
                "claims": [
                    {
                        "text": "Loan A was delinquent.",
                        "citations": [
                            {
                                "source_id": "S99" if self.bad else evidence[0].ref,
                                "quote": SOURCE,
                            }
                        ],
                    }
                ],
            }
        )


def test_answer_requires_a_configured_generator(tmp_path, monkeypatch):
    monkeypatch.setenv("AM_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.delenv("AM_OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    with TestClient(app) as client:
        response = client.post("/answer", json={"question": "Was Loan A delinquent?"})
    assert response.status_code == 503


def test_answer_returns_only_verified_citations(tmp_path, monkeypatch):
    monkeypatch.setenv("AM_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    get_settings.cache_clear()
    app.dependency_overrides[get_answer_generator] = lambda: CitedGenerator()
    try:
        with TestClient(app) as client:
            assert client.post(
                "/remember",
                json={
                    "content": SOURCE,
                    "namespace": "cmbs",
                    "metadata": {"source": "synthetic-servicer-report.txt", "page": 4},
                },
            ).status_code == 200
            response = client.post(
                "/answer", json={"question": "Was Loan A delinquent?", "namespace": "cmbs"}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Loan A was delinquent. [S1]"
        assert data["citations"][0]["source"] == "synthetic-servicer-report.txt"
        assert data["citations"][0]["metadata"]["page"] == 4
    finally:
        app.dependency_overrides.pop(get_answer_generator, None)


def test_answer_rejects_fake_source_ids_at_http_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("AM_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    get_settings.cache_clear()
    app.dependency_overrides[get_answer_generator] = lambda: CitedGenerator(bad=True)
    try:
        with TestClient(app) as client:
            client.post("/remember", json={"content": SOURCE, "namespace": "cmbs"})
            response = client.post(
                "/answer", json={"question": "Was Loan A delinquent?", "namespace": "cmbs"}
            )
        assert response.status_code == 502
        assert "Loan A was delinquent." not in response.text
    finally:
        app.dependency_overrides.pop(get_answer_generator, None)
