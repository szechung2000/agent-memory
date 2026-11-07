"""Operator-facing traces, readiness, and sanitized dependency failure responses."""

import json
import logging

import httpx
from fastapi.testclient import TestClient
from openai import APIConnectionError
from sqlalchemy.exc import OperationalError

from agent_memory.api.main import app, get_answer_generator, get_db
from agent_memory.core.config import get_settings


def test_request_tracing_and_readiness(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("AM_DATABASE_URL", f"sqlite:///{tmp_path}/traces.db")
    monkeypatch.setenv("AM_OPENAI_API_KEY", "")
    get_settings.cache_clear()
    with caplog.at_level(logging.INFO, logger="agent_memory.api"):
        with TestClient(app) as client:
            response = client.get("/readyz", headers={"X-Request-ID": "interview-demo-1"})
            invalid = client.get("/healthz", headers={"X-Request-ID": "invalid\nvalue"})
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert response.headers["X-Request-ID"] == "interview-demo-1"
    assert invalid.status_code == 200
    assert invalid.headers["X-Request-ID"] != "invalid\nvalue"
    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.message.startswith("{")
    ]
    assert any(
        event["event"] == "http_request"
        and event["route"] == "/readyz"
        and event["request_id"] == "interview-demo-1"
        and event["duration_ms"] >= 0
        for event in events
    )


def test_database_failure_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setenv("AM_DATABASE_URL", f"sqlite:///{tmp_path}/failure.db")
    get_settings.cache_clear()

    class FailingSession:
        def execute(self, statement):
            raise OperationalError("SELECT secret", {}, RuntimeError("secret-password"))

    def broken_db():
        yield FailingSession()

    app.dependency_overrides[get_db] = broken_db
    try:
        with TestClient(app) as client:
            response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json() == {"detail": "Database is temporarily unavailable"}
        assert "secret" not in response.text
        assert response.headers["X-Request-ID"]
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_model_failure_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setenv("AM_DATABASE_URL", f"sqlite:///{tmp_path}/provider.db")
    monkeypatch.setenv("AM_OPENAI_API_KEY", "")
    get_settings.cache_clear()

    class BrokenGenerator:
        def generate(self, question, evidence):
            raise APIConnectionError(
                message="secret-provider-connection-error",
                request=httpx.Request("POST", "https://example.invalid/chat"),
            )

    app.dependency_overrides[get_answer_generator] = lambda: BrokenGenerator()
    try:
        with TestClient(app) as client:
            client.post("/remember", json={"content": "Synthetic CMBS: loan was modified."})
            response = client.post("/answer", json={"question": "Was the loan modified?"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Model provider is temporarily unavailable"}
        assert "secret" not in response.text
        assert response.headers["X-Request-ID"]
    finally:
        app.dependency_overrides.pop(get_answer_generator, None)


def test_answer_logs_metrics_without_question_or_evidence(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("AM_DATABASE_URL", f"sqlite:///{tmp_path}/metrics.db")
    monkeypatch.setenv("AM_OPENAI_API_KEY", "")
    get_settings.cache_clear()
    evidence_text = "Synthetic CMBS Loan Q has DSCR 1.42 for 2026."

    class MeasuredGenerator:
        model = "stubbed-model"
        last_usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}

        def generate(self, question, evidence):
            return json.dumps(
                {
                    "insufficient_evidence": False,
                    "claims": [
                        {
                            "text": "Loan Q has DSCR 1.42.",
                            "citations": [
                                {"source_id": evidence[0].ref, "quote": evidence_text}
                            ],
                        }
                    ],
                }
            )

    app.dependency_overrides[get_answer_generator] = lambda: MeasuredGenerator()
    try:
        with caplog.at_level(logging.INFO, logger="agent_memory.api"):
            with TestClient(app) as client:
                assert client.post("/remember", json={"content": evidence_text}).status_code == 200
                response = client.post(
                    "/answer", json={"question": "What is the private DSCR question?"}
                )
        assert response.status_code == 200
        events = [
            json.loads(record.message)
            for record in caplog.records
            if record.message.startswith("{")
        ]
        answer_event = next(event for event in events if event["event"] == "rag_answer")
        assert answer_event["total_tokens"] == 120
        assert answer_event["retrieved_count"] == 1
        assert answer_event["status"] == "answered"
        assert answer_event["request_id"] == response.headers["X-Request-ID"]
        assert "private DSCR question" not in caplog.text
        assert evidence_text not in caplog.text
    finally:
        app.dependency_overrides.pop(get_answer_generator, None)
