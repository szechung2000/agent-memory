"""Feature 001 HTTP contract test with the sibling second-brain adapter."""

import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

from agent_memory.api.main import app

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "second-brain" / "src"))
import second_brain.memory as second_brain_memory  # noqa: E402


class ClientHTTPResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def test_second_brain_adapter_upserts_a_slash_key_through_fastapi(tmp_path, monkeypatch):
    monkeypatch.setenv("AM_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from agent_memory.core.config import get_settings

    get_settings.cache_clear()
    with TestClient(app) as client:

        def testclient_urlopen(request, timeout):
            response = client.request(
                request.get_method(),
                urlsplit(request.full_url).path,
                content=request.data,
                headers=dict(request.header_items()),
            )
            if response.is_error:
                raise HTTPError(
                    request.full_url,
                    response.status_code,
                    response.text,
                    response.headers,
                    None,
                )
            return ClientHTTPResponse(response.content)

        monkeypatch.setattr(second_brain_memory, "urlopen", testclient_urlopen)
        adapter = second_brain_memory.AgentMemoryAdapter("http://testserver")
        memory_id = adapter.upsert(
            "obsidian:Atomic/claim.md",
            {
                "content": "verified claim with source provenance",
                "namespace": "brain",
                "metadata": {"provenance": "source"},
            },
        )

        hits = adapter.recall("verified claim", scopes=["brain"], k=1)

    assert len(hits) == 1
    assert hits[0]["id"] == memory_id
    assert hits[0]["content"] == "verified claim with source provenance"
    assert hits[0]["namespace"] == "brain"
    assert hits[0]["metadata"] == {"provenance": "source"}
