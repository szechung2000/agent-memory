"""Offline tests for GitHub-star discovery: no network or paid model calls."""

import base64
import json
import logging

import httpx
import numpy as np
import pytest

from agent_memory import cli
from agent_memory.evals.run_golden import make_eval_repo
from agent_memory.integrations.github_stars import (
    GitHubStarsClient,
    stars_namespace,
    sync_stars,
)


class CountingEmbedder:
    def __init__(self):
        self.batches = []

    def embed(self, texts):
        self.batches.append(texts)
        return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]


def star(number, **overrides):
    result = {
        "id": number,
        "full_name": f"example/repo-{number}",
        "description": f"A Python agent workflow number {number}",
        "topics": ["agents", "python"],
        "language": "Python",
        "license": {"spdx_id": "MIT"},
        "archived": False,
        "pushed_at": "2026-11-01T10:00:00Z",
    }
    return {**result, **overrides}


def transport_for(stars, *, readmes=None):
    readmes = readmes or {}
    pages = []

    def handler(request):
        if request.url.path.endswith("/starred"):
            assert request.url.host == "api.github.com"
            page = int(request.url.params["page"])
            size = int(request.url.params["per_page"])
            pages.append(page)
            start = (page - 1) * size
            return httpx.Response(200, json=stars[start : start + size])
        if request.url.path.endswith("/readme"):
            name = request.url.path.split("/repos/", 1)[1].removesuffix("/readme")
            if name not in readmes:
                return httpx.Response(404)
            return httpx.Response(200, text=readmes[name], headers={"content-type": "text/plain"})
        raise AssertionError(f"Unexpected request: {request.url}")

    return httpx.MockTransport(handler), pages


def test_paginated_sync_deduplicates_on_repeat(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/stars.db")
    embedder = CountingEmbedder()
    data = [star(i) for i in range(1, 206)]
    transport, pages = transport_for(data)

    with GitHubStarsClient(transport=transport) as client:
        first = sync_stars(repo, embedder, client, "szechung2000")
        second = sync_stars(repo, embedder, client, "szechung2000")

    assert (first.seen, first.added, first.updated) == (205, 205, 0)
    assert (second.seen, second.unchanged) == (205, 205)
    assert pages == [1, 2, 3, 1, 2, 3]
    assert sum(len(batch) for batch in embedder.batches) == 205
    assert repo.get_by_external_id("github:205", namespace="github-stars:szechung2000")


def test_limit_is_not_confused_with_pagination(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/limited.db")
    transport, pages = transport_for([star(i) for i in range(1, 206)])
    with GitHubStarsClient(transport=transport) as client:
        stats = sync_stars(repo, CountingEmbedder(), client, "alice", max_repos=150)
    assert stats.added == 150
    assert pages == [1, 2]
    assert repo.get_by_external_id("github:150", namespace="github-stars:alice")
    assert repo.get_by_external_id("github:151", namespace="github-stars:alice") is None


def test_refresh_updates_content_and_preserves_readme(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/update.db")
    embedder = CountingEmbedder()
    old = star(12)
    transport, _ = transport_for([old], readmes={"example/repo-12": "Uses BM25 reranking."})
    with GitHubStarsClient(transport=transport) as client:
        assert sync_stars(repo, embedder, client, "alice", include_readmes=True).added == 1
    first = repo.get_by_external_id("github:12", namespace="github-stars:alice")
    assert "BM25" in first.content
    repo.update_metadata(first.id, {**first.metadata, "personal_note": "review later"})

    updated = star(12, description="A search and retrieval project", archived=True)
    transport, _ = transport_for([updated])
    with GitHubStarsClient(transport=transport) as client:
        result = sync_stars(repo, embedder, client, "alice")
        assert result.updated == 1
        assert sync_stars(repo, embedder, client, "alice").unchanged == 1
    current = repo.get_by_external_id("github:12", namespace="github-stars:alice")
    assert current.id == first.id
    assert current.metadata["archived"] is True
    assert current.metadata["personal_note"] == "review later"
    assert "BM25" in current.content
    assert sum(map(len, embedder.batches)) == 2

    # A metadata-only refresh must not hide that upstream README content changed.
    pushed = star(12, description="A search and retrieval project", pushed_at="2026-11-02")
    transport, _ = transport_for([pushed])
    with GitHubStarsClient(transport=transport) as client:
        assert sync_stars(repo, embedder, client, "alice").unchanged == 1
    transport, _ = transport_for(
        [pushed], readmes={"example/repo-12": "Adds a new citation verifier."}
    )
    with GitHubStarsClient(transport=transport) as client:
        assert sync_stars(repo, embedder, client, "alice", include_readmes=True).updated == 1
    current = repo.get_by_external_id("github:12", namespace="github-stars:alice")
    assert "new citation verifier" in current.content
    assert "BM25" not in current.content


def test_missing_readme_and_no_token_are_safe(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/missing.db")
    transport, _ = transport_for([star(1)])
    with GitHubStarsClient(transport=transport) as client:
        result = sync_stars(repo, CountingEmbedder(), client, "alice", include_readmes=True)
        assert result.added == 1
    assert (
        "README excerpt"
        not in repo.get_by_external_id("github:1", namespace="github-stars:alice").content
    )
    with GitHubStarsClient(transport=transport) as client:
        assert sync_stars(repo, CountingEmbedder(), client, "alice", include_readmes=True).unchanged


def test_starred_references_are_scoped_per_user(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/namespaces.db")
    transport, _ = transport_for([star(19)])
    with GitHubStarsClient(transport=transport) as client:
        sync_stars(repo, CountingEmbedder(), client, "Alice")
        sync_stars(repo, CountingEmbedder(), client, "bob")
    alice = repo.get_by_external_id("github:19", namespace="github-stars:alice")
    bob = repo.get_by_external_id("github:19", namespace="github-stars:bob")
    assert alice.id != bob.id


def test_readme_supports_github_base64_payload():
    def handler(_request):
        return httpx.Response(
            200,
            json={"encoding": "base64", "content": base64.b64encode(b"Example README").decode()},
        )

    with GitHubStarsClient(transport=httpx.MockTransport(handler)) as client:
        assert client.readme("example/repo") == "Example README"


def test_http_errors_fail_and_repo_names_are_validated(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/errors.db")
    client_transport = httpx.MockTransport(lambda request: httpx.Response(403, text="rate limit"))
    with GitHubStarsClient(transport=client_transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            sync_stars(repo, CountingEmbedder(), client, "alice")

    bad_transport, _ = transport_for([star(1, full_name="../attacker")])
    with GitHubStarsClient(transport=bad_transport) as client:
        with pytest.raises(ValueError, match="Invalid GitHub repository"):
            sync_stars(repo, CountingEmbedder(), client, "alice")

    with pytest.raises(ValueError, match="username"):
        stars_namespace("alice/../../etc")


def test_cli_search_uses_existing_retrieval_and_emits_sources(tmp_path, monkeypatch, capsys):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/search.db")
    embedder = CountingEmbedder()
    transport, _ = transport_for([star(42, description="RAG citation evaluation")])
    with GitHubStarsClient(transport=transport) as client:
        sync_stars(repo, embedder, client, "alice")
    monkeypatch.setattr(cli, "_repo_and_embedder", lambda: (repo, embedder))

    assert cli.main(["stars", "search", "citation", "--user", "alice", "--json"]) == 0
    results = json.loads(capsys.readouterr().out)
    assert results[0]["repository"] == "example/repo-42"
    assert results[0]["url"] == "https://github.com/example/repo-42"
    assert results[0]["description"] == "RAG citation evaluation"


@pytest.mark.parametrize("outcome", ["success", "http_error", "network_error", "value_error"])
def test_cli_never_exposes_auth_token(tmp_path, monkeypatch, capsys, caplog, outcome):
    """A token must not enter the memory DB, stdout, stderr, or debug logs."""
    from agent_memory.integrations import github_stars

    token = "sentinel-not-a-real-token"
    monkeypatch.setenv("GITHUB_TOKEN", token)
    repo = make_eval_repo(f"sqlite:///{tmp_path}/secrets.db")
    monkeypatch.setattr(cli, "_repo_and_embedder", lambda: (repo, CountingEmbedder()))

    def handler(request):
        assert request.url.host == "api.github.com"
        assert request.headers.get("authorization") == f"Bearer {token}"
        if outcome == "http_error":
            return httpx.Response(403, text=f"Sensitive response containing {token}")
        if outcome == "network_error":
            raise httpx.ConnectError(f"Sensitive transport failure containing {token}")
        if outcome == "value_error":
            raise ValueError(f"Sensitive parsing failure containing {token}")
        return httpx.Response(200, json=[star(7)])

    original_client = GitHubStarsClient
    monkeypatch.setattr(
        github_stars,
        "GitHubStarsClient",
        lambda token: original_client(token=token, transport=httpx.MockTransport(handler)),
    )

    with caplog.at_level(logging.DEBUG):
        exit_code = cli.main(["stars", "sync", "--user", "alice", "--limit", "1"])
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err + caplog.text

    if outcome == "success":
        assert exit_code == 0
        memory = repo.get_by_external_id("github:7", namespace="github-stars:alice")
        assert token not in memory.content + json.dumps(memory.metadata)
    else:
        assert exit_code == 1
        assert "sync failed" in captured.err
        expected_error = {
            "http_error": "HTTP 403",
            "network_error": "network request error",
            "value_error": "invalid input or GitHub repository data",
        }[outcome]
        assert expected_error in captured.err
