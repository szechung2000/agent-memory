"""Index GitHub stars as source-linked memories; do not clone or execute repositories."""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from agent_memory.core.models import Memory

API_URL = "https://api.github.com"
README_LIMIT = 3000
_USERNAME = re.compile(r"[A-Za-z0-9-]{1,39}\Z")
_REPO_NAME = re.compile(r"[A-Za-z0-9-]{1,39}/[A-Za-z0-9_.-]+\Z")


def _valid_full_name(full_name: str | None) -> bool:
    return bool(
        isinstance(full_name, str)
        and _REPO_NAME.fullmatch(full_name)
        and full_name.split("/", 1)[1] not in {".", ".."}
    )


def stars_namespace(username: str) -> str:
    if not _USERNAME.fullmatch(username):
        raise ValueError("GitHub username must contain 1–39 letters, digits, or hyphens")
    return f"github-stars:{username.lower()}"


@dataclass
class SyncStats:
    seen: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0


class GitHubStarsClient:
    """Read-only GitHub REST client with a bounded request timeout."""

    def __init__(self, token: str | None = None, transport: httpx.BaseTransport | None = None):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "agent-memory-github-stars",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.Client(
            base_url=API_URL, headers=headers, timeout=20, transport=transport
        )

    def __enter__(self) -> GitHubStarsClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.client.close()

    def starred(self, username: str, max_repos: int | None = None) -> Iterator[dict[str, Any]]:
        stars_namespace(username)
        if max_repos is not None and max_repos < 1:
            raise ValueError("max_repos must be greater than zero")
        page = 1
        seen = 0
        while True:
            page_size = 100
            response = self.client.get(
                f"/users/{username}/starred", params={"per_page": page_size, "page": page}
            )
            response.raise_for_status()
            items = response.json()
            if not isinstance(items, list):
                raise ValueError("GitHub starred response must be a list")
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("GitHub starred response contains an invalid repository")
                yield item
                seen += 1
                if max_repos is not None and seen >= max_repos:
                    return
            if len(items) < page_size:
                return
            page += 1

    def readme(self, full_name: str) -> str:
        if not _valid_full_name(full_name):
            raise ValueError("Invalid repository name in GitHub response")
        response = self.client.get(
            f"/repos/{full_name}/readme", headers={"Accept": "application/vnd.github.raw+json"}
        )
        if response.status_code == 404:
            return ""
        response.raise_for_status()
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return response.text[:README_LIMIT]
        if isinstance(payload, dict) and payload.get("encoding") == "base64":
            return base64.b64decode(payload["content"]).decode("utf-8", errors="replace")[
                :README_LIMIT
            ]
        return response.text[:README_LIMIT]


def _reference(
    repo: dict[str, Any],
    readme: str,
    namespace: str,
    readme_checked: bool,
    readme_pushed_at: str | None,
) -> Memory:
    full_name = repo["full_name"]
    if not _valid_full_name(full_name) or not isinstance(repo.get("id"), int):
        raise ValueError("GitHub repository has an invalid ID or name")
    topics = repo.get("topics") or []
    if not isinstance(topics, list):
        topics = []
    description = repo.get("description") or "(No description)"
    language = repo.get("language") or "Unknown"
    license_data = repo.get("license") or {}
    license_id = license_data.get("spdx_id") if isinstance(license_data, dict) else None
    source = f"https://github.com/{full_name}"
    content = "\n".join(
        [
            f"GitHub repository: {full_name}",
            f"Description: {description}",
            f"Topics: {', '.join(str(tag) for tag in topics)}",
            f"Language: {language}",
            f"License: {license_id or 'Unknown'}",
            *([f"README excerpt (untrusted):\n{readme}"] if readme else []),
        ]
    )
    return Memory(
        namespace=namespace,
        title=full_name,
        content=content,
        metadata={
            "source": source,
            "github_repo_id": repo["id"],
            "full_name": full_name,
            "description": description,
            "language": language,
            "topics": topics,
            "license": license_id,
            "archived": bool(repo.get("archived", False)),
            "pushed_at": repo.get("pushed_at"),
            "readme_excerpt": readme,
            "readme_checked": readme_checked,
            "readme_pushed_at": readme_pushed_at,
        },
    )


def sync_stars(
    repo: Any,
    embedder: Any,
    client: GitHubStarsClient,
    username: str,
    *,
    include_readmes: bool = False,
    max_repos: int | None = None,
    batch_size: int = 32,
) -> SyncStats:
    """Import an account's stars via the memory repository's keyed upsert contract.

    On repeated runs, unchanged descriptions do not incur embedding calls.  Existing
    README excerpts survive a metadata-only sync. This is an additive sync: an
    unstarred repository is not removed until a separate prune feature is built.
    """
    namespace = stars_namespace(username)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    stats = SyncStats()
    pending: list[tuple[str, Memory, bool]] = []

    def flush() -> None:
        if not pending:
            return
        vectors = embedder.embed([item.content for _, item, _ in pending])
        if len(vectors) != len(pending):
            raise ValueError("Embedder returned the wrong number of vectors")
        for (external_id, memory, is_update), vector in zip(pending, vectors, strict=True):
            repo.upsert(external_id, memory, np.asarray(vector, dtype=np.float32))
            if is_update:
                stats.updated += 1
            else:
                stats.added += 1
        pending.clear()

    for item in client.starred(username, max_repos=max_repos):
        stats.seen += 1
        if not _valid_full_name(item.get("full_name", "")) or not isinstance(item.get("id"), int):
            raise ValueError("Invalid GitHub repository metadata")
        external_id = f"github:{item['id']}"
        previous = repo.get_by_external_id(external_id, namespace=namespace)
        readme = previous.metadata.get("readme_excerpt", "") if previous else ""
        readme_checked = bool(previous and previous.metadata.get("readme_checked"))
        readme_pushed_at = previous.metadata.get("readme_pushed_at") if previous else None
        if include_readmes and (
            previous is None or readme_pushed_at != item.get("pushed_at") or not readme_checked
        ):
            readme = client.readme(item["full_name"])
            readme_checked = True
            readme_pushed_at = item.get("pushed_at")
        memory = _reference(item, readme, namespace, readme_checked, readme_pushed_at)
        if previous is not None and previous.content == memory.content:
            if any(previous.metadata.get(key) != value for key, value in memory.metadata.items()):
                repo.update_metadata(previous.id, {**previous.metadata, **memory.metadata})
            stats.unchanged += 1
            continue
        pending.append((external_id, memory, previous is not None))
        if len(pending) >= batch_size:
            flush()
    flush()
    return stats


def token_from_environment() -> str | None:
    """Use the caller's existing GitHub token; never persist it with memories."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
