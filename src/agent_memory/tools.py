"""Agent tool definitions for memory operations.

Exposes the memory service as OpenAI/Anthropic-compatible function-calling
tools so any agent can remember, search, and consolidate. The executor maps
tool calls to repository/embedder calls — no HTTP hop needed in-process.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from agent_memory.core.models import Memory, MemoryKind
from agent_memory.core.temporal import temporal_penalty  # noqa: F401 (re-export convenience)

MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": (
                "Store a durable fact or episodic event in long-term memory. "
                "Use for user preferences, decisions, and notable events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The fact/event to remember"},
                    "kind": {
                        "type": "string",
                        "enum": ["semantic", "episodic"],
                        "description": "semantic = durable fact; episodic = event/log",
                    },
                    "namespace": {
                        "type": "string", "description": "Memory namespace", "default": "brain"
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search long-term memory semantically. Returns ranked memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                    "kind": {"type": "string", "enum": ["semantic", "episodic"]},
                    "namespace": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_context",
            "description": (
                "Assemble a prompt-ready context block about a topic: relevant facts "
                "plus related episodes. Use at the start of a task needing background."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["topic"],
            },
        },
    },
]


class MemoryToolExecutor:
    """Executes memory tool calls against a repo+embedder pair."""

    def __init__(self, repo, embedder) -> None:
        self.repo = repo
        self.embedder = embedder

    @property
    def tools(self) -> list[dict[str, Any]]:
        return MEMORY_TOOLS

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        handler = {
            "memory_write": self._write,
            "memory_search": self._search,
            "memory_context": self._context,
        }.get(name)
        if handler is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return handler(**arguments)
        except Exception as exc:  # noqa: BLE001 — agents need errors as text
            return json.dumps({"error": str(exc)})

    def _write(self, content: str, kind: str = "semantic", namespace: str = "brain") -> str:
        vec = np.asarray(self.embedder.embed([content])[0], dtype=np.float32)
        mid = self.repo.write(
            Memory(content=content, kind=MemoryKind(kind), namespace=namespace), vec
        )
        return json.dumps({"id": mid, "status": "remembered"})

    def _search(
        self, query: str, k: int = 5, kind: str | None = None, namespace: str | None = None
    ) -> str:
        qvec = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        results = self.repo.recall(qvec, query, k=k, kind=kind, namespace=namespace)
        return json.dumps(
            [
                {"content": m.content, "kind": m.kind.value, "score": score}
                for m, score in results
            ],
            indent=2,
        )

    def _context(self, topic: str, k: int = 5) -> str:
        qvec = np.asarray(self.embedder.embed([topic])[0], dtype=np.float32)
        results = self.repo.recall(qvec, topic, k=k)
        facts = [m.content for m, _ in results if m.kind == MemoryKind.SEMANTIC]
        episodes = [m.content for m, _ in results if m.kind == MemoryKind.EPISODIC]
        block = "\n".join(f"- {f}" for f in facts)
        if episodes:
            block += "\n\nRelated events:\n" + "\n".join(f"- {e}" for e in episodes)
        return json.dumps({"topic": topic, "context": block or "(no relevant memories)"})
