"""Tests for agent tools + demo agent (no keys, no network)."""

import json

import pytest

from agent_memory.evals.run_golden import make_eval_repo
from agent_memory.tools import MEMORY_TOOLS, MemoryToolExecutor


@pytest.fixture()
def executor(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/tools.db")
    from agent_memory.core.embedding import get_embedder

    return MemoryToolExecutor(repo, get_embedder())


def test_tool_definitions_valid():
    names = {t["function"]["name"] for t in MEMORY_TOOLS}
    assert {"memory_write", "memory_search", "memory_context"} <= names
    for t in MEMORY_TOOLS:
        params = t["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params


def test_write_and_search_roundtrip(executor):
    out = json.loads(executor.execute("memory_write", {"content": "Simon prefers uv over pip"}))
    assert out["status"] == "remembered"

    hits = json.loads(executor.execute("memory_search", {"query": "package manager preference"}))
    assert any("uv over pip" in h["content"] for h in hits)


def test_search_kind_filter(executor):
    executor.execute("memory_write", {"content": "semantic fact gamma", "kind": "semantic"})
    executor.execute("memory_write", {"content": "episodic delta event", "kind": "episodic"})
    hits = json.loads(
        executor.execute("memory_search", {"query": "delta", "k": 5, "kind": "episodic"})
    )
    assert all(h["kind"] == "episodic" for h in hits)


def test_context_assembles_block(executor):
    executor.execute("memory_write", {"content": "Trading bot uses momentum entries on TSLA"})
    out = json.loads(executor.execute("memory_context", {"topic": "trading strategy"}))
    assert "momentum" in out["context"]


def test_unknown_tool_returns_error(executor):
    out = json.loads(executor.execute("nonexistent_tool", {}))
    assert "error" in out


def test_invalid_args_return_error_not_raise(executor):
    out = json.loads(executor.execute("memory_write", {}))  # missing content
    assert "error" in out


# ---------- demo agent ----------

async def test_demo_reply_remembers_and_recalls(executor, tmp_path, monkeypatch):
    monkeypatch.delenv("AM_OPENAI_API_KEY", raising=False)
    from agent_memory.core.config import get_settings

    get_settings.cache_clear()
    from agent_memory.demo.telegram_agent import MemoryAgent

    agent = MemoryAgent(executor, llm=None)

    reply1 = await agent.handle_message("remember that Simon's trading bot uses momentum on TSLA")
    assert "Remembered" in reply1

    reply2 = await agent.handle_message("what does the trading bot do?")
    assert "From memory:" in reply2
    assert "TSLA" in reply2 or "momentum" in reply2
