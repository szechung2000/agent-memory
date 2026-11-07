"""Shared scoring/context and central runtime bootstrap stay small and testable."""

import json

import numpy as np
import pytest
from sqlalchemy import inspect

from agent_memory.core.context import context_items
from agent_memory.core.embedding import LocalHashEmbedder
from agent_memory.core.models import Memory, MemoryKind
from agent_memory.core.scoring import hybrid_score
from agent_memory.core.temporal import temporal_penalty
from agent_memory.db.engine import create_repository_for_url
from agent_memory.tools import MemoryToolExecutor


def test_hybrid_score_uses_one_shared_weighting_rule():
    content = "Synthetic CMBS loan A was delinquent in September."
    query = "CMBS loan"
    expected = 0.7 * 0.8 + 0.3 - temporal_penalty(query, content)
    assert hybrid_score(query, content, None, 0.8) == pytest.approx(expected)


def test_sqlite_backend_uses_shared_score(tmp_path):
    repo = create_repository_for_url(f"sqlite:///{tmp_path}/score.db")
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    content = "Synthetic CMBS loan A has DSCR 1.42."
    repo.write(Memory(content=content), vector)
    results = repo.recall(vector, "CMBS loan", k=1)
    assert len(results) == 1
    assert results[0][1] == pytest.approx(hybrid_score("CMBS loan", content, None, 1.0))
    with pytest.raises(ValueError, match="not implemented"):
        repo.recall(vector, "CMBS loan", filters={"source": "report.txt"})


def test_api_and_agent_context_share_evidence(tmp_path):
    repo = create_repository_for_url(f"sqlite:///{tmp_path}/context.db")
    embedder = LocalHashEmbedder()
    content = "Synthetic remittance confirms Loan A was delinquent."
    memory_id = repo.write(
        Memory(
            content=content,
            kind=MemoryKind.SEMANTIC,
            metadata={"source": "synthetic/servicer.txt", "page": 2},
        ),
        embedder.embed([content])[0],
    )
    items = context_items(repo, embedder, "Loan A delinquency", 3)
    tool_result = json.loads(
        MemoryToolExecutor(repo, embedder).execute(
            "memory_context", {"topic": "Loan A delinquency", "k": 3}
        )
    )
    assert items[0]["id"] == tool_result["sources"][0]["memory_id"] == memory_id
    assert items[0]["source"] == tool_result["sources"][0]["source"]
    assert content in tool_result["context"]


def test_runtime_bootstrap_does_not_create_unused_entity_graph_tables(tmp_path):
    repo = create_repository_for_url(f"sqlite:///{tmp_path}/schema.db")
    names = set(inspect(repo.session.get_bind()).get_table_names())
    assert "memories" in names
    assert "entities" not in names
    assert "edges" not in names
