"""Tests for cross-encoder reranking (skips scoring when model unavailable)."""

from agent_memory.core.models import Memory, MemoryKind
from agent_memory.core.rerank import rerank


def _mem(content):
    return Memory(content=content, kind=MemoryKind.SEMANTIC)


def test_rerank_preserves_temporal_gate():
    """A temporally-penalized fact must not beat an unpenalized one on wording."""
    q = "Who owned Bear Stearns before the financial crisis?"
    leak = "In March 2008 JPMorgan Chase acquired Bear Stearns in a fire sale"
    clean = "Bear Stearns was an independent investment bank in New York"

    # fused scores: penalized fact has low fused (temporal penalty applied upstream)
    cands = [(_mem(leak), 0.16), (_mem(clean), 0.55)]
    out = rerank(q, cands, top_k=2)
    assert out[0][0].content == clean


def test_rerank_returns_top_k():
    cands = [(_mem(f"fact {i}"), float(i)) for i in range(5)]
    out = rerank("facts", list(cands), top_k=3)
    assert len(out) <= 3


def test_rerank_empty_candidates():
    assert rerank("q", [], top_k=3) == []
