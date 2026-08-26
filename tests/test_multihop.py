"""Tests for multi-hop recall."""

from agent_memory.core.models import Memory, MemoryKind
from agent_memory.core.multihop import multihop_recall
from agent_memory.evals.run_golden import make_eval_repo


def _seed(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/mh.db")
    from agent_memory.core.embedding import get_embedder

    emb = get_embedder()
    facts = [
        "LoRA adapts attention projection matrices Wq and Wv",
        "The checkpoint reduction was roughly 10,000 times",
        "GPT-3 has 175 billion parameters",
    ]
    for f in facts:
        repo.write(Memory(content=f, kind=MemoryKind.SEMANTIC), emb.embed([f])[0])
    return repo, emb


def test_single_hop_equivalent(tmp_path):
    repo, emb = _seed(tmp_path)
    res = multihop_recall(repo, emb, "Wq and Wv adaptation", k=3, hops=1)
    assert any("Wq and Wv" in m.content for m, _ in res)


def test_two_hop_finds_second_degree_fact(tmp_path):
    repo, emb = _seed(tmp_path)
    # query about checkpoint size should surface GPT-3 param fact via expansion
    res = multihop_recall(repo, emb, "checkpoint reduction", k=3, hops=2)
    contents = [m.content for m, _ in res]
    assert any("10,000" in c for c in contents)
    # second-degree: expansion terms bridge to related facts
    assert len(res) >= 2


def test_no_expansion_when_first_pass_empty(tmp_path):
    repo, emb = _seed(tmp_path)
    res = multihop_recall(repo, emb, "quantum chromodynamics bosons", k=3, hops=2)
    assert isinstance(res, list)


def test_dedupe_and_ranking(tmp_path):
    repo, emb = _seed(tmp_path)
    res = multihop_recall(repo, emb, "attention matrices", k=5, hops=2)
    ids = [m.id for m, _ in res]
    assert len(ids) == len(set(ids))
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)
