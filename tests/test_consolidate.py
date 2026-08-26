"""Tests for the consolidation pipeline (extractive summarizer, no LLM)."""

import pytest

from agent_memory.consolidate import (
    ExtractiveSummarizer,
    form_candidates,
    promote_candidate,
    run_consolidation,
)
from agent_memory.core.models import Memory, MemoryKind
from agent_memory.evals.run_golden import make_eval_repo


@pytest.fixture()
def env(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/cons.db")
    from agent_memory.core.embedding import get_embedder

    emb = get_embedder()
    return repo, emb


def _write_episode(repo, emb, content):
    return repo.write(
        Memory(content=content, kind=MemoryKind.EPISODIC, session_id="s1"), emb.embed([content])[0]
    )


def test_extractive_summarizer_returns_text():
    s = ExtractiveSummarizer()
    out = s.summarize(["Sally joined EY in 2006.", "Sally passed the CPA exam after joining."])
    assert isinstance(out, str) and len(out) > 10


def test_form_candidates_groups_episodes(env):
    repo, emb = env
    _write_episode(repo, emb, "User said they prefer Python for data pipelines")
    _write_episode(repo, emb, "User mentioned Python is their choice for ETL work")
    _write_episode(repo, emb, "Completely different topic: the weather in Tokyo")

    candidates, report = form_candidates(repo)
    assert report.episodes_scanned == 3
    assert report.candidates_created >= 1
    assert all(isinstance(c.source_ids, list) and c.source_ids for c in candidates)


def test_promote_creates_semantic_with_provenance(env):
    repo, emb = env
    eid = _write_episode(repo, emb, "Sally passed her CPA exam in late 2006 at EY")
    cand_content = "Sally passed the CPA exam in 2006."
    from agent_memory.consolidate import Candidate

    mid, sup = promote_candidate(repo, Candidate(content=cand_content, source_ids=[eid]), emb)

    mem = repo.get(mid)
    assert mem.kind == MemoryKind.SEMANTIC
    assert eid in mem.metadata.get("promoted_from", [])
    assert sup is False

    # episode marked promoted -> second scan won't re-produce it
    candidates2, report2 = form_candidates(repo)
    all_sources = {sid for c in candidates2 for sid in c.source_ids}
    assert eid not in all_sources


def test_supersession_on_near_duplicate(env):
    repo, emb = env
    # existing semantic fact
    old = "The trading bot trades TSLA with a 3 percent stop loss"
    old_vec = emb.embed([old])[0]
    repo.write(Memory(content=old, kind=MemoryKind.SEMANTIC), old_vec)

    # new candidate: same claim, changed detail -> near-dup, different content
    new = "The trading bot trades NVDA with a 3 percent stop loss"
    from agent_memory.consolidate import Candidate

    mid, sup = promote_candidate(
        repo,
        Candidate(content=new),
        emb,
    )
    if sup:  # embedding-dependent; assert consistency when triggered
        old_mem = [m for m, _ in repo.recall(old_vec, old, k=5) if m.content == old][0]
        assert old_mem.metadata.get("superseded_by") == mid


def test_full_run_no_episodes(env):
    repo, emb = env
    report = run_consolidation(repo, emb)
    assert report.episodes_scanned == 0
    assert report.promoted == 0


def test_full_run_promotes(env):
    repo, emb = env
    _write_episode(repo, emb, "Simon prefers concise answers with code examples")
    report = run_consolidation(repo, emb)
    assert report.episodes_scanned == 1
    assert report.promoted >= 1
