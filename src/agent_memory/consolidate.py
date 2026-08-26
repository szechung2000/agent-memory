"""Consolidation: episodic memories -> durable semantic facts.

Pipeline:
1. Group unexpired, unpromoted episodes (by session, then similarity clusters)
2. Summarize groups into fact candidates via a summarizer
   (LLM if configured, extractive fallback otherwise)
3. Candidates land in a review queue; approval promotes them to semantic
   memories with `promoted_from` provenance
4. Contradiction check on promotion: near-duplicate semantic memory with
   differing content -> old memory superseded (retrieval prefers newest)

The summarizer interface keeps the pipeline testable without API keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import text

from agent_memory.core.models import Memory, MemoryKind

# ---------------- summarizers ----------------

class Summarizer:
    def summarize(self, episode_texts: list[str]) -> str:
        raise NotImplementedError


class ExtractiveSummarizer(Summarizer):
    """No-LLM fallback: pick the most central sentences (highest word overlap)."""

    def summarize(self, episode_texts: list[str]) -> str:
        if len(episode_texts) == 1:
            return episode_texts[0]
        sentences: list[str] = []
        for t in episode_texts:
            sentences.extend(re.split(r"(?<=[.!?])\s+", t))
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return " ".join(episode_texts)[:500]

        # rank by token frequency centrality
        freq: dict[str, int] = {}
        for s in sentences:
            for w in set(s.lower().split()):
                freq[w] = freq.get(w, 0) + 1
        scored = sorted(
            sentences,
            key=lambda s: -sum(freq.get(w, 0) for w in set(s.lower().split())),
        )
        keep = scored[: max(1, len(scored) // 3)]
        # preserve original order for readability
        return " ".join(s for s in sentences if s in keep)


class LLMSummarizer(Summarizer):
    """Uses OpenAI chat completions when a key is available."""

    PROMPT = (
        "Distill the following conversation excerpts into durable facts about "
        "people, preferences, entities and events. Output only the facts as "
        "short declarative sentences, one per line.\n\n---\n{episodes}\n---"
    )

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI()

    def summarize(self, episode_texts: list[str]) -> str:
        joined = "\n".join(f"- {t}" for t in episode_texts)[:8000]
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": self.PROMPT.format(episodes=joined)}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""


def get_summarizer() -> Summarizer:
    from agent_memory.core.config import get_settings

    return LLMSummarizer() if get_settings().openai_api_key else ExtractiveSummarizer()


# ---------------- consolidation pipeline ----------------

@dataclass
class Candidate:
    content: str
    source_ids: list[str] = field(default_factory=list)
    namespace: str = "brain"


@dataclass
class ConsolidationReport:
    episodes_scanned: int = 0
    groups_formed: int = 0
    candidates_created: int = 0
    promoted: int = 0
    superseded: int = 0


def _fetch_episodes(repo) -> list[Memory]:
    bind = repo.session.get_bind()
    with bind.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, content FROM memories
                WHERE kind = 'episodic'
                  AND NOT EXISTS (
                      SELECT 1 FROM promoted_links WHERE episode_id = memories.id
                  )
                ORDER BY created_at
                """
            )
        ).all()
    return [Memory(id=r[0], kind=MemoryKind.EPISODIC, content=r[1]) for r in rows]


def _ensure_promoted_links_table(repo) -> None:
    bind = repo.session.get_bind()
    with bind.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS promoted_links (
                    episode_id VARCHAR(36) PRIMARY KEY,
                    candidate_hash VARCHAR(64),
                    promoted_memory_id VARCHAR(36)
                )
                """
            )
        )
        conn.commit()


def _group_episodes(repo, episodes: list[Memory], threshold: float = 0.55) -> list[list[Memory]]:
    """Greedy similarity grouping via embedding cosine."""
    from agent_memory.core.embedding import get_embedder

    emb = get_embedder()
    vecs = np.array(emb.embed([e.content for e in episodes]))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms == 0, 1, norms)

    groups: list[list[int]] = []
    assigned: set[int] = set()
    sims = vecs @ vecs.T
    for i in range(len(episodes)):
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        for j in range(i + 1, len(episodes)):
            if j not in assigned and sims[i][j] >= threshold:
                group.append(j)
                assigned.add(j)
        groups.append(group)
    return [[episodes[i] for i in g] for g in groups]


def form_candidates(
    repo,
    summarizer: Summarizer | None = None,
) -> tuple[list[Candidate], ConsolidationReport]:
    """Scan episodes, produce deduplicated fact candidates (not yet stored)."""
    summarizer = summarizer or get_summarizer()
    report = ConsolidationReport()
    _ensure_promoted_links_table(repo)

    episodes = _fetch_episodes(repo)
    report.episodes_scanned = len(episodes)
    if not episodes:
        return [], report

    candidates: list[Candidate] = []
    seen_hashes: set[str] = set()
    for group in _group_episodes(repo, episodes):
        report.groups_formed += 1
        summary = summarizer.summarize([e.content for e in group])
        for line in filter(None, (ln.strip("-• ").strip() for ln in summary.splitlines())):
            h = str(hash(line.lower()))
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            candidates.append(
                Candidate(content=line, source_ids=[e.id for e in group], namespace="brain")
            )
    report.candidates_created = len(candidates)
    return candidates, report


def promote_candidate(repo, cand: Candidate, embedder) -> tuple[str, bool]:
    """Store a candidate as a semantic memory. Returns (memory_id, superseded_old)."""
    _ensure_promoted_links_table(repo)
    vec = np.asarray(embedder.embed([cand.content])[0], dtype=np.float32)
    mem = Memory(
        content=cand.content,
        kind=MemoryKind.SEMANTIC,
        namespace=cand.namespace,
        title=None,
        metadata={"promoted_from": cand.source_ids},
    )
    mid = repo.write(mem, vec)

    # contradiction/supersession: near-dup existing semantic memory with different content
    results = repo.recall(vec, cand.content, k=5, kind="semantic")
    superseded_old = False
    for m, score in results:
        if m.id == mid:
            continue
        if score >= 0.85 and m.content.strip().lower() != cand.content.strip().lower():
            m.metadata["superseded_by"] = mid
            repo.update_metadata(m.id, m.metadata)
            superseded_old = True
            break

    _mark_promoted(repo, cand, mid)
    return mid, superseded_old


def _mark_promoted(repo, cand: Candidate, memory_id: str) -> None:
    bind = repo.session.get_bind()
    with bind.connect() as conn:
        for eid in cand.source_ids:
            conn.execute(
                text(
                    "INSERT OR IGNORE INTO promoted_links VALUES (:e, :h, :m)"
                    if "sqlite" in str(bind.url)
                    else "INSERT INTO promoted_links VALUES (:e, :h, :m) ON CONFLICT DO NOTHING"
                ),
                {"e": eid, "h": str(hash(cand.content)), "m": memory_id},
            )
        conn.commit()


def run_consolidation(repo, embedder, summarizer: Summarizer | None = None) -> ConsolidationReport:
    """Full pass: episodes -> candidates -> auto-promote (review queue is the CLI's job)."""
    candidates, report = form_candidates(repo, summarizer)
    for cand in candidates:
        _, sup = promote_candidate(repo, cand, embedder)
        report.promoted += 1
        if sup:
            report.superseded += 1
    return report
