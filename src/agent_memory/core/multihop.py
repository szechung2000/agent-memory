"""Multi-hop retrieval: expand a query through intermediate memories.

L2/L3 questions need evidence spread across several facts. Strategy:
1. First-pass recall with the original query (k1 candidates)
2. Extract expansion terms from top hits — entities/phrases in the hits that
   aren't in the query — and run a second recall pass with query + expansions
3. Fuse: union of both passes, re-scored, deduplicated

This is retrieval-side multi-hop: cheap (two recalls), no LLM required,
and composable with the reranker.
"""

from __future__ import annotations

import re

import numpy as np

from agent_memory.core.models import Memory

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "for", "to",
    "and", "or", "with", "by", "what", "which", "who", "how", "why", "does", "do",
    "did", "that", "this", "it", "its", "from", "at", "as", "be", "been", "their",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 2}


def multihop_recall(
    repo,
    embedder,
    query: str,
    k: int = 10,
    hops: int = 2,
    kind: str | None = None,
    namespace: str | None = None,
    expansion_terms: int = 6,
) -> list[tuple[Memory, float]]:
    """Recall with iterative query expansion. Returns fused, deduped results."""

    def _recall(qtext: str, k_n: int) -> list[tuple[Memory, float]]:
        qvec = np.asarray(embedder.embed([qtext])[0], dtype=np.float32)
        return repo.recall(qvec, qtext, k=k_n, kind=kind, namespace=namespace)

    # hop 0: original query
    first = _recall(query, k)
    results: dict[str, tuple[Memory, float]] = {}
    for m, s in first:
        results[m.id] = (m, s)

    if hops < 2 or not first:
        return sorted(results.values(), key=lambda x: -x[1])[:k]

    # build expansion terms from top hits: content tokens not already in query
    qtoks = _tokens(query)
    expansion: list[str] = []
    seen_terms: set[str] = set()
    for m, _s in sorted(first, key=lambda x: -x[1])[:4]:
        for tok in sorted(_tokens(m.content)):
            if tok not in qtoks and tok not in seen_terms:
                seen_terms.add(tok)
                expansion.append(tok)
            if len(expansion) >= expansion_terms:
                break
        if len(expansion) >= expansion_terms:
            break

    if not expansion:
        return sorted(results.values(), key=lambda x: -x[1])[:k]

    # hop 1: expanded query
    expanded_query = query + " " + " ".join(expansion)
    second = _recall(expanded_query, k)
    for m, s in second:
        if m.id in results:
            # keep best score, small bonus for being found by both paths
            old_m, old_s = results[m.id]
            results[m.id] = (m, max(old_s, s) + 0.03)
        else:
            results[m.id] = (m, s * 0.9)  # slight discount vs direct hits

    return sorted(results.values(), key=lambda x: -x[1])[:k]
