"""Cross-encoder reranking: refine fused candidates by query-document relevance.

Uses bge-reranker-base (small, CPU-friendly) via sentence-transformers
CrossEncoder. Optional dependency — rerank() degrades to identity when the
package/model isn't available.

Reranking fixes the *ordering* problem: dense+keyword fusion can bury an
evidence fact below k when its embedding is mediocre but its textual relevance
to the question is high — exactly the L3 multi-part case.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

_RERANK_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _load_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(_RERANK_MODEL, max_length=512)
    except Exception:  # ImportError or model download failure
        return None


def rerank(
    query: str,
    candidates: list[tuple],
    top_k: int,
    score_attr: str = "content",
) -> list[tuple]:
    """Resort candidates by cross-encoder relevance. Identity fallback.

    candidates: list of (memory, fused_score) tuples; memory.content is scored.
    The incoming fused score ALREADY includes the temporal penalty — blending
    preserves it so the reranker cannot resurface temporally-leaking facts.

    Returns top_k by blended score (ties keep fused order).
    """
    if not candidates:
        return []

    ce = _load_cross_encoder()
    if ce is None:
        return sorted(candidates, key=lambda x: -x[1])[:top_k]

    pairs = [(query, getattr(m, score_attr)) for m, _ in candidates]
    scores = ce.predict(pairs)  # relevance logits

    # normalize CE logits to [0,1] so the blend is scale-comparable with fused [0,1]
    arr = np.asarray(scores, dtype=np.float32)
    rng = float(arr.max() - arr.min())
    norm = (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

    from agent_memory.core.temporal import temporal_penalty

    rescored = []
    for (m, fused), rel in zip(candidates, norm, strict=True):
        # blend; fused carries the temporal penalty signal, and we re-apply it so the
        # cross-encoder cannot resurface temporally-leaking facts on textual similarity
        penalty = temporal_penalty(query, getattr(m, score_attr))
        blended = 0.6 * float(rel) + 0.4 * max(fused, 0.0)
        blended -= penalty * 2.0  # hard-ish gate: leaks must not win on wording alone
        rescored.append((m, blended))
    rescored.sort(key=lambda x: -x[1])
    return rescored[:top_k]
