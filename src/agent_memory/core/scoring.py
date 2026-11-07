"""Backend-independent ranking rule for dense, keyword, and temporal evidence."""

from agent_memory.core.temporal import temporal_penalty


def hybrid_score(query: str, content: str, title: str | None, cosine: float) -> float:
    """Score a candidate after its backend computes cosine similarity.

    Storage-specific candidate selection can still change the returned set.
    """
    query_terms = set(query.lower().split())
    document_terms = set(content.lower().split()) | set((title or "").lower().split())
    overlap = len(query_terms & document_terms) / max(len(query_terms), 1)
    return 0.7 * max(cosine, 0.0) + 0.3 * overlap - temporal_penalty(query, content)
