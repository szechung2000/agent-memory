"""Shared evidence assembly for HTTP and in-process agent tools."""

import numpy as np


def context_items(repo, embedder, topic: str, k: int) -> list[dict]:
    """Return the same ranked evidence and source metadata to both interfaces."""
    qvec = np.asarray(embedder.embed([topic])[0], dtype=np.float32)
    hits = repo.recall(qvec, topic, k=k)
    return [
        {
            "id": memory.id,
            "kind": memory.kind.value,
            "content": memory.content,
            "title": memory.title,
            "score": float(score),
            "source": memory.metadata.get("source"),
            "metadata": memory.metadata,
        }
        for memory, score in hits
    ]
