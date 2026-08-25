"""Embedding providers.

Tier order (first available wins):
1. OpenAI (AM_OPENAI_API_KEY) — text-embedding-3-small
2. Local BGE (BAAI/bge-small-en-v1.5 via sentence-transformers) — real semantic
   embeddings offline, CPU-friendly; used for evals and dev
3. Hash fallback — deterministic bag-of-chars; last-resort only

The local BGE model is an optional dependency: install with
`uv sync --extra local-embeddings` or `uv pip install sentence-transformers`.
"""

import hashlib
from functools import lru_cache

import numpy as np

from agent_memory.core.config import get_settings


class Embedder:
    """Interface: embed(texts) -> list[np.ndarray]."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        raise NotImplementedError


class LocalHashEmbedder(Embedder):
    """Deterministic bag-of-chars hashing embedding. Dev/test fallback."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.dim = dim

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            tokens = t.lower().split()
            for tok in tokens + list(t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                v[h % self.dim] += 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return out


class LocalBGEEmbedder(Embedder):
    """sentence-transformers bge-small-en-v1.5: 384-dim, runs on CPU."""

    model_name = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer  # optional dep

        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [np.asarray(v, dtype=np.float32) for v in vecs]


class OpenAIEmbedder(Embedder):
    def __init__(self) -> None:
        super().__init__()
        from openai import OpenAI

        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.embedding_model

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


@lru_cache(maxsize=1)
def _try_bge() -> Embedder | None:
    try:
        return LocalBGEEmbedder()
    except ImportError:
        return None


def get_embedder() -> Embedder:
    if get_settings().openai_api_key:
        return OpenAIEmbedder()
    bge = _try_bge()
    return bge if bge is not None else LocalHashEmbedder()
