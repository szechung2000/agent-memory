"""Embedding providers.

Uses OpenAI when AM_OPENAI_API_KEY is set; falls back to a deterministic
local hash-based embedder so the service is fully functional offline.
The local embedder is for dev/tests only — retrieval quality is dummy.
"""

import hashlib
import struct

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


class OpenAIEmbedder(Embedder):
    def __init__(self) -> None:
        super().__init__()
        from openai import OpenAI

        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.embedding_model

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


def get_embedder() -> Embedder:
    return OpenAIEmbedder() if get_settings().openai_api_key else LocalHashEmbedder()


# keep struct import used for future packed formats
_ = struct
