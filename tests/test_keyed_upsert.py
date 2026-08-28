"""Source-scoped idempotent write contract."""

import numpy as np

from agent_memory.core.models import Memory
from agent_memory.evals.run_golden import make_eval_repo


def test_keyed_upsert_is_idempotent_and_merges_metadata(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/upsert.db")
    vector = np.array([1.0, 0.0], dtype=np.float32)

    first_id = repo.upsert(
        "sha256:paper-a:page:1:chunk:0",
        Memory(
            content="The method assumes bounded inputs.",
            namespace="paper-corpus",
            metadata={"document_id": "sha256:paper-a", "page_start": 1},
        ),
        vector,
    )
    second_id = repo.upsert(
        "sha256:paper-a:page:1:chunk:0",
        Memory(
            content="The method assumes bounded inputs.",
            namespace="paper-corpus",
            metadata={"citation": {"locator": "p. 1"}},
        ),
        vector,
    )

    assert second_id == first_id
    saved = repo.get(first_id)
    assert saved is not None
    assert saved.metadata == {
        "document_id": "sha256:paper-a",
        "page_start": 1,
        "citation": {"locator": "p. 1"},
    }
    assert len(repo.recall(vector, "bounded inputs", namespace="paper-corpus")) == 1


def test_keyed_upsert_keeps_identical_text_from_distinct_sources(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/provenance.db")
    vector = np.array([1.0, 0.0], dtype=np.float32)
    content = "A repeated definition appears in two papers."

    paper_a = repo.upsert(
        "sha256:paper-a:page:1:chunk:0",
        Memory(
            content=content,
            namespace="paper-corpus",
            metadata={"document_id": "sha256:paper-a", "page_start": 1},
        ),
        vector,
    )
    paper_b = repo.upsert(
        "sha256:paper-b:page:1:chunk:0",
        Memory(
            content=content,
            namespace="paper-corpus",
            metadata={"document_id": "sha256:paper-b", "page_start": 1},
        ),
        vector,
    )

    assert paper_a != paper_b
    hits = repo.recall(vector, "repeated definition", namespace="paper-corpus")
    assert {hit.metadata["document_id"] for hit, _ in hits} == {
        "sha256:paper-a",
        "sha256:paper-b",
    }
