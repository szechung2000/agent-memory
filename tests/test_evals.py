"""CI gate: golden suites must clear minimum recall thresholds.

Note: with the local hash embedder these are smoke-level gates (retrieval
plumbing works). Real quality thresholds kick in once OpenAI embeddings are
enabled — the same suites then act as regression tests for embedding/retrieval
changes.
"""

import pytest

from agent_memory.core.embedding import get_embedder
from agent_memory.evals.run_golden import make_eval_repo, run_suite

THRESHOLDS = {
    "multihop": 0.4,  # hash embedder is weak; real gates once OpenAI embeddings on
    "glossary": 0.6,
    "temporal": 0.4,  # strict temporal logic needs real embeddings + reranking
}


@pytest.fixture()
def repo():
    return make_eval_repo()


@pytest.mark.parametrize("suite_name", list(THRESHOLDS))
def test_golden_suite(repo, suite_name):
    score, _ = run_suite(repo, get_embedder(), suite_name)
    assert score >= THRESHOLDS[suite_name], (
        f"{suite_name}: {score:.0%} < {THRESHOLDS[suite_name]:.0%}"
    )
