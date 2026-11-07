"""Citation verification and safe abstention for retrieval-augmented answers."""

import json
from types import SimpleNamespace

import pytest

from agent_memory.core.answer import (
    ABSTENTION,
    CitationValidationError,
    Evidence,
    OpenAIAnswerGenerator,
    RerankerUnavailableError,
    answer_question,
)
from agent_memory.core.embedding import LocalHashEmbedder
from agent_memory.core.models import Memory, MemoryKind
from agent_memory.demo.cited_answer import run_demo
from agent_memory.evals.run_golden import make_eval_repo

SOURCE_TEXT = "CMBS Loan A: September 2026 remittance reports a 30-day delinquency."


class FakeGenerator:
    def __init__(self, response: dict | str):
        self.response = response
        self.calls = 0

    def generate(self, question, evidence):
        self.calls += 1
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


def draft(source_id="S1", quote=SOURCE_TEXT, text="Loan A was delinquent."):
    return {
        "insufficient_evidence": False,
        "claims": [
            {"text": text, "citations": [{"source_id": source_id, "quote": quote}]}
        ],
    }


@pytest.fixture()
def corpus():
    repo = make_eval_repo()
    embedder = LocalHashEmbedder()
    repo.write(
        Memory(
            content=SOURCE_TEXT,
            kind=MemoryKind.SEMANTIC,
            namespace="cmbs",
            metadata={"source": "synthetic/remittance.txt", "page": 2},
        ),
        embedder.embed([SOURCE_TEXT])[0],
    )
    yield repo, embedder
    repo.session.close()


def test_valid_claim_has_verified_source_and_exact_quote(corpus):
    repo, embedder = corpus
    result = answer_question(
        repo, embedder, FakeGenerator(draft()), "Which loan was delinquent?",
        namespace="cmbs",
    )
    assert result.status == "answered"
    assert result.answer == "Loan A was delinquent. [S1]"
    assert result.claims[0].citations == ["S1"]
    assert result.citations[0].quote == SOURCE_TEXT
    assert result.citations[0].source == "synthetic/remittance.txt"
    assert result.citations[0].metadata["page"] == 2
    assert result.citations[0].memory_id
    assert result.retrieved_count == 1


def test_no_evidence_abstains_without_using_the_generator(corpus):
    repo, embedder = corpus
    generator = FakeGenerator(draft())
    result = answer_question(
        repo, embedder, generator, "Which loan was delinquent?",
        namespace="missing-namespace",
    )
    assert result.status == "insufficient_evidence"
    assert result.answer == ABSTENTION
    assert result.citations == []
    assert generator.calls == 0


def test_model_can_abstain_when_evidence_is_insufficient(corpus):
    repo, embedder = corpus
    generator = FakeGenerator({"insufficient_evidence": True, "claims": []})
    result = answer_question(
        repo, embedder, generator, "What was Loan A's appraisal?",
        namespace="cmbs",
    )
    assert result.status == "insufficient_evidence"
    assert result.citations == []
    assert generator.calls == 1


@pytest.mark.parametrize(
    "output",
    [
        draft(source_id="S99"),
        draft(quote="Loan Z had a $40 million balance."),
        draft(quote=""),
        draft(quote="Loan A"),
        draft(text="Loan A was delinquent. [S99]"),
        {"insufficient_evidence": False, "claims": [{"text": "Unsupported", "citations": []}]},
        {"insufficient_evidence": True, "claims": draft()["claims"]},
        {"insufficient_evidence": False, "claims": []},
        "not valid JSON",
    ],
)
def test_invalid_citations_fail_closed(corpus, output):
    repo, embedder = corpus
    with pytest.raises(CitationValidationError):
        answer_question(
            repo, embedder, FakeGenerator(output), "Was Loan A delinquent?",
            namespace="cmbs",
        )


def test_reranker_request_fails_if_model_unavailable(corpus, monkeypatch):
    import agent_memory.core.rerank as rerank_module

    monkeypatch.setattr(rerank_module, "_load_cross_encoder", lambda: None)
    repo, embedder = corpus
    with pytest.raises(RerankerUnavailableError):
        answer_question(
            repo, embedder, FakeGenerator(draft()), "Was Loan A delinquent?",
            namespace="cmbs", rerank=True,
        )


def test_offline_demo_uses_real_verification_with_a_scripted_generator():
    trace = run_demo()
    assert trace["mode"] == "scripted_generator_not_live_llm"
    assert trace["status"] == "answered"
    assert "CMBS Loan A" in trace["answer"]
    assert trace["citations"][0]["source"] == "synthetic/servicer-remittance-sep-2026.txt"
    assert trace["citations"][0]["quote"] == SOURCE_TEXT


def test_openai_generator_uses_json_mode_and_delimited_evidence_without_network():
    recorded = {}

    class StubCompletions:
        def create(self, **kwargs):
            recorded.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(draft())))]
            )

    generator = object.__new__(OpenAIAnswerGenerator)
    generator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=StubCompletions())
    )
    generator.model = "test-model"
    evidence = [
        Evidence(
            ref="S1",
            memory_id="memory-123",
            content=SOURCE_TEXT,
            score=0.9,
            title=None,
            source="synthetic/remittance.txt",
            metadata={"source": "synthetic/remittance.txt"},
        )
    ]

    assert json.loads(generator.generate("What happened?", evidence)) == draft()
    assert recorded["response_format"] == {"type": "json_object"}
    assert recorded["model"] == "test-model"
    assert "untrusted" in recorded["messages"][0]["content"]
    payload = json.loads(recorded["messages"][1]["content"])
    assert payload["evidence"][0]["source_id"] == "S1"
    assert payload["evidence"][0]["source"] == "synthetic/remittance.txt"
