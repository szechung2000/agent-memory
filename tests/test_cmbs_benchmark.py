"""CMBS benchmark invariants, fair A/B ranking, and citation/abstention scoring."""

import json
from collections import Counter
from dataclasses import replace

import pytest

from agent_memory.core.embedding import LocalHashEmbedder
from agent_memory.evals.cmbs_benchmark import (
    Benchmark,
    bm25_rerank,
    compare_live_variants,
    load_dataset,
    main,
    score_predictions,
)


@pytest.fixture(scope="module")
def data():
    return load_dataset()


def test_dataset_has_held_out_labels_and_balanced_domain_cases(data):
    docs, questions = data
    assert len(docs) == 32
    assert len(questions) == 45
    assert Counter(q.category for q in questions) == {
        "direct": 9,
        "multi_document": 9,
        "temporal": 9,
        "unanswerable": 9,
        "conflict": 9,
    }
    assert all(not q.support_ids and not q.expected_answer for q in questions if not q.answerable)
    assert all(q.expected_answer and q.support_ids for q in questions if q.answerable)
    assert len({d.id for d in docs}) == len(docs)


def test_dataset_rejects_unknown_gold_documents(tmp_path, data):
    docs, questions = data
    (tmp_path / "corpus.json").write_text(json.dumps({"documents": [d.__dict__ for d in docs]}))
    cases = [
        {
            **q.__dict__,
            "support_ids": list(q.support_ids),
            "alternative_support_sets": [list(s) for s in q.alternative_support_sets],
            "must_include": list(q.must_include),
        }
        for q in questions
    ]
    cases[0]["support_ids"] = ["made-up-document"]
    (tmp_path / "questions.json").write_text(json.dumps({"cases": cases}))
    with pytest.raises(ValueError, match="support set"):
        load_dataset(tmp_path)


def test_reranker_cannot_see_gold_labels_and_uses_same_candidate_pool(data):
    docs, questions = data
    benchmark = Benchmark(docs, LocalHashEmbedder())
    try:
        question = questions[0]
        base, reranked, _, _ = benchmark.rankings(question.question, "bm25")
        # Changing gold IDs cannot affect ranking. Neither candidate path receives labels.
        different_gold = replace(question, support_ids=("terms-e",))
        base2, reranked2, _, _ = benchmark.rankings(different_gold.question, "bm25")
        assert [m.id for m, _ in base] == [m.id for m, _ in base2]
        assert [m.id for m, _ in reranked] == [m.id for m, _ in reranked2]
        assert {m.id for m, _ in base} == {m.id for m, _ in reranked}
        assert all(set(m.metadata) == {"doc_id", "source"} for m, _ in base)
    finally:
        benchmark.close()


def test_retrieval_reports_baseline_and_reranking_with_both_k_values(data):
    docs, questions = data
    benchmark = Benchmark(docs, LocalHashEmbedder())
    try:
        report = benchmark.evaluate_retrieval(questions, "bm25")
    finally:
        benchmark.close()
    assert report["questions"] == 45
    assert report["answerable_scored"] == 36
    assert report["reranker"] == "bm25"
    for variant in ("baseline", "reranked"):
        metric = report["metrics"][variant]
        assert 0 <= metric["evidence_recall_at_5"] <= metric["evidence_recall_at_10"] <= 1
        assert 0 <= metric["full_support_at_5"] <= metric["full_support_at_10"] <= 1
        assert 0 <= metric["mrr_at_10"] <= 1
    # Conservative smoke gates for the fixed hash corpus, not production-quality targets.
    assert report["metrics"]["baseline"]["evidence_recall_at_5"] >= 0.65
    assert report["metrics"]["reranked"]["evidence_recall_at_5"] >= 0.70
    assert len(report["cases"]) == 36
    assert "conflict" in report["by_category"]
    assert "unanswerable" not in report["by_category"]
    assert bm25_rerank("empty", []) == []


def _prediction(case_id, status, answer="", doc_id=None, quote=None):
    cites = [{"doc_id": doc_id, "quote": quote}] if doc_id else []
    return {"case_id": case_id, "status": status, "answer": answer, "citations": cites}


def test_perfect_citations_and_correct_abstention_score_one(data):
    docs, questions = data
    result = score_predictions(
        questions,
        docs,
        [
            _prediction(
                "direct-01", "answered", "$20.0 million",
                doc_id="terms-a", quote=docs[0].content,
            ),
            _prediction("unknown-01", "insufficient_evidence"),
        ],
        require_complete=False,
    )
    assert result["complete"] is False
    assert result["evaluated"] == 2
    assert result["citation_quote_anchor_rate"] == 1
    assert result["gold_citation_precision"] == 1
    assert result["gold_citation_recall"] == 1
    assert result["key_fact_coverage_proxy"] == 1
    assert result["abstention_precision"] == 1
    assert result["abstention_recall"] == 1


def test_wrong_or_fabricated_citation_and_unsupported_answer_reduce_scores(data):
    docs, questions = data
    result = score_predictions(
        questions,
        docs,
        [
            _prediction(
                "direct-01", "answered", "An unrelated claim",
                doc_id="terms-b", quote=docs[1].content,
            ),
            _prediction(
                "conflict-01", "answered", "$18.9 million",
                doc_id="correction-a", quote="fake quote from no document",
            ),
            _prediction("unknown-01", "answered", "Invented answer"),
        ],
        require_complete=False,
    )
    assert result["gold_citation_precision"] == 0
    assert result["citation_quote_anchor_rate"] == 0.5
    assert result["key_fact_coverage_proxy"] == 0.5
    assert result["abstention_recall"] == 0


def test_correctly_anchored_quote_does_not_establish_claim_truth(data):
    docs, questions = data
    result = score_predictions(
        questions, docs,
        [
            _prediction(
                "direct-01", "answered", "Loan A principal was $1.",
                "terms-a", docs[0].content,
            )
        ],
        require_complete=False,
    )
    assert result["gold_citation_precision"] == 1
    assert result["key_fact_coverage_proxy"] == 0


def test_false_abstention_and_invalid_prediction_status_are_reported(data):
    docs, questions = data
    result = score_predictions(
        questions, docs,
        [
            _prediction("direct-01", "insufficient_evidence"),
            _prediction("unknown-01", "insufficient_evidence"),
        ],
        require_complete=False,
    )
    assert result["false_abstentions"] == 1
    assert result["abstention_precision"] == 0.5
    with pytest.raises(ValueError, match="status"):
        score_predictions(
            questions, docs, [_prediction("direct-01", "random")], require_complete=False
        )


def test_missing_or_duplicate_predictions_cannot_appear_complete(data):
    docs, questions = data
    one = _prediction("unknown-01", "insufficient_evidence")
    with pytest.raises(ValueError, match="every"):
        score_predictions(questions, docs, [one])
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions(questions, docs, [one, one], require_complete=False)


def test_complete_prediction_set_is_required_and_scoreable(data, tmp_path):
    docs, questions = data
    by_id = {d.id: d for d in docs}
    # Synthetic scorer fixture built from labels; NEVER a reported model result.
    predictions = [
        (
            _prediction(
                q.id, "answered", q.expected_answer,
                q.support_ids[0], by_id[q.support_ids[0]].content,
            )
            if q.answerable else _prediction(q.id, "insufficient_evidence")
        )
        for q in questions
    ]
    result = score_predictions(questions, docs, predictions)
    assert result["complete"] is True
    assert result["evaluated"] == 45
    assert result["abstention_precision"] == 1
    assert result["abstention_recall"] == 1
    assert result["gold_citation_precision"] == 1
    assert result["gold_citation_recall"] < 1  # one citation cannot cover every multi-hop set
    fixture_path = tmp_path / "synthetic-scorer-fixture.json"
    report_path = tmp_path / "report.json"
    fixture_path.write_text(json.dumps(predictions))
    assert main([
        "--predictions", str(fixture_path),
        "--output", str(report_path),
    ]) == 0
    saved = json.loads(report_path.read_text())
    assert saved["answer_evaluation"]["evaluated"] == 45
    assert saved["answer_evaluation_type"] == "external_predictions"


def test_alternate_evidence_set_counts_as_full_support(data):
    docs, questions = data
    q = next(item for item in questions if item.id == "temporal-07")
    assert q.alternative_support_sets == (("workout-a",),)
    from agent_memory.evals.cmbs_benchmark import _coverage

    assert _coverage(["workout-a"], q.acceptable_support_sets, 1) == 1


def test_rejects_paid_live_run_without_explicit_key(monkeypatch):
    monkeypatch.delenv("AM_OPENAI_API_KEY", raising=False)
    from agent_memory.core.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(SystemExit):
        main(["--live"])


def test_cross_encoder_unavailable_fails_instead_of_silent_fallback(data, monkeypatch):
    import agent_memory.core.rerank as rerank_module

    monkeypatch.setattr(rerank_module, "_load_cross_encoder", lambda: None)
    docs, _ = data
    benchmark = Benchmark(docs, LocalHashEmbedder())
    try:
        with pytest.raises(RuntimeError, match="unavailable"):
            benchmark.rankings("Loan A maturity", "cross-encoder")
    finally:
        benchmark.close()


def test_prediction_generation_does_not_read_gold_fields(data):
    docs, questions = data
    benchmark = Benchmark(docs, LocalHashEmbedder())

    class AbstainingGenerator:
        def generate(self, question, evidence):
            assert question and evidence
            assert all("expected_answer" not in e.metadata for e in evidence)
            return json.dumps({"insufficient_evidence": True, "claims": []})

    try:
        generated = benchmark.generate_predictions(
            questions[:2], AbstainingGenerator(), "bm25", "reranked"
        )
    finally:
        benchmark.close()
    assert all(p["status"] == "insufficient_evidence" for p in generated)
    assert [p["case_id"] for p in generated] == [q.id for q in questions[:2]]


def test_generation_to_citation_scoring_path_uses_only_retrieved_evidence(data):
    docs, questions = data
    benchmark = Benchmark(docs, LocalHashEmbedder())

    class QuoteOnlyGenerator:
        def generate(self, question, evidence):
            source = evidence[0]
            return json.dumps({
                "insufficient_evidence": False,
                "claims": [{
                    "text": "A fact was retrieved.",
                    "citations": [{"source_id": source.ref, "quote": source.content}],
                }],
            })

    try:
        predictions = benchmark.generate_predictions(
            questions[:1], QuoteOnlyGenerator(), "bm25", "reranked"
        )
    finally:
        benchmark.close()
    assert predictions[0]["status"] == "answered"
    result = score_predictions(questions, docs, predictions, require_complete=False)
    assert result["citation_quote_anchor_rate"] == 1


class MeteredGenerator:
    """Offline stand-in for an API call; tests never invoke a paid provider."""

    def __init__(self):
        self.calls = []
        self.last_usage = {}

    def generate(self, question, evidence):
        self.calls.append((question, [item.metadata["doc_id"] for item in evidence]))
        self.last_usage = {
            "prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42,
        }
        if "insurance premium" in question.lower() or "appraised value" in question.lower():
            return json.dumps({"insufficient_evidence": True, "claims": []})
        return json.dumps({
            "insufficient_evidence": False,
            "claims": [{
                "text": "Evidence was returned.",
                "citations": [{"source_id": evidence[0].ref, "quote": evidence[0].content}],
            }],
        })


def test_live_pair_uses_one_retrieval_per_question_and_scores_both(data, monkeypatch):
    docs, questions = data
    selected = [questions[0], next(q for q in questions if q.id == "unknown-01")]
    benchmark = Benchmark(docs, LocalHashEmbedder())
    generator = MeteredGenerator()
    calls = []
    original = benchmark.rankings

    def tracked(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(benchmark, "rankings", tracked)
    try:
        predictions = benchmark.generate_paired_predictions(selected, generator, "bm25")
    finally:
        benchmark.close()
    assert len(calls) == 2  # Same retrieval and candidate pool, two generation calls.
    assert len(generator.calls) == 4
    assert generator.calls[0][0] == generator.calls[1][0] == questions[0].question
    assert all("expected_answer" not in label for _, labels in generator.calls for label in labels)
    assert all(p["generation_latency_ms"] >= 0 for rows in predictions.values() for p in rows)
    report = compare_live_variants(questions, docs, predictions)
    assert report["paired_questions"] == 2
    assert report["complete"] is False
    assert report["baseline"]["evaluated"] == report["reranked"]["evaluated"] == 2
    assert report["baseline"]["citation_quote_anchor_rate"] == 1
    assert report["baseline"]["abstention_recall"] == 1
    assert report["token_usage"]["baseline"]["total_tokens"] == 84
    assert report["token_usage"]["reranked"]["reported_calls"] == 2
    assert "generation_latency_ms_p95" in report["latency_ms"]["baseline"]


def test_paired_comparison_rejects_different_case_sets(data):
    docs, questions = data
    with pytest.raises(ValueError, match="same question IDs"):
        compare_live_variants(questions, docs, {
            "baseline": [{"case_id": "direct-01"}],
            "reranked": [{"case_id": "direct-02"}],
        })


def test_live_cli_paired_partial_smoke_run_is_explicit(data, tmp_path, monkeypatch):
    from agent_memory.core.config import get_settings

    monkeypatch.setenv("AM_OPENAI_API_KEY", "test-only-mocked")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "agent_memory.evals.cmbs_benchmark.OpenAIAnswerGenerator",
        lambda **kwargs: MeteredGenerator(),
    )
    output = tmp_path / "paired.json"
    try:
        assert main([
            "--live", "--live-mode", "both", "--max-live-cases", "2",
            "--output", str(output),
        ]) == 0
    finally:
        get_settings.cache_clear()
    report = json.loads(output.read_text())
    assert report["answer_evaluation_type"] == "live_model_paired"
    assert report["generation_requests"] == 4
    assert report["live_questions_evaluated"] == 2
    assert report["answer_evaluation"]["complete"] is False
    assert set(report["predictions"]) == {"baseline", "reranked"}
    assert len(report["predictions"]["baseline"]) == 2


def test_live_case_cap_requires_opt_in_and_valid_range(monkeypatch):
    with pytest.raises(SystemExit):
        main(["--max-live-cases", "3"])
    from agent_memory.core.config import get_settings

    monkeypatch.setenv("AM_OPENAI_API_KEY", "test-only-mocked")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit):
            main(["--live", "--max-live-cases", "0"])
        with pytest.raises(SystemExit):
            main(["--live", "--max-live-cases", "46"])
    finally:
        get_settings.cache_clear()
