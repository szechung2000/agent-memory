"""Offline interviewer dashboard exposes measured, non-fabricated workflows."""

from fastapi.testclient import TestClient

from agent_memory.demo.web import app


def test_dashboard_html_is_self_contained_and_clear_about_limits():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Evidence first." in response.text
    assert "scripted generator" in response.text.lower()
    assert "Labeled synthetic CMBS" in response.text or "labeled synthetic CMBS" in response.text
    assert 'id="live-report" type="file"' in response.text
    assert 'data-panel="live"' in response.text
    assert "never uploads it or calls a model" in response.text


def test_dashboard_reuses_original_labeled_dataset():
    with TestClient(app) as client:
        data = client.get("/api/questions").json()
        overview = client.get("/api/benchmark").json()
    assert data["synthetic"] is True
    assert len(data["cases"]) == 45
    assert overview["documents"] == 32
    assert overview["questions"] == 45
    assert overview["answerable_scored"] == 36
    assert overview["generation_evaluated"] is False
    assert overview["metrics"]["reranked"]["evidence_recall_at_5"] >= 0.70


def test_case_inspection_exposes_actual_sources_and_held_out_metrics():
    with TestClient(app) as client:
        response = client.get("/api/questions/conflict-01")
        missing = client.get("/api/questions/does-not-exist")
    assert response.status_code == 200
    assert missing.status_code == 404
    data = response.json()
    assert data["case"]["gold_support_ids"] == ["correction-a"]
    assert data["case"]["expected_answer"] == "$18.9 million"
    for variant in ("baseline", "reranked"):
        assert len(data[variant]["top_5"]) == 5
        assert data[variant]["metrics"]["evidence_recall_at_5"] is not None
        for hit in data[variant]["top_5"]:
            assert hit["source"].startswith("synthetic/")
            assert hit["content"]
            assert hit["gold_evidence"] == (hit["doc_id"] == "correction-a")


def test_unanswerable_question_has_no_fabricated_retrieval_quality_score():
    with TestClient(app) as client:
        data = client.get("/api/questions/unknown-01").json()
    assert data["case"]["answerable"] is False
    assert data["case"]["expected_answer"] is None
    assert data["baseline"]["metrics"] is None
    assert data["reranked"]["metrics"] is None


def test_scripted_citation_demo_verifies_quotes_and_abstains_without_llm():
    with TestClient(app) as client:
        good = client.get("/api/citations/supported")
        missing = client.get("/api/citations/abstain")
        unknown = client.get("/api/citations/invalid")
    assert good.status_code == missing.status_code == 200
    assert unknown.status_code == 404
    verified = good.json()
    assert verified["status"] == "answered"
    assert verified["mode"] == "scripted_generator_not_live_llm"
    assert verified["ingest"]["chunks_written"] == 3
    assert verified["citations"][0]["quote"] in next(
        source["content"] for source in verified["retrieved_evidence"]
        if source["id"] == verified["citations"][0]["memory_id"]
    )
    abstained = missing.json()
    assert abstained["status"] == "insufficient_evidence"
    assert abstained["citations"] == []
