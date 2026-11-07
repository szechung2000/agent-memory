"""Interview demo and provenance contract tests."""

from agent_memory.demo.fde_workflow import SAMPLE_DOCS, run_demo


def test_fde_demo_traces_retrieval_and_source_metadata():
    trace = run_demo()

    assert trace["synthetic_data"] is True
    assert trace["ingest"] == {"chunks_written": 3, "duplicates_skipped": 3}
    assert trace["tool_calls"] == ["memory_search", "memory_context", "memory_write"]
    assert len(trace["retrieved_evidence"]) == len(SAMPLE_DOCS)
    assert {r["source"] for r in trace["retrieved_evidence"]} == {
        source for source, _ in SAMPLE_DOCS
    }
    assert all(r["id"] and r["metadata"]["source"] for r in trace["retrieved_evidence"])
    assert "CMBS Loan A" in trace["context"]["context"]
    for source in trace["context"]["sources"]:
        assert f"[memory:{source['memory_id']}]" in trace["context"]["context"]
    assert trace["episode"]["status"] == "remembered"
