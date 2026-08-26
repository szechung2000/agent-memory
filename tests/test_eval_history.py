"""Tests for eval history persistence + trend readout."""


from agent_memory.evals import run_golden
from agent_memory.evals.run_golden import append_history, print_trend, read_history


def test_append_and_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(run_golden, "HISTORY_FILE", tmp_path / "history.jsonl")
    append_history({"papers": 0.75, "glossary": 1.0}, {"papers": 22.5, "glossary": 18.0})
    append_history({"papers": 0.80, "glossary": 1.0}, {"papers": 21.0, "glossary": 17.5})

    hist = read_history()
    assert len(hist) == 2
    assert hist[1]["scores"]["papers"] == 0.8
    assert hist[0]["latency_ms_mean"]["glossary"] == 18.0
    assert "ts" in hist[0]

    # limit works
    assert len(read_history(limit=1)) == 1


def test_trend_output_shows_suites(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_golden, "HISTORY_FILE", tmp_path / "history.jsonl")
    append_history({"papers": 0.7}, {"papers": 20.0})
    append_history({"papers": 0.75}, {"papers": 22.0})
    print_trend()
    out = capsys.readouterr().out
    assert "eval history" in out
    assert "papers" in out
    assert "70%" in out and "75%" in out


def test_read_history_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run_golden, "HISTORY_FILE", tmp_path / "missing.jsonl")
    assert read_history() == []
