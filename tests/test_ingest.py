"""Tests for chunker + ingestion pipeline."""



from agent_memory.core.chunker import chunk_text
from agent_memory.evals.run_golden import make_eval_repo
from agent_memory.ingest.pipeline import Ingester, parse_frontmatter


def _ingester(tmp_path):
    repo = make_eval_repo(f"sqlite:///{tmp_path}/ingest.db")
    from agent_memory.core.embedding import get_embedder

    return Ingester(repo, get_embedder())


# ---------- chunker ----------

def test_chunk_short_text_single_chunk():
    chunks = chunk_text("This is a short document about pgvector.")
    assert len(chunks) == 1
    assert chunks[0].index == 0


def test_chunk_respects_target_size():
    para = "Sentence one. " * 30  # ~420 chars
    text = "\n\n".join([para] * 5)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 900 for c in chunks)  # hard ceiling sanity


def test_chunk_keeps_heading_with_section():
    md = "# Alpha\n\nContent for alpha section.\n\n## Beta\n\nDifferent content here."
    chunks = chunk_text(md)
    assert any(c.heading == "Alpha" for c in chunks)
    assert any(c.heading == "Beta" for c in chunks)


def test_chunk_indices_sequential():
    text = "\n\n".join("word " * 50 for _ in range(6))
    chunks = chunk_text(text)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


# ---------- frontmatter ----------

def test_frontmatter_parsed_and_stripped():
    meta, body = parse_frontmatter("---\ntitle: Notes\ntags: [rl, cs6601]\n---\n\nBody here.")
    assert meta["title"] == "Notes"
    assert meta["tags"] == ["rl", "cs6601"]
    assert body.lstrip().startswith("Body here.")


def test_no_frontmatter():
    meta, body = parse_frontmatter("Just text.")
    assert meta == {}
    assert body == "Just text."


# ---------- pipeline ----------

def test_ingest_writes_chunks_and_dedupes(tmp_path):
    ing = _ingester(tmp_path)
    doc = "# Topic\n\n" + ("Paragraph about retrieval quality. " * 20)
    r1 = ing.ingest_text(doc)
    assert r1.chunks_written > 0
    n_first = r1.chunks_written

    r2 = ing.ingest_text(doc)  # identical content -> all dupes
    assert r2.chunks_written == 0
    assert r2.duplicates_skipped == n_first


def test_ingest_markdown_file_with_metadata(tmp_path):
    ing = _ingester(tmp_path)
    f = tmp_path / "note.md"
    f.write_text(
        "---\ntags: [market-data]\n---\n\n# Bloomberg notes\n\n"
        "Feed handlers normalize exchange data."
    )
    res = ing.ingest_path(f)
    assert res.files_seen == 1
    assert res.chunks_written >= 1

    # verify metadata landed on the memory
    from agent_memory.core.embedding import get_embedder
    emb = get_embedder()
    results = ing.repo.recall(emb.embed(["feed handlers"])[0], "feed handlers", k=1)
    mem = results[0][0]
    assert "market-data" in mem.metadata.get("tags", [])
    assert mem.metadata["doc_type"] == "md"


def test_ingest_directory(tmp_path):
    ing = _ingester(tmp_path)
    (tmp_path / "a.md").write_text("Alpha doc content about databases.")
    (tmp_path / "b.txt").write_text("Beta file content about embeddings.")
    (tmp_path / "c.pdf").write_text("ignored placeholder")  # pdf extraction may fail gracefully
    (tmp_path / "skip.me").write_text("not supported")
    res = ing.ingest_path(tmp_path)
    assert res.files_seen >= 2  # a.md and b.txt at minimum; c.pdf may error but counted
