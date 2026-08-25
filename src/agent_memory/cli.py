"""`am` CLI: ingest documents, query memory.

Usage:
    uv run am ingest <path> [--namespace brain]
    uv run am recall "query text" [--k 5] [--namespace brain]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_memory.core.config import get_settings


def _repo_and_embedder():
    from agent_memory.core.embedding import get_embedder
    from agent_memory.ingest.pipeline import get_repo_for_url

    repo = get_repo_for_url(get_settings().database_url)
    return repo, get_embedder()


def cmd_ingest(args: argparse.Namespace) -> int:
    from agent_memory.ingest.pipeline import Ingester

    path = Path(args.path)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1

    repo, embedder = _repo_and_embedder()
    ing = Ingester(repo, embedder)
    result = ing.ingest_path(path, namespace=args.namespace)
    print(
        f"files: {result.files_seen}  chunks written: {result.chunks_written}  "
        f"dupes skipped: {result.duplicates_skipped}"
    )
    for err in result.errors:
        print(f"error: {err}", file=sys.stderr)
    return 1 if result.errors else 0


def cmd_recall(args: argparse.Namespace) -> int:
    repo, embedder = _repo_and_embedder()
    qvec = embedder.embed([args.query])[0]
    results = repo.recall(
        qvec,
        args.query,
        k=args.k,
        namespace=args.namespace,
    )
    if not results:
        print("no results")
        return 0
    for m, score in results:
        title = f" [{m.title}]" if m.title else ""
        print(f"{score:.3f}{title} | {m.content[:120].replace(chr(10), ' ')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="am", description="agent-memory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a file or directory of documents")
    p_ingest.add_argument("path", help="file or directory (.md .txt .pdf)")
    p_ingest.add_argument("--namespace", default="brain")
    p_ingest.set_defaults(func=cmd_ingest)

    p_recall = sub.add_parser("recall", help="query memory")
    p_recall.add_argument("query")
    p_recall.add_argument("--k", type=int, default=5)
    p_recall.add_argument("--namespace", default=None)
    p_recall.set_defaults(func=cmd_recall)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
