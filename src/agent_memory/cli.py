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
    from agent_memory.core.models import Memory as _M  # noqa: F401

    repo, embedder = _repo_and_embedder()
    if getattr(args, "multihop", False):
        from agent_memory.core.multihop import multihop_recall

        results = multihop_recall(repo, embedder, args.query, k=args.k, hops=2)
    else:
        qvec = embedder.embed([args.query])[0]
        results = repo.recall(qvec, args.query, k=args.k, namespace=args.namespace)
    if not results:
        print("no results")
        return 0
    for m, score in results:
        title = f" [{m.title}]" if m.title else ""
        print(f"{score:.3f}{title} | {m.content[:120].replace(chr(10), ' ')}")
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    from agent_memory.consolidate import form_candidates, promote_candidate

    repo, embedder = _repo_and_embedder()
    candidates, report = form_candidates(repo)
    print(
        f"episodes scanned: {report.episodes_scanned}  groups: {report.groups_formed}  "
        f"candidates: {report.candidates_created}"
    )
    promoted = superseded = 0
    for i, cand in enumerate(candidates):
        if args.review:
            ans = input(f"[{i + 1}/{len(candidates)}] promote? {cand.content[:100]} [y/N] ")
            if not ans.strip().lower().startswith("y"):
                continue
        _, sup = promote_candidate(repo, cand, embedder)
        promoted += 1
        superseded += 1 if sup else 0
    print(f"promoted: {promoted}  superseded: {superseded}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    from agent_memory.evals.run_golden import print_trend

    print_trend(limit=args.limit)
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
    p_recall.add_argument("--multihop", action="store_true", help="iterative query expansion")
    p_recall.set_defaults(func=cmd_recall)

    p_cons = sub.add_parser("consolidate", help="distill episodes into semantic facts")
    p_cons.add_argument(
        "--review", action="store_true", help="interactive approval before each promotion"
    )
    p_cons.set_defaults(func=cmd_consolidate)

    p_hist = sub.add_parser("eval-history", help="show golden-eval score trends")
    p_hist.add_argument("--limit", type=int, default=10)
    p_hist.set_defaults(func=cmd_history)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
