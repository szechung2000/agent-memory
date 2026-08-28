# Second-Brain Workflow

This document defines the workflow around `agent-memory` for technical books,
arXiv papers, and personal knowledge. `agent-memory` is the retrieval and
memory engine; the original PDFs and reviewed Obsidian notes remain the
human-readable knowledge base.

## System boundaries

```text
sources/       Immutable PDFs and bibliographic manifests
corpus/        Replaceable extraction artifacts and page-level evidence
agent-memory/  Searchable source and memory index
Obsidian       Curated notes, decisions, and review state
```

The original source is never overwritten. Extraction output can be regenerated
when the extractor or configuration changes. Obsidian notes are drafts until
reviewed, and only reviewed durable claims are promoted to the `brain`
namespace.

## Namespaces

| Namespace | Contents | Trust level |
|---|---|---|
| `paper-corpus` | Extracted arXiv source blocks | Evidence, not interpretation |
| `book-corpus` | Extracted book source blocks | Evidence, not interpretation |
| `agent-derived` | LLM summaries and candidate claims | Unreviewed |
| `brain` | Reviewed, durable personal knowledge | Trusted for normal retrieval |

## Kanban lifecycle

Use one board card per source. The card links to the source note and shows the
next human action. State transitions are recorded in source-note frontmatter so
the board can be rendered manually or by an Obsidian plugin later.

```text
Inbox
  -> Registered
  -> Extracting
  -> Extraction Review
  -> Reading Brief Ready
  -> Claims Need Review
  -> Approved / Promoted
```

Additional terminal or exception states:

- `Blocked`: extraction, metadata, access, or copyright issue needs attention.
- `Archived`: intentionally retained but no longer active.

### State definitions

- **Inbox**: source received but not registered.
- **Registered**: stable source ID, hash, bibliographic metadata, and canonical
  path exist.
- **Extracting**: an ingestion worker is producing normalized evidence.
- **Extraction Review**: quality warnings require review or a better extractor.
- **Reading Brief Ready**: a source-grounded orientation note is available;
  claims are not automatically trusted.
- **Claims Need Review**: selected claims, equations, tables, figures, or
  project applications need verification.
- **Approved / Promoted**: the literature note is reviewed for its intended
  use and selected durable claims may enter `brain`.

## Agent and human responsibilities

### Ingestion worker

May register sources, compute hashes, extract text/layout/OCR, create evidence
records, and report quality warnings. It must not interpret the source or
write trusted memories.

### Reading worker

May produce a reading brief and candidate claims, definitions, methods,
connections, and open questions. Every factual proposal must include evidence
IDs and page/section locators.

### Reviewer worker

Checks entailment, citation completeness, extraction warnings, numerical
claims, and distinction between source claims and inference. It can reject or
request revision, but should not silently turn weak evidence into certainty.

### Human owner

Decides relevance and priority, verifies important claims, approves personal
interpretations, and controls promotion into `brain`.

## Approval policy

Approval is purpose-specific, not a single assertion that every sentence is
true:

```yaml
source_status: extracted
reading_status: brief-ready
review_status: unreviewed
promotion_status: blocked
```

An unread source may be approved as a useful reading brief while its claims
remain unverified. Promotion into `brain` is blocked until the relevant claims
have been checked against evidence.

## Minimum audit fields

Every source workflow should retain:

- `source_id` and byte-level SHA-256
- source URL/path and retrieval date
- document version or book edition
- extractor name, version, and configuration
- extraction warnings and quality status
- agent, timestamp, and action taken
- claim IDs, evidence IDs, and review decisions
- Obsidian note path and memory namespace

## Initial implementation scope

1. Add this workflow specification.
2. Add a deterministic literature-note template.
3. Add source-note frontmatter and review checklists.
4. Add PDF extraction/provenance requirements.
5. Later implement a worker/orchestrator that updates frontmatter and board
   cards idempotently.

The workflow should remain usable with plain Markdown before introducing a
specialized dashboard or database-backed Kanban UI.
