# Technical PDF Ingestion Design

## Goal

Ingest technical books and arXiv papers containing dual-column layouts,
equations, charts, tables, and scanned pages without losing the evidence needed
to verify claims later.

`agent-memory` should remain responsible for embedding, deduplication, storage,
retrieval, ranking, and context assembly. PDF parsing and normalization should
be an upstream adapter that emits structured, provenance-bearing blocks.

## Storage layers

```text
sources/       Original immutable PDF and manifest
corpus/        Versioned extraction artifacts and page-level evidence
agent-memory/  Search index over corpus and approved memories
Obsidian       Curated literature notes and review state
```

The original PDF must never be overwritten. A new arXiv revision, book edition,
or extraction configuration creates a new document or corpus version.

## Tiered extraction

### Tier 0: classify

Record whether the PDF has a text layer, approximate text density, likely column
count, equation/table/figure presence, language, and scan/OCR likelihood.

### Tier 1: ordinary text PDFs

Use a page-aware extractor such as PyMuPDF. Keep `pypdf` as a lightweight
fallback. Extract one page at a time and preserve page boundaries, headings,
and conservative header/footer warnings.

### Tier 2: complex technical PDFs

Escalate to an optional layout-aware extractor such as Marker when column order,
equations, tables, or figure captions are materially important.

### Tier 3: scanned PDFs

Route image-only or low-coverage pages through OCR. Persist OCR confidence and
page-level warnings. Empty extraction must be a visible warning, not a silently
successful document.

Heavy OCR/layout dependencies should remain optional rather than mandatory for
all users.

## Normalized block model

A normalized extraction should preserve blocks such as:

- `heading`
- `paragraph`
- `equation`
- `table`
- `table_caption`
- `figure_caption`
- `figure_reference`
- `code`
- `list`

Markdown is a useful rendered output, but should not be the only representation
for equations or tables.

## Required document metadata

```json
{
  "document_id": "sha256:<source-bytes>",
  "source_uri": "books/attention.pdf",
  "source_sha256": "...",
  "title": "Attention Is All You Need",
  "authors": ["Vaswani et al."],
  "published_at": "2017",
  "arxiv_id": "1706.03762",
  "edition": null,
  "page_count": 15,
  "extractor": {
    "name": "pymupdf",
    "version": "...",
    "config_hash": "..."
  },
  "extraction_status": "complete",
  "warnings": []
}
```

## Required chunk metadata

Use the existing memory metadata field initially rather than immediately
redesigning the database:

```json
{
  "document_id": "sha256:<source-bytes>",
  "document_version": "sha256:<source-bytes>:<extractor-config>",
  "source_uri": "books/attention.pdf",
  "source_type": "pdf",
  "page_start": 3,
  "page_end": 4,
  "section_path": ["3 Model Architecture", "3.2 Attention"],
  "block_types": ["paragraph", "equation"],
  "reading_order_confidence": 0.98,
  "extraction_confidence": 0.95,
  "ocr": false,
  "citation": {
    "label": "Attention Is All You Need, pp. 3-4",
    "locator": "p. 3, section 3.2"
  }
}
```

Dedupe must be scoped by source identity. A common passage appearing in two
books must retain both provenance links; global content-only dedupe is unsafe.

## Structure-specific requirements

### Dual columns

Preserve page/block coordinates or a reading-order confidence. Detect likely
column ordering failures and route them to extraction review.

### Equations

Store the equation, LaTeX when available, variable definitions, nearby
explanation, equation number, page, and bounding box. An equation should not be
indexed without enough surrounding prose to explain it.

### Tables

Store both structured cells/rows and a searchable rendering. Preserve captions,
headers, footnotes, page, bounding box, and extraction confidence.

### Figures

Index the figure caption and nearby discussion. Store page and bounding box. Any
agent-generated visual description must be marked `derived`, not treated as a
source fact.

## Quality gates

The ingestion pipeline should validate:

- Stable source hash and recorded page count
- Expected pages represented in normalized output
- Monotonic page order
- Non-empty extraction for text PDFs
- Scan detection and OCR routing
- Reading-order warnings for columns
- Formula/table/caption retention for sampled documents
- Chunk indices and sizes
- Provenance on every indexed chunk
- Idempotent re-ingestion with the same source and extractor configuration
- Preservation of duplicate passages from different sources

## Retrieval evaluation fixtures

Add fixtures for:

- A two-column paper with a fact spanning columns
- An equation requiring nearby explanatory prose
- A table question requiring row/column semantics
- A figure-caption question
- A duplicate passage in two documents
- Two versions of the same paper
- An OCR page with low confidence

Measure retrieval recall, first relevant rank, citation completeness, page
locator preservation, duplicate-source retention, temporal leakage, latency,
and extraction cost separately.

## Incremental implementation

1. Add `pypdf` or PyMuPDF as an explicit optional dependency.
2. Refactor PDF extraction to be page-aware.
3. Add document/source hashes and page/section metadata.
4. Fix dedupe scope to include document identity.
5. Add extraction reports and warnings.
6. Add PyMuPDF as the normal text-PDF adapter.
7. Add an optional layout/OCR adapter.
8. Add structure-aware fixtures and retrieval tests.
9. Add a versioned evidence sidecar before introducing new database tables.
10. Add a structured block table only if sidecar files and metadata no longer
    satisfy retrieval or audit requirements.

Do not make a heavyweight layout model or multimodal vector store a prerequisite
for the first usable version.
