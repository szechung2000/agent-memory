"""Improve temporal penalty: anchor inference for year-less queries.

'before the financial crisis' -> 2008, 'before the rebrand' etc. handled via
event-anchored known dates + explicit years. Keep it simple and testable.
"""
import re

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

# canonical event anchors: if query references these phrases without a year,
# use the well-known date of the event
EVENT_ANCHORS = {
    "financial crisis": 2008,
    "the crisis": 2008,
}

TEMPORAL_MARKERS = (
    "acquired", "acquisition", "merged", "merger", "bought", "purchased",
    "rebranded", "renamed", "became", "formed", "sold", "absorbed", "fire sale",
)


def _years(text: str) -> list[int]:
    return [int(y) for y in _YEAR_RE.findall(text)]


def query_anchor_year(query: str) -> int | None:
    ys = _years(query)
    ql = query.lower()
    if ys:
        return max(ys)
    for phrase, year in EVENT_ANCHORS.items():
        if phrase in ql:
            return year
    return None


def temporal_penalty(query: str, content: str) -> float:
    """Penalize change-of-state facts dated at/after the query's time anchor.

    'Before the crisis' anchors to 2008 — a merger happening in 2008 is part of
    the post/pre boundary, so events dated >= anchor count as later state
    changes when the query looks backward ("before", "prior", "previously",
    or a plain past-state question like "who owned X in 2002").
    """
    anchor = query_anchor_year(query)
    cy = _years(content)
    if anchor is None or not cy:
        return 0.0
    ql = query.lower()
    if any(w in ql for w in ("after", "since")):
        return 0.0
    if not any(m in content.lower() for m in TEMPORAL_MARKERS):
        return 0.0
    # "before"-style queries: any event at/after the anchor is a later state change
    if min(cy) >= anchor:
        return 0.5
    return 0.0
