"""Chunker: split documents into overlapping chunks, respecting markdown structure.

Target 400-800 chars of content per chunk (token-approximate: ~4 chars/token
gives 100-200 tokens — small enough for precise retrieval, big enough for
context). Splits preferentially at headings, then paragraphs, then sentences,
then hard-wraps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_SIZE = 700
MIN_SIZE = 250

_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_PARA_SPLIT = "\n\n"


@dataclass
class Chunk:
    text: str
    index: int          # position within the source doc
    heading: str | None  # nearest preceding markdown heading, if any


def _split_long_block(block: str) -> list[str]:
    """Hard-split an oversized paragraph at sentence/word boundaries."""
    if len(block) <= TARGET_SIZE:
        return [block]
    parts = re.split(r"(?<=[.!?])\s+", block)
    out: list[str] = []
    cur = ""
    for part in parts:
        if cur and len(cur) + len(part) + 1 > TARGET_SIZE:
            out.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str) -> list[Chunk]:
    """Split text into ordered, optionally overlapping chunks."""
    text = text.strip()
    if not text:
        return []

    # split into sections at markdown headings (keep heading with its section)
    if _HEADING_RE.search(text):
        lines = text.split("\n")
        sections: list[tuple[str | None, list[str]]] = []
        current_heading: str | None = None
        buf: list[str] = []
        for line in lines:
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m:
                if buf:
                    sections.append((current_heading, buf))
                current_heading = m.group(2).strip()
                buf = [line]
            else:
                buf.append(line)
        if buf:
            sections.append((current_heading, buf))
    else:
        sections = [(None, text.split("\n"))]

    chunks: list[Chunk] = []
    for heading, lines in sections:
        block = "\n".join(lines).strip()
        if not block:
            continue
        paragraphs = [p for p in block.split(_PARA_SPLIT) if p.strip()]

        cur_parts: list[str] = []
        cur_size = 0

        def flush(h: str | None) -> None:
            nonlocal cur_parts, cur_size
            if not cur_parts:
                return
            body = _PARA_SPLIT.join(cur_parts).strip()
            if body:
                chunks.append(Chunk(text=body, index=len(chunks), heading=h))
            cur_parts = []
            cur_size = 0

        for para in paragraphs:
            pieces = _split_long_block(para)
            for piece in pieces:
                plen = len(piece)
                if cur_size and cur_size + plen > TARGET_SIZE:
                    flush(heading)
                cur_parts.append(piece)
                cur_size += plen + 2
        flush(heading)

    return chunks
