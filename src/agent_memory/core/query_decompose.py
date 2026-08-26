"""Query decomposition for multi-aspect retrieval.

INRAExplorer-style query understanding: split a multi-part question into aspect
sub-queries so each evidence family gets its own retrieval quota.

Two decomposers behind one interface:
- HeuristicDecomposer: clause splitting on punctuation/markers — offline baseline
- LLMDecomposer: gpt-4o-mini generates sub-queries when AM_OPENAI_API_KEY is set

The eval harness can also run in "oracle" mode (aspects taken from golden labels)
to measure the ceiling of decomposition-based retrieval.
"""

from __future__ import annotations

import re


class Decomposer:
    def decompose(self, question: str) -> list[str]:
        raise NotImplementedError


class HeuristicDecomposer(Decomposer):
    """Clause splitting: markers, colons, semicolons. No dependencies."""

    _SPLIT_RE = re.compile(r"(?:; |: | but | whereas | and also )")

    def decompose(self, question: str) -> list[str]:
        parts = [p.strip(" ?.") for p in self._SPLIT_RE.split(question)]
        parts = [p for p in parts if len(p.split()) >= 3]
        return parts[:4] if len(parts) > 1 else []


class LLMDecomposer(Decomposer):
    PROMPT = (
        "Break this question into independent sub-questions, each answerable "
        "from a separate fact. One per line, no numbering.\n\nQuestion: {q}"
    )

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI()

    def decompose(self, question: str) -> list[str]:
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": self.PROMPT.format(q=question)}],
            temperature=0,
        )
        raw = [
            line.strip("-•0123456789. ").strip()
            for line in (resp.choices[0].message.content or "").splitlines()
        ]
        return [line for line in raw if len(line.split()) >= 3][:4]


def get_decomposer() -> Decomposer:
    from agent_memory.core.config import get_settings

    return LLMDecomposer() if get_settings().openai_api_key else HeuristicDecomposer()


def auto_aspects(question: str) -> list[str]:
    """Non-empty aspect list for a question ([] = single-aspect)."""
    return get_decomposer().decompose(question)
