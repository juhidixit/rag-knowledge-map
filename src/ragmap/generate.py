"""Grounded answer generation.

With ``ANTHROPIC_API_KEY`` set, Claude writes an answer that cites the retrieved passages
as [1], [2], ... Without a key the service still works and returns an extractive answer
(the best passage), so the demo and the tests never depend on a paid API.
"""

from __future__ import annotations

import os
import re

from ragmap.types import Hit

LLM_MODEL = os.environ.get("RAGMAP_LLM_MODEL", "claude-opus-5-5")

SYSTEM = (
    "You answer questions about a user's documents using only the numbered passages provided. "
    "Cite every claim with the passage number in square brackets, e.g. [2]. "
    "If the passages do not contain the answer, say so plainly instead of guessing. "
    "Keep answers under 150 words."
)


def _context(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        loc = f"section: {h.chunk.section}" if h.chunk.section else ""
        if h.chunk.page:
            loc += f", page {h.chunk.page}"
        parts.append(f"[{i}] ({loc.strip(', ')})\n{h.chunk.text}")
    return "\n\n".join(parts)


def extractive_answer(query: str, hits: list[Hit]) -> dict:
    if not hits:
        return {"answer": "No relevant passage found.", "citations": [], "generator": "extractive"}
    best = hits[0].chunk.text
    sentences = re.split(r"(?<=[.!?])\s+", best)
    q_terms = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 3}
    ranked = sorted(sentences, key=lambda s: -len(q_terms & set(re.findall(r"[a-z0-9]+", s.lower()))))
    answer = " ".join(ranked[:2]).strip()
    return {"answer": f"{answer} [1]", "citations": [1], "generator": "extractive"}


def generate_answer(query: str, hits: list[Hit]) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY") or not hits:
        return extractive_answer(query, hits)
    try:
        import anthropic
    except ImportError:
        return extractive_answer(query, hits)

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=SYSTEM,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": f"Passages:\n\n{_context(hits)}\n\nQuestion: {query}"}],
        )
    except anthropic.APIError as exc:  # degrade gracefully: retrieval results are still useful
        out = extractive_answer(query, hits)
        out["warning"] = f"LLM call failed ({type(exc).__name__}); showing extractive answer."
        return out
    if response.stop_reason == "refusal":
        out = extractive_answer(query, hits)
        out["warning"] = "The model declined this request; showing extractive answer."
        return out
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    cites = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
    return {"answer": text, "citations": cites, "generator": response.model}
