"""Re-ranking: a Protocol + an Ollama-LLM reranker.

The retriever returns chunks best-first by vector similarity, but a cheap
semantic rerank with the chat model often improves precision. The seam is the
:class:`Reranker` Protocol; :class:`OllamaReranker` asks the LLM to sort the top
candidate chunks by relevance to the query and returns them in that order.

Gated by ``cfg.reranker_model``: when unset (the default), the retriever never
builds a reranker and pays zero overhead. On any error (model unavailable, bad
JSON, malformed indices) the reranker falls back to the original order so a
rerank failure can never make retrieval worse than vector-only.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from code_explain.chunker import Chunk
    from code_explain.config import Config
    from code_explain.llm import LLMClient

log = logging.getLogger(__name__)

# How many top chunks to ask the LLM to sort. Beyond this the LLM's ordering
# signal is weak and the prompt grows; the tail is appended unchanged.
MAX_CANDIDATES = 20

RERANK_SYSTEM = (
    "You are a code search re-ranker. Given a query and a list of code chunks, "
    "return the chunk indices ordered by relevance to the query, most relevant "
    "first. Respond ONLY as JSON: {\"order\": [idx, ...]} where each idx is the "
    "0-based position from the list. Include every index exactly once."
)


class Reranker(Protocol):
    """Re-order retrieved chunks by relevance to the query."""

    def rerank(self, query: str, chunks: "list[Chunk]") -> "list[Chunk]": ...


class OllamaReranker:
    """Asks the chat model to sort the top chunks; falls back on any error."""

    def __init__(self, cfg: "Config", llm: "LLMClient") -> None:
        self.cfg = cfg
        self.llm = llm
        self.model = cfg.reranker_model

    def rerank(self, query: str, chunks: "list[Chunk]") -> "list[Chunk]":
        if not chunks or not self.model:
            return chunks
        candidates = chunks[:MAX_CANDIDATES]
        tail = chunks[MAX_CANDIDATES:]
        try:
            order = self._ask_llm(query, candidates)
        except Exception as exc:  # noqa: BLE001 — rerank must never break retrieval
            log.debug("rerank failed (%s); keeping vector order.", exc)
            return chunks
        if not order:
            return chunks
        reordered = [candidates[i] for i in order if 0 <= i < len(candidates)]
        # Append any candidates the model omitted, preserving vector order.
        seen = set(order)
        for i, c in enumerate(candidates):
            if i not in seen:
                reordered.append(c)
        return reordered + tail

    def _ask_llm(self, query: str, candidates: "list[Chunk]") -> list[int]:
        lines = [f"[{i}] {c.rel_path}:L{c.start_line} ({c.kind}: {c.symbol or '-'})"
                 f"\n{(c.text or '').strip()[:300]}" for i, c in enumerate(candidates)]
        user = f"Query: {query}\n\nChunks:\n" + "\n\n".join(lines)
        resp = self.llm.chat_turn(
            RERANK_SYSTEM, [{"role": "user", "content": user}], format="json"
        )
        content = getattr(resp.message, "content", None) or ""
        if not content:
            return []
        data = json.loads(content)
        order = data.get("order", []) if isinstance(data, dict) else []
        if not isinstance(order, list):
            return []
        out: list[int] = []
        for v in order:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
        return out