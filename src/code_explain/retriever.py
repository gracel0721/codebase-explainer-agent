"""Retrieval: query embedding -> vector search -> re-ranking -> context assembly.

The retriever returns :class:`Chunk` objects (not strings) so the future agent
layer can inspect metadata to choose tool calls without re-querying.

Re-ranking (v1): a per-file cap prevents one huge file from starving context
diversity. A future cross-encoder reranker slots in behind ``rerank()`` without
touching the LLM layer.

Context-window budgeting greedily packs retrieved chunks (highest relevance
first) into the budget left after reserving room for the system prompt,
conversation history, the question, and the answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from code_explain.chunker import Chunk
from code_explain.embedder import Embedder
from code_explain.store import VectorStore

if TYPE_CHECKING:
    from code_explain.config import Config


SYSTEM_PROMPT_TOKENS = 400
QUESTION_TOKENS = 100
HISTORY_TURNS_KEPT = 6


class Retriever:
    def __init__(self, cfg: "Config", store: VectorStore, embedder: Embedder) -> None:
        self.cfg = cfg
        self.store = store
        self.embedder = embedder

    # -- search ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        history: list[dict] | None = None,
    ) -> list[Chunk]:
        """Return chunks for the query, packed into the context budget."""
        qvec = self.embedder.embed_query(query)
        hits = self.store.search(qvec, top_k=self.cfg.top_k)
        if not hits:
            return []
        chunks = self.store.get_chunks([h.chunk_id for h in hits])
        # ``store.search`` returns hits best-first (lowest cosine distance first),
        # and ``get_chunks`` preserves the requested order.

        chunks = self._apply_per_file_cap(chunks)
        chunks = self._maybe_add_file_headers(chunks)
        budget = self._context_budget(history or [])
        packed = self._pack_budget(chunks, budget)
        return packed

    def _apply_per_file_cap(self, chunks: list[Chunk]) -> list[Chunk]:
        cap = self.cfg.per_file_cap
        if cap <= 0:
            return chunks
        kept: list[Chunk] = []
        per_file: dict[str, int] = {}
        for c in chunks:
            n = per_file.get(c.rel_path, 0)
            if n < cap:
                kept.append(c)
                per_file[c.rel_path] = n + 1
        # Refill from the remaining chunks if we dropped any.
        kept_set = {id(c) for c in kept}
        leftover = [c for c in chunks if id(c) not in kept_set]
        return kept + leftover

    def _maybe_add_file_headers(self, chunks: list[Chunk]) -> list[Chunk]:
        if not self.cfg.with_file_headers:
            return chunks
        # Find small module/header chunks per file; if a file has function/method
        # chunks but its header wasn't retrieved, pull it in from the store.
        files_in_context: dict[str, list[Chunk]] = {}
        for c in chunks:
            files_in_context.setdefault(c.rel_path, []).append(c)

        additions: list[Chunk] = []
        addition_ids: set[str] = set()
        for rel, file_chunks in files_in_context.items():
            has_header = any(c.kind == "module" for c in file_chunks)
            if has_header:
                continue
            # Look for a small module chunk of this file that wasn't retrieved.
            header = self._find_file_header(rel, exclude={c.chunk_id for c in chunks})
            if header is not None and header.chunk_id not in addition_ids:
                additions.append(header)
                addition_ids.add(header.chunk_id)
        if not additions:
            return chunks
        # Insert each header just before its file's first chunk.
        result = list(chunks)
        for h in additions:
            # place header at the position of its file's first occurrence
            idx = next((i for i, c in enumerate(result) if c.rel_path == h.rel_path), len(result))
            result.insert(idx, h)
        return result

    def _find_file_header(self, rel_path: str, exclude: set[str]) -> Chunk | None:
        conn = getattr(self.store, "_conn", None)
        if conn is None:
            return None
        row = conn.execute(
            "SELECT chunk_id, rel_path, lang, kind, symbol, parent_symbol, "
            "start_line, end_line, start_byte, end_byte, text, n_tokens, file_hash, mtime "
            "FROM chunks WHERE rel_path = ? AND kind = 'module' "
            "ORDER BY start_line LIMIT 1",
            (rel_path,),
        ).fetchone()
        if row is None:
            return None
        from code_explain.store import row_to_chunk

        chunk = row_to_chunk(row)
        if chunk.n_tokens > 200:
            return None
        if chunk.chunk_id in exclude:
            return None
        return chunk

    # -- budgeting ------------------------------------------------------

    def _context_budget(self, history: list[dict]) -> int:
        history_tokens = 0
        for msg in history[-HISTORY_TURNS_KEPT * 2 :]:
            history_tokens += len(str(msg.get("content", ""))) // 4
        reserved = SYSTEM_PROMPT_TOKENS + history_tokens + QUESTION_TOKENS + self.cfg.answer_max_tokens
        return max(512, self.cfg.llm_n_ctx - reserved)

    def _pack_budget(self, chunks: list[Chunk], budget: int) -> list[Chunk]:
        packed: list[Chunk] = []
        used = 0
        for c in chunks:
            if used + c.n_tokens > budget and packed:
                break
            packed.append(c)
            used += c.n_tokens
        return packed

    # -- rendering ------------------------------------------------------

    @staticmethod
    def render_context(chunks: list[Chunk]) -> str:
        blocks: list[str] = []
        for c in chunks:
            symbol = c.symbol if c.symbol else "-"
            header = f"=== FILE: {c.rel_path}  L{c.start_line}-{c.end_line}  ({c.kind}: {symbol}) ==="
            blocks.append(header + "\n" + c.text)
        return "\n\n".join(blocks)