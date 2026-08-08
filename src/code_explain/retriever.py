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

        # Hybrid search: fuse vector hits with FTS5 BM25 hits via reciprocal
        # rank fusion. Only when the store supports FTS and hybrid is on; a
        # LanceDB backend (no search_fts) degrades to vector-only.
        if self.cfg.hybrid_search and hasattr(self.store, "search_fts"):
            fts_hits = self.store.search_fts(query, self.cfg.top_k)
            fused_ids = self._fuse(hits, fts_hits)
            chunks = self.store.get_chunks(fused_ids)
        else:
            chunks = self.store.get_chunks([h.chunk_id for h in hits])
        # ``store.search`` returns hits best-first (lowest cosine distance first),
        # and ``get_chunks`` preserves the requested order.

        # Optional LLM rerank of the seed vector hits (before per-file cap and
        # graph expansion, so graph interleaving is preserved). Zero overhead
        # when ``reranker_model`` is unset.
        if self.cfg.reranker_model:
            chunks = self._rerank(query, chunks)

        chunks = self._apply_per_file_cap(chunks)
        if self.cfg.with_graph:
            chunks = self._expand_via_graph(chunks)
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

    def _rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Lazy-build and apply the configured reranker. Falls back to the
        original order on any error (see :class:`OllamaReranker`)."""
        if not hasattr(self, "_reranker"):
            from code_explain.llm import LLMClient
            from code_explain.reranker import OllamaReranker

            self._reranker = OllamaReranker(self.cfg, LLMClient(self.cfg))
        return self._reranker.rerank(query, chunks)

    @staticmethod
    def _fuse(hits, fts_hits) -> list[str]:
        """Reciprocal rank fusion of vector + FTS hits -> ordered chunk_ids.

        ``score = 1/(60+rank_vec) + 1/(60+rank_fts)``; the 60 constant is the
        standard RRF damping. Each list is best-first; a chunk appearing in both
        gets both contributions. Returns chunk_ids sorted by fused score desc.
        """
        scores: dict[str, float] = {}
        for rank, h in enumerate(hits):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (60 + rank)
        for rank, (cid, _score) in enumerate(fts_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
        return [cid for cid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    # -- graph expansion (Stage 2) --------------------------------------

    def _expand_via_graph(self, chunks: list[Chunk]) -> list[Chunk]:
        """Pull graph neighbors (callers/callees/contained) of retrieved chunks
        into context. No-op unless ``cfg.with_graph`` and a graph is present.

        Expanded chunks are interleaved right after the seed that produced them,
        so a call chain reads in order. They then flow through the normal token
        budget packing; ``cap_per_seed`` / ``graph_depth`` bound the cost.
        """
        from code_explain import graph

        conn = getattr(self.store, "_conn", None)
        if conn is None or not chunks:
            return chunks
        if not graph.is_graph_present(self.store):
            return chunks
        neighbor_map = graph.expand(
            self.store,
            [c.chunk_id for c in chunks],
            depth=self.cfg.graph_depth,
            cap_per_seed=3,
        )
        if not neighbor_map:
            return chunks
        # Flatten neighbor ids and fetch their chunks in one query.
        all_ids: list[str] = []
        for ids in neighbor_map.values():
            all_ids.extend(ids)
        by_id = {c.chunk_id: c for c in self.store.get_chunks(all_ids)}
        return self._interleave_graph_chunks(chunks, neighbor_map, by_id)

    def _interleave_graph_chunks(
        self,
        seeds: list[Chunk],
        neighbor_map: dict[str, list[str]],
        by_id: dict[str, Chunk],
    ) -> list[Chunk]:
        """Insert each seed's neighbor chunks immediately after the seed."""
        result: list[Chunk] = []
        have: set[str] = {c.chunk_id for c in seeds}
        for seed in seeds:
            result.append(seed)
            for nid in neighbor_map.get(seed.chunk_id, []):
                if nid in have:
                    continue
                chunk = by_id.get(nid)
                if chunk is None:
                    continue
                result.append(chunk)
                have.add(nid)
        return result

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