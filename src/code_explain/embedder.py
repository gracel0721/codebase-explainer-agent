"""Ollama embedding client.

Two pitfalls are handled here (both from the plan):
1. ``nomic-embed-text`` silently truncates at 2048 tokens unless ``num_ctx`` is
   raised, so every embed call passes ``options={"num_ctx": embed_n_ctx}``.
2. The current API is ``ollama.embed(input=[...])`` returning ``embeddings``
   (plural list); the deprecated ``ollama.embeddings()`` returns ``embedding``
   (singular) and must not be used.

Batches of ~32 texts per call keep cold-start amortized and throughput sane.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ollama

if TYPE_CHECKING:
    from code_explain.config import Config

log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


class Embedder:
    def __init__(self, cfg: "Config", batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.cfg = cfg
        self.model = cfg.embed_model
        self.num_ctx = cfg.embed_n_ctx
        self.batch_size = batch_size
        self._client = ollama.Client(host=cfg.ollama_host)

    def _embed_batch_raw(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embed(
            model=self.model,
            input=texts,
            options={"num_ctx": self.num_ctx},
        )
        # ollama>=0.3 returns a dict-like response with "embeddings" (plural).
        embs = getattr(resp, "embeddings", None)
        if embs is None and isinstance(resp, dict):
            embs = resp.get("embeddings")
        if embs is None:
            # Fallback for the deprecated singular shape (one input).
            single = getattr(resp, "embedding", None) or (resp.get("embedding") if isinstance(resp, dict) else None)
            if single is not None:
                embs = [single]
        if embs is None:
            raise RuntimeError(f"Unexpected ollama.embed response shape: {type(resp)}")
        return [list(map(float, v)) for v in embs]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, in batches. Returns one vector per input text."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(self._embed_batch_raw(batch))
        if len(out) != len(texts):
            raise RuntimeError(
                f"embedding count mismatch: requested {len(texts)}, got {len(out)}"
            )
        return out

    def embed_query(self, query: str) -> list[float]:
        return self._embed_batch_raw([query])[0]