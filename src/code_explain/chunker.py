"""Chunk data model + line-based fallback chunker.

The :class:`Chunk` dataclass is the shared currency of the whole pipeline:
``parser`` produces chunks, ``store`` persists them, ``indexer`` orchestrates
them, and ``retriever`` returns them. Its metadata fields (``kind``,
``symbol``, ``parent_symbol``, line/byte ranges) are deliberately rich so the
future code-graph stage can build edges from chunks without re-parsing files.

Token counts use the cheap ``len(text) // 4`` proxy (no tiktoken dependency).
This is only a budgeting heuristic — the embedder truncates at its own
``num_ctx`` anyway.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------


@dataclass
class Chunk:
    """A retrievable unit of source code.

    ``kind`` is one of: ``function``, ``class``, ``method``, ``module``,
    ``block``, ``text``. ``symbol``/``parent_symbol`` are nullable (a module
    header chunk has ``symbol=None``). ``file_hash``/``mtime`` record the
    provenance of the file this chunk was extracted from and drive staleness.
    """

    chunk_id: str
    rel_path: str
    lang: str
    kind: str
    text: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    n_tokens: int
    file_hash: str
    mtime: float
    symbol: str | None = None
    parent_symbol: str | None = None

    def as_store_row(self) -> dict:
        """Return a dict matching the ``chunks`` table column order."""
        return {
            "chunk_id": self.chunk_id,
            "rel_path": self.rel_path,
            "lang": self.lang,
            "kind": self.kind,
            "symbol": self.symbol,
            "parent_symbol": self.parent_symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "text": self.text,
            "n_tokens": self.n_tokens,
            "file_hash": self.file_hash,
            "mtime": self.mtime,
        }


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 chars/token for Latin text."""
    return max(1, len(text) // 4)


def new_chunk_id() -> str:
    return uuid.uuid4().hex


def file_sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def _line_offsets(source: str) -> list[int]:
    """Byte offset of the start of each line (for translating char->byte)."""
    offsets = [0]
    for ch in source:
        offsets.append(offsets[-1] + len(ch.encode("utf-8")))
    return offsets


# ----------------------------------------------------------------------------
# Line-based fallback chunker
# ----------------------------------------------------------------------------


def line_chunk(
    source: str,
    rel_path: str,
    lang: str,
    file_hash: str,
    mtime: float,
    *,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Sliding-window chunker over whole lines, used as a fallback when AST
    chunking isn't available (unknown language, parse failure, prose files).

    Never splits a line. Windows target ``target_tokens`` with ``overlap_tokens``
    of overlap; stride = ``target - overlap``.
    """
    if not source:
        return []

    lines = source.splitlines(keepends=True)
    if not lines:
        return []

    # Precompute per-line token estimate and byte length.
    line_tokens = [estimate_tokens(ln) for ln in lines]
    stride = max(1, target_tokens - overlap_tokens)

    chunks: list[Chunk] = []
    n = len(lines)
    start = 0
    # char offsets for byte mapping
    char_offset = 0
    line_char_start = []  # char offset at start of each line
    for ln in lines:
        line_char_start.append(char_offset)
        char_offset += len(ln)

    while start < n:
        # accumulate lines until we reach target_tokens
        end = start
        acc = 0
        while end < n and acc < target_tokens:
            acc += line_tokens[end]
            end += 1
        end = min(end, n)  # exclusive end line index

        text = "".join(lines[start:end])
        char_start = line_char_start[start]
        char_end = line_char_start[end - 1] + len(lines[end - 1]) if end > start else char_start
        byte_start = len(source[:char_start].encode("utf-8"))
        byte_end = len(source[:char_end].encode("utf-8"))

        chunks.append(
            Chunk(
                chunk_id=new_chunk_id(),
                rel_path=rel_path,
                lang=lang,
                kind="text",
                text=text,
                start_line=start + 1,
                end_line=end,
                start_byte=byte_start,
                end_byte=byte_end,
                n_tokens=estimate_tokens(text),
                file_hash=file_hash,
                mtime=mtime,
            )
        )

        if end >= n:
            break
        # advance by stride lines for overlap
        next_start = start
        moved = 0
        while next_start < end and moved < stride:
            moved += line_tokens[next_start]
            next_start += 1
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def module_chunk(
    text: str,
    rel_path: str,
    lang: str,
    file_hash: str,
    mtime: float,
    start_line: int,
    start_byte: int,
    end_byte: int,
    end_line: int,
) -> Chunk:
    """Construct a ``module`` kind chunk (file header / top-level statements)."""
    return Chunk(
        chunk_id=new_chunk_id(),
        rel_path=rel_path,
        lang=lang,
        kind="module",
        text=text,
        symbol=None,
        start_line=start_line,
        end_line=end_line,
        start_byte=start_byte,
        end_byte=end_byte,
        n_tokens=estimate_tokens(text),
        file_hash=file_hash,
        mtime=mtime,
    )