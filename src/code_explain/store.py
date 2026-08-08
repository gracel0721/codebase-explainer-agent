"""Vector store: a ``VectorStore`` Protocol and a ``SQLiteVecStore`` impl.

This module is the persistence layer. It owns the SQLite schema (the ``chunks``,
``files``, ``meta`` tables and the sqlite-vec ``chunk_vec`` virtual table) and
exposes a small protocol surface. The indexer and retriever depend on the
Protocol, never on the concrete class — so a future LanceDB store (or a
graph-augmented store) can replace this implementation without touching the
rest of the pipeline.

macOS note: the SQLite bundled with the *system* Python lacks loadable-
extension support, so sqlite-vec cannot load. We detect that and fall back to
``pysqlite3`` if it's installed (install the ``macos-system-python`` extra);
otherwise we raise a clear, actionable error.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import sqlite_vec

from code_explain.chunker import Chunk


SCHEMA_VERSION = "1"


# Stage 2 graph tables. Additive and lazy (created via
# ``SQLiteVecStore.ensure_graph_tables``, like ``chunk_vec``) — NOT part of
# ``SCHEMA_SQL`` and do NOT bump ``SCHEMA_VERSION``. An older DB without them
# simply means graph features no-op.
GRAPH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS edges (
    source_chunk_id TEXT NOT NULL,
    target_chunk_id TEXT NOT NULL,
    edge_kind       TEXT NOT NULL,
    via_symbol      TEXT,
    created_at      REAL NOT NULL,
    PRIMARY KEY(source_chunk_id, target_chunk_id, edge_kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_chunk_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_chunk_id);

CREATE TABLE IF NOT EXISTS imports (
    rel_path         TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    alias            TEXT,
    module_path      TEXT NOT NULL,
    target_rel_path  TEXT,
    PRIMARY KEY(rel_path, symbol, module_path)
);
"""


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    rel_path      TEXT NOT NULL,
    lang          TEXT NOT NULL,
    kind          TEXT NOT NULL,
    symbol        TEXT,
    parent_symbol TEXT,
    start_line    INTEGER NOT NULL,
    end_line      INTEGER NOT NULL,
    start_byte    INTEGER NOT NULL,
    end_byte      INTEGER NOT NULL,
    text          TEXT NOT NULL,
    n_tokens      INTEGER NOT NULL,
    file_hash     TEXT NOT NULL,
    mtime         REAL NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_rel_path ON chunks(rel_path);

CREATE TABLE IF NOT EXISTS files (
    rel_path   TEXT PRIMARY KEY,
    lang       TEXT,
    file_hash  TEXT NOT NULL,
    mtime      REAL NOT NULL,
    n_chunks   INTEGER NOT NULL,
    indexed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class ChunkHit:
    chunk_id: str
    distance: float  # cosine distance (lower == better, sqlite-vec)


@dataclass
class FileRecord:
    rel_path: str
    lang: str | None
    file_hash: str
    mtime: float
    n_chunks: int
    indexed_at: float


class VectorStore(Protocol):
    """Storage seam: indexer/retriever depend on this, not the concrete class."""

    def open(self) -> None: ...
    def upsert_file(
        self, file: FileRecord, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None: ...
    def delete_file(self, rel_path: str) -> None: ...
    def search(self, query_vec: list[float], top_k: int) -> list[ChunkHit]: ...
    def get_chunk(self, chunk_id: str) -> Chunk | None: ...
    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]: ...
    def get_file_records(self) -> dict[str, FileRecord]: ...
    def get_meta(self, key: str) -> str | None: ...
    def set_meta(self, key: str, value: str) -> None: ...
    def count_chunks(self) -> int: ...
    def count_files(self) -> int: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------

_CHUNK_COLUMNS = (
    "chunk_id",
    "rel_path",
    "lang",
    "kind",
    "symbol",
    "parent_symbol",
    "start_line",
    "end_line",
    "start_byte",
    "end_byte",
    "text",
    "n_tokens",
    "file_hash",
    "mtime",
    "created_at",
)


def row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        rel_path=row["rel_path"],
        lang=row["lang"],
        kind=row["kind"],
        symbol=row["symbol"],
        parent_symbol=row["parent_symbol"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        start_byte=row["start_byte"],
        end_byte=row["end_byte"],
        text=row["text"],
        n_tokens=row["n_tokens"],
        file_hash=row["file_hash"],
        mtime=row["mtime"],
    )


# ---------------------------------------------------------------------------
# SQLite + sqlite-vec connection bootstrap
# ---------------------------------------------------------------------------


def _open_connection(db_path: Path) -> sqlite3.Connection:
    """Open a sqlite3 connection with loadable-extension support.

    Falls back to pysqlite3 when the stdlib sqlite3 lacks
    ``enable_load_extension`` (macOS system Python). Raises a clear error if
    no extension-capable sqlite is available.
    """
    has_native = hasattr(sqlite3.Connection, "enable_load_extension")
    conn = None
    if has_native:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            # Extension load failed with native sqlite; close and try pysqlite3.
            conn.close()
            conn = None

    if conn is None:
        try:
            import pysqlite3 as _pysqlite3  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sqlite-vec could not be loaded. The current Python's sqlite3 "
                "does not support loadable extensions (common on the macOS "
                "system Python). Install the fallback with "
                "`pip install \"code-explain[macos-system-python]\"` or use "
                "Homebrew Python (`brew install python@3.12`)."
            ) from exc
        conn = _pysqlite3.connect(str(db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ---------------------------------------------------------------------------
# SQLiteVecStore
# ---------------------------------------------------------------------------


class SQLiteVecStore:
    """Concrete ``VectorStore`` over SQLite + sqlite-vec."""

    def __init__(self, db_path: Path, embed_dim: int) -> None:
        self.db_path = db_path
        self.embed_dim = embed_dim
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle --------------------------------------------------------

    def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = _open_connection(self.db_path)
        self._conn.executescript(SCHEMA_SQL)
        self._ensure_vec_table()
        self.ensure_fts_table()
        self.backfill_fts()
        # Seed schema_version if absent.
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", SCHEMA_VERSION)

    def _ensure_vec_table(self) -> None:
        """Create the vec0 table if missing. Dimension is fixed at creation."""
        exists = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_vec'"
        ).fetchone()
        if not exists:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE chunk_vec USING vec0("
                f"chunk_id TEXT PRIMARY KEY, embedding FLOAT[{self.embed_dim}])"
            )
            self._conn.commit()

    def ensure_graph_tables(self) -> None:
        """Create the Stage 2 ``edges``/``imports`` tables if missing.

        Additive and optional — calling this on a fresh DB does not affect
        Stage 1 behavior. Graph code calls this before reading/writing edges.
        """
        self._conn.executescript(GRAPH_SCHEMA_SQL)
        self._conn.commit()

    def ensure_fts_table(self) -> None:
        """Create the hybrid-search FTS5 table if missing.

        A regular (content-bearing) FTS5 table over chunk text/symbol/path.
        ``chunk_id`` is UNINDEXED (stored, not matched) so we can map hits back
        to chunks and delete rows by id on re-index. Created lazily in
        :meth:`open`; an older DB without it simply means FTS adds nothing until
        a re-index populates it.
        """
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "chunk_id UNINDEXED, text, symbol, rel_path)"
        )
        self._conn.commit()

    def backfill_fts(self) -> None:
        """Idempotently ensure every chunk has an FTS row.

        Incremental re-indexing skips unchanged files (the fast mtime path), so
        chunks inserted before the FTS table existed — or by an older version
        that didn't write FTS — would otherwise be absent from hybrid search
        even though they're in the vector index. This repairs the gap on open
        so an upgraded/incremental index gets full FTS coverage without a
        ``--force`` rebuild.
        """
        conn = self._conn
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        if n_fts >= n_chunks:
            return  # complete (or close enough; --force recovers rare drift)
        conn.execute(
            "INSERT INTO chunks_fts(chunk_id, text, symbol, rel_path) "
            "SELECT chunk_id, text, symbol, rel_path FROM chunks "
            "WHERE chunk_id NOT IN (SELECT chunk_id FROM chunks_fts)"
        )
        conn.commit()

    def drop_all(self) -> None:
        """Drop all stored data and recreate the (empty) vector + FTS tables.

        Called by the indexer when the embedding model/dim changed or on
        ``--force``. Operates via this method (not raw ``_conn`` access in the
        indexer) so a non-SQLite backend can implement its own teardown.
        """
        conn = self._conn
        conn.executescript(
            "DELETE FROM chunks; DELETE FROM files; DELETE FROM meta; "
            "DROP TABLE IF EXISTS chunk_vec; "
            "DROP TABLE IF EXISTS chunks_fts; "
            "DROP TABLE IF EXISTS edges; DROP TABLE IF EXISTS imports;"
        )
        conn.commit()
        self._ensure_vec_table()  # recreate with the configured dim
        self.ensure_fts_table()

    def stored_embed_dim(self) -> int | None:
        """The embedding dimension this store was created with, or None."""
        v = self.get_meta("embed_dim")
        return int(v) if v is not None else None

    def stored_embed_model(self) -> str | None:
        return self.get_meta("embed_model")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- meta -------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    # -- file/chunk writes ------------------------------------------------

    def upsert_file(
        self, file: FileRecord, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        assert len(chunks) == len(embeddings), "chunk/embedding count mismatch"
        conn = self._conn
        try:
            conn.execute("BEGIN")
            # Remove any prior chunks/vectors for this file.
            self._delete_file_rows(file.rel_path)
            # Insert chunks + vectors.
            now = time.time()
            for chunk, emb in zip(chunks, embeddings):
                conn.execute(
                    f"INSERT INTO chunks ({', '.join(_CHUNK_COLUMNS)}) "
                    f"VALUES ({', '.join('?' * len(_CHUNK_COLUMNS))})",
                    _chunk_params(chunk, now),
                )
                conn.execute(
                    "INSERT INTO chunk_vec(chunk_id, embedding) VALUES (?, ?)",
                    (chunk.chunk_id, sqlite_vec.serialize_float32(emb)),
                )
                conn.execute(
                    "INSERT INTO chunks_fts(chunk_id, text, symbol, rel_path) "
                    "VALUES (?, ?, ?, ?)",
                    (chunk.chunk_id, chunk.text, chunk.symbol or "", chunk.rel_path),
                )
            # Upsert the file provenance row.
            conn.execute(
                "INSERT INTO files(rel_path, lang, file_hash, mtime, n_chunks, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(rel_path) DO UPDATE SET "
                "lang=excluded.lang, file_hash=excluded.file_hash, mtime=excluded.mtime, "
                "n_chunks=excluded.n_chunks, indexed_at=excluded.indexed_at",
                (
                    file.rel_path,
                    file.lang,
                    file.file_hash,
                    file.mtime,
                    len(chunks),
                    file.indexed_at,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _delete_file_rows(self, rel_path: str) -> None:
        """Delete all chunks + vectors + file row for a path. Caller holds txn."""
        conn = self._conn
        ids = [
            r["chunk_id"]
            for r in conn.execute(
                "SELECT chunk_id FROM chunks WHERE rel_path = ?", (rel_path,)
            ).fetchall()
        ]
        if ids:
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM chunk_vec WHERE chunk_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", ids)
        conn.execute("DELETE FROM chunks WHERE rel_path = ?", (rel_path,))
        conn.execute("DELETE FROM files WHERE rel_path = ?", (rel_path,))

    def delete_file(self, rel_path: str) -> None:
        conn = self._conn
        try:
            conn.execute("BEGIN")
            self._delete_file_rows(rel_path)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def touch_file(self, rel_path: str, mtime: float, indexed_at: float) -> None:
        """Update only the ``files`` row mtime/indexed_at, leaving chunks intact.

        Used when a file's mtime changed but its content hash is identical (e.g.
        a ``touch``), so we don't want to re-chunk/re-embed — and critically must
        NOT delete the existing chunks.
        """
        self._conn.execute(
            "UPDATE files SET mtime = ?, indexed_at = ? WHERE rel_path = ?",
            (mtime, indexed_at, rel_path),
        )
        self._conn.commit()

    # -- reads ------------------------------------------------------------

    def search(self, query_vec: list[float], top_k: int) -> list[ChunkHit]:
        qblob = sqlite_vec.serialize_float32(query_vec)
        rows = self._conn.execute(
            "SELECT chunk_id, distance FROM chunk_vec "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (qblob, top_k),
        ).fetchall()
        return [ChunkHit(chunk_id=r["chunk_id"], distance=float(r["distance"])) for r in rows]

    def search_fts(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Hybrid-search: BM25-ranked chunk_ids for ``query`` (best first).

        Returns ``(chunk_id, bm25_score)`` tuples. The query is wrapped as a
        quoted phrase so FTS5 operator characters don't break the match; any
        ``OperationalError`` (malformed query, table absent) returns ``[]`` so
        the retriever falls back to vector-only.
        """
        if not query.strip():
            return []
        # Escape double quotes and wrap as a phrase to dodge FTS5 syntax errors.
        q = '"' + query.replace('"', '""') + '"'
        try:
            rows = self._conn.execute(
                "SELECT chunk_id, bm25(chunks_fts) AS score FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (q, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["chunk_id"], float(r["score"])) for r in rows]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute(
            f"SELECT {', '.join(_CHUNK_COLUMNS)} FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT {', '.join(_CHUNK_COLUMNS)} FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        by_id = {r["chunk_id"]: r for r in rows}
        return [row_to_chunk(by_id[c]) for c in chunk_ids if c in by_id]

    def get_file_records(self) -> dict[str, FileRecord]:
        rows = self._conn.execute(
            "SELECT rel_path, lang, file_hash, mtime, n_chunks, indexed_at FROM files"
        ).fetchall()
        return {
            r["rel_path"]: FileRecord(
                rel_path=r["rel_path"],
                lang=r["lang"],
                file_hash=r["file_hash"],
                mtime=float(r["mtime"]),
                n_chunks=int(r["n_chunks"]),
                indexed_at=float(r["indexed_at"]),
            )
            for r in rows
        }

    def count_chunks(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def count_files(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])

    def count_edges(self) -> int:
        """Number of graph edges, or 0 if the ``edges`` table doesn't exist yet."""
        try:
            return int(self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        except sqlite3.OperationalError:
            return 0

    def count_imports(self) -> int:
        """Number of resolved/unresolved import rows, or 0 if absent."""
        try:
            return int(self._conn.execute("SELECT COUNT(*) FROM imports").fetchone()[0])
        except sqlite3.OperationalError:
            return 0


def _chunk_params(chunk: Chunk, now: float) -> tuple:
    """Return chunk fields in ``_CHUNK_COLUMNS`` order for INSERT.

    ``created_at`` is not part of the :class:`Chunk` model; it's stamped here
    at insert time.
    """
    row = chunk.as_store_row()
    return tuple(row[c] for c in _CHUNK_COLUMNS if c != "created_at") + (now,)