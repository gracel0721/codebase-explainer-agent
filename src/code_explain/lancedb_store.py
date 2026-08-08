"""LanceDB vector store — an alternative ``VectorStore`` backend.

Stores chunk rows + their vectors in a single Lance table; file provenance and
meta live in a small SQLite sidecar at ``cfg.db_path`` (the same ``files``/
``meta`` schema as :class:`SQLiteVecStore`, but without sqlite-vec — the
sidecar is a plain sqlite3 connection). Selected by ``cfg.vector_backend ==
"lancedb"`` via :func:`code_explain.indexer.open_store`.

Limitations (by design — see the plan):
* The code graph queries ``store._conn``'s chunks/edges tables, so it is
  SQLite-only. This store has no ``_conn``; :func:`code_explain.graph.build_graph`
  no-ops when ``_conn`` is absent, and ``is_graph_present`` returns False.
* Hybrid FTS5 search is SQLite-only. This store has no ``search_fts``; the
  retriever guards with ``hasattr(store, "search_fts")`` and degrades to
  vector-only.

``lancedb``/``pyarrow`` are imported lazily inside methods so this module
imports cleanly without the ``lancedb`` extra installed.
"""

from __future__ import annotations

import sqlite3
import time
from typing import TYPE_CHECKING

from code_explain.chunker import Chunk
from code_explain.store import ChunkHit, FileRecord

if TYPE_CHECKING:
    from code_explain.config import Config


# The SQLite sidecar schema (files + meta only — chunks live in Lance).
_SIDECAR_SQL = """
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

# Lance table columns (Chunk metadata + the vector), in addition order.
_LANCE_COLUMNS = (
    "chunk_id", "rel_path", "lang", "kind", "symbol", "parent_symbol",
    "start_line", "end_line", "start_byte", "end_byte", "text", "n_tokens",
    "file_hash", "mtime",
)


class LanceDBStore:
    """``VectorStore`` over LanceDB (vectors + chunk rows) + a SQLite sidecar."""

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self.db_path = cfg.db_path
        self.embed_dim = cfg.embed_dim
        self.lance_dir = cfg.index_dir / "lance"
        self._db = None  # lancedb connection
        self._tbl = None  # the chunks LanceTable
        self._side: sqlite3.Connection | None = None  # files + meta sidecar
        # No ``_conn``: graph/FTS code detects its absence and no-ops.

    # -- lifecycle --------------------------------------------------------

    def open(self) -> None:
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise RuntimeError(
                "The 'lancedb' extra is required for vector_backend='lancedb'. "
                "Install it with `pip install \"code-explain[lancedb]\"`."
            ) from exc

        self.lance_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.lance_dir))
        self._side = sqlite3.connect(str(self.db_path))
        self._side.row_factory = sqlite3.Row
        self._side.executescript(_SIDECAR_SQL)
        self._side.commit()

        schema = self._arrow_schema(pa)
        # ``list_tables()`` returns a response object whose ``in`` check doesn't
        # match plain strings, so probe by opening and fall back to creating.
        try:
            self._tbl = self._db.open_table("chunks")
        except ValueError:
            self._tbl = self._db.create_table("chunks", schema=schema, mode="overwrite")

    def _arrow_schema(self, pa):
        return pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("rel_path", pa.string()),
            pa.field("lang", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("symbol", pa.string()),
            pa.field("parent_symbol", pa.string()),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
            pa.field("start_byte", pa.int32()),
            pa.field("end_byte", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("n_tokens", pa.int32()),
            pa.field("file_hash", pa.string()),
            pa.field("mtime", pa.float64()),
            pa.field("vector", pa.list_(pa.float32(), self.embed_dim)),
        ])

    def close(self) -> None:
        if self._side is not None:
            self._side.close()
            self._side = None
        self._tbl = None
        self._db = None

    def drop_all(self) -> None:
        """Drop the Lance table + sidecar data and recreate empty."""
        import pyarrow as pa

        try:
            self._db.drop_table("chunks")
        except Exception:  # noqa: BLE001 - table may not exist yet
            pass
        self._tbl = self._db.create_table(
            "chunks", schema=self._arrow_schema(pa), mode="overwrite"
        )
        self._side.executescript("DELETE FROM files; DELETE FROM meta;")
        self._side.commit()

    # -- meta (sidecar) --------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self._side.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._side.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._side.commit()

    def stored_embed_dim(self) -> int | None:
        v = self.get_meta("embed_dim")
        return int(v) if v is not None else None

    def stored_embed_model(self) -> str | None:
        return self.get_meta("embed_model")

    # -- writes ----------------------------------------------------------

    def upsert_file(
        self, file: FileRecord, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        assert len(chunks) == len(embeddings), "chunk/embedding count mismatch"
        # Remove any prior chunks for this file from Lance.
        self._tbl.delete(self._rel_filter(file.rel_path))
        # Insert new rows.
        rows = [self._row_for(c, e) for c, e in zip(chunks, embeddings)]
        if rows:
            self._tbl.add(rows)
        # Upsert the sidecar file row.
        self._side.execute(
            "INSERT INTO files(rel_path, lang, file_hash, mtime, n_chunks, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(rel_path) DO UPDATE SET "
            "lang=excluded.lang, file_hash=excluded.file_hash, mtime=excluded.mtime, "
            "n_chunks=excluded.n_chunks, indexed_at=excluded.indexed_at",
            (file.rel_path, file.lang, file.file_hash, file.mtime, len(chunks), file.indexed_at),
        )
        self._side.commit()

    def delete_file(self, rel_path: str) -> None:
        self._tbl.delete(self._rel_filter(rel_path))
        self._side.execute("DELETE FROM files WHERE rel_path = ?", (rel_path,))
        self._side.commit()

    def touch_file(self, rel_path: str, mtime: float, indexed_at: float) -> None:
        self._side.execute(
            "UPDATE files SET mtime = ?, indexed_at = ? WHERE rel_path = ?",
            (mtime, indexed_at, rel_path),
        )
        self._side.commit()

    # -- reads -----------------------------------------------------------

    def search(self, query_vec: list[float], top_k: int) -> list[ChunkHit]:
        rows = self._tbl.search(list(map(float, query_vec))).metric("cosine").limit(top_k).to_list()
        return [ChunkHit(chunk_id=r["chunk_id"], distance=float(r["_distance"])) for r in rows]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        for r in self._scan():
            if r["chunk_id"] == chunk_id:
                return self._row_to_chunk(r)
        return None

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        wanted = set(chunk_ids)
        by_id = {r["chunk_id"]: r for r in self._scan() if r["chunk_id"] in wanted}
        return [self._row_to_chunk(by_id[c]) for c in chunk_ids if c in by_id]

    def get_file_records(self) -> dict[str, FileRecord]:
        rows = self._side.execute(
            "SELECT rel_path, lang, file_hash, mtime, n_chunks, indexed_at FROM files"
        ).fetchall()
        return {
            r["rel_path"]: FileRecord(
                rel_path=r["rel_path"], lang=r["lang"], file_hash=r["file_hash"],
                mtime=float(r["mtime"]), n_chunks=int(r["n_chunks"]),
                indexed_at=float(r["indexed_at"]),
            )
            for r in rows
        }

    def count_chunks(self) -> int:
        return int(self._tbl.count_rows())

    def count_files(self) -> int:
        return int(self._side.execute("SELECT COUNT(*) FROM files").fetchone()[0])

    def count_edges(self) -> int:
        return 0  # graph is SQLite-only

    def count_imports(self) -> int:
        return 0  # graph is SQLite-only

    # -- helpers ---------------------------------------------------------

    def _scan(self):
        """Full scan of the Lance table as a list of dicts."""
        return self._tbl.to_arrow().to_pylist()

    @staticmethod
    def _rel_filter(rel_path: str) -> str:
        """A SQL filter for ``rel_path`` safe for Lance's delete()."""
        escaped = rel_path.replace("'", "''")
        return f"rel_path = '{escaped}'"

    @staticmethod
    def _row_for(chunk: Chunk, emb: list[float]) -> dict:
        row = {c: getattr(chunk, c) for c in _LANCE_COLUMNS}
        row["vector"] = list(map(float, emb))
        return row

    @staticmethod
    def _row_to_chunk(r: dict) -> Chunk:
        return Chunk(
            chunk_id=r["chunk_id"], rel_path=r["rel_path"], lang=r["lang"], kind=r["kind"],
            text=r["text"], start_line=int(r["start_line"]), end_line=int(r["end_line"]),
            start_byte=int(r["start_byte"]), end_byte=int(r["end_byte"]),
            n_tokens=int(r["n_tokens"]), file_hash=r["file_hash"], mtime=float(r["mtime"]),
            symbol=r["symbol"], parent_symbol=r["parent_symbol"],
        )