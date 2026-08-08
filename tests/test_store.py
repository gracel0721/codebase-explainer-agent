"""Tests for SQLiteVecStore: roundtrip, re-upsert, delete, rebuild, touch."""

import time

from code_explain.chunker import Chunk, new_chunk_id
from code_explain.store import FileRecord, SQLiteVecStore


def _vec(i=0):
    v = [0.0] * 8
    v[i] = 1.0
    return v


def _chunk(rel, symbol, kind="function", start=1, end=2):
    return Chunk(
        chunk_id=new_chunk_id(), rel_path=rel, lang="python", kind=kind,
        text=f"def {symbol}(): pass", symbol=symbol, parent_symbol=None,
        start_line=start, end_line=end, start_byte=0, end_byte=10,
        n_tokens=3, file_hash="h", mtime=0.0,
    )


def _file(rel, n):
    return FileRecord(rel_path=rel, lang="python", file_hash="h", mtime=0.0, n_chunks=n, indexed_at=0.0)


def test_roundtrip_upsert_and_get(store):
    c1, c2 = _chunk("a.py", "f"), _chunk("a.py", "g")
    store.upsert_file(_file("a.py", 2), [c1, c2], [_vec(0), _vec(1)])
    assert store.count_chunks() == 2
    assert store.count_files() == 1
    got = store.get_chunks([c1.chunk_id, c2.chunk_id])
    assert [c.symbol for c in got] == ["f", "g"]


def test_re_upsert_replaces_chunks(store):
    c1, c2 = _chunk("a.py", "f"), _chunk("a.py", "g")
    store.upsert_file(_file("a.py", 2), [c1, c2], [_vec(0), _vec(1)])
    assert store.count_chunks() == 2
    # Re-upsert the same file with a single chunk -> old two are gone.
    c3 = _chunk("a.py", "h")
    store.upsert_file(_file("a.py", 1), [c3], [_vec(2)])
    assert store.count_chunks() == 1
    assert store.get_chunk(c1.chunk_id) is None
    assert store.get_chunk(c3.chunk_id) is not None


def test_delete_file_removes_chunks(store):
    c = _chunk("a.py", "f")
    store.upsert_file(_file("a.py", 1), [c], [_vec(0)])
    store.delete_file("a.py")
    assert store.count_chunks() == 0
    assert store.count_files() == 0


def test_search_returns_best_first(store):
    c1, c2 = _chunk("a.py", "f"), _chunk("a.py", "g")
    store.upsert_file(_file("a.py", 2), [c1, c2], [_vec(0), _vec(1)])
    # Query equal to c1's vector -> c1 is nearest (distance 0).
    hits = store.search(_vec(0), top_k=2)
    assert hits[0].chunk_id == c1.chunk_id
    assert hits[0].distance <= hits[1].distance


def test_touch_file_keeps_chunks_and_updates_mtime(store):
    c = _chunk("a.py", "f")
    store.upsert_file(_file("a.py", 1), [c], [_vec(0)])
    before = store.count_chunks()
    store.touch_file("a.py", 999.0, 1234.0)
    assert store.count_chunks() == before
    rec = store.get_file_records()["a.py"]
    assert rec.mtime == 999.0


def test_embed_dim_mismatch_forces_rebuild(tmp_path):
    from code_explain.indexer import open_store

    from code_explain.config import Config

    repo = tmp_path / "repo"
    repo.mkdir()
    cfg8 = Config.resolve(repo, overrides={"embed_dim": 8})
    s8 = open_store(cfg8)
    c = _chunk("a.py", "f")
    s8.upsert_file(_file("a.py", 1), [c], [_vec(0)])
    assert s8.count_chunks() == 1
    s8.close()

    # Reopen with a different dim and no force -> drops and recreates empty.
    cfg16 = Config.resolve(repo, overrides={"embed_dim": 16})
    s16 = open_store(cfg16)
    assert s16.count_chunks() == 0
    assert s16.stored_embed_dim() == 16
    s16.close()


def test_get_meta_set_meta_roundtrip(store):
    assert store.get_meta("k") is None
    store.set_meta("k", "v")
    assert store.get_meta("k") == "v"


def test_backfill_fts_repairs_old_index(store):
    """An index built before FTS existed has chunks but no FTS rows. Opening it
    (or calling backfill_fts) should populate FTS so hybrid search works."""
    c1, c2 = _chunk("a.py", "greet"), _chunk("b.py", "farewell")
    store.upsert_file(_file("a.py", 1), [c1], [_vec(0)])
    store.upsert_file(_file("b.py", 1), [c2], [_vec(1)])
    # Simulate the pre-FTS state: wipe FTS, leave chunks.
    store._conn.execute("DELETE FROM chunks_fts")
    store._conn.commit()
    assert store._conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 0
    assert store.count_chunks() == 2

    store.backfill_fts()
    n_fts = store._conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert n_fts == 2  # every chunk now has an FTS row
    # FTS search now returns the chunk.
    hits = store.search_fts("greet", 5)
    assert hits  # was empty before backfill


def test_backfill_fts_idempotent_when_complete(store):
    c = _chunk("a.py", "greet")
    store.upsert_file(_file("a.py", 1), [c], [_vec(0)])
    before = store._conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    store.backfill_fts()  # FTS already complete via upsert
    after = store._conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert before == after == 1