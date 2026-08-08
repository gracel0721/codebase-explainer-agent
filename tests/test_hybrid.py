"""Tests for hybrid FTS5 search + reciprocal rank fusion."""

from code_explain.retriever import Retriever
from code_explain.store import ChunkHit, FileRecord, SQLiteVecStore
from tests.conftest import make_chunk


def _vec(i=0):
    v = [0.0] * 8
    v[i] = 1.0
    return v


def _store_with_chunk(tmp_path, chunk, vec=None):
    store = SQLiteVecStore(tmp_path / "idx.db", 8)
    store.open()
    store.upsert_file(
        FileRecord(chunk.rel_path, "python", "h", 0.0, 1, 0.0), [chunk], [vec or _vec(0)]
    )
    return store


def test_search_fts_finds_keyword(tmp_path, make_cfg):
    cfg = make_cfg(tmp_path / "repo", embed_dim=8)
    c = make_chunk("app.py", "greet", text="def greet(name):\n    return hi\n")
    store = _store_with_chunk(tmp_path, c)
    fts = store.search_fts("greet", 5)
    assert len(fts) == 1
    assert fts[0][0] == c.chunk_id
    store.close()


def test_search_fts_malformed_query_returns_empty(tmp_path, make_cfg):
    store = SQLiteVecStore(tmp_path / "idx.db", 8)
    store.open()
    assert store.search_fts('"unbalanced', 5) == []
    store.close()


def test_search_fts_empty_query_returns_empty(tmp_path, make_cfg):
    store = SQLiteVecStore(tmp_path / "idx.db", 8)
    store.open()
    assert store.search_fts("   ", 5) == []
    store.close()


def test_fuse_rrf_ranks_overlap_first():
    # 'c' appears in both lists -> highest fused score -> first.
    hits = [ChunkHit("a", 0.0), ChunkHit("b", 0.1), ChunkHit("c", 0.2)]
    fts = [("c", -1.0), ("d", -2.0)]
    ids = Retriever._fuse(hits, fts)
    assert ids[0] == "c"
    assert set(ids) == {"a", "b", "c", "d"}


def test_hybrid_enabled_calls_fts(tmp_path, make_cfg, mock_embedder, mocker):
    cfg = make_cfg(tmp_path / "repo", embed_dim=8, hybrid_search=True, top_k=1, per_file_cap=0)
    c = make_chunk("app.py", "greet", text="def greet(): pass")
    store = _store_with_chunk(tmp_path, c, vec=_vec(0))
    spy = mocker.spy(store, "search_fts")

    from code_explain.embedder import Embedder

    r = Retriever(cfg, store, Embedder(cfg))
    mocker.patch.object(r.embedder, "embed_query", return_value=_vec(0))
    r.retrieve("greet")
    spy.assert_called_once()
    store.close()


def test_hybrid_disabled_skips_fts(tmp_path, make_cfg, mock_embedder, mocker):
    cfg = make_cfg(tmp_path / "repo", embed_dim=8, hybrid_search=False, top_k=1, per_file_cap=0)
    c = make_chunk("app.py", "greet", text="def greet(): pass")
    store = _store_with_chunk(tmp_path, c, vec=_vec(0))
    spy = mocker.patch.object(store, "search_fts", return_value=[])

    from code_explain.embedder import Embedder

    r = Retriever(cfg, store, Embedder(cfg))
    mocker.patch.object(r.embedder, "embed_query", return_value=_vec(0))
    r.retrieve("greet")
    spy.assert_not_called()
    store.close()