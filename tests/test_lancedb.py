"""Tests for the LanceDB vector backend (skipped without the `lancedb` extra)."""

import os
import time

import pytest

lancedb = pytest.importorskip("lancedb")  # noqa: F841 - skip the module if absent

from code_explain import graph
from code_explain.indexer import index_repo, is_stale, open_store


def _lance_cfg(make_cfg, tmp_repo, **extra):
    ov = {"embed_dim": 8, "vector_backend": "lancedb"}
    ov.update(extra)
    return make_cfg(tmp_repo, **ov)


def test_lancedb_indexes_and_retrieves(tmp_repo, make_cfg, mock_embedder):
    cfg = _lance_cfg(make_cfg, tmp_repo)
    report, store = index_repo(cfg, console=None)
    assert report.n_indexed >= 1
    assert store.count_chunks() >= 1

    from code_explain.embedder import Embedder
    from code_explain.retriever import Retriever

    chunks = Retriever(cfg, store, Embedder(cfg)).retrieve("greet")
    assert chunks  # retrieval returns something
    store.close()


def test_lancedb_graph_noops(tmp_repo, make_cfg, mock_embedder):
    cfg = _lance_cfg(make_cfg, tmp_repo)
    _, store = index_repo(cfg, console=None)
    assert graph.is_graph_present(store) is False
    report = graph.build_graph(store, cfg, console=None)
    assert report.n_edges == 0  # SQLite-only graph degrades to a no-op
    store.close()


def test_lancedb_has_no_fts_so_hybrid_degrades(tmp_repo, make_cfg, mock_embedder):
    cfg = _lance_cfg(make_cfg, tmp_repo, hybrid_search=True)
    _, store = index_repo(cfg, console=None)
    assert not hasattr(store, "search_fts")  # FTS is SQLite-only
    store.close()


def test_lancedb_incremental_touch_keeps_chunks(tmp_repo, make_cfg, mock_embedder):
    cfg = _lance_cfg(make_cfg, tmp_repo)
    _, store = index_repo(cfg, console=None)
    n = store.count_chunks()
    store.close()

    app = tmp_repo / "src" / "app.py"
    os.utime(app, (time.time() + 5, time.time() + 5))
    time.sleep(0.01)
    report, store = index_repo(cfg, console=None)
    assert store.count_chunks() == n  # content unchanged -> chunks preserved
    assert report.n_indexed == 0
    store.close()


def test_lancedb_is_stale_after_change(tmp_repo, make_cfg, mock_embedder):
    cfg = _lance_cfg(make_cfg, tmp_repo)
    _, store = index_repo(cfg, console=None)
    assert is_stale(cfg, store) is False
    store.close()

    app = tmp_repo / "src" / "app.py"
    app.write_text(app.read_text() + "\n# new\n")
    os.utime(app, (time.time() + 10, time.time() + 10))
    store = open_store(cfg)
    assert is_stale(cfg, store) is True
    store.close()