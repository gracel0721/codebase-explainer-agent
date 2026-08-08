"""Tests for the indexer: incremental re-index, deletion, staleness."""

import os
import time

from code_explain.indexer import index_repo, is_stale, open_store


def _index(cfg, **kw):
    return index_repo(cfg, console=None, **kw)


def test_index_repo_indexes_files(tmp_repo, make_cfg, mock_embedder):
    cfg = make_cfg(tmp_repo)
    report, store = _index(cfg)
    assert report.n_indexed >= 1
    assert store.count_files() >= 1
    assert store.count_chunks() >= 1
    store.close()


def test_incremental_touch_keeps_chunks(tmp_repo, make_cfg, mock_embedder):
    cfg = make_cfg(tmp_repo)
    _, store = _index(cfg)
    n_before = store.count_chunks()
    store.close()

    # Bump mtime without changing content (a `touch`).
    app = tmp_repo / "src" / "app.py"
    os.utime(app, (time.time() + 5, time.time() + 5))
    time.sleep(0.01)

    report, store = _index(cfg)
    # Content hash matches -> skipped (re-chunked=False), chunks preserved.
    assert store.count_chunks() == n_before
    assert report.n_indexed == 0
    store.close()


def test_deleted_file_is_removed(tmp_repo, make_cfg, mock_embedder):
    cfg = make_cfg(tmp_repo)
    _, store = _index(cfg)
    assert "src/app.py" in store.get_file_records()
    store.close()

    (tmp_repo / "src" / "app.py").unlink()
    report, store = _index(cfg)
    assert "src/app.py" not in store.get_file_records()
    store.close()


def test_is_stale_after_change(tmp_repo, make_cfg, mock_embedder):
    cfg = make_cfg(tmp_repo)
    _, store = _index(cfg)
    assert is_stale(cfg, store) is False
    store.close()

    # Real content change -> stale.
    app = tmp_repo / "src" / "app.py"
    app.write_text(app.read_text() + "\n# new line\n")
    os.utime(app, (time.time() + 10, time.time() + 10))
    store = open_store(cfg)
    assert is_stale(cfg, store) is True
    store.close()


def test_is_stale_when_empty(make_cfg, tmp_path):
    cfg = make_cfg(tmp_path / "empty")
    store = open_store(cfg)
    assert is_stale(cfg, store) is True
    store.close()