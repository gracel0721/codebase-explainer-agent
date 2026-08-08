"""Tests for the code-relationship graph: build, callers, absence no-op."""

from code_explain import graph
from code_explain.indexer import index_repo


def _index_and_build(cfg):
    _, store = index_repo(cfg, console=None)
    report = graph.build_graph(store, cfg)
    return store, report


def test_graph_absent_until_built(tmp_repo, make_cfg, mock_embedder):
    cfg = make_cfg(tmp_repo)
    _, store = index_repo(cfg, console=None)
    assert graph.is_graph_present(store) is False
    # callers_of on a graph-less store returns [] (no _conn-less crash).
    assert graph.callers_of(store, "greet") == []
    store.close()


def test_build_graph_produces_edges(tmp_repo, make_cfg, mock_embedder):
    cfg = make_cfg(tmp_repo)
    store, report = _index_and_build(cfg)
    assert graph.is_graph_present(store) is True
    assert report.n_edges >= 1
    assert store.count_edges() == report.n_edges
    store.close()


def test_callers_of_finds_caller(tmp_repo, make_cfg, mock_embedder):
    cfg = make_cfg(tmp_repo)
    store, _ = _index_and_build(cfg)
    # `greet` is called inside Greeter.hello.
    callers = graph.callers_of(store, "greet")
    syms = {c.symbol for c in callers}
    assert "hello" in syms
    store.close()


def test_expand_returns_neighbors(tmp_repo, make_cfg, mock_embedder):
    cfg = make_cfg(tmp_repo)
    store, _ = _index_and_build(cfg)
    # Find the greet function chunk id, then expand its neighbors.
    conn = store._conn
    row = conn.execute("SELECT chunk_id FROM chunks WHERE symbol='greet'").fetchone()
    assert row is not None
    neighbor_map = graph.expand(store, [row["chunk_id"]], depth=1, cap_per_seed=5)
    assert row["chunk_id"] in neighbor_map
    assert len(neighbor_map[row["chunk_id"]]) >= 1
    store.close()


def test_is_graph_stale_after_reindex(tmp_repo, make_cfg, mock_embedder):
    cfg = make_cfg(tmp_repo)
    store, _ = _index_and_build(cfg)
    assert graph.is_graph_stale(store) is False
    store.close()
    # Re-index (no changes) -> still not stale (indexed_at not bumped beyond graph).
    _, store2 = index_repo(cfg, console=None)
    # No files changed, so MAX(indexed_at) unchanged -> not stale.
    assert graph.is_graph_stale(store2) is False
    store2.close()