"""Tests for the retriever: per-file cap, budget, file headers, graph no-op."""

from code_explain.chunker import new_chunk_id
from code_explain.retriever import Retriever
from code_explain.store import ChunkHit, FileRecord, SQLiteVecStore
from tests.conftest import make_chunk, seed_fake_store


def _vec(i=0):
    v = [0.0] * 8
    v[i] = 1.0
    return v


def _retriever(cfg, store, mock_embedder):
    from code_explain.embedder import Embedder

    return Retriever(cfg, store, Embedder(cfg))


def test_per_file_cap_keeps_one_per_file_first(make_cfg, fake_store, mock_embedder):
    cfg = make_cfg(embed_dim=8, per_file_cap=1)
    a1 = make_chunk("a.py", "a1", start_line=1, end_line=1)
    a2 = make_chunk("a.py", "a2", start_line=2, end_line=2)
    b1 = make_chunk("b.py", "b1", start_line=1, end_line=1)
    b2 = make_chunk("b.py", "b2", start_line=2, end_line=2)
    seed_fake_store(fake_store, [a1, a2, b1, b2])

    r = _retriever(cfg, fake_store, mock_embedder)
    chunks = r.retrieve("q")
    # First two results are one-per-file (cap=1), then the refill.
    first_files = {c.rel_path for c in chunks[:2]}
    assert first_files == {"a.py", "b.py"}
    assert len(chunks) == 4


def test_budget_packs_few_chunks(make_cfg, fake_store, mock_embedder):
    # Small context + large chunks -> only one fits.
    cfg = make_cfg(embed_dim=8, llm_n_ctx=600, per_file_cap=0)
    big = make_chunk("a.py", "big", text="x" * 4000, start_line=1, end_line=1)
    big.n_tokens = 600
    big2 = make_chunk("a.py", "big2", text="y" * 4000, start_line=2, end_line=2)
    big2.n_tokens = 600
    seed_fake_store(fake_store, [big, big2])

    r = _retriever(cfg, fake_store, mock_embedder)
    chunks = r.retrieve("q")
    assert len(chunks) == 1


def test_graph_expansion_noops_without_conn(make_cfg, fake_store, mock_embedder):
    cfg = make_cfg(embed_dim=8, with_graph=True, per_file_cap=0)
    c1 = make_chunk("a.py", "f", start_line=1, end_line=1)
    seed_fake_store(fake_store, [c1])
    r = _retriever(cfg, fake_store, mock_embedder)
    chunks = r.retrieve("q")
    # FakeStore has _conn=None -> graph expansion is a no-op; seed returned.
    assert [c.chunk_id for c in chunks] == [c1.chunk_id]


def test_file_header_pulled_in_when_missing(tmp_path, make_cfg, mock_embedder, mocker):
    # Real store: query returns only the function chunk; the retriever should
    # inject the file's module header chunk that wasn't retrieved.
    cfg = make_cfg(tmp_path / "repo", embed_dim=8, top_k=1, per_file_cap=0)
    store = SQLiteVecStore(tmp_path / "idx.db", 8)
    store.open()

    mod = make_chunk("app.py", None, kind="module", text='"""doc"""\n', start_line=1, end_line=1)
    func = make_chunk("app.py", "greet", kind="function", text="def greet(): pass", start_line=3, end_line=4)
    store.upsert_file(
        FileRecord("app.py", "python", "h", 0.0, 2, 0.0), [mod, func], [_vec(0), _vec(1)]
    )

    from code_explain.embedder import Embedder

    emb = Embedder(cfg)
    # Force the query vector to match the function chunk so search returns it.
    mocker.patch.object(emb, "embed_query", return_value=_vec(1))
    r = Retriever(cfg, store, emb)
    chunks = r.retrieve("greet")
    kinds = {c.kind for c in chunks}
    assert "module" in kinds  # header injected
    assert "function" in kinds
    store.close()