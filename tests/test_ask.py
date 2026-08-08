"""Tests for the answer pipeline: Ctrl-C cancellation flushes partial text."""

from rich.console import Console

from code_explain.ask import answer_question_stream
from code_explain.retriever import Retriever
from tests.conftest import FakeStore, make_chunk, seed_fake_store


class _InterruptingLLM:
    """Streams a couple of deltas, then raises KeyboardInterrupt mid-stream."""

    def __init__(self, deltas):
        self._deltas = list(deltas)

    def chat_stream(self, system, messages):
        for d in self._deltas:
            yield d
        raise KeyboardInterrupt


class _ErrorLLM:
    """Streams one delta, then raises a non-interrupt exception (dropped conn)."""

    def __init__(self, deltas, exc):
        self._deltas = list(deltas)
        self._exc = exc

    def chat_stream(self, system, messages):
        for d in self._deltas:
            yield d
        raise self._exc


def _setup(cfg, fake_store, mock_embedder):
    c = make_chunk("app.py", "greet", text="def greet(): pass")
    seed_fake_store(fake_store, [c])
    retriever = Retriever(cfg, fake_store, mock_embedder(cfg))
    return retriever


def test_ctrl_c_flushes_partial_answer(make_cfg, fake_store, mock_embedder):
    cfg = make_cfg(embed_dim=8, per_file_cap=0)
    retriever = _setup(cfg, fake_store, mock_embedder)
    console = Console(quiet=True, highlight=False)
    llm = _InterruptingLLM(["partial ", "answer"])

    out = answer_question_stream(
        "q", cfg, fake_store, retriever, llm, history=[], console=console,
        render_markdown=False,
    )
    assert out == "partial answer"  # buffered text is returned, no traceback


def test_mid_stream_error_flushes_then_reraises(make_cfg, fake_store, mock_embedder):
    cfg = make_cfg(embed_dim=8, per_file_cap=0)
    retriever = _setup(cfg, fake_store, mock_embedder)
    console = Console(quiet=True, highlight=False)

    class Boom(Exception):
        pass

    llm = _ErrorLLM(["partial "], Boom("connection dropped"))
    try:
        answer_question_stream(
            "q", cfg, fake_store, retriever, llm, history=[], console=console,
            render_markdown=True,
        )
    except Boom:
        pass
    else:
        raise AssertionError("non-interrupt error should re-raise after flushing")


class _StubLLM:
    """LLM that records the user message it was called with."""

    def __init__(self):
        self.received = None

    def chat_stream(self, system, messages):
        self.received = messages
        yield "ok"


def test_empty_index_prints_actionable_hint(make_cfg, mock_embedder, capsys):
    """When the index has 0 chunks, the user gets a clear 'index is empty' hint
    instead of just an LLM ramble about 'no indexed context'."""
    cfg = make_cfg(embed_dim=8, per_file_cap=0)
    empty_store = FakeStore(dim=8)  # no chunks seeded
    retriever = Retriever(cfg, empty_store, mock_embedder(cfg))
    console = Console(highlight=False, width=200)
    llm = _StubLLM()

    answer_question_stream(
        "what does this repo do", cfg, empty_store, retriever, llm,
        history=[], console=console, render_markdown=False,
    )
    out = capsys.readouterr().out
    assert "index is empty" in out
    assert "code-explain index" in out


def test_no_match_prints_rephrase_hint(make_cfg, fake_store, mock_embedder, capsys):
    """When the store has chunks but retrieval misses, the hint suggests
    rephrasing rather than claiming the index is empty."""
    cfg = make_cfg(embed_dim=8, per_file_cap=0)
    c = make_chunk("app.py", "greet", text="def greet(): pass")
    seed_fake_store(fake_store, [c])
    # Script an empty search result so retrieval returns [] despite a chunk.
    fake_store.search_result = []
    retriever = Retriever(cfg, fake_store, mock_embedder(cfg))
    console = Console(highlight=False, width=200)
    llm = _StubLLM()

    answer_question_stream(
        "totally unrelated query", cfg, fake_store, retriever, llm,
        history=[], console=console, render_markdown=False,
    )
    out = capsys.readouterr().out
    assert "No indexed chunks matched" in out
    assert "index is empty" not in out