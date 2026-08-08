"""Tests for the Ollama-LLM reranker: reorder, fallback, no-op."""

import json

from code_explain.reranker import OllamaReranker
from tests.conftest import FakeMessage, FakeResponse, make_chunk


def _llm_with_content(content):
    class _LLM:
        def chat_turn(self, system, messages, *, format=None, tools=None, keep_alive=None):
            return FakeResponse(FakeMessage(content=content))

    return _LLM()


def test_rerank_reorders_by_llm_order(make_cfg):
    cfg = make_cfg(reranker_model="reranker")
    chunks = [make_chunk("a.py", "f0"), make_chunk("a.py", "f1"), make_chunk("a.py", "f2")]
    llm = _llm_with_content(json.dumps({"order": [2, 0, 1]}))
    out = OllamaReranker(cfg, llm).rerank("q", chunks)
    assert [c.symbol for c in out] == ["f2", "f0", "f1"]


def test_rerank_appends_omitted_candidates(make_cfg):
    cfg = make_cfg(reranker_model="reranker")
    chunks = [make_chunk("a.py", "f0"), make_chunk("a.py", "f1"), make_chunk("a.py", "f2")]
    # LLM only orders two of three; the third is appended in vector order.
    llm = _llm_with_content(json.dumps({"order": [1, 0]}))
    out = OllamaReranker(cfg, llm).rerank("q", chunks)
    assert out[0].symbol == "f1"
    assert out[1].symbol == "f0"
    assert "f2" in [c.symbol for c in out]
    assert len(out) == 3


def test_rerank_falls_back_on_bad_json(make_cfg):
    cfg = make_cfg(reranker_model="reranker")
    chunks = [make_chunk("a.py", "f0"), make_chunk("a.py", "f1")]
    out = OllamaReranker(cfg, _llm_with_content("not json")).rerank("q", chunks)
    assert [c.symbol for c in out] == ["f0", "f1"]  # original order preserved


def test_rerank_falls_back_on_exception(make_cfg):
    cfg = make_cfg(reranker_model="reranker")
    chunks = [make_chunk("a.py", "f0")]

    class _Boom:
        def chat_turn(self, *a, **k):
            raise RuntimeError("model on fire")

    out = OllamaReranker(cfg, _Boom()).rerank("q", chunks)
    assert out == chunks


def test_rerank_noop_without_model(make_cfg):
    cfg = make_cfg()  # reranker_model is None by default
    chunks = [make_chunk("a.py", "f0")]
    out = OllamaReranker(cfg, _llm_with_content("{}")).rerank("q", chunks)
    assert out == chunks


def test_rerank_noop_on_empty(make_cfg):
    cfg = make_cfg(reranker_model="reranker")
    out = OllamaReranker(cfg, _llm_with_content("{}")).rerank("q", [])
    assert out == []