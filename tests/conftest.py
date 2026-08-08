"""Shared pytest fixtures for code-explain.

All tests run against mocked Ollama (no live server). The embedder is replaced
with a deterministic, stable-hash embedder so retrieval is reproducible without
a model. A real on-disk ``SQLiteVecStore`` is used for store/indexer/graph
roundtrip tests; a scripted ``FakeStore`` is used where a test needs to control
search results (retriever/agent).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from code_explain.chunker import Chunk, new_chunk_id
from code_explain.store import ChunkHit, FileRecord, SQLiteVecStore


@pytest.fixture(autouse=True)
def no_ollama(mocker):
    """Stop any test from reaching a live Ollama server."""
    mocker.patch("ollama.Client")


# ---------------------------------------------------------------------------
# Repo + config
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_repo(tmp_path) -> Path:
    """A tiny git repo with one python module + a .gitignore."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    (repo / ".gitignore").write_text("*.log\nignore_me/\n")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text(
        '"""module doc"""\n'
        "import os\n\n\n"
        "def greet(name):\n"
        '    return f"hi {name}"\n\n\n'
        "class Greeter:\n"
    "    def hello(self):\n"
    '        return greet("world")\n'
    )
    return repo


@pytest.fixture
def make_cfg(tmp_path):
    """Factory: Config.resolve(repo, **overrides). embed_dim defaults to 8 for speed."""
    from code_explain.config import Config

    def _make(repo=None, **overrides):
        repo = Path(repo) if repo else tmp_path / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        ov = {"embed_dim": 8}
        ov.update(overrides)
        return Config.resolve(repo, overrides=ov)

    return _make


# ---------------------------------------------------------------------------
# Real store (file-backed in tmp)
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = SQLiteVecStore(tmp_path / "index.db", 8)
    s.open()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Deterministic embedder
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder(mocker):
    """Patch Embedder._embed_batch_raw with a stable, dim-aware fake.

    Returns the Embedder class; tests construct ``Embedder(cfg)`` as usual.
    Vectors are deterministic per text (sha256-derived) so search results are
    reproducible without a model.
    """
    from code_explain.embedder import Embedder

    def fake_raw(self, texts):
        dim = self.cfg.embed_dim
        out = []
        for t in texts:
            h = int(hashlib.sha256(t.encode()).hexdigest(), 16)
            vec = [0.0] * dim
            vec[0] = 1.0
            for i in range(1, min(8, dim)):
                vec[i] = float((h >> i) & 1)
            out.append(vec)
        return out

    mocker.patch("code_explain.embedder.Embedder._embed_batch_raw", fake_raw)
    return Embedder


# ---------------------------------------------------------------------------
# Scripted in-memory store (retriever/agent tests)
# ---------------------------------------------------------------------------


class FakeStore:
    """Minimal VectorStore with scripted search results and no SQL connection.

    ``_conn`` is None so graph expansion and file-header injection no-op —
    exactly what retriever tests need to isolate cap/budget behavior.
    """

    def __init__(self, dim: int = 8):
        self._dim = dim
        self._chunks: dict[str, Chunk] = {}
        self._files: dict[str, FileRecord] = {}
        self._meta: dict[str, str] = {}
        self._conn = None
        self.search_result: list[ChunkHit] = []
        self.last_query_vec = None

    def open(self) -> None:
        pass

    def upsert_file(self, file, chunks, embeddings):
        for c in chunks:
            self._chunks[c.chunk_id] = c
        self._files[file.rel_path] = file

    def delete_file(self, rel_path):
        self._files.pop(rel_path, None)
        self._chunks = {k: v for k, v in self._chunks.items() if v.rel_path != rel_path}

    def search(self, query_vec, top_k):
        self.last_query_vec = query_vec
        return self.search_result[:top_k]

    def get_chunk(self, chunk_id):
        return self._chunks.get(chunk_id)

    def get_chunks(self, chunk_ids):
        return [self._chunks[c] for c in chunk_ids if c in self._chunks]

    def get_file_records(self):
        return dict(self._files)

    def get_meta(self, key):
        return self._meta.get(key)

    def set_meta(self, key, value):
        self._meta[key] = value

    def count_chunks(self):
        return len(self._chunks)

    def count_files(self):
        return len(self._files)

    def close(self):
        pass


@pytest.fixture
def fake_store():
    return FakeStore(dim=8)


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------


def make_chunk(rel_path, symbol, kind="function", text=None, start_line=1, end_line=1, parent=None):
    """Build a Chunk with sensible defaults for tests."""
    text = text if text is not None else f"def {symbol}(): pass"
    return Chunk(
        chunk_id=new_chunk_id(),
        rel_path=rel_path,
        lang="python",
        kind=kind,
        text=text,
        symbol=symbol,
        parent_symbol=parent,
        start_line=start_line,
        end_line=end_line,
        start_byte=0,
        end_byte=len(text),
        n_tokens=max(1, len(text) // 4),
        file_hash="deadbeef",
        mtime=0.0,
    )


def seed_fake_store(store, chunks):
    """Put chunks in a FakeStore and script search to return them in order."""
    for c in chunks:
        store._chunks[c.chunk_id] = c
    store.search_result = [ChunkHit(chunk_id=c.chunk_id, distance=float(i)) for i, c in enumerate(chunks)]


@pytest.fixture
def make_chunk_fn():
    return make_chunk


# ---------------------------------------------------------------------------
# Fake LLM response objects (agent tests)
# ---------------------------------------------------------------------------


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeResponse:
    def __init__(self, message):
        self.message = message