# code-explain

A **local, RAG-powered CLI** that indexes a codebase and answers questions about
it — "How does authentication work?", "Where are payments processed?", "What
happens when `POST /orders` is called?" — with accurate `path:line` citations.

Everything runs locally via [Ollama](https://ollama.com): your code never leaves
your machine, no API keys required.

```
code-explain ./my-project
>>> How does authentication work?
```

> **Stage 1 of a larger vision.** This release implements RAG question-answering.
> The architecture is built so a code-relationship **graph** layer and an
> **agentic edit** layer can be added later without rewrites (see
> [Architecture](#architecture)).

---

## How it works

```
Repository → File discovery → AST-aware chunking → Embeddings → sqlite-vec
   → vector search → context assembly (token-budgeted) → local LLM → answer
```

- **AST-aware chunking** via [tree-sitter](https://tree-sitter.github.io/): files
  are split into semantic units (functions, classes, methods, types) — not
  arbitrary line windows. Each chunk carries metadata (`kind`, `symbol`,
  `parent_symbol`, line/byte range) that a future graph layer can build on.
- **Embeddings** via Ollama (`nomic-embed-text` by default).
- **Vector store** via [sqlite-vec](https://alexgarcia.xyz/sqlite-vec/) — a single
  `.db` file inside `<repo>/.code-explain/`.
- **Retrieval** is token-budgeted and capped per-file so one large file can't
  starve context diversity.
- **LLM** via Ollama (`qwen2.5-coder:7b` by default, configurable).

---

## Install

### Prerequisites

- **Python 3.10+**. On macOS, prefer Homebrew Python
  (`brew install python@3.12`) — the *system* Python's SQLite lacks loadable-
  extension support, which sqlite-vec needs. If you must use the system Python,
  install the fallback: `pip install "code-explain[macos-system-python]"`.
- **[Ollama](https://ollama.com)** running locally (`ollama serve` or the app).

### Pull models

```bash
ollama pull qwen2.5-coder:7b     # or any model you prefer
ollama pull nomic-embed-text     # embeddings
```

### Install code-explain

```bash
git clone <this repo> && cd codebase-explainer-agent
python -m venv .venv && . .venv/bin/activate
pip install -e .
```

---

## Usage

```bash
# Analyze a repo, then drop into an interactive chat (default):
code-explain ./my-project

# One-shot question with citations:
code-explain ask ./my-project "Where are payments processed?"

# Build/rebuild the index only:
code-explain index ./my-project --force

# Show index stats and staleness:
code-explain status ./my-project

# Print the resolved config:
code-explain config
```

In the chat REPL: type questions, `:reset` to clear history, `:exit` (or Ctrl-D)
to quit.

### Options

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--model` | `CODE_EXPLAIN_LLM_MODEL` | `qwen2.5-coder:7b` | Ollama LLM model |
| `--embed-model` | `CODE_EXPLAIN_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `--num-ctx` | `CODE_EXPLAIN_LLM_NCTX` | `8192` | LLM context window |
| `--db` | — | `<repo>/.code-explain/index.db` | Override index db path |
| `--verbose` | — | off | Debug logging |
| `--no-color` | — | off | Plain output |

Config is resolved **defaults → env vars → `<repo>/.code-explain/config.json` →
CLI flags** (highest priority).

---

## Indexing & staleness

The index lives in `<repo>/.code-explain/` (relocatable with `--db`). A
`.code-explain/.gitignore` is written so the index is never accidentally
committed — your top-level `.gitignore` is never touched.

Re-indexing is **incremental and two-tier**:

- **Fast path (mtime):** files whose mtime is unchanged are skipped entirely.
- **Content path (sha256):** a file whose mtime changed is re-hashed; only a real
  content change re-chunks/re-embeds that file (a plain `touch` is a no-op).
- New files are added; deleted files are removed. `--force` rebuilds from
  scratch.

If the embedding model or dimension changes, the index is auto-rebuilt so you
never serve vectors from a different model.

---

## Architecture

```
src/code_explain/
  cli.py        # typer app; fallback group enables `code-explain <path>`
  config.py     # single resolution point for all settings
  discovery.py  # git ls-files (+ untracked) or walk+gitignore fallback
  chunker.py    # Chunk dataclass + line-based fallback chunker
  parser.py     # tree-sitter AST chunking (per-language config)
  embedder.py   # Ollama embeddings (batched, num_ctx-safe)
  store.py      # VectorStore Protocol + SQLiteVecStore (pysqlite3 fallback)
  indexer.py    # incremental discovery→chunk→embed→store orchestration
  retriever.py  # search → per-file cap → file headers → token-budget packing
  llm.py        # thin streaming Ollama chat client
  ask.py        # answer pipeline + interactive chat loop
  prompts.py    # system prompt (the only place prompt text lives)
```

### Seams for the future stages

- **`VectorStore` Protocol** (`store.py`) — a LanceDB or graph-augmented store
  can replace `SQLiteVecStore` without touching the indexer/retriever.
- **Rich chunk metadata** — the future **code graph** builds edges from chunks
  (matching `symbol` names referenced in other chunks' `text`); no re-parse
  needed. The `chunks` table is the graph's node source.
- **Retriever returns `list[Chunk]`**, not strings — the **agent** layer can
  inspect metadata to choose tool calls (open file at line, list neighbors).
- **`llm.py` is a thin streaming client** — the agent wraps it with a
  `tools=` dispatch loop; both single-shot `ask` and the future agent share it.
- **`config.py`** is the single place to add `graph_enabled`/`agent_enabled`/
  `reranker_model` later.

---

## Supported languages (AST chunking)

Python, JavaScript/JSX, TypeScript/TSX, Go, Rust, Java, Kotlin, Scala, C, C++,
C#, Ruby, PHP, Swift. Other text files (Markdown, YAML, TOML, JSON, etc.) use the
line-based fallback chunker.

---

## Verification

```bash
# On a real, well-structured repo:
git clone https://github.com/pallets/click /tmp/click
code-explain index /tmp/click --force
code-explain status /tmp/click
code-explain ask /tmp/click "How does Click parse a command line into a Command object?"
```

Expect a concise answer with multiple `path:line` citations; verify the cited
line ranges actually contain the named symbols. An out-of-scope question (e.g.
"how does the Rust async runtime work?" against a Python repo) should be
**declined**, not hallucinated.

---

## License

See [LICENSE](LICENSE).