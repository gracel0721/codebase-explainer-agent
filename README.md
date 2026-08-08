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

> **Three stages, all implemented.** Stage 1 is RAG question-answering. Stage 2
> adds a code-relationship **graph** that expands retrieval with caller/callee
> context. Stage 3 adds an **agentic-edit** layer that explores the codebase
> with tools and proposes patches. Each stage slots in behind small protocols
> (`VectorStore`, `Reranker`) without rewrites (see [Architecture](#architecture)).

---

## How it works

```
Repository → File discovery → AST-aware chunking → Embeddings → vector store
   → (hybrid: vector + FTS5 fusion) → optional LLM rerank → context assembly
   (token-budgeted, optionally graph-expanded) → local LLM → answer
```

- **AST-aware chunking** via [tree-sitter](https://tree-sitter.github.io/): files
  are split into semantic units (functions, classes, methods, types) — not
  arbitrary line windows. Each chunk carries metadata (`kind`, `symbol`,
  `parent_symbol`, line/byte range) that the graph layer builds on.
- **Embeddings** via Ollama (`nomic-embed-text` by default).
- **Vector store** via [sqlite-vec](https://alexgarcia.xyz/sqlite-vec/) — a single
  `.db` file inside `<repo>/.code-explain/` — or [LanceDB](https://lancedb.github.io/)
  behind an optional extra (`vector_backend`).
- **Hybrid search** (on by default): reciprocal-rank-fuses vector results with
  FTS5 keyword hits so symbol/identifier-heavy queries match precisely.
- **Optional LLM reranker** (`reranker_model`): asks the LLM to reorder the top
  chunks; degrades to the original order on any error.
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
pip install -e .                 # core (sqlite-vec backend)
pip install -e ".[lancedb]"      # optional LanceDB vector backend
pip install -e ".[dev]"         # tests (pytest, pytest-mock)
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

# Build the code-relationship graph (caller/callee context for ask/chat):
code-explain graph ./my-project
code-explain --with-graph ask ./my-project "What calls process_payment?"

# Explore + propose edits with an LLM agent (writes nothing by default):
code-explain agent ./my-project "Add a docstring to parse_token if missing"
code-explain agent ./my-project --apply "Fix the off-by-one in range()"   # confirm to write

# Show index stats and staleness (machine-readable with --json):
code-explain status ./my-project
code-explain status ./my-project --json

# Drop the index + graph without re-indexing:
code-explain reset ./my-project

# Print the resolved config (machine-readable with --json):
code-explain config
code-explain config --json
```

In the chat REPL: type questions, `:reset` to clear history, `:exit` (or Ctrl-D)
to quit. Ctrl-C during a streamed answer interrupts cleanly and keeps the partial
text — no traceback.

### Options

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--model` | `CODE_EXPLAIN_LLM_MODEL` | `qwen2.5-coder:7b` | Ollama LLM model |
| `--embed-model` | `CODE_EXPLAIN_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `--num-ctx` | `CODE_EXPLAIN_LLM_NCTX` | `8192` | LLM context window |
| `--db` | — | `<repo>/.code-explain/index.db` | Override index db path |
| `--with-graph` | `CODE_EXPLAIN_WITH_GRAPH` | off | Expand context with caller/callee chunks |
| `--verbose` | — | off | Debug logging |
| `--no-color` | — | off | Plain output |

Agent command flags: `--apply` (write patches via `git apply`, with confirmation),
`--allow-tests` (permit `run_tests`, with per-run confirmation), `--max-iterations`
(cap on tool turns).

Additional env vars: `CODE_EXPLAIN_RERANKER_MODEL` (LLM reranker; off by default),
`CODE_EXPLAIN_HYBRID_SEARCH` (on by default), `CODE_EXPLAIN_VECTOR_BACKEND`
(`sqlite` or `lancedb`), `CODE_EXPLAIN_GRAPH_DEPTH` (1–3), `CODE_EXPLAIN_OLLAMA_HOST`.

Config is resolved **defaults → env vars → `<repo>/.code-explain/config.json` →
CLI flags** (highest priority) and validated on resolve (bad values exit with a
clear error, code 2).

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

### Ignoring files

Discovery respects `.gitignore` (via `git ls-files --exclude-standard`). Drop a
`.codeexplainignore` at the repo root for code-explain–specific exclusions — same
gitignore syntax, applied on top of the git listing and the walk fallback. Use it
to keep files tracked in git out of the index (generated code, vendored stubs,
secrets).

---

## Agent (Stage 3)

`code-explain agent [PATH] "task"` runs a tool-calling loop over the LLM. The
agent explores with `search_code` / `list_symbols` / `read_file` / `find_callers`,
then proposes a unified diff via `propose_patch`.

- **Propose-only by default** — the diff is printed and nothing is written.
- **`--apply`** validates the patch with `git apply --check`, prompts for
  confirmation, then writes via `git apply`. Bad patches are rejected before any
  prompt, so you're never asked to confirm a patch that can't apply.
- **`--allow-tests`** gates the `run_tests` tool behind a per-run confirmation.

The agent needs a tool-calling model (`qwen2.5-coder:7b` works). Some models
declare tool support but emit tool calls as JSON in the reply text instead of the
structured channel — the agent recovers those so the loop still dispatches.

```bash
code-explain agent ./my-project "Find the function that builds the graph and add a docstring if missing."
code-explain agent ./my-project --apply "Fix the off-by-one in range()"
```

---

## Architecture

```
src/code_explain/
  cli.py          # typer app; fallback group enables `code-explain <path>`
  config.py       # single resolution + validation point for all settings
  discovery.py    # git ls-files (+ untracked) or walk+gitignore/.codeexplainignore
  chunker.py      # Chunk dataclass + line-based fallback chunker
  parser.py       # tree-sitter AST chunking (per-language config)
  embedder.py     # Ollama embeddings (batched, num_ctx-safe)
  store.py        # VectorStore Protocol + SQLiteVecStore (FTS5, pysqlite3 fallback)
  lancedb_store.py# optional LanceDB backend (vectors + SQLite sidecar)
  indexer.py      # incremental discovery→chunk→embed→store; open_store() factory
  graph.py        # code-relationship graph (caller/callee, SQLite-only)
  retriever.py    # vector+FTS fusion → rerank → per-file cap → graph → budget
  reranker.py     # Reranker Protocol + OllamaReranker (LLM reorders top chunks)
  llm.py          # streaming + non-streaming (tool-turn) Ollama chat client
  ask.py          # answer pipeline + interactive chat loop (Ctrl-C safe)
  agent.py        # Stage 3: tool-dispatch loop + tools (read_file, search_code, …)
  errors.py       # OllamaUnavailableError + friendly translation of ollama errors
  prompts.py      # system prompts (RAG, graph, agent — the only place text lives)
```

### Seams that made the stages slot in

- **`VectorStore` Protocol** (`store.py`) — the LanceDB backend implements the
  same surface; `open_store()` picks one from `vector_backend`. Graph + hybrid
  FTS query `store._conn`, so they're SQLite-only and degrade to vector-only
  under LanceDB (the retriever guards with `hasattr`).
- **`Reranker` Protocol** (`reranker.py`) — `OllamaReranker` reorders top chunks;
  the retriever builds it lazily and it no-ops on any error.
- **Rich chunk metadata** — the **code graph** builds edges from chunks (matching
  `symbol` names referenced in other chunks' `text`); no re-parse needed.
- **Retriever returns `list[Chunk]`**, not strings — the **agent** inspects
  metadata to choose tool calls (open file at line, list neighbors).
- **`llm.py` is a thin client** with both streaming (`chat_stream`) and
  non-streaming tool-turn (`chat_turn`) methods; `ask` and the agent share it.
- **`config.py`** is the single place all knobs live, validated on resolve.

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