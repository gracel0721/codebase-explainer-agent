# Changelog

## 0.2.0

Stage 3 (agentic edits), the quality/hardening backlog, and polish.

### Stage 3 — Agentic edits
- New `code-explain agent [PATH] "task"` command: a tool-calling loop over the
  LLM with tools `read_file`, `list_symbols`, `find_callers`, `search_code`,
  `propose_patch`, `run_tests` (`agent.py`).
- Propose-only by default; `--apply` writes via `git apply` after `--check` +
  confirmation. `--allow-tests` gates `run_tests` behind a per-run prompt.
- Recovers tool calls that some models emit as JSON in the reply text instead of
  the structured `tool_calls` channel (e.g. qwen2.5-coder).
- `LLMClient.chat_turn` (non-streaming tool turns) + `supports_tools()`.

### Quality / hardening
- Test suite (pytest, mocked Ollama — no live server): chunker, parser,
  discovery, store, indexer, graph, retriever, reranker, hybrid, agent, ask,
  LanceDB.
- Optional LLM reranker (`reranker_model`; `Reranker` Protocol + `OllamaReranker`,
  no-op on error).
- Hybrid search: reciprocal-rank-fuse vector results with FTS5 keyword hits
  (`hybrid_search`, on by default).
- `OllamaUnavailableError` + friendly translation of connection/ollama errors;
  cheap reachability check before `index`/`ask`/`chat`/`agent`.
- Optional LanceDB vector backend (`vector_backend=lancedb`, `lancedb` extra);
  `open_store()` factory. Graph + hybrid FTS are SQLite-only and degrade to
  vector-only under LanceDB.
- PyPI packaging polish: 3.13/3.14 classifiers, Beta status, project URLs,
  `lancedb`/`dev` extras.

### Polish
- Rich `status` table + `--json`; `config --json`.
- `reset [PATH]` command (drops index + graph).
- `Config.validate()` on resolve; bad values exit with a clear error (code 2).
- `.codeexplainignore` (gitignore syntax, applied on top of the git listing).
- Ctrl-C during a streamed answer / agent run interrupts cleanly with partial
  text and no traceback.

## 0.1.0

Stage 1 — RAG question-answering over an indexed codebase with `path:line`
citations, plus the Stage 2 code-graph groundwork (rich chunk metadata).