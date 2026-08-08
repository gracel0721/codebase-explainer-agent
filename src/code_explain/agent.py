"""Stage 3 — agentic edits: a tool-dispatch loop over the LLM.

The agent explores the indexed codebase with a small set of tools
(``read_file``, ``list_symbols``, ``find_callers``, ``search_code``,
``propose_patch``, ``run_tests``), gathers evidence, and proposes a unified
diff. It writes nothing by default — ``propose_patch`` only prints the diff; the
``--apply`` gate (``ToolContext.apply``) asks for confirmation and runs
``git apply``. ``run_tests`` is a separate ``--allow-tests`` gate with a
per-run confirmation prompt.

Tools are declared as explicit OpenAI-shape dicts (``TOOLS_SPEC``) rather than
relying on ``ollama._utils.convert_function_to_tool``'s docstring parsing, so
the JSON schema and parameter descriptions are fully under our control. The
actual handlers are per-``ToolContext`` closures built by :func:`build_handlers`
so they can capture the store/retriever/console without exposing those as
"parameters" to the model.

Loop shape: non-streaming :meth:`LLMClient.chat_turn` for every tool turn
(Ollama does not reliably assemble tool-call JSON across stream chunks); when a
turn returns prose with no tool calls, that prose *is* the final answer and is
rendered directly. An iteration cap plus a repeated-identical-tool-call guard
bound the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rich.markdown import Markdown
from rich.panel import Panel

from code_explain.prompts import SYSTEM_PROMPT_AGENT

if TYPE_CHECKING:
    from rich.console import Console

    from code_explain.config import Config
    from code_explain.llm import LLMClient
    from code_explain.retriever import Retriever
    from code_explain.store import VectorStore


# Cap on tool result size (chars) so a huge file can't blow the LLM context.
MAX_TOOL_RESULT = 6000
# Max lines read_file returns when no range is given.
READ_FILE_DEFAULT_LINES = 200


@dataclass
class ToolContext:
    """Everything a tool handler needs, captured (not passed) into closures."""

    cfg: "Config"
    store: "VectorStore"
    retriever: "Retriever"
    console: "Console"
    allow_tests: bool = False
    apply: bool = False
    repo_path: Path | None = None


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function shape; passed straight to ollama's `tools=`)
# ---------------------------------------------------------------------------

TOOLS_SPEC: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a slice of a file from the repo, returned as numbered "
                "lines (`path:line` matches the citations format). Use this to "
                "read exact lines before proposing an edit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative path, e.g. `src/auth/login.py`.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed first line (inclusive). Omit to start at 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed last line (inclusive). Omit to read to EOF (capped).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_symbols",
            "description": (
                "List the indexed symbols (functions/classes/methods/modules) "
                "in a file with their line ranges. Faster than read_file for "
                "orientation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative path.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_callers",
            "description": (
                "Find code that calls or references a given symbol. Uses the "
                "code graph if built; otherwise falls back to semantic search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "A function/class/method name, e.g. `build_graph`.",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Semantic search over the indexed codebase. Returns the most "
                "relevant code chunks (path:line + symbol + a short excerpt)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A natural-language or keyword query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_patch",
            "description": (
                "Propose a change to a file as a standard unified diff. By "
                "default the diff is only displayed; with --apply it is written "
                "via `git apply` after confirmation. Call this when the task is "
                "done (or state no change is needed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative path of the file being changed.",
                    },
                    "diff": {
                        "type": "string",
                        "description": "A unified diff with enough context to apply cleanly.",
                    },
                },
                "required": ["path", "diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the project's test command. Disabled unless --allow-tests "
                "is set, and prompts for confirmation before each run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_args": {
                        "type": "string",
                        "description": "Optional extra args appended to the test command.",
                    }
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """One task, one tool-dispatch loop."""

    def __init__(
        self,
        cfg: "Config",
        llm: "LLMClient",
        ctx: ToolContext,
        *,
        max_iterations: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.ctx = ctx
        self.max_iterations = max_iterations or cfg.agent_max_iterations
        self._handlers: dict[str, Callable[[dict], str]] = build_handlers(ctx)
        self._last_tool_calls: tuple | None = None  # repeat detection

    def run(self, task: str) -> str:
        """Run the agent loop for ``task``; return the final answer text."""
        console = self.ctx.console
        messages: list[dict] = [{"role": "user", "content": task}]

        try:
            for _ in range(self.max_iterations):
                resp = self.llm.chat_turn(SYSTEM_PROMPT_AGENT, messages, tools=TOOLS_SPEC)
                msg = resp.message
                tool_calls = _tool_calls(msg)
                content = _msg_content(msg)

                # Some models (e.g. qwen2.5-coder) declare tool support but emit
                # the tool call as JSON in `content` instead of via the structured
                # `tool_calls` channel. Recover those so the loop still dispatches.
                if not tool_calls:
                    tool_calls = _extract_content_tool_calls(content)

                if not tool_calls:
                    # A turn with prose and no tool calls is the final answer.
                    return self._render_final(content)

                # Feed the assistant turn (with its tool_calls) back into history so
                # Ollama links the following tool results to these calls.
                messages.append(
                    {
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": [_tool_call_dict(tc) for tc in tool_calls],
                    }
                )

                # Repeat detection: identical consecutive tool-call set => stop.
                sig = tuple((tc["function"]["name"], _args_key(tc["function"]["arguments"])) for tc in messages[-1]["tool_calls"])
                if sig == self._last_tool_calls:
                    console.print("[yellow]repeated identical tool calls; stopping.[/yellow]")
                    return self._render_final(content)
                self._last_tool_calls = sig

                for tc in tool_calls:
                    name, args = _tool_call(tc)
                    result = self._invoke(name, args)
                    messages.append({"role": "tool", "content": result, "tool_name": name})

            # Iteration cap exhausted: force a final prose turn without tools.
            console.print("[yellow]hit iteration cap; wrapping up.[/yellow]")
            resp = self.llm.chat_turn(SYSTEM_PROMPT_AGENT, messages, tools=None)
            return self._render_final(_msg_content(resp.message))
        except KeyboardInterrupt:
            console.print("\n[dim][interrupted][/dim]")
            return ""

    # -- internals ------------------------------------------------------

    def _invoke(self, name: str, args: dict) -> str:
        console = self.ctx.console
        handler = self._handlers.get(name)
        if handler is None:
            return f"error: unknown tool {name!r}"
        pretty = _format_args(args)
        console.print(f"[dim]tool: {name}({pretty})[/dim]")
        try:
            result = handler(args)
        except Exception as exc:  # noqa: BLE001 — surface to the model, don't crash the loop
            result = f"error: {type(exc).__name__}: {exc}"
        return _truncate(result, MAX_TOOL_RESULT)

    def _render_final(self, content: str) -> str:
        console = self.ctx.console
        if not content:
            content = "(no final answer produced)."
        console.print()  # blank line after the tool-call trace
        console.print(Markdown(content))
        return content


# ---------------------------------------------------------------------------
# Tool handlers (per-ctx closures)
# ---------------------------------------------------------------------------


def build_handlers(ctx: ToolContext) -> dict[str, Callable[[dict], str]]:
    """Build the name -> handler map for ``ctx``. Handlers take an args dict."""

    def read_file(args: dict) -> str:
        rel = str(args.get("path", "")).strip()
        p = _safe_path(ctx.repo_path, rel)
        if p is None or not p.is_file():
            return f"error: cannot read {rel!r} (outside repo or missing)."
        lines = p.read_text(errors="replace").splitlines()
        start = _as_int(args.get("start_line"), 1)
        end = _as_int(args.get("end_line"), len(lines))
        start = max(1, start)
        end = min(len(lines), end)
        if end < start:
            return f"error: empty range {start}-{end} (file has {len(lines)} lines)."
        sel = lines[start - 1 : end]
        out = "\n".join(f"{i + start:6d}: {ln}" for i, ln in enumerate(sel))
        if end < len(lines):
            out += f"\n... ({len(lines) - end} more lines)"
        return out or "(empty file)"

    def list_symbols(args: dict) -> str:
        rel = str(args.get("path", "")).strip()
        conn = getattr(ctx.store, "_conn", None)
        if conn is None:
            return "error: store has no SQL connection (vector backend may be LanceDB)."
        rows = conn.execute(
            "SELECT symbol, kind, start_line, end_line FROM chunks "
            "WHERE rel_path = ? ORDER BY start_line",
            (rel,),
        ).fetchall()
        if not rows:
            # Fallback: match by basename in case the model used a slightly
            # different path.
            base = Path(rel).name
            rows = conn.execute(
                "SELECT symbol, kind, start_line, end_line, rel_path FROM chunks "
                "WHERE rel_path LIKE ? ORDER BY start_line",
                (f"%{base}%",),
            ).fetchall()
            if not rows:
                return f"no indexed symbols for {rel!r}."
            return "\n".join(
                f"{r['kind']:8} {(r['symbol'] or '-'):30} L{r['start_line']}-{r['end_line']}  ({r['rel_path']})"
                for r in rows
            )
        return "\n".join(
            f"{r['kind']:8} {(r['symbol'] or '-'):30} L{r['start_line']}-{r['end_line']}" for r in rows
        )

    def find_callers(args: dict) -> str:
        symbol = str(args.get("symbol", "")).strip()
        if not symbol:
            return "error: symbol is required."
        from code_explain import graph

        if graph.is_graph_present(ctx.store):
            chunks = graph.callers_of(ctx.store, symbol)
            if chunks:
                return _format_chunks(chunks)
            return f"no callers found for {symbol!r} via the graph."
        # Vector fallback when no graph is built.
        chunks = ctx.retriever.retrieve(symbol)
        out = _format_chunks(chunks)
        return out or f"no results for {symbol!r}."

    def search_code(args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "error: query is required."
        chunks = ctx.retriever.retrieve(query)
        out = _format_chunks(chunks)
        return out or "no results."

    def propose_patch(args: dict) -> str:
        rel = str(args.get("path", "")).strip()
        diff = str(args.get("diff", ""))
        if not diff.strip():
            return "error: empty diff."
        ctx.console.print(Panel(diff, title=f"proposed patch: {rel}", border_style="cyan"))
        if not ctx.apply:
            return (
                "patch proposed (not applied). Re-run with --apply to write it, "
                "or apply it yourself with `git apply`."
            )
        p = _safe_path(ctx.repo_path, rel)
        if p is None:
            return "error: target path is outside the repo."
        # Validate before prompting so we never ask the user to confirm a patch
        # that can't apply cleanly (git rejects bad context/hunks here).
        ok, detail = _git_apply(ctx.repo_path, diff, check_only=True)
        if not ok:
            return f"error: patch does not apply cleanly: {detail}"
        confirm = ctx.console.input(f"Apply this patch to {rel}? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            return "patch not applied (declined)."
        ok, detail = _git_apply(ctx.repo_path, diff)
        return "applied." if ok else f"error applying patch: {detail}"

    def run_tests(args: dict) -> str:
        if not ctx.allow_tests:
            return "run_tests is disabled (pass --allow-tests to enable)."
        cmd = _sniff_test_cmd(ctx.repo_path, args.get("test_args"))
        shown = " ".join(cmd)
        confirm = ctx.console.input(f"Run tests: {shown}? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            return "tests not run (declined)."
        try:
            r = subprocess.run(
                cmd, cwd=str(ctx.repo_path), capture_output=True, timeout=300
            )
        except FileNotFoundError as exc:
            return f"error: command not found: {exc}"
        except subprocess.TimeoutExpired:
            return "error: tests timed out after 300s."
        out = (r.stdout or b"").decode(errors="replace") + (r.stderr or b"").decode(errors="replace")
        tail = out[-4000:] if len(out) > 4000 else out
        return f"exit {r.returncode}\n{tail}"

    return {
        "read_file": read_file,
        "list_symbols": list_symbols,
        "find_callers": find_callers,
        "search_code": search_code,
        "propose_patch": propose_patch,
        "run_tests": run_tests,
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _format_chunks(chunks) -> str:
    """Compact `path:line symbol (kind)` + first-text-line listing."""
    lines: list[str] = []
    for c in chunks[:20]:
        first = (c.text or "").splitlines()[0] if c.text else ""
        sym = c.symbol or "-"
        lines.append(f"{c.rel_path}:L{c.start_line}  {sym} ({c.kind})  {first[:80]}")
    if len(chunks) > 20:
        lines.append(f"... (+{len(chunks) - 20} more)")
    return "\n".join(lines)


def _safe_path(repo: Path | None, rel: str) -> Path | None:
    """Resolve ``rel`` under ``repo``; reject traversal / paths outside repo."""
    if not rel or repo is None:
        return None
    repo = repo.resolve()
    candidate = (repo / rel).resolve() if not os.path.isabs(rel) else Path(rel).resolve()
    try:
        candidate.relative_to(repo)
    except ValueError:
        return None
    return candidate


def _as_int(val, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _format_args(args: dict) -> str:
    if not args:
        return ""
    try:
        return json.dumps(args, default=str)
    except Exception:
        return str(args)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text) - limit} more chars)"


def _sniff_test_cmd(repo: Path, extra: str | None) -> list[str]:
    """Pick a test command from repo markers; default to ``pytest``."""
    extra = (extra or "").strip()
    extra_args = extra.split() if extra else []
    if (repo / "pyproject.toml").exists() or (repo / "pytest.ini").exists() or (repo / "setup.py").exists():
        return ["pytest", *extra_args]
    if (repo / "package.json").exists():
        return ["npm", "test", "--", *extra_args]
    if (repo / "go.mod").exists():
        return ["go", "test", *extra_args]
    return ["pytest", *extra_args]


def _git_apply(repo: Path, diff: str, *, check_only: bool = False) -> tuple[bool, str]:
    """Validate (and optionally apply) a unified diff via ``git apply``.

    With ``check_only`` only runs ``git apply --check`` (no writes). Safe: git
    rejects bad patches (wrong context/hunks) during ``--check`` before any
    write happens.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(diff)
        patch_path = f.name
    try:
        subprocess.run(
            ["git", "-C", str(repo), "apply", "--check", patch_path],
            capture_output=True,
            check=True,
            timeout=60,
        )
        if check_only:
            return True, ""
        subprocess.run(
            ["git", "-C", str(repo), "apply", patch_path],
            capture_output=True,
            check=True,
            timeout=60,
        )
        return True, ""
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
        return False, err.strip() or str(exc)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    finally:
        try:
            os.unlink(patch_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Response-shape helpers (work for pydantic messages and plain dicts)
# ---------------------------------------------------------------------------


def _msg_content(msg) -> str:
    c = getattr(msg, "content", None)
    if c:
        return c
    if isinstance(msg, dict):
        return msg.get("content") or ""
    return ""


def _extract_content_tool_calls(content: str):
    """Recover tool calls a model emitted as JSON in ``content``.

    Some models (qwen2.5-coder) declare tool support but put the tool call in the
    message text instead of the structured ``tool_calls`` channel. This scans the
    content for JSON tool-call objects (a single object, an array, an OpenAI
    ``{"tool_calls": [...]}`` envelope, or JSON embedded in prose) and returns
    them as OpenAI-shape dicts ``{"function": {"name", "arguments"}}``, or
    ``None`` if none are found (so the caller treats the turn as final prose).
    """
    if not content:
        return None
    text = _strip_code_fences(content)
    candidates: list = []
    whole = _try_loads(text)
    if isinstance(whole, dict):
        candidates = [whole]
    elif isinstance(whole, list):
        candidates = [o for o in whole if isinstance(o, dict)]
    else:
        # Scan for JSON objects embedded in prose (e.g. narration around a call).
        dec = json.JSONDecoder()
        i, n = 0, len(text)
        while i < n:
            if text[i] == "{":
                try:
                    obj, end = dec.raw_decode(text[i:])
                    if isinstance(obj, dict):
                        candidates.append(obj)
                    i += end
                    continue
                except json.JSONDecodeError:
                    pass
            i += 1

    calls: list[dict] = []
    for obj in candidates:
        calls.extend(_coerce_tool_call_obj(obj))
    return calls or None


def _coerce_tool_call_obj(obj: dict) -> list[dict]:
    """Turn one parsed JSON object into OpenAI-shape tool-call dicts (or [])."""
    tcs = obj.get("tool_calls")
    if isinstance(tcs, list):
        out = []
        for tc in tcs:
            if isinstance(tc, dict):
                fn = tc.get("function") if "function" in tc else tc
                if isinstance(fn, dict) and fn.get("name"):
                    out.append({"function": {"name": fn["name"], "arguments": fn.get("arguments", {})}})
        return out
    name = obj.get("name")
    if name and ("arguments" in obj or "parameters" in obj):
        return [{"function": {"name": name, "arguments": obj.get("arguments", obj.get("parameters", {}))}}]
    return []


def _strip_code_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _try_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _tool_calls(msg):
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        return tcs
    if isinstance(msg, dict):
        return msg.get("tool_calls") or []
    return []


def _tool_call(tc) -> tuple[str, dict]:
    fn = getattr(tc, "function", None)
    if fn is None and isinstance(tc, dict):
        fn = tc.get("function")
    name = getattr(fn, "name", None)
    args = getattr(fn, "arguments", None)
    if isinstance(fn, dict):
        name = name or fn.get("name")
        args = args if args is not None else fn.get("arguments")
    if isinstance(args, dict):
        parsed = args
    elif args is None or args == "":
        parsed = {}
    else:
        try:
            parsed = json.loads(args) if isinstance(args, str) else dict(args)
        except Exception:
            parsed = {}
    return name or "", parsed


def _tool_call_dict(tc) -> dict:
    name, args = _tool_call(tc)
    return {"function": {"name": name, "arguments": args}}


def _args_key(args) -> str:
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except Exception:
        return str(args)