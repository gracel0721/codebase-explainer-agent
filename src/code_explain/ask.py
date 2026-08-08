"""Answer pipeline and interactive chat loop.

``answer_question`` is the single-shot RAG turn: retrieve -> assemble context ->
stream the LLM answer with citations. The chat loop keeps conversation history
between turns and is what the default ``code-explain <path>`` invocation drops
you into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from code_explain.llm import LLMClient
from code_explain.prompts import CONTEXT_HEADER_TEMPLATE, SYSTEM_PROMPT, SYSTEM_PROMPT_GRAPH
from code_explain.retriever import Retriever

if TYPE_CHECKING:
    from code_explain.config import Config
    from code_explain.store import VectorStore


def _build_user_message(query: str, context: str) -> str:
    if not context:
        return (
            f"{query}\n\n"
            "(No indexed context was retrieved for this question. Say so and "
            "explain what kind of file/symbol would be needed to answer it.)"
        )
    return f"{query}\n\n" + CONTEXT_HEADER_TEMPLATE.format(context=context)


def answer_question_stream(
    query: str,
    cfg: "Config",
    store: "VectorStore",
    retriever: Retriever,
    llm: LLMClient,
    history: list[dict],
    *,
    console: Console,
    render_markdown: bool = True,
) -> str:
    """Retrieve context for ``query``, stream the answer, return the full text."""
    chunks = retriever.retrieve(query, history=history)
    if not chunks:
        # Surface an actionable hint instead of letting the LLM ramble about
        # "no indexed context". An empty index vs. a miss get different advice.
        if store.count_chunks() == 0:
            console.print(
                "[yellow]The index is empty.[/yellow] "
                "Run `code-explain index <path>` to build it, then ask again."
            )
        else:
            console.print(
                "[yellow]No indexed chunks matched your question.[/yellow] "
                "Try rephrasing, or run `code-explain index <path>` to refresh the index."
            )
    context = Retriever.render_context(chunks)
    user_msg = _build_user_message(query, context)
    messages = history + [{"role": "user", "content": user_msg}]
    system = SYSTEM_PROMPT_GRAPH if cfg.with_graph else SYSTEM_PROMPT

    full_parts: list[str] = []
    interrupted = False
    try:
        if render_markdown:
            with Live("", console=console, refresh_per_second=20, vertical_overflow="visible") as live:
                buf = ""
                for delta in llm.chat_stream(system, messages):
                    buf += delta
                    full_parts.append(delta)
                    live.update(Markdown(buf))
        else:
            for delta in llm.chat_stream(system, messages):
                full_parts.append(delta)
                console.print(delta, end="")
    except KeyboardInterrupt:
        interrupted = True
        # Flush whatever was buffered before the interrupt (markdown may have an
        # unrendered tail; raw mode printed incrementally already).
        if render_markdown and full_parts:
            console.print(Markdown("".join(full_parts)))
    except Exception as exc:
        # A dropped connection mid-stream (e.g. httpx.ReadError) — flush the
        # partial text we did receive rather than dropping it, then re-raise so
        # the caller's Ollama-error handler can print a friendly message.
        if full_parts and render_markdown:
            console.print(Markdown("".join(full_parts)))
        raise
    full = "".join(full_parts)
    if interrupted:
        console.print("\n[dim][interrupted][/dim]")

    # Show the citations actually used (path:line) under the answer.
    if chunks:
        cites = Text("Sources: " + ", ".join(f"{c.rel_path}:L{c.start_line}" for c in chunks[:8]), style="dim")
        if len(chunks) > 8:
            cites.append(f" (+{len(chunks) - 8} more)", style="dim")
        console.print(cites)
    return full


def answer_question(
    query: str,
    cfg: "Config",
    store: "VectorStore",
    retriever: Retriever,
    llm: LLMClient,
    history: list[dict] | None = None,
    *,
    console: Console,
) -> str:
    """Non-streaming convenience wrapper (still streams internally)."""
    return answer_question_stream(
        query, cfg, store, retriever, llm, history or [], console=console
    )


def chat_loop(
    cfg: "Config",
    store: "VectorStore",
    *,
    console: Console | None = None,
) -> None:
    """Interactive multi-turn REPL."""
    from code_explain.embedder import Embedder

    console = console or Console()
    embedder = Embedder(cfg)
    retriever = Retriever(cfg, store, embedder)
    llm = LLMClient(cfg)
    history: list[dict] = []

    console.print(
        Panel.fit(
            f"[bold]code-explain[/bold] — chatting about [cyan]{cfg.repo_path}[/cyan]\n"
            "Ask questions about the codebase. Type :exit (or Ctrl-D) to quit, "
            ":reset to clear history.",
            border_style="cyan",
        )
    )

    while True:
        try:
            line = console.input("[bold green]>>>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        if line in (":exit", ":quit"):
            break
        if line == ":reset":
            history = []
            console.print("[dim]history cleared[/dim]")
            continue
        history.append({"role": "user", "content": line})
        answer = answer_question_stream(
            line, cfg, store, retriever, llm, history[:-1], console=console
        )
        history.append({"role": "assistant", "content": answer})
        console.print()