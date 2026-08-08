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
from code_explain.prompts import CONTEXT_HEADER_TEMPLATE, SYSTEM_PROMPT
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
    context = Retriever.render_context(chunks)
    user_msg = _build_user_message(query, context)
    messages = history + [{"role": "user", "content": user_msg}]

    full_parts: list[str] = []
    if render_markdown:
        with Live("", console=console, refresh_per_second=20, vertical_overflow="visible") as live:
            buf = ""
            for delta in llm.chat_stream(SYSTEM_PROMPT, messages):
                buf += delta
                full_parts.append(delta)
                live.update(Markdown(buf))
    else:
        for delta in llm.chat_stream(SYSTEM_PROMPT, messages):
            full_parts.append(delta)
            console.print(delta, end="")
    full = "".join(full_parts)

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