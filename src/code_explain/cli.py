"""code-explain CLI.

Commands:
    code-explain [PATH]                 index-if-stale, then chat (default)
    code-explain index [PATH] [--force] build/rebuild the index
    code-explain ask   [PATH] "question" one-shot answer with citations
    code-explain chat  [PATH]           interactive multi-turn chat
    code-explain status [PATH]          index stats + staleness
    code-explain config                 print resolved config for the current dir

The bare form ``code-explain ./my-project`` works via a custom Typer group that
falls back to a hidden ``default`` command (index-if-stale then chat) whenever
the first token isn't a known subcommand. This avoids the classic conflict
between a group-level positional argument and subcommand dispatch.

Global options (also accepted via env vars / .code-explain/config.json):
    --model --embed-model --num-ctx --db --verbose --no-color
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from typer._click.exceptions import UsageError
from typer.core import TyperGroup

from code_explain import __version__
from code_explain.config import Config

log = logging.getLogger("code_explain")

DEFAULT_COMMAND = "default"


class _FallbackGroup(TyperGroup):
    """Route unknown first tokens to the hidden ``default`` command.

    Lets ``code-explain ./my-project`` work (path isn't a subcommand) while
    keeping normal subcommand dispatch for ``index``/``ask``/``chat``/etc.
    """

    def resolve_command(self, ctx, args):
        if not args:
            cmd = self.get_command(ctx, DEFAULT_COMMAND)
            return DEFAULT_COMMAND, cmd, args
        try:
            return super().resolve_command(ctx, args)
        except UsageError:
            cmd = self.get_command(ctx, DEFAULT_COMMAND)
            if cmd is None:
                raise
            return DEFAULT_COMMAND, cmd, args


app = typer.Typer(
    name="code-explain",
    help="A local, RAG-powered CLI that answers questions about a codebase.",
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
    cls=_FallbackGroup,
)


# ---------------------------------------------------------------------------
# Shared option wiring
# ---------------------------------------------------------------------------

PathArg = Annotated[
    Optional[Path],
    typer.Argument(help="Path to the repository to analyze. Defaults to the current directory."),
]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"code-explain {__version__}")
        raise typer.Exit()


def _setup(verbose: bool, no_color: bool) -> Console:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return Console(no_color=no_color, highlight=False)


def _make_config(
    path: Path | None,
    model: str | None,
    embed_model: str | None,
    num_ctx: int | None,
    db: Path | None,
) -> Config:
    repo = (path or Path(".")).resolve()
    overrides = {"llm_model": model, "embed_model": embed_model, "llm_n_ctx": num_ctx}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return Config.resolve(repo, db_path=db, overrides=overrides)


def _print_report(report, console: Console) -> None:
    console.print(
        f"[green]Indexed[/green] {report.n_indexed} files "
        f"({report.n_skipped} unchanged, {report.n_errors} errors) "
        f"in {report.duration:.1f}s — {report.n_chunks} chunks."
    )


# ---------------------------------------------------------------------------
# Root callback: set global options on ctx.obj (no default behavior here —
# the default command handles the bare form).
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: Annotated[Optional[str], typer.Option("--model", help="Ollama LLM model.")] = None,
    embed_model: Annotated[Optional[str], typer.Option("--embed-model", help="Ollama embedding model.")] = None,
    num_ctx: Annotated[Optional[int], typer.Option("--num-ctx", help="LLM context window.")] = None,
    db: Annotated[Optional[Path], typer.Option("--db", help="Override index db path.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose logging.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored output.")] = False,
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version and exit."),
    ] = None,
) -> None:
    """Analyze a codebase and answer questions about it."""
    ctx.obj = {
        "model": model,
        "embed_model": embed_model,
        "num_ctx": num_ctx,
        "db": db,
        "verbose": verbose,
        "no_color": no_color,
    }


def _ctx_opts(ctx: typer.Context) -> dict:
    return ctx.obj or {}


def _cfg_from_ctx(ctx: typer.Context, path: Path | None) -> Config:
    o = _ctx_opts(ctx)
    return _make_config(path, o.get("model"), o.get("embed_model"), o.get("num_ctx"), o.get("db"))


def _console_from_ctx(ctx: typer.Context) -> Console:
    o = _ctx_opts(ctx)
    return _setup(o.get("verbose", False), o.get("no_color", False))


# ---------------------------------------------------------------------------
# Default command (hidden): the bare `code-explain [PATH]` form.
# ---------------------------------------------------------------------------


@app.command(name=DEFAULT_COMMAND, hidden=True)
def default_cmd(
    ctx: typer.Context,
    path: PathArg = None,
    no_markdown: Annotated[bool, typer.Option("--no-markdown", help="Print raw text.")] = False,
) -> None:
    """Index-if-stale, then enter chat."""
    console = _console_from_ctx(ctx)
    cfg = _cfg_from_ctx(ctx, path)
    from code_explain.embedder import Embedder
    from code_explain.indexer import index_repo, is_stale, open_store

    console.print(f"[bold]code-explain[/bold] — [cyan]{cfg.repo_path}[/cyan]")
    if not cfg.db_path.exists() or is_stale(cfg):
        console.print("[dim]Indexing…[/dim]")
        report, store = index_repo(cfg, console=console, embedder=Embedder(cfg))
        _print_report(report, console)
    else:
        console.print("[dim]Index is up to date.[/dim]")
        store = open_store(cfg)

    from code_explain.ask import chat_loop

    chat_loop(cfg, store, console=console)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command()
def index(
    ctx: typer.Context,
    path: PathArg = None,
    force: Annotated[bool, typer.Option("--force", help="Drop and rebuild everything.")] = False,
) -> None:
    """Build (or incrementally update) the index."""
    console = _console_from_ctx(ctx)
    cfg = _cfg_from_ctx(ctx, path)
    from code_explain.embedder import Embedder
    from code_explain.indexer import index_repo

    report, _store = index_repo(cfg, force=force, console=console, embedder=Embedder(cfg))
    _print_report(report, console)
    if report.n_errors:
        raise typer.Exit(code=1)


@app.command()
def ask(
    ctx: typer.Context,
    path: PathArg = None,
    question: Annotated[str, typer.Argument(help="Question to ask about the codebase.")] = "",
    no_markdown: Annotated[bool, typer.Option("--no-markdown", help="Print raw text, no rendering.")] = False,
) -> None:
    """Answer one question about the codebase, with citations."""
    console = _console_from_ctx(ctx)
    if not question:
        console.print("[red]Error:[/red] a question is required. Usage: code-explain ask [PATH] \"question\"")
        raise typer.Exit(code=2)
    cfg = _cfg_from_ctx(ctx, path)
    _ensure_indexed(cfg, console)
    _run_ask(cfg, question, console, render_markdown=not no_markdown)


@app.command()
def chat(
    ctx: typer.Context,
    path: PathArg = None,
) -> None:
    """Interactive multi-turn chat about the codebase."""
    console = _console_from_ctx(ctx)
    cfg = _cfg_from_ctx(ctx, path)
    _ensure_indexed(cfg, console)
    from code_explain.ask import chat_loop
    from code_explain.indexer import open_store

    store = open_store(cfg)
    chat_loop(cfg, store, console=console)


@app.command()
def status(
    ctx: typer.Context,
    path: PathArg = None,
) -> None:
    """Show index stats and staleness."""
    console = _console_from_ctx(ctx)
    cfg = _cfg_from_ctx(ctx, path)
    from code_explain.indexer import is_stale, open_store

    if not cfg.db_path.exists():
        console.print("[yellow]No index found.[/yellow] Run `code-explain index`.")
        return
    store = open_store(cfg)
    n_files = store.count_files()
    n_chunks = store.count_chunks()
    stale = is_stale(cfg, store)
    console.print(f"Repo:        [cyan]{cfg.repo_path}[/cyan]")
    console.print(f"Index:       {cfg.db_path}")
    console.print(f"LLM model:   {cfg.llm_model}")
    console.print(f"Embed model: {cfg.embed_model} (dim {cfg.embed_dim})")
    console.print(f"Files:       {n_files}")
    console.print(f"Chunks:      {n_chunks}")
    console.print(f"Stale:       {'[red]yes[/red]' if stale else '[green]no[/green]'}")
    store.close()


def config_cmd(
    ctx: typer.Context,
) -> None:
    """Print the resolved config for the current directory."""
    console = _console_from_ctx(ctx)
    cfg = _cfg_from_ctx(ctx, Path("."))
    for k, v in cfg.to_display().items():
        console.print(f"{k:20} {v}")


# Register the function-based command under the name "config".
app.command(name="config")(config_cmd)


# ---------------------------------------------------------------------------
# Helpers shared by subcommands
# ---------------------------------------------------------------------------


def _ensure_indexed(cfg: Config, console: Console) -> None:
    from code_explain.embedder import Embedder
    from code_explain.indexer import index_repo, is_stale

    if not cfg.db_path.exists() or is_stale(cfg):
        console.print("[dim]Indexing…[/dim]")
        report, _store = index_repo(cfg, console=console, embedder=Embedder(cfg))
        _print_report(report, console)


def _run_ask(cfg: Config, question: str, console: Console, *, render_markdown: bool) -> None:
    from code_explain.ask import answer_question_stream
    from code_explain.embedder import Embedder
    from code_explain.indexer import open_store
    from code_explain.llm import LLMClient
    from code_explain.retriever import Retriever

    store = open_store(cfg)
    embedder = Embedder(cfg)
    retriever = Retriever(cfg, store, embedder)
    llm = LLMClient(cfg)
    answer_question_stream(
        question, cfg, store, retriever, llm, history=[], console=console,
        render_markdown=render_markdown,
    )
    store.close()


if __name__ == "__main__":
    app()