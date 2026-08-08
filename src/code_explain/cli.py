"""code-explain CLI.

Commands:
    code-explain [PATH]                 index-if-stale, then chat (default)
    code-explain index [PATH] [--force] build/rebuild the index
    code-explain ask   [PATH] "question" one-shot answer with citations
    code-explain chat  [PATH]           interactive multi-turn chat
    code-explain graph [PATH] [--force] build the code-relationship graph
    code-explain agent [PATH] "task"    explore + propose edits with an LLM agent
    code-explain status [PATH]          index stats + staleness
    code-explain config                 print resolved config for the current dir

The bare form ``code-explain ./my-project`` works via a custom Typer group that
falls back to a hidden ``default`` command (index-if-stale then chat) whenever
the first token isn't a known subcommand. This avoids the classic conflict
between a group-level positional argument and subcommand dispatch.

Global options (also accepted via env vars / .code-explain/config.json):
    --model --embed-model --num-ctx --db --with-graph --verbose --no-color
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
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
    with_graph: bool = False,
) -> Config:
    repo = (path or Path(".")).resolve()
    overrides: dict = {"llm_model": model, "embed_model": embed_model, "llm_n_ctx": num_ctx}
    if with_graph:
        overrides["with_graph"] = True
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
    with_graph: Annotated[
        bool, typer.Option("--with-graph", help="Use the code graph to expand context with caller/callee chunks (run `code-explain graph` first).")
    ] = False,
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
        "with_graph": with_graph,
        "verbose": verbose,
        "no_color": no_color,
    }


def _ctx_opts(ctx: typer.Context) -> dict:
    return ctx.obj or {}


def _cfg_from_ctx(ctx: typer.Context, path: Path | None) -> Config:
    o = _ctx_opts(ctx)
    try:
        return _make_config(
            path, o.get("model"), o.get("embed_model"), o.get("num_ctx"), o.get("db"),
            with_graph=bool(o.get("with_graph", False)),
        )
    except ValueError as exc:
        console = _console_from_ctx(ctx)
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=2)


def _console_from_ctx(ctx: typer.Context) -> Console:
    o = _ctx_opts(ctx)
    return _setup(o.get("verbose", False), o.get("no_color", False))


def _check_ollama(cfg: Config, console: Console) -> None:
    """Cheap reachability check: is `ollama serve` up? Prints a friendly error
    and exits if not, so a down server never produces a raw traceback."""
    import ollama

    from code_explain.errors import OllamaUnavailableError, raise_ollama_or_reraise

    try:
        ollama.Client(host=cfg.ollama_host).list()
    except Exception as exc:
        try:
            raise_ollama_or_reraise(exc, model=None, what="reach Ollama")
        except OllamaUnavailableError as err:
            console.print(f"[red]Error:[/red] {err}")
            raise typer.Exit(code=1)


def _print_ollama_error(exc: BaseException, console: Console) -> None:
    """Print an :class:`OllamaUnavailableError` and exit; re-raise anything else."""
    from code_explain.errors import OllamaUnavailableError

    if isinstance(exc, OllamaUnavailableError):
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    raise exc


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
    _check_ollama(cfg, console)
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

    _warn_graph_missing(cfg, console)
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
    _check_ollama(cfg, console)
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
    _check_ollama(cfg, console)
    _ensure_indexed(cfg, console)
    try:
        _run_ask(cfg, question, console, render_markdown=not no_markdown)
    except Exception as exc:
        _print_ollama_error(exc, console)


@app.command()
def chat(
    ctx: typer.Context,
    path: PathArg = None,
) -> None:
    """Interactive multi-turn chat about the codebase."""
    console = _console_from_ctx(ctx)
    cfg = _cfg_from_ctx(ctx, path)
    _check_ollama(cfg, console)
    _ensure_indexed(cfg, console)
    _warn_graph_missing(cfg, console)
    from code_explain.ask import chat_loop
    from code_explain.indexer import open_store

    store = open_store(cfg)
    chat_loop(cfg, store, console=console)


@app.command()
def graph(
    ctx: typer.Context,
    path: PathArg = None,
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if a graph exists.")] = False,
) -> None:
    """Build (or rebuild) the code-relationship graph from the chunk index."""
    console = _console_from_ctx(ctx)
    cfg = _cfg_from_ctx(ctx, path)
    _ensure_indexed(cfg, console)
    from code_explain.graph import build_graph
    from code_explain.indexer import open_store

    store = open_store(cfg)
    report = build_graph(store, cfg, force=force, console=console)
    console.print(
        f"[green]Graph built[/green] — {report.n_nodes} nodes, {report.n_edges} edges, "
        f"{report.n_imports} imports ({report.n_unresolved_imports} unresolved) "
        f"in {report.duration:.1f}s."
    )
    store.close()


@app.command()
def agent(
    ctx: typer.Context,
    path: PathArg = None,
    task: Annotated[str, typer.Argument(help="Editing task for the agent.")] = "",
    apply: Annotated[bool, typer.Option("--apply", help="Write proposed patches via `git apply` (with confirmation).")] = False,
    allow_tests: Annotated[bool, typer.Option("--allow-tests", help="Permit the agent to run tests (with per-run confirmation).")] = False,
    max_iterations: Annotated[Optional[int], typer.Option("--max-iterations", help="Cap on agent tool turns.")] = None,
) -> None:
    """Explore the codebase and propose edits with an LLM agent."""
    console = _console_from_ctx(ctx)
    if not task:
        console.print("[red]Error:[/red] a task is required. Usage: code-explain agent [PATH] \"task\"")
        raise typer.Exit(code=2)
    cfg = _cfg_from_ctx(ctx, path)
    _check_ollama(cfg, console)
    _ensure_indexed(cfg, console)
    _warn_graph_missing(cfg, console)

    from code_explain.agent import Agent, ToolContext
    from code_explain.embedder import Embedder
    from code_explain.indexer import open_store
    from code_explain.llm import LLMClient
    from code_explain.retriever import Retriever

    store = open_store(cfg)
    embedder = Embedder(cfg)
    retriever = Retriever(cfg, store, embedder)
    llm = LLMClient(cfg)

    if not llm.supports_tools():
        console.print(
            f"[red]Error:[/red] model {cfg.llm_model!r} does not declare tool support. "
            "The agent needs a tool-calling model (e.g. qwen2.5-coder:7b). "
            "Pull it with `ollama pull qwen2.5-coder:7b` and re-run."
        )
        store.close()
        raise typer.Exit(code=1)

    # CLI agent flags override the resolved config (and thus env/file). Re-resolve
    # once so all layers still apply, with these flags winning.
    agent_overrides: dict = {}
    if apply:
        agent_overrides["agent_apply"] = True
    if allow_tests:
        agent_overrides["agent_allow_tests"] = True
    if max_iterations is not None:
        agent_overrides["agent_max_iterations"] = max_iterations
    if agent_overrides:
        try:
            cfg = Config.resolve(
                cfg.repo_path, db_path=cfg.db_path, overrides=agent_overrides
            )
        except ValueError as exc:
            console.print(f"[red]Config error:[/red] {exc}")
            store.close()
            raise typer.Exit(code=2)

    tool_ctx = ToolContext(
        cfg=cfg, store=store, retriever=retriever, console=console,
        allow_tests=cfg.agent_allow_tests, apply=cfg.agent_apply,
        repo_path=cfg.repo_path,
    )
    ag = Agent(cfg, llm, tool_ctx, max_iterations=cfg.agent_max_iterations)
    console.print(Panel.fit(
        f"[bold]code-explain agent[/bold] — [cyan]{cfg.repo_path}[/cyan]\n"
        f"task: {task}\n"
        f"mode: {'apply (with confirm)' if cfg.agent_apply else 'propose-only'}"
        + (" | tests allowed" if cfg.agent_allow_tests else ""),
        border_style="magenta",
    ))
    try:
        ag.run(task)
    except Exception as exc:
        _print_ollama_error(exc, console)
    finally:
        store.close()


@app.command()
def status(
    ctx: typer.Context,
    path: PathArg = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print stats as JSON.")] = False,
) -> None:
    """Show index stats and staleness."""
    console = _console_from_ctx(ctx)
    cfg = _cfg_from_ctx(ctx, path)
    from code_explain.indexer import is_stale, open_store

    if not cfg.db_path.exists():
        if json_output:
            typer.echo(json.dumps({"indexed": False}))
        else:
            console.print("[yellow]No index found.[/yellow] Run `code-explain index`.")
        return
    store = open_store(cfg)
    n_files = store.count_files()
    n_chunks = store.count_chunks()
    stale = is_stale(cfg, store)
    last_indexed = store.get_meta("last_index_at")

    from code_explain.graph import is_graph_present, is_graph_stale

    graph_present = is_graph_present(store)
    n_edges = store.count_edges() if graph_present else 0
    n_imports = store.count_imports() if graph_present else 0
    gstale = is_graph_stale(store) if graph_present else None
    store.close()

    if json_output:
        typer.echo(json.dumps({
            "indexed": True,
            "repo_path": str(cfg.repo_path),
            "db_path": str(cfg.db_path),
            "llm_model": cfg.llm_model,
            "embed_model": cfg.embed_model,
            "embed_dim": cfg.embed_dim,
            "files": n_files,
            "chunks": n_chunks,
            "stale": stale,
            "last_indexed": last_indexed,
            "graph": {"present": graph_present, "edges": n_edges, "imports": n_imports, "stale": gstale},
        }))
        return

    from rich.table import Table

    table = Table(title="code-explain status", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="dim", no_wrap=True)
    table.add_column("Value")
    table.add_row("Repo", f"[cyan]{cfg.repo_path}[/cyan]")
    table.add_row("Index", str(cfg.db_path))
    table.add_row("LLM model", cfg.llm_model)
    table.add_row("Embed model", f"{cfg.embed_model} (dim {cfg.embed_dim})")
    table.add_row("Files", str(n_files))
    table.add_row("Chunks", str(n_chunks))
    table.add_row("Stale", "[red]yes[/red]" if stale else "[green]no[/green]")
    table.add_row("Last indexed", last_indexed or "[dim]—[/dim]")
    if graph_present:
        stale_str = (
            "[red]yes[/red]" if gstale else "[green]no[/green]" if gstale is False else "[dim]?[/dim]"
        )
        table.add_row("Graph", f"{n_edges} edges, {n_imports} imports (stale {stale_str})")
    else:
        table.add_row("Graph", "[dim]not built[/dim] (run `code-explain graph`)")
    console.print(table)


@app.command()
def reset(
    ctx: typer.Context,
    path: PathArg = None,
) -> None:
    """Drop the index and graph without re-indexing."""
    console = _console_from_ctx(ctx)
    cfg = _cfg_from_ctx(ctx, path)
    if not cfg.db_path.exists():
        console.print("[yellow]No index found.[/yellow] Nothing to reset.")
        return
    confirm = console.input(f"Drop index for {cfg.repo_path}? [y/N] ").strip().lower()
    if confirm not in ("y", "yes"):
        console.print("[dim]reset cancelled[/dim]")
        return
    from code_explain.indexer import open_store

    store = open_store(cfg, force=True)  # force=True drops everything and recreates empty
    store.close()
    console.print("[green]Index reset.[/green] Run `code-explain index` to rebuild.")


def config_cmd(
    ctx: typer.Context,
    json_output: Annotated[bool, typer.Option("--json", help="Print config as JSON.")] = False,
) -> None:
    """Print the resolved config for the current directory."""
    console = _console_from_ctx(ctx)
    cfg = _cfg_from_ctx(ctx, Path("."))
    if json_output:
        typer.echo(json.dumps(cfg.to_display(), indent=2))
        return
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


def _warn_graph_missing(cfg: Config, console: Console) -> None:
    """If --with-graph was requested but no graph is built, say so once."""
    if not cfg.with_graph:
        return
    from code_explain.graph import is_graph_present
    from code_explain.indexer import open_store

    store = open_store(cfg)
    try:
        if not is_graph_present(store):
            console.print(
                "[yellow]--with-graph set, but no graph is built[/yellow] "
                "(run `code-explain graph`); answering without graph expansion."
            )
    finally:
        store.close()


def _run_ask(cfg: Config, question: str, console: Console, *, render_markdown: bool) -> None:
    from code_explain.ask import answer_question_stream
    from code_explain.embedder import Embedder
    from code_explain.indexer import open_store
    from code_explain.llm import LLMClient
    from code_explain.retriever import Retriever

    _warn_graph_missing(cfg, console)
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