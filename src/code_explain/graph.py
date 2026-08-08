"""Stage 2 — code-relationship graph built on top of the chunk index.

The chunk table already stores the metadata a code graph needs (``kind``,
``symbol``, ``parent_symbol``, line ranges) — see :mod:`code_explain.parser`.
This module derives a caller→callee / containment graph from those chunks
**without re-parsing files for edges**: identifier tokens in each chunk's
``text`` are matched against the global symbol index, and an edge is emitted
only when the target is unambiguous (same file, imported into the file, or
globally unique). Cross-file resolution uses a per-file import table parsed
from source (cheap, no embedding) for python / javascript / typescript / go.

All graph state lives in two additive SQLite tables (``edges``, ``imports``)
created lazily via :meth:`SQLiteVecStore.ensure_graph_tables` — they are NOT
part of ``SCHEMA_SQL`` and do not bump ``SCHEMA_VERSION``. Graph code operates
on ``store._conn`` directly (mirroring ``retriever._find_file_header``); a
future non-SQLite store simply no-ops the graph hook. Everything here is
gated behind ``cfg.with_graph`` / the ``graph_built`` meta key, so Stage 1
behavior is untouched when the graph is absent.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from code_explain.chunker import Chunk
from code_explain.parser import lang_for_path

if TYPE_CHECKING:
    from code_explain.config import Config
    from code_explain.store import SQLiteVecStore

log = logging.getLogger(__name__)

IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Cap on symbols pulled in by a wildcard import (`from x import *`).
WILDCARD_CAP = 50


# ---------------------------------------------------------------------------
# Public report + data
# ---------------------------------------------------------------------------


@dataclass
class GraphReport:
    n_nodes: int = 0
    n_edges: int = 0
    n_imports: int = 0
    n_unresolved_imports: int = 0
    duration: float = 0.0

    @property
    def ok(self) -> bool:
        return True


@dataclass
class ImportSpec:
    """One imported name from a single import statement.

    ``name`` is the symbol as written in the exporting module (or ``"*"`` for a
    wildcard, or the module dotted path for a bare ``import X``). ``alias`` is
    the locally bound name (equal to ``name`` unless aliased) — this is what
    identifier matching looks for in chunk text. ``is_module`` marks a bare
    module import (no specific symbol); ``wildcard`` marks ``import *``.
    """

    name: str
    alias: str
    module_path: str
    wildcard: bool = False
    is_module: bool = False


# ---------------------------------------------------------------------------
# Graph presence / staleness
# ---------------------------------------------------------------------------


def is_graph_present(store: "SQLiteVecStore") -> bool:
    """True if a graph has been built (the ``graph_built`` meta key is set)."""
    return store.get_meta("graph_built") is not None


def is_graph_stale(store: "SQLiteVecStore") -> bool | None:
    """None if no graph; True if any file was (re)indexed after the graph build."""
    ts = store.get_meta("graph_built")
    if ts is None:
        return None
    try:
        built = float(ts)
    except (TypeError, ValueError):
        return None
    conn = getattr(store, "_conn", None)
    if conn is None:
        return False
    row = conn.execute("SELECT MAX(indexed_at) FROM files").fetchone()
    latest = row[0] if row is not None else None
    if latest is not None and float(latest) > built:
        return True
    return False


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_graph(
    store: "SQLiteVecStore",
    cfg: "Config",
    *,
    force: bool = False,  # noqa: ARG001 — accepted for CLI symmetry; build is always a full rebuild
    console=None,
) -> GraphReport:
    """Build the edges/imports tables from the existing chunk index.

    Always rebuilds from scratch (DELETE then repopulate) — it's an explicit,
    cheap O(chunks) command, so edges never dangle after a re-index. Requires
    an existing index; call ``_ensure_indexed`` first from the CLI.
    """
    start = time.time()
    conn = getattr(store, "_conn", None)
    if conn is None:
        # Non-SQLite backend (e.g. LanceDB): the graph is SQLite-only. No-op
        # with a clear message rather than an AttributeError.
        if console is not None:
            console.print(
                "[yellow]Graph is SQLite-only; the current vector backend "
                "(lancedb) does not support it. Skipping graph build.[/yellow]"
            )
        return GraphReport(duration=time.time() - start)
    store.ensure_graph_tables()

    # Full rebuild.
    conn.executescript("DELETE FROM edges; DELETE FROM imports;")
    conn.commit()

    files = store.get_file_records()  # rel_path -> FileRecord
    known_files: set[str] = set(files)

    # --- 1. Parse + resolve imports per file --------------------------------
    imports_by_file: dict[str, list[tuple[ImportSpec, str | None]]] = {}
    n_unresolved = 0
    for rel, rec in files.items():
        lang = rec.lang or lang_for_path(Path(rel)) or ""
        try:
            src = (cfg.repo_path / rel).read_bytes()
        except OSError:
            continue
        for spec in _parse_imports(lang, src):
            tgt = _resolve_module_path(spec.module_path, rel, lang, known_files)
            if tgt is None:
                n_unresolved += 1
            conn.execute(
                "INSERT OR IGNORE INTO imports(rel_path, symbol, alias, module_path, target_rel_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (rel, spec.name, spec.alias, spec.module_path, tgt),
            )
            imports_by_file.setdefault(rel, []).append((spec, tgt))
    conn.commit()

    # --- 2. Symbol index + per-file lookup maps ------------------------------
    # symbol -> list of (chunk_id, rel_path, kind)
    symbol_index: dict[str, list[tuple[str, str, str]]] = {}
    # (rel_path, symbol) -> chunk_id, preferring class/function/method over module
    file_symbol: dict[tuple[str, str], str] = {}
    # rel_path -> list of (chunk_id, kind)
    file_chunks: dict[str, list[tuple[str, str]]] = {}
    # rel_path -> a representative module chunk_id (fallback target)
    file_module_chunk: dict[str, str] = {}

    rows = conn.execute(
        "SELECT chunk_id, rel_path, kind, symbol FROM chunks"
    ).fetchall()
    _kind_rank = {"class": 0, "function": 1, "method": 1, "block": 2, "module": 3}
    file_symbol_rank: dict[tuple[str, str], int] = {}
    for r in rows:
        cid, rel, kind, sym = r["chunk_id"], r["rel_path"], r["kind"], r["symbol"]
        file_chunks.setdefault(rel, []).append((cid, kind))
        if kind == "module" and rel not in file_module_chunk:
            file_module_chunk[rel] = cid
        if sym is not None:
            symbol_index.setdefault(sym, []).append((cid, rel, kind))
            key = (rel, sym)
            rank = _kind_rank.get(kind, 9)
            if key not in file_symbol or rank < file_symbol_rank[key]:
                file_symbol[key] = cid
                file_symbol_rank[key] = rank

    # --- 3. Identifier-match edges ------------------------------------------
    now = time.time()
    chunk_rows = conn.execute(
        "SELECT chunk_id, rel_path, lang, kind, symbol, text FROM chunks"
    ).fetchall()
    n_edges = 0
    for r in chunk_rows:
        cid = r["chunk_id"]
        rel = r["rel_path"]
        lang = r["lang"] or ""
        sym = r["symbol"]
        text = r["text"] or ""
        stop = _lang_stoplist(lang)
        calls = set(CALL_RE.findall(text))  # tokens appearing as `token(`
        imports = imports_by_file.get(rel, [])
        # alias -> (spec, target_rel) for quick import lookup
        import_by_alias: dict[str, tuple[ImportSpec, str | None]] = {
            sp.alias: (sp, tgt) for (sp, tgt) in imports if sp.alias
        }

        seen_targets: set[str] = set()
        for m in IDENT_RE.finditer(text):
            tok = m.group(0)
            if len(tok) < 2 or tok in stop or tok == sym:
                continue
            target = _resolve_target(
                tok, cid, rel, symbol_index, file_symbol, file_module_chunk,
                import_by_alias, file_chunks,
            )
            if target is None or target == cid or target in seen_targets:
                continue
            edge_kind = "calls" if tok in calls else "references"
            cur = conn.execute(
                "INSERT OR IGNORE INTO edges(source_chunk_id, target_chunk_id, edge_kind, via_symbol, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (cid, target, edge_kind, tok, now),
            )
            if cur.rowcount:
                n_edges += 1
            seen_targets.add(target)

    # --- 4. Contains edges: class header -> its method chunks ----------------
    class_rows = conn.execute(
        "SELECT chunk_id, rel_path, symbol FROM chunks WHERE kind = 'class' AND symbol IS NOT NULL"
    ).fetchall()
    for cr in class_rows:
        methods = conn.execute(
            "SELECT chunk_id FROM chunks WHERE rel_path = ? AND kind = 'method' AND parent_symbol = ?",
            (cr["rel_path"], cr["symbol"]),
        ).fetchall()
        for mr in methods:
            cur = conn.execute(
                "INSERT OR IGNORE INTO edges(source_chunk_id, target_chunk_id, edge_kind, via_symbol, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (cr["chunk_id"], mr["chunk_id"], "contains", cr["symbol"], now),
            )
            if cur.rowcount:
                n_edges += 1

    conn.commit()
    built_at = time.time()
    store.set_meta("graph_built", str(built_at))

    report = GraphReport(
        n_nodes=store.count_chunks(),
        n_edges=store.count_edges(),
        n_imports=store.count_imports(),
        n_unresolved_imports=n_unresolved,
        duration=time.time() - start,
    )
    return report


def _resolve_target(
    token: str,
    source_cid: str,
    source_rel: str,
    symbol_index: dict[str, list[tuple[str, str, str]]],
    file_symbol: dict[tuple[str, str], str],
    file_module_chunk: dict[str, str],
    import_by_alias: dict[str, tuple[ImportSpec, str | None]],
    file_chunks: dict[str, list[tuple[str, str]]],
) -> str | None:
    """Resolve an identifier token to a target chunk_id, or None.

    Priority: same-file symbol > imported name > globally-unique symbol.
    Ambiguous names (≥2 files, not imported) are skipped — the main precision
    lever.
    """
    cands = symbol_index.get(token)

    # 1. Same-file definition.
    if cands:
        same_file = [c for c in cands if c[1] == source_rel]
        if same_file:
            # Prefer class/function/method over module/block.
            same_file.sort(key=lambda c: {"class": 0, "function": 1, "method": 1}.get(c[2], 9))
            return same_file[0][0]

    # 2. Imported name (from-imports, module imports, aliases).
    imp = import_by_alias.get(token)
    if imp is not None:
        return _resolve_import_target(imp, file_symbol, file_module_chunk, file_chunks)

    # 3. Globally unique symbol.
    if cands and len(cands) == 1:
        return cands[0][0]

    # Ambiguous or unknown -> no edge.
    return None


def _resolve_import_target(
    imp_tuple: tuple[ImportSpec, str | None],
    file_symbol: dict[tuple[str, str], str],
    file_module_chunk: dict[str, str],
    file_chunks: dict[str, list[tuple[str, str]]],
) -> str | None:
    spec, tgt_rel = imp_tuple
    if tgt_rel is None:
        return None
    if spec.wildcard:
        # Wildcard symbols resolve via the global unique/same-file rules
        # elsewhere; no single target here.
        return None
    if spec.is_module:
        return file_module_chunk.get(tgt_rel) or _first_def_chunk(file_chunks.get(tgt_rel, []))
    # from-import of a named symbol.
    cid = file_symbol.get((tgt_rel, spec.name))
    if cid:
        return cid
    if spec.name == "default":
        return _first_def_chunk(file_chunks.get(tgt_rel, []))
    return file_module_chunk.get(tgt_rel) or _first_def_chunk(file_chunks.get(tgt_rel, []))


def _first_def_chunk(items: list[tuple[str, str]]) -> str | None:
    """First non-module chunk in a file, else the first chunk, else None."""
    for cid, kind in items:
        if kind in ("function", "class", "method"):
            return cid
    return items[0][0] if items else None


# ---------------------------------------------------------------------------
# Traversal / queries
# ---------------------------------------------------------------------------


def expand(
    store: "SQLiteVecStore",
    seed_chunk_ids: list[str],
    *,
    depth: int = 1,
    cap_per_seed: int = 3,
) -> dict[str, list[str]]:
    """BFS over edges (both directions) from each seed.

    Returns ``{seed_chunk_id: [neighbor_chunk_ids]}`` (seeds excluded). Empty
    dict if the ``edges`` table is absent or ``_conn`` is None.
    """
    conn = getattr(store, "_conn", None)
    if conn is None or not seed_chunk_ids:
        return {}
    try:
        conn.execute("SELECT 1 FROM edges LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return {}

    result: dict[str, list[str]] = {}
    all_seeds = set(seed_chunk_ids)
    for seed in seed_chunk_ids:
        seen = set(all_seeds)
        frontier = [seed]
        neighbors: list[str] = []
        for _ in range(max(depth, 0)):
            next_frontier: list[str] = []
            for sid in frontier:
                rows = conn.execute(
                    "SELECT target_chunk_id FROM edges WHERE source_chunk_id = ? "
                    "UNION SELECT source_chunk_id FROM edges WHERE target_chunk_id = ?",
                    (sid, sid),
                ).fetchall()
                for r in rows:
                    tid = r[0]
                    if tid in seen:
                        continue
                    seen.add(tid)
                    neighbors.append(tid)
                    next_frontier.append(tid)
                    if len(neighbors) >= cap_per_seed:
                        break
                if len(neighbors) >= cap_per_seed:
                    break
            frontier = next_frontier
            if len(neighbors) >= cap_per_seed or not frontier:
                break
        if neighbors:
            result[seed] = neighbors
    return result


def callees_of(store: "SQLiteVecStore", symbol: str) -> list[Chunk]:
    """Chunks called/referenced by the chunks defining ``symbol``."""
    return _neighbors(store, symbol, forward=True)


def callers_of(store: "SQLiteVecStore", symbol: str) -> list[Chunk]:
    """Chunks that call/reference the chunks defining ``symbol``."""
    return _neighbors(store, symbol, forward=False)


def _neighbors(store: "SQLiteVecStore", symbol: str, *, forward: bool) -> list[Chunk]:
    conn = getattr(store, "_conn", None)
    if conn is None:
        return []
    seeds = [
        r["chunk_id"]
        for r in conn.execute(
            "SELECT chunk_id FROM chunks WHERE symbol = ?", (symbol,)
        ).fetchall()
    ]
    if not seeds:
        return []
    # No graph built yet -> edges table absent -> no callers/callees.
    try:
        conn.execute("SELECT 1 FROM edges LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return []
    placeholders = ",".join("?" * len(seeds))
    if forward:
        sql = (
            f"SELECT DISTINCT target_chunk_id FROM edges WHERE source_chunk_id IN ({placeholders}) "
            f"AND edge_kind IN ('calls', 'references')"
        )
    else:
        sql = (
            f"SELECT DISTINCT source_chunk_id FROM edges WHERE target_chunk_id IN ({placeholders}) "
            f"AND edge_kind IN ('calls', 'references')"
        )
    ids = [r[0] for r in conn.execute(sql, seeds).fetchall()]
    return store.get_chunks(ids)


# ---------------------------------------------------------------------------
# Identifier extraction + per-language stoplists
# ---------------------------------------------------------------------------


_COMMON_STOP = {
    "true", "false", "null", "none", "nil", "undefined", "self", "this", "cls",
    "super", "return", "const", "let", "var", "new", "if", "else", "for", "while",
    "import", "from", "export", "default", "async", "await", "yield", "raise",
    "try", "catch", "finally", "throw", "class", "def", "func", "fn", "fun",
    "public", "private", "protected", "static", "void", "int", "str", "bool",
    "float", "string", "number", "object", "any", "type", "struct", "enum",
    "interface", "extends", "implements", "package", "module", "get", "set",
}

_PY_STOP = _COMMON_STOP | {
    "lambda", "with", "as", "pass", "break", "continue", "global", "nonlocal",
    "elif", "in", "not", "and", "or", "is", "del", "assert", "print", "len",
    "range", "isinstance", "hasattr", "getattr", "setattr", "dict", "list",
    "tuple", "Exception", "ValueError", "TypeError", "KeyError",
}

_JS_STOP = _COMMON_STOP | {
    "function", "typeof", "instanceof", "delete", "in", "of", "do", "switch",
    "case", "break", "continue", "console", "log", "require", "exports",
    "module", "Promise", "Array", "Object", "String", "Number", "Boolean",
    "JSON", "Math",
}

_GO_STOP = _COMMON_STOP | {
    "func", "go", "chan", "select", "switch", "case", "defer", "range", "map",
    "make", "append", "len", "cap", "fmt", "Println", "Printf", "error",
    "Error", "string", "byte", "rune", "bool", "int", "int64", "float64",
}

_STOPLISTS = {
    "python": _PY_STOP,
    "javascript": _JS_STOP,
    "typescript": _JS_STOP,
    "go": _GO_STOP,
}


def _lang_stoplist(lang: str) -> set[str]:
    return _STOPLISTS.get(lang, _COMMON_STOP)


# ---------------------------------------------------------------------------
# Import parsing (regex per language; robust against grammar drift)
# ---------------------------------------------------------------------------


def _parse_imports(lang: str, source: bytes) -> list[ImportSpec]:
    text = source.decode("utf-8", errors="replace")
    if lang == "python":
        return _py_imports(text)
    if lang in ("javascript", "typescript"):
        return _js_imports(text)
    if lang == "go":
        return _go_imports(text)
    return []


def _py_imports(text: str) -> list[ImportSpec]:
    specs: list[ImportSpec] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        m = re.match(r"from\s+([.\w]*)\s+import\s+(.+)$", line)
        if m:
            mod, names = m.group(1), m.group(2)
            names = names.split("#")[0].strip().rstrip("\\").strip()
            if names == "*":
                specs.append(ImportSpec("*", "*", mod, wildcard=True))
                continue
            for part in _split_commas(names):
                if " as " in part:
                    name, alias = part.split(" as ", 1)
                    specs.append(ImportSpec(name.strip(), alias.strip(), mod))
                else:
                    specs.append(ImportSpec(part, part, mod))
            continue
        m = re.match(r"import\s+([\w.,\s]+)$", line)
        if m:
            for part in _split_commas(m.group(1)):
                if " as " in part:
                    name, alias = part.split(" as ", 1)
                    specs.append(ImportSpec(name.strip(), alias.strip(), name.strip(), is_module=True))
                else:
                    bare = part.strip()
                    specs.append(ImportSpec(bare, bare.split(".")[0], bare, is_module=True))
    return specs


def _js_imports(text: str) -> list[ImportSpec]:
    specs: list[ImportSpec] = []
    # ESM: import { a, b as c } from "mod"; import x from "mod"; import * as ns from "mod"
    esm = re.compile(
        r"import\s+(?:({[^}]*})|(\*\s+as\s+\w+)|(\w+))?\s*(?:from\s+)?[\"']([^\"']+)[\"']"
    )
    for m in esm.finditer(text):
        names_block, ns_block, default_name, mod = m.groups()
        if names_block:
            for part in _split_commas(names_block):
                if " as " in part:
                    name, alias = part.split(" as ", 1)
                    specs.append(ImportSpec(name.strip(), alias.strip(), mod))
                else:
                    specs.append(ImportSpec(part, part, mod))
        if ns_block:
            alias = ns_block.split()[-1]
            specs.append(ImportSpec("*", alias, mod, wildcard=True))
        if default_name:
            specs.append(ImportSpec("default", default_name, mod))
    # CommonJS: const/let/var x = require("mod")
    for m in re.finditer(
        r"(?:const|let|var)\s+(\w+)\s*=\s*require\([\"']([^\"']+)[\"']\)", text
    ):
        specs.append(ImportSpec("default", m.group(1), m.group(2)))
    return specs


def _go_imports(text: str) -> list[ImportSpec]:
    specs: list[ImportSpec] = []
    # single: import "path"  /  import alias "path"
    for m in re.finditer(r"import\s+(?:\w+\s+)?[\"']([^\"']+)[\"']", text):
        specs.append(_go_spec(m.group(1)))
    # block: import ( ... )
    for block in re.finditer(r"import\s*\(([^)]*)\)", text, re.S):
        for m in re.finditer(r"(?:\w+\s+)?[\"']([^\"']+)[\"']", block.group(1)):
            specs.append(_go_spec(m.group(1)))
    return specs


def _go_spec(mod: str) -> ImportSpec:
    alias = mod.split("/")[-1] or mod
    return ImportSpec(mod, alias, mod, is_module=True)


def _split_commas(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


# ---------------------------------------------------------------------------
# Module path -> rel_path resolution
# ---------------------------------------------------------------------------


def _resolve_module_path(
    module_path: str, source_rel: str, lang: str, known_files: set[str]
) -> str | None:
    if not module_path:
        return None
    if lang == "python":
        return _resolve_py(module_path, source_rel, known_files)
    if lang in ("javascript", "typescript"):
        return _resolve_js(module_path, source_rel, known_files)
    if lang == "go":
        return _resolve_go(module_path, known_files)
    return None


def _resolve_py(mod: str, source_rel: str, known: set[str]) -> str | None:
    leading = len(mod) - len(mod.lstrip("."))
    rest = mod[leading:]
    if leading > 0:
        base = Path(source_rel).parent
        for _ in range(max(leading - 1, 0)):
            base = base.parent
        if rest:
            base = base / rest.replace(".", "/")
        rel = base.as_posix()
    else:
        rel = rest.replace(".", "/")
    candidates = [rel + ".py", rel + "/__init__.py", "src/" + rel + ".py", "src/" + rel + "/__init__.py"]
    for c in candidates:
        if c in known:
            return c
    return None


def _resolve_js(mod: str, source_rel: str, known: set[str]) -> str | None:
    if not (mod.startswith("./") or mod.startswith("../")):
        return None  # bare / external package
    base = Path(source_rel).parent.as_posix()
    rel = _normalize_rel(base + "/" + mod)
    exts = [".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"]
    candidates = [rel + e for e in exts] + [rel + "/index" + e for e in exts]
    for c in candidates:
        if c in known:
            return c
    return None


def _resolve_go(mod: str, known: set[str]) -> str | None:
    last = mod.split("/")[-1]
    if not last:
        return None
    # Best effort: find a .go file whose parent directory equals the package name.
    for f in known:
        p = Path(f)
        if p.suffix == ".go" and p.parent.name == last:
            return f
    return None


def _normalize_rel(p: str) -> str:
    parts: list[str] = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts)