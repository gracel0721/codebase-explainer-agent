"""AST-aware chunking via tree-sitter.

For each supported language we walk the tree-sitter tree and emit one chunk per
top-level definition (function / class / method / type). Definitions that exceed
``max_chunk`` tokens are either split into their nested definitions (classes,
impls) or line-chunked (oversized functions). Non-definition top-level nodes
(imports, comments, module statements) become a single ``module`` chunk.

Unknown languages, parse failures, and prose files fall back to the line-based
chunker in :mod:`code_explain.chunker`.

The per-chunk metadata (``kind``/``symbol``/``parent_symbol``/line range) is the
seam for the future code-graph stage: a graph builder can later iterate chunks
and add edges by matching ``symbol`` names referenced inside other chunks'
``text`` — no re-parse needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from code_explain.chunker import (
    Chunk,
    estimate_tokens,
    file_sha256,
    line_chunk,
    module_chunk,
    new_chunk_id,
)

log = logging.getLogger(__name__)

MAX_CHUNK_TOKENS = 1024


# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------

# file extension -> tree-sitter-language-pack language name
EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    # prose / config (no AST chunking -> line fallback, lang kept for metadata)
    ".md": "markdown",
    ".rst": "text",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".sh": "bash",
    ".bash": "bash",
    ".sql": "sql",
}

# Languages with an AST chunker configured below.
_AST_LANGS: set[str] = set()

# node type -> base kind for leaf definitions (functions/methods/types)
_LEAF_TYPES: dict[str, dict[str, str]] = {}

# node type -> kind for container definitions (class/impl) that get split into a
# header chunk + member chunks
_CONTAINER_TYPES: dict[str, dict[str, str]] = {}

# node types that wrap a single inner definition (decorators, export). Unwrap to
# the inner def; use the wrapper's byte range so decorators/export keyword are
# included in the chunk text.
_WRAPPER_TYPES: dict[str, set[str]] = {}


def _register(
    lang: str,
    *,
    leaf: dict[str, str] | None = None,
    container: dict[str, str] | None = None,
    wrapper: set[str] | None = None,
) -> None:
    _AST_LANGS.add(lang)
    _LEAF_TYPES[lang] = leaf or {}
    _CONTAINER_TYPES[lang] = container or {}
    _WRAPPER_TYPES[lang] = wrapper or set()


_register(
    "python",
    leaf={"function_definition": "function"},
    container={"class_definition": "class"},
    wrapper={"decorated_definition"},
)
_register(
    "javascript",
    leaf={
        "function_declaration": "function",
        "method_definition": "method",
        "arrow_function": "function",
        "class_declaration": "class",
    },
    container={"class_declaration": "class"},
    wrapper={"export_statement"},
)
_register(
    "typescript",
    leaf={
        "function_declaration": "function",
        "method_definition": "method",
        "arrow_function": "function",
        "class_declaration": "class",
        "interface_declaration": "class",
        "type_alias_declaration": "class",
    },
    container={"class_declaration": "class"},
    wrapper={"export_statement", "ambient_declaration"},
)
_register(
    "go",
    leaf={
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "class",
    },
    container={},
    wrapper=set(),
)
_register(
    "rust",
    leaf={
        "function_item": "function",
        "struct_item": "class",
        "enum_item": "class",
        "trait_item": "class",
    },
    container={"impl_item": "class"},
    wrapper=set(),
)
_register(
    "java",
    leaf={
        "method_declaration": "method",
        "constructor_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "class",
        "enum_declaration": "class",
    },
    container={"class_declaration": "class", "interface_declaration": "class", "enum_declaration": "class"},
    wrapper=set(),
)
_register(
    "csharp",
    leaf={
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "class",
        "struct_declaration": "class",
    },
    container={"class_declaration": "class", "struct_declaration": "class"},
    wrapper=set(),
)
_register(
    "cpp",
    leaf={"function_definition": "function", "struct_specifier": "class", "class_specifier": "class"},
    container={"class_specifier": "class", "struct_specifier": "class"},
    wrapper=set(),
)
_register("c", leaf={"function_definition": "function", "struct_specifier": "class"}, container={}, wrapper=set())
_register(
    "ruby",
    leaf={"method": "method", "singleton_method": "method", "class": "class", "module": "class"},
    container={"class": "class", "module": "class"},
    wrapper=set(),
)
_register("php", leaf={"function_definition": "function", "class_declaration": "class"}, container={"class_declaration": "class"}, wrapper=set())


# node types that hold a container's members (Python `block`, JS `class_body`, etc.).
# Member definitions may be nested one level inside these rather than being direct
# children of the container node.
_BODY_TYPES = {
    "block",
    "class_body",
    "statement_block",
    "declaration_list",
    "field_declaration_list",
    "interface_body",
    "enum_body",
    "module",
    "declaration",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lang_for_path(path: Path) -> str | None:
    return EXT_TO_LANG.get(path.suffix.lower())


def supports_ast(lang: str) -> bool:
    return lang in _AST_LANGS


def parse_file(
    path: Path,
    source: bytes,
    file_hash: str,
    mtime: float,
    rel_path: str,
    *,
    max_chunk_tokens: int = MAX_CHUNK_TOKENS,
) -> list[Chunk]:
    """Return AST-aware chunks for a file, or line-based chunks as a fallback."""
    lang = lang_for_path(path)
    if lang is None:
        # Unknown extension -> treat as prose text.
        return line_chunk(source.decode("utf-8", errors="replace"), rel_path, "text", file_hash, mtime)

    if not supports_ast(lang):
        return line_chunk(source.decode("utf-8", errors="replace"), rel_path, lang, file_hash, mtime)

    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(lang)
        tree = parser.parse(source)
    except Exception as exc:  # corrupted grammar, parse error, etc.
        log.debug("tree-sitter parse failed for %s (%s); using line chunker", rel_path, exc)
        return line_chunk(source.decode("utf-8", errors="replace"), rel_path, lang, file_hash, mtime)

    if tree.root_node is None:
        return line_chunk(source.decode("utf-8", errors="replace"), rel_path, lang, file_hash, mtime)

    try:
        chunker = _ASTChunker(lang, rel_path, file_hash, mtime, max_chunk_tokens)
        chunks = chunker.collect(tree.root_node)
    except Exception as exc:  # never let a single bad file kill indexing
        log.debug("AST chunking failed for %s (%s); using line chunker", rel_path, exc)
        return line_chunk(source.decode("utf-8", errors="replace"), rel_path, lang, file_hash, mtime)

    if not chunks:
        return line_chunk(source.decode("utf-8", errors="replace"), rel_path, lang, file_hash, mtime)
    return chunks


# ---------------------------------------------------------------------------
# Internal walker
# ---------------------------------------------------------------------------


class _ASTChunker:
    def __init__(self, lang, rel_path, file_hash, mtime, max_tokens):
        self.lang = lang
        self.rel_path = rel_path
        self.file_hash = file_hash
        self.mtime = mtime
        self.max_tokens = max_tokens
        self.leaf = _LEAF_TYPES.get(lang, {})
        self.container = _CONTAINER_TYPES.get(lang, {})
        self.wrappers = _WRAPPER_TYPES.get(lang, set())

    # -- helpers ---------------------------------------------------------

    def _node_text(self, node, source: bytes) -> str:
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _name_of(self, node, source: bytes) -> str | None:
        n = node.child_by_field_name("name")
        if n is not None:
            return self._node_text(n, source)
        # fallback: first identifier-like child
        for c in node.children:
            if c.type in ("identifier", "property_identifier", "type_identifier", "field_identifier"):
                return self._node_text(c, source)
        return None

    def _unwrap(self, node, source: bytes):
        """If node is a wrapper (decorator/export), return (inner_def, use_node)
        where use_node is the node whose byte range to use for text (the wrapper,
        so decorators are included). Otherwise return (node, node)."""
        if node.type in self.wrappers:
            for c in node.children:
                if c.type in self.leaf or c.type in self.container:
                    return c, node
        return node, node

    def _is_def(self, node) -> bool:
        return node.type in self.leaf or node.type in self.container or node.type in self.wrappers

    # -- chunk construction ----------------------------------------------

    def _make_chunk(self, node, source, kind, symbol, parent_symbol) -> Chunk:
        text = self._node_text(node, source)
        return Chunk(
            chunk_id=new_chunk_id(),
            rel_path=self.rel_path,
            lang=self.lang,
            kind=kind,
            text=text,
            symbol=symbol,
            parent_symbol=parent_symbol,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            n_tokens=estimate_tokens(text),
            file_hash=self.file_hash,
            mtime=self.mtime,
        )

    # -- main collection -------------------------------------------------

    def collect(self, root) -> list[Chunk]:
        source: bytes = root.text if root.text is not None else b""
        chunks: list[Chunk] = []
        remainder_nodes = []

        for child in root.children:
            inner, use_node = self._unwrap(child, source)
            if inner.type in self.container:
                self._emit_container(inner, use_node, source, parent_symbol=None, into=chunks)
            elif inner.type in self.leaf:
                self._emit_leaf(inner, use_node, source, parent_symbol=None, into=chunks)
            elif inner is not child and self._is_def(inner):
                # wrapper around something we don't otherwise classify
                chunks.append(self._make_chunk(use_node, source, "block", self._name_of(inner, source), None))
            else:
                remainder_nodes.append(child)

        self._emit_remainder(remainder_nodes, source, into=chunks)
        return chunks

    def _def_children(self, node) -> list:
        """Definition children of ``node``, including ones nested one level inside
        a body node (Python ``block`` / JS ``class_body`` / etc.)."""
        members: list = []
        for c in node.children:
            if self._is_def(c):
                members.append(c)
            elif c.type in _BODY_TYPES:
                for gc in c.children:
                    if self._is_def(gc):
                        members.append(gc)
        return members

    def _emit_container(self, inner, use_node, source, parent_symbol, into) -> None:
        kind = self.container[inner.type]
        symbol = self._name_of(inner, source)
        members = self._def_children(inner)
        if not members:
            # No member definitions -> emit the whole container as one chunk.
            into.append(self._make_chunk(use_node, source, kind, symbol, parent_symbol))
            return
        # Header: everything from the container start up to the first member
        # (decorators, signature, docstring, class-level attrs/imports).
        first_member = members[0]
        header = source[use_node.start_byte : first_member.start_byte].decode("utf-8", errors="replace")
        if header.strip():
            into.append(
                Chunk(
                    chunk_id=new_chunk_id(),
                    rel_path=self.rel_path,
                    lang=self.lang,
                    kind=kind,
                    text=header,
                    symbol=symbol,
                    parent_symbol=parent_symbol,
                    start_line=use_node.start_point[0] + 1,
                    end_line=first_member.start_point[0] + 1,
                    start_byte=use_node.start_byte,
                    end_byte=first_member.start_byte,
                    n_tokens=estimate_tokens(header),
                    file_hash=self.file_hash,
                    mtime=self.mtime,
                )
            )
        # Recurse into each member (members are the real def nodes, possibly
        # nested inside a body child of the container).
        for c in members:
            m_inner, m_use = self._unwrap(c, source)
            if m_inner.type in self.container:
                self._emit_container(m_inner, m_use, source, parent_symbol=symbol, into=into)
            elif m_inner.type in self.leaf:
                base = self.leaf[m_inner.type]
                kind_m = "method" if base in ("function", "method") else base
                self._emit_leaf(m_inner, m_use, source, parent_symbol=symbol, kind=kind_m, into=into)

    def _emit_leaf(self, inner, use_node, source, parent_symbol, into, kind=None) -> None:
        base = kind or self.leaf.get(inner.type, "function")
        if kind is None and parent_symbol is not None and base == "function":
            base = "method"
        symbol = self._name_of(inner, source)
        text = self._node_text(use_node, source)
        if estimate_tokens(text) <= self.max_tokens:
            into.append(self._make_chunk(use_node, source, base, symbol, parent_symbol))
            return
        # Oversized leaf: try to split on nested definitions.
        nested = self._def_children(inner)
        if nested:
            first_nested = nested[0]
            header = source[use_node.start_byte : first_nested.start_byte].decode("utf-8", errors="replace")
            if header.strip():
                into.append(
                    Chunk(
                        chunk_id=new_chunk_id(),
                        rel_path=self.rel_path,
                        lang=self.lang,
                        kind="block",
                        text=header,
                        symbol=symbol,
                        parent_symbol=parent_symbol,
                        start_line=use_node.start_point[0] + 1,
                        end_line=first_nested.start_point[0] + 1,
                        start_byte=use_node.start_byte,
                        end_byte=first_nested.start_byte,
                        n_tokens=estimate_tokens(header),
                        file_hash=self.file_hash,
                        mtime=self.mtime,
                    )
                )
            for c in nested:
                n_inner, n_use = self._unwrap(c, source)
                if n_inner.type in self.container:
                    self._emit_container(n_inner, n_use, source, parent_symbol=symbol, into=into)
                elif n_inner.type in self.leaf:
                    nbase = self.leaf[n_inner.type]
                    nkind = "method" if nbase in ("function", "method") else nbase
                    self._emit_leaf(n_inner, n_use, source, parent_symbol=symbol, kind=nkind, into=into)
        else:
            # No nested defs: line-chunk the oversized leaf's text as "block".
            sub = line_chunk(
                text, self.rel_path, self.lang, self.file_hash, self.mtime,
                target_tokens=800, overlap_tokens=100,
            )
            for c in sub:
                c.kind = "block"
                c.symbol = symbol
                c.parent_symbol = parent_symbol
                into.append(c)

    def _emit_remainder(self, nodes, source, into) -> None:
        if not nodes:
            return
        text = "".join(self._node_text(n, source) for n in nodes)
        if not text.strip():
            return
        first, last = nodes[0], nodes[-1]
        into.append(
            module_chunk(
                text=text,
                rel_path=self.rel_path,
                lang=self.lang,
                file_hash=self.file_hash,
                mtime=self.mtime,
                start_line=first.start_point[0] + 1,
                end_line=last.end_point[0] + 1,
                start_byte=first.start_byte,
                end_byte=last.end_byte,
            )
        )