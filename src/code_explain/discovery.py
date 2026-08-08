"""File discovery for a repository.

Uses ``git ls-files`` when inside a git repo (fast, respects .gitignore). Falls
back to a recursive walk filtered by a pathspec gitignore when not. Skips
commonly-ignored directories, the index dir, and binary files.

This module is intentionally dependency-light — the AST/embedding work happens
in :mod:`code_explain.parser` and :mod:`code_explain.indexer`.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

try:
    import pathspec

    _HAS_PATHSPEC = True
except ImportError:  # pragma: no cover - pathspec is a hard dep, but be safe
    _HAS_PATHSPEC = False

log = logging.getLogger(__name__)

# Directories never to descend into.
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".code-explain",
    "coverage",
    ".tox",
    ".idea",
    ".vscode",
    "bower_components",
    "vendor",
}

# Extensions we treat as text/code worth indexing (others are still attempted
# via binary-or-not sniffing, but this allowlist is a fast path).
TEXT_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".rs",
    ".java", ".kt", ".scala", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh",
    ".cs", ".rb", ".php", ".swift", ".lua", ".pl", ".r", ".dart", ".ex", ".exs",
    ".erl", ".clj", ".cljs", ".hs", ".ml", ".fs", ".nim", ".zig", ".v", ".d",
    ".md", ".rst", ".txt", ".markdown", ".org",
    ".yaml", ".yml", ".toml", ".json", ".jsonc", ".ini", ".cfg", ".conf",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".sql", ".graphql", ".gql", ".proto", ".thrift",
    ".dockerfile", ".makefile", ".cmake",
}

# Explicit filenames (no extension) to include.
TEXT_BASENAMES = {
    "Makefile", "makefile", "Dockerfile", "dockerfile", "Rakefile", "Gemfile",
    "Jenkinsfile", "Vagrantfile", ".gitignore", ".gitattributes", ".env.example",
}

MAX_FILE_BYTES = 1_500_000  # skip files larger than ~1.5MB


def discover_files(repo_path: Path) -> list[Path]:
    """Return repo-relative-ish Paths of text files to index.

    Returns absolute paths sorted by str. Respects ``.gitignore`` when possible,
    and always respects a ``.codeexplainignore`` file at the repo root (git's
    ``--exclude-standard`` honors ``.gitignore`` but not ``.codeexplainignore``,
    so we filter the git listing through it explicitly).
    """
    repo_path = repo_path.resolve()
    codeexplain_spec = _load_codeexplain_ignore(repo_path)
    rels = _git_ls_files(repo_path)
    if rels is None:
        rels = _walk(repo_path)  # walk already applies .gitignore + .codeexplainignore
    elif codeexplain_spec is not None:
        # git respects .gitignore but not .codeexplainignore → filter here.
        rels = [r for r in rels if not codeexplain_spec.match_file(r)]
    files = [(repo_path / r) for r in rels]
    files = [f for f in files if _is_indexable(f)]
    return sorted(files, key=lambda p: str(p))


def _is_indexable(path: Path) -> bool:
    rel_name = path.name
    if path.is_dir():
        return False
    # Quick extension/basename allowlist for speed, then binary sniff.
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS or rel_name in TEXT_BASENAMES:
        return True
    # Unknown extension: sniff for binary content.
    return _looks_text(path)


def _looks_text(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return False
        with path.open("rb") as fh:
            chunk = fh.read(8192)
        if b"\x00" in chunk:
            return False
        # Heuristic: mostly printable + whitespace
        text_chars = bytes(range(9, 13)) + bytes(range(32, 127)) + b"\xa0"
        if not chunk:
            return False
        nontext = sum(1 for b in chunk if b not in text_chars)
        return nontext / len(chunk) < 0.30
    except OSError:
        return False


def _git_ls_files(repo_path: Path) -> list[str] | None:
    """Return repo-relative file paths via ``git ls-files``, or None if not a
    git repo / git unavailable."""
    try:
        result = subprocess.run(
            [
                "git", "-C", str(repo_path), "ls-files", "-z",
                "--cached", "--others", "--exclude-standard",
            ],
            capture_output=True,
            check=True,
            text=False,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    files = [f.decode("utf-8", errors="replace") for f in result.stdout.split(b"\x00") if f]
    # Filter out files inside ignored dirs (git ls-files may include e.g. build
    # outputs that were force-added). Also skip our own index dir.
    out = []
    for f in files:
        parts = Path(f).parts
        if any(p in IGNORED_DIRS for p in parts):
            continue
        out.append(f)
    return out


def _walk(repo_path: Path) -> list[str]:
    """Fallback walk honoring a .gitignore if present and IGNORED_DIRS."""
    ignore_specs = _load_gitignores(repo_path)
    out: list[str] = []
    for root, dirs, files in _iter_filtered(repo_path, ignore_specs):
        for name in files:
            full = Path(root) / name
            rel = full.relative_to(repo_path).as_posix()
            if _ignored(rel, ignore_specs):
                continue
            out.append(rel)
    return out


def _iter_filtered(repo_path: Path, ignore_specs):
    """os.walk that prunes ignored dirs and applies gitignore patterns."""
    import os

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        rel_root = Path(root).relative_to(repo_path).as_posix()
        dirs[:] = [
            d for d in dirs
            if not _ignored((f"{rel_root}/{d}" if rel_root != "." else d) + "/", ignore_specs)
        ]
        yield root, dirs, files


def _load_gitignores(repo_path: Path):
    """Load ``.gitignore`` and ``.codeexplainignore`` (if present) as pathspecs.

    Used by the walk fallback, which honors both. Git already respects
    ``.gitignore`` via ``--exclude-standard``; ``.codeexplainignore`` is applied
    to the git listing separately in :func:`discover_files`.
    """
    if not _HAS_PATHSPEC:
        return []
    specs = []
    for name in (".gitignore", ".codeexplainignore"):
        f = repo_path / name
        if not f.exists():
            continue
        try:
            specs.append(pathspec.PathSpec.from_lines("gitignore", f.read_text().splitlines()))
        except OSError:
            continue
    return specs


def _load_codeexplain_ignore(repo_path: Path):
    """Load just the ``.codeexplainignore`` spec, or ``None`` if absent/unsuppported."""
    if not _HAS_PATHSPEC:
        return None
    f = repo_path / ".codeexplainignore"
    if not f.exists():
        return None
    try:
        return pathspec.PathSpec.from_lines("gitignore", f.read_text().splitlines())
    except OSError:
        return None


def _ignored(rel_path: str, specs) -> bool:
    return any(s.match_file(rel_path) for s in specs) if specs else False