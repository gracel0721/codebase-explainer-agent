"""Configuration resolution.

A single resolution point for all settings: defaults -> environment variables
-> ``<repo>/.code-explain/config.json`` -> CLI flags (highest priority). This is
also the natural place to add future knobs (``agent_enabled``,
``graph_enabled``, ``reranker_model``) without changing any module signatures.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_LLM_MODEL = "qwen2.5-coder:7b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_EMBED_DIM = 768
DEFAULT_EMBED_NCTX = 8192
DEFAULT_LLM_NCTX = 8192
DEFAULT_TOP_K = 12
DEFAULT_PER_FILE_CAP = 4
DEFAULT_ANSWER_MAX_TOKENS = 1024
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
SCHEMA_VERSION = "1"


@dataclass
class Config:
    repo_path: Path
    db_path: Path
    llm_model: str = DEFAULT_LLM_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dim: int = DEFAULT_EMBED_DIM
    embed_n_ctx: int = DEFAULT_EMBED_NCTX
    llm_n_ctx: int = DEFAULT_LLM_NCTX
    top_k: int = DEFAULT_TOP_K
    per_file_cap: int = DEFAULT_PER_FILE_CAP
    answer_max_tokens: int = DEFAULT_ANSWER_MAX_TOKENS
    ollama_host: str = DEFAULT_OLLAMA_HOST
    with_file_headers: bool = True
    keep_alive: str = "10m"

    # ---- paths --------------------------------------------------------

    @property
    def index_dir(self) -> Path:
        return self.db_path.parent

    @property
    def config_file(self) -> Path:
        return self.index_dir / "config.json"

    # ---- resolution ---------------------------------------------------

    @classmethod
    def resolve(
        cls,
        repo_path: Path,
        *,
        db_path: Path | None = None,
        overrides: dict | None = None,
    ) -> "Config":
        repo_path = repo_path.resolve()
        index_dir = (db_path.parent if db_path else repo_path / ".code-explain")
        if db_path is None:
            db_path = index_dir / "index.db"

        # Start from environment.
        env = _env_config()
        # Layer repo config file (if present).
        file_cfg_path = index_dir / "config.json"
        file_cfg: dict = {}
        if file_cfg_path.exists():
            try:
                file_cfg = json.loads(file_cfg_path.read_text())
            except Exception:
                file_cfg = {}

        merged: dict = {}
        merged.update(env)
        merged.update(file_cfg)
        if overrides:
            merged.update({k: v for k, v in overrides.items() if v is not None})

        return cls(
            repo_path=repo_path,
            db_path=db_path,
            llm_model=merged.get("llm_model", DEFAULT_LLM_MODEL),
            embed_model=merged.get("embed_model", DEFAULT_EMBED_MODEL),
            embed_dim=int(merged.get("embed_dim", DEFAULT_EMBED_DIM)),
            embed_n_ctx=int(merged.get("embed_n_ctx", DEFAULT_EMBED_NCTX)),
            llm_n_ctx=int(merged.get("llm_n_ctx", DEFAULT_LLM_NCTX)),
            top_k=int(merged.get("top_k", DEFAULT_TOP_K)),
            per_file_cap=int(merged.get("per_file_cap", DEFAULT_PER_FILE_CAP)),
            answer_max_tokens=int(merged.get("answer_max_tokens", DEFAULT_ANSWER_MAX_TOKENS)),
            ollama_host=merged.get("ollama_host", DEFAULT_OLLAMA_HOST),
            with_file_headers=bool(merged.get("with_file_headers", True)),
            keep_alive=merged.get("keep_alive", "10m"),
        )

    # ---- persistence --------------------------------------------------

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        data = {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(self).items()
        }
        self.config_file.write_text(json.dumps(data, indent=2))

    def to_display(self) -> dict:
        return {
            "repo_path": str(self.repo_path),
            "db_path": str(self.db_path),
            "llm_model": self.llm_model,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "embed_n_ctx": self.embed_n_ctx,
            "llm_n_ctx": self.llm_n_ctx,
            "top_k": self.top_k,
            "per_file_cap": self.per_file_cap,
            "answer_max_tokens": self.answer_max_tokens,
            "ollama_host": self.ollama_host,
            "with_file_headers": self.with_file_headers,
            "keep_alive": self.keep_alive,
        }


def _env_config() -> dict:
    """Read supported settings from environment variables."""
    out: dict = {}
    env_map = {
        "CODE_EXPLAIN_LLM_MODEL": "llm_model",
        "CODE_EXPLAIN_EMBED_MODEL": "embed_model",
        "CODE_EXPLAIN_EMBED_DIM": "embed_dim",
        "CODE_EXPLAIN_EMBED_NCTX": "embed_n_ctx",
        "CODE_EXPLAIN_LLM_NCTX": "llm_n_ctx",
        "CODE_EXPLAIN_TOP_K": "top_k",
        "CODE_EXPLAIN_PER_FILE_CAP": "per_file_cap",
        "CODE_EXPLAIN_ANSWER_MAX_TOKENS": "answer_max_tokens",
        "CODE_EXPLAIN_OLLAMA_HOST": "ollama_host",
        "OLLAMA_HOST": "ollama_host",
        "CODE_EXPLAIN_KEEP_ALIVE": "keep_alive",
    }
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        if cfg_key in ("embed_dim", "embed_n_ctx", "llm_n_ctx", "top_k", "per_file_cap", "answer_max_tokens"):
            try:
                out[cfg_key] = int(val)
            except ValueError:
                continue
        else:
            out[cfg_key] = val
    return out