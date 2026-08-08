"""Ollama error translation.

Network failures (server not running) and model errors (model not pulled) both
surface as low-level exceptions (``httpx.ConnectError``, ollama's
``ResponseError``). This module gives them one clear, actionable error type so
the CLI can print a friendly message instead of a traceback.
"""

from __future__ import annotations


class OllamaUnavailableError(RuntimeError):
    """Ollama isn't reachable, or the requested model isn't available locally."""


def _is_ollama_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a connection/model error worth translating."""
    if isinstance(exc, ConnectionError):
        return True
    try:
        import httpx

        if isinstance(
            exc,
            (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.RemoteProtocolError),
        ):
            return True
    except ImportError:  # pragma: no cover - httpx is a transitive dep of ollama
        pass
    import ollama

    for name in ("ResponseError", "RequestError"):
        t = getattr(ollama, name, None)
        if t is not None and isinstance(exc, t):
            return True
    return False


def raise_ollama_or_reraise(exc: BaseException, *, model: str | None, what: str) -> None:
    """Translate a connection/model error into :class:`OllamaUnavailableError`,
    or re-raise ``exc`` unchanged for anything else (so programming errors
    aren't masked). Always called from an ``except`` block; never returns."""
    if _is_ollama_error(exc):
        msg = f"Could not {what}. "
        if model:
            msg += f"Model {model!r} not available — did you `ollama pull {model}`? "
        msg += "Is `ollama serve` running? (configure the host via OLLAMA_HOST.)"
        raise OllamaUnavailableError(msg) from exc
    raise exc