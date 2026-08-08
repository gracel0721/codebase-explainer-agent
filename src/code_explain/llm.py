"""Thin Ollama chat client with streaming.

The same client serves both the single-shot ``ask`` pipeline and the future
agent stage: the agent wraps ``chat_stream`` with a tool-call dispatch loop,
passing the same ``messages``/``options`` it already accepts.

Pitfall handled: Ollama defaults to a small ``num_ctx`` (4096 for many models)
and silently truncates long inputs. We always pass ``options={"num_ctx": ...}``
and keep the model warm with ``keep_alive`` so interactive sessions don't pay
cold-start on every turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import ollama

if TYPE_CHECKING:
    from code_explain.config import Config


class LLMClient:
    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self._client = ollama.Client(host=cfg.ollama_host)

    def _options(self) -> dict:
        return {
            "num_ctx": self.cfg.llm_n_ctx,
            "num_predict": self.cfg.answer_max_tokens,
        }

    def chat_stream(
        self,
        system: str,
        messages: list[dict],
        *,
        keep_alive: str | None = None,
    ) -> Iterator[str]:
        full = [{"role": "system", "content": system}] + list(messages)
        try:
            stream = self._client.chat(
                model=self.cfg.llm_model,
                messages=full,
                stream=True,
                options=self._options(),
                keep_alive=keep_alive or self.cfg.keep_alive,
            )
        except Exception as exc:
            from code_explain.errors import raise_ollama_or_reraise

            raise_ollama_or_reraise(exc, model=self.cfg.llm_model, what="stream a chat completion from Ollama")
        try:
            for chunk in stream:
                content = _extract_content(chunk)
                if content:
                    yield content
        except Exception as exc:
            from code_explain.errors import raise_ollama_or_reraise

            raise_ollama_or_reraise(exc, model=self.cfg.llm_model, what="read a chat stream from Ollama")

    def chat(
        self,
        system: str,
        messages: list[dict],
        *,
        keep_alive: str | None = None,
    ) -> str:
        return "".join(self.chat_stream(system, messages, keep_alive=keep_alive))

    def chat_turn(
        self,
        system: str,
        messages: list[dict],
        *,
        tools: list | None = None,
        keep_alive: str | None = None,
        format: str | None = None,
    ):
        """Non-streaming chat turn that may invoke tools.

        Returns the raw ``ChatResponse`` so the caller can read
        ``message.tool_calls`` and ``message.content`` directly. Streaming is
        disabled because Ollama does not reliably assemble tool-call JSON across
        stream chunks; use :meth:`chat_stream` only for the final prose answer.

        ``format`` is passed through to Ollama (e.g. ``"json"``) to force a
        structured response; used by the LLM reranker.
        """
        full = [{"role": "system", "content": system}] + list(messages)
        try:
            return self._client.chat(
                model=self.cfg.llm_model,
                messages=full,
                tools=tools,
                stream=False,
                options=self._options(),
                keep_alive=keep_alive or self.cfg.keep_alive,
                format=format,
            )
        except Exception as exc:
            from code_explain.errors import raise_ollama_or_reraise

            raise_ollama_or_reraise(exc, model=self.cfg.llm_model, what="send a chat turn to Ollama")

    def supports_tools(self) -> bool:
        """True if the configured model declares the ``tools`` capability.

        Cached on the instance. Any error (Ollama not running, model missing)
        returns False so callers can fail with a clear message rather than crash.
        """
        if hasattr(self, "_supports_tools_cache"):
            return self._supports_tools_cache  # type: ignore[no-any-return]
        try:
            resp = self._client.show(self.cfg.llm_model)
            caps = getattr(resp, "capabilities", None)
            ok = bool(caps) and "tools" in caps
        except Exception:
            ok = False
        self._supports_tools_cache = ok
        return ok


def _extract_content(chunk) -> str:
    """Pull the text delta out of a streamed chat chunk (supports dict and
    pydantic response objects across ollama-python versions)."""
    # pydantic-style: chunk.message.content
    msg = getattr(chunk, "message", None)
    if msg is not None:
        c = getattr(msg, "content", None)
        if c:
            return c
    if isinstance(chunk, dict):
        msg = chunk.get("message")
        if isinstance(msg, dict):
            c = msg.get("content")
            if c:
                return c
        c = chunk.get("content")
        if c:
            return c
    return ""