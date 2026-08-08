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
        stream = self._client.chat(
            model=self.cfg.llm_model,
            messages=full,
            stream=True,
            options=self._options(),
            keep_alive=keep_alive or self.cfg.keep_alive,
        )
        for chunk in stream:
            content = _extract_content(chunk)
            if content:
                yield content

    def chat(
        self,
        system: str,
        messages: list[dict],
        *,
        keep_alive: str | None = None,
    ) -> str:
        return "".join(self.chat_stream(system, messages, keep_alive=keep_alive))


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