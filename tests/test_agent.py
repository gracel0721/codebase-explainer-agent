"""Tests for the agent loop + tools (scripted LLM, no live Ollama)."""

from rich.console import Console

from code_explain.agent import Agent, ToolContext, build_handlers
from code_explain.retriever import Retriever
from tests.conftest import FakeMessage, FakeResponse, FakeToolCall, make_chunk, seed_fake_store


def _ctx(cfg, fake_store, console=None, **kw):
    from code_explain.embedder import Embedder

    retriever = Retriever(cfg, fake_store, Embedder(cfg))
    return ToolContext(
        cfg=cfg, store=fake_store, retriever=retriever,
        console=console or Console(quiet=True), **kw,
    )


class _ScriptedLLM:
    """Returns scripted FakeResponses in order; records call count."""

    def __init__(self, responses):
        self._iter = iter(responses)
        self.calls = 0

    def chat_turn(self, system, messages, *, tools=None, keep_alive=None, format=None):
        self.calls += 1
        return next(self._iter)

    def supports_tools(self):
        return True


def test_agent_calls_search_then_answers(make_cfg, fake_store, mock_embedder):
    cfg = make_cfg(embed_dim=8, per_file_cap=0)
    c = make_chunk("app.py", "greet", text="def greet(): pass")
    seed_fake_store(fake_store, [c])
    ctx = _ctx(cfg, fake_store)

    llm = _ScriptedLLM([
        FakeResponse(FakeMessage(content=None, tool_calls=[FakeToolCall("search_code", {"query": "greet"})])),
        FakeResponse(FakeMessage(content="Done. No change needed.", tool_calls=None)),
    ])
    out = Agent(cfg, llm, ctx).run("find greet")
    assert "Done" in out
    assert llm.calls == 2


def test_agent_propose_patch_does_not_write(tmp_path, make_cfg, fake_store, mock_embedder, mocker):
    cfg = make_cfg(tmp_path / "repo", embed_dim=8, per_file_cap=0, agent_apply=False)
    (tmp_path / "repo" / "app.py").write_text("x = 1\n")
    ctx = _ctx(cfg, fake_store, apply=False, repo_path=cfg.repo_path)

    git_apply = mocker.patch("code_explain.agent._git_apply", return_value=(True, ""))
    llm = _ScriptedLLM([
        FakeResponse(FakeMessage(content=None, tool_calls=[
            FakeToolCall("propose_patch", {"path": "app.py", "diff": "@@ \n-x = 1\n+x = 2\n"}),
        ])),
        FakeResponse(FakeMessage(content="proposed the change", tool_calls=None)),
    ])
    Agent(cfg, llm, ctx).run("edit app.py")
    # Propose-only: the diff is printed but never applied.
    git_apply.assert_not_called()


def test_agent_propose_patch_applies_when_enabled(tmp_path, make_cfg, fake_store, mock_embedder, mocker):
    cfg = make_cfg(tmp_path / "repo", embed_dim=8, per_file_cap=0, agent_apply=True)
    (tmp_path / "repo" / "app.py").write_text("x = 1\n")
    # Auto-confirm the apply prompt.
    mocker.patch.object(Console, "input", return_value="y")
    ctx = _ctx(cfg, fake_store, apply=True, repo_path=cfg.repo_path)

    git_apply = mocker.patch("code_explain.agent._git_apply", return_value=(True, ""))
    llm = _ScriptedLLM([
        FakeResponse(FakeMessage(content=None, tool_calls=[
            FakeToolCall("propose_patch", {"path": "app.py", "diff": "@@ \n-x\n+x\n"}),
        ])),
        FakeResponse(FakeMessage(content="applied", tool_calls=None)),
    ])
    Agent(cfg, llm, ctx).run("edit app.py")
    # check-only validation, then the real apply.
    assert git_apply.call_count == 2
    assert git_apply.call_args_list[0].kwargs.get("check_only") is True


def test_run_tests_disabled_by_default(make_cfg, fake_store, mock_embedder):
    cfg = make_cfg(embed_dim=8, per_file_cap=0, agent_allow_tests=False)
    ctx = _ctx(cfg, fake_store, allow_tests=False, repo_path=cfg.repo_path)
    handlers = build_handlers(ctx)
    out = handlers["run_tests"]({})
    assert "disabled" in out


def test_agent_stops_on_repeated_tool_calls(make_cfg, fake_store, mock_embedder):
    cfg = make_cfg(embed_dim=8, per_file_cap=0)
    c = make_chunk("app.py", "greet", text="def greet(): pass")
    seed_fake_store(fake_store, [c])
    ctx = _ctx(cfg, fake_store)

    # The same tool call every turn -> repeat detection stops the loop.
    resp = FakeResponse(FakeMessage(content=None, tool_calls=[FakeToolCall("search_code", {"query": "greet"})]))
    llm = _ScriptedLLM([resp, resp, resp])
    Agent(cfg, llm, ctx).run("loop")
    # Should stop after the second identical turn, well under the iteration cap.
    assert llm.calls == 2


def test_agent_unknown_tool_returns_error(make_cfg, fake_store, mock_embedder):
    cfg = make_cfg(embed_dim=8, per_file_cap=0)
    ctx = _ctx(cfg, fake_store)
    llm = _ScriptedLLM([
        FakeResponse(FakeMessage(content=None, tool_calls=[FakeToolCall("nope", {})])),
        FakeResponse(FakeMessage(content="ok", tool_calls=None)),
    ])
    out = Agent(cfg, llm, ctx).run("do something")
    assert "ok" in out  # loop survives the unknown tool and reaches the final answer


def test_agent_recovers_tool_call_from_content(make_cfg, fake_store, mock_embedder):
    """qwen2.5-coder emits tool calls as JSON in `content` (tool_calls=None).
    The agent should recover and dispatch them rather than treating the turn
    as a final answer."""
    import json

    cfg = make_cfg(embed_dim=8, per_file_cap=0)
    c = make_chunk("app.py", "greet", text="def greet(): pass")
    seed_fake_store(fake_store, [c])
    ctx = _ctx(cfg, fake_store)

    content_with_call = json.dumps({"name": "search_code", "arguments": {"query": "greet"}})
    llm = _ScriptedLLM([
        FakeResponse(FakeMessage(content=content_with_call, tool_calls=None)),
        FakeResponse(FakeMessage(content="Done.", tool_calls=None)),
    ])
    out = Agent(cfg, llm, ctx).run("find greet")
    assert "Done" in out
    assert llm.calls == 2  # recovered the call -> looped -> then final answer


def test_extract_content_tool_calls_shapes():
    from code_explain.agent import _extract_content_tool_calls

    # Single object.
    r = _extract_content_tool_calls('{"name": "read_file", "arguments": {"path": "a.py"}}')
    assert r and r[0]["function"]["name"] == "read_file"
    # Array.
    r = _extract_content_tool_calls('[{"name":"search_code","arguments":{"query":"x"}}]')
    assert r and r[0]["function"]["name"] == "search_code"
    # OpenAI envelope.
    r = _extract_content_tool_calls('{"tool_calls":[{"function":{"name":"read_file","arguments":{"path":"b.py"}}}]}')
    assert r and r[0]["function"]["name"] == "read_file"
    # Embedded in prose.
    r = _extract_content_tool_calls('I will search.\n{"name":"search_code","arguments":{"query":"greet"}}\nthen read.')
    assert r and r[0]["function"]["name"] == "search_code"
    # Plain prose / non-tool JSON -> None.
    assert _extract_content_tool_calls("Just a normal answer.") is None
    assert _extract_content_tool_calls('{"summary": "added a docstring"}') is None
    assert _extract_content_tool_calls("") is None