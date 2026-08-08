"""Tests for file discovery: ignored dirs, .gitignore, walk fallback, binary sniff."""

from pathlib import Path

from code_explain import discovery


def test_ignored_dirs_are_skipped(tmp_repo, mocker):
    # Force the walk fallback so IGNORED_DIRS pruning is exercised.
    mocker.patch.object(discovery, "_git_ls_files", return_value=None)
    (tmp_repo / "node_modules").mkdir()
    (tmp_repo / "node_modules" / "pkg.py").write_text("x = 1\n")
    (tmp_repo / "__pycache__").mkdir()
    (tmp_repo / "__pycache__" / "junk.pyc").write_text("x = 1\n")

    rels = [p.relative_to(tmp_repo).as_posix() for p in discovery.discover_files(tmp_repo)]
    assert not any(r.startswith("node_modules/") for r in rels)
    assert not any(r.startswith("__pycache__/") for r in rels)
    assert "src/app.py" in rels


def test_gitignore_is_respected_on_walk(tmp_repo, mocker):
    mocker.patch.object(discovery, "_git_ls_files", return_value=None)
    (tmp_repo / "ignore_me").mkdir()
    (tmp_repo / "ignore_me" / "secret.py").write_text("x = 1\n")
    (tmp_repo / "trace.log").write_text("log\n")

    rels = [p.relative_to(tmp_repo).as_posix() for p in discovery.discover_files(tmp_repo)]
    assert not any(r.startswith("ignore_me/") for r in rels)
    assert "trace.log" not in rels  # *.log ignored
    assert "src/app.py" in rels


def test_git_ls_files_path_used_when_available(tmp_repo, mocker):
    # When _git_ls_files returns a list (already filtered, as the real impl does),
    # discover_files uses it directly.
    mocker.patch.object(discovery, "_git_ls_files", return_value=["src/app.py"])
    rels = [p.relative_to(tmp_repo).as_posix() for p in discovery.discover_files(tmp_repo)]
    assert rels == ["src/app.py"]


def test_git_ls_files_filters_ignored_dirs(tmp_repo):
    # Real git: an untracked file under node_modules is listed by git but filtered
    # out by _git_ls_files' IGNORED_DIRS check.
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_repo)], check=True)
    (tmp_repo / "node_modules").mkdir()
    (tmp_repo / "node_modules" / "pkg.py").write_text("x = 1\n")
    rels = discovery._git_ls_files(tmp_repo)
    assert rels is not None
    assert not any(r.startswith("node_modules/") for r in rels)
    assert "src/app.py" in rels


def test_git_ls_files_returns_none_falls_back_to_walk(tmp_repo, mocker):
    spy = mocker.patch.object(discovery, "_walk", wraps=discovery._walk)
    mocker.patch.object(discovery, "_git_ls_files", return_value=None)
    rels = [p.relative_to(tmp_repo).as_posix() for p in discovery.discover_files(tmp_repo)]
    assert "src/app.py" in rels
    spy.assert_called_once()


def test_binary_file_is_not_indexable(tmp_path):
    f = tmp_path / "blob.dat"
    f.write_bytes(b"\x00\x01\x02\x03" * 100)
    assert discovery._looks_text(f) is False


def test_text_file_is_indexable(tmp_path):
    f = tmp_path / "code.unknownext"
    f.write_text("print('hello')\n" * 10)
    assert discovery._looks_text(f) is True


def test_oversized_file_is_not_indexable(tmp_path):
    f = tmp_path / "huge.txt"
    f.write_bytes(b"a" * (discovery.MAX_FILE_BYTES + 1))
    assert discovery._looks_text(f) is False


def test_codeexplainignore_respected_on_walk(tmp_repo, mocker):
    mocker.patch.object(discovery, "_git_ls_files", return_value=None)
    (tmp_repo / "secrets").mkdir()
    (tmp_repo / "secrets" / "key.py").write_text("x = 1\n")
    (tmp_repo / "skip_me.py").write_text("x = 1\n")
    (tmp_repo / ".codeexplainignore").write_text("secrets/\nskip_me.py\n")

    rels = [p.relative_to(tmp_repo).as_posix() for p in discovery.discover_files(tmp_repo)]
    assert not any(r.startswith("secrets/") for r in rels)
    assert "skip_me.py" not in rels
    assert "src/app.py" in rels


def test_codeexplainignore_filters_git_listing(tmp_repo):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_repo)], check=True)
    (tmp_repo / "git_tracked_but_ignored.py").write_text("x = 1\n")
    (tmp_repo / ".codeexplainignore").write_text("git_tracked_but_ignored.py\n")
    # .codeexplainignore itself is listed by git but is binary-ish/ignored; the
    # point is the tracked file it names is dropped from discovery.

    rels = [p.relative_to(tmp_repo).as_posix() for p in discovery.discover_files(tmp_repo)]
    assert "git_tracked_but_ignored.py" not in rels
    assert "src/app.py" in rels