"""Tests for the AST-aware parser + line fallback."""

from code_explain.parser import lang_for_path, parse_file, supports_ast


def _parse(path, source):
    return parse_file(path, source, "h", 0.0, path.name)


def test_lang_for_path_known_and_unknown():
    from pathlib import Path

    assert lang_for_path(Path("a.py")) == "python"
    assert lang_for_path(Path("a.unknownext")) is None


def test_python_file_emits_function_class_method_and_module():
    from pathlib import Path

    src = (
        '"""doc"""\n'
        "import os\n\n"
        "def greet(name):\n"
        '    return f"hi {name}"\n\n'
        "class Greeter:\n"
        "    def hello(self):\n"
        '        return greet("world")\n'
    ).encode()
    chunks = _parse(Path("app.py"), src)
    kinds = {c.kind for c in chunks}
    symbols = {c.symbol for c in chunks if c.symbol}
    assert "function" in kinds
    assert "class" in kinds
    assert "method" in kinds
    assert "greet" in symbols
    assert "Greeter" in symbols
    assert "hello" in symbols
    # The method's parent_symbol is the class.
    methods = [c for c in chunks if c.kind == "method"]
    assert methods and methods[0].parent_symbol == "Greeter"


def test_unknown_extension_falls_back_to_line_chunker():
    from pathlib import Path

    src = b"some prose\nmore prose\n"
    chunks = _parse(Path("readme.unknownext"), src)
    assert chunks
    assert all(c.kind == "text" for c in chunks)
    assert chunks[0].lang == "text"


def test_prose_file_uses_line_fallback_with_kept_lang():
    from pathlib import Path

    chunks = _parse(Path("notes.md"), b"# Title\n\nbody text\n")
    assert chunks
    assert chunks[0].lang == "markdown"
    assert all(c.kind == "text" for c in chunks)


def test_supports_ast():
    assert supports_ast("python")
    assert not supports_ast("text")
    assert not supports_ast("markdown")