"""Tests for chunker: data model, line chunker, helpers."""

from code_explain.chunker import Chunk, estimate_tokens, file_sha256, line_chunk, new_chunk_id


def test_estimate_tokens_is_chars_over_four_min_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_file_sha256_is_deterministic():
    assert file_sha256(b"hello") == file_sha256(b"hello")
    assert file_sha256(b"hello") != file_sha256(b"world")


def test_new_chunk_id_is_unique():
    ids = {new_chunk_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_line_chunk_covers_all_lines():
    src = "\n".join(f"line {i}" for i in range(1, 21)) + "\n"
    chunks = line_chunk(src, "a.py", "python", "h", 0.0, target_tokens=4, overlap_tokens=1)
    assert chunks
    # First chunk starts at line 1; the union of ranges covers line 1..20.
    assert chunks[0].start_line == 1
    covered_end = max(c.end_line for c in chunks)
    assert covered_end == 20
    # Every chunk is kind "text" and carries provenance.
    for c in chunks:
        assert c.kind == "text"
        assert c.rel_path == "a.py"
        assert c.file_hash == "h"


def test_line_chunk_has_overlap_between_windows():
    # Small uniform lines so several fit per window and overlap is produced.
    src = "ab\n" * 40
    chunks = line_chunk(src, "a.py", "python", "h", 0.0, target_tokens=4, overlap_tokens=1)
    assert len(chunks) >= 2
    # Overlap: the next window starts at or before the previous window's end line.
    assert chunks[1].start_line <= chunks[0].end_line


def test_line_chunk_empty_source_returns_empty():
    assert line_chunk("", "a.py", "python", "h", 0.0) == []


def test_chunk_as_store_row_has_all_columns():
    c = Chunk(
        chunk_id="c1", rel_path="a.py", lang="python", kind="function", text="x",
        start_line=1, end_line=1, start_byte=0, end_byte=1, n_tokens=1,
        file_hash="h", mtime=0.0, symbol="f", parent_symbol=None,
    )
    row = c.as_store_row()
    for key in (
        "chunk_id", "rel_path", "lang", "kind", "symbol", "parent_symbol",
        "start_line", "end_line", "start_byte", "end_byte", "text", "n_tokens",
        "file_hash", "mtime",
    ):
        assert key in row