from pathlib import Path

import pytest

from src.documents import TextFileLoader


def test_text_file_loader_accepts_text_files(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello search", encoding="utf-8")

    document = TextFileLoader().load(file_path)

    assert document.source_path == str(file_path)
    assert document.content == "hello search"
    assert document.metadata["filename"] == "sample.txt"
    assert document.metadata["extension"] == ".txt"


def test_text_file_loader_rejects_non_text_files(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.md"
    file_path.write_text("hello search", encoding="utf-8")

    with pytest.raises(ValueError, match="Only .txt files"):
        TextFileLoader().load(file_path)
