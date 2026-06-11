from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TextDocument:
    source_path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TextFileLoader:
    def load(self, path: str | Path) -> TextDocument:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        if not file_path.is_file():
            raise ValueError(f"Expected a file path: {file_path}")
        if file_path.suffix.lower() != ".txt":
            raise ValueError(f"Only .txt files are supported: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        return TextDocument(
            source_path=str(file_path),
            content=content,
            metadata={
                "filename": file_path.name,
                "extension": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size,
            },
        )
