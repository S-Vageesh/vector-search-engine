import json
import sys
import types

import pytest

from src.documents import TextDocument
from src.repository import (
    EmbeddingStorage,
    InMemoryDocumentRepository,
    PostgresDocumentRepository,
    _embedding_json,
    _vector_literal,
)


def test_vector_literal_formats_pgvector_values() -> None:
    assert _vector_literal([1, 2.5, -3]) == "[1.0,2.5,-3.0]"


def test_vector_literal_rejects_empty_embeddings() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        _vector_literal([])


def test_embedding_json_formats_json_values() -> None:
    assert _embedding_json([1, 2.5, -3]) == "[1.0, 2.5, -3.0]"


def test_postgres_repository_falls_back_to_json_when_pgvector_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_psycopg = _FakePsycopg(extension_available=False)
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    repository = PostgresDocumentRepository("postgresql://test", 2)

    repository.initialize_schema()
    first_id = repository.save(TextDocument("a.txt", "alpha"), [1, 0], "fake-model")
    second_id = repository.save(TextDocument("b.txt", "beta"), [0, 1], "fake-model")
    results = repository.search([1, 0], top_k=1)

    assert first_id == 1
    assert second_id == 2
    assert repository._embedding_storage is EmbeddingStorage.JSON
    assert fake_psycopg.state.embedding_storage == EmbeddingStorage.JSON
    assert fake_psycopg.state.rollbacks == 1
    assert fake_psycopg.state.vectors[0]["embedding"] == [1.0, 0.0]
    assert [result.text for result in results] == ["alpha"]
    assert results[0].similarity == pytest.approx(1.0)


def test_postgres_repository_uses_pgvector_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_psycopg = _FakePsycopg(extension_available=True)
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    repository = PostgresDocumentRepository("postgresql://test", 2)

    repository.initialize_schema()
    repository.save(TextDocument("a.txt", "alpha"), [1, 0], "fake-model")

    assert repository._embedding_storage is EmbeddingStorage.PGVECTOR
    assert fake_psycopg.state.embedding_storage == EmbeddingStorage.PGVECTOR
    assert fake_psycopg.state.index_created is True
    assert fake_psycopg.state.vectors[0]["embedding"] == "[1.0,0.0]"


def test_in_memory_repository_stores_documents_and_vectors_separately() -> None:
    repository = InMemoryDocumentRepository()
    document = TextDocument(source_path="doc.txt", content="body")

    document_id = repository.save(document, [0.4, 0.5], "fake-model")

    assert document_id == 1
    assert repository.saved[0]["document"] == document
    assert repository.saved[0]["embedding"] == [0.4, 0.5]


class _FakeDatabaseState:
    def __init__(self, extension_available: bool) -> None:
        self.extension_available = extension_available
        self.embedding_storage: EmbeddingStorage | None = None
        self.documents: list[dict[str, object]] = []
        self.vectors: list[dict[str, object]] = []
        self.index_created = False
        self.rollbacks = 0


class _FakePsycopg(types.ModuleType):
    def __init__(self, extension_available: bool) -> None:
        super().__init__("psycopg")
        self.state = _FakeDatabaseState(extension_available)

    def connect(self, database_url: str) -> "_FakeConnection":
        assert database_url == "postgresql://test"
        return _FakeConnection(self.state)


class _FakeConnection:
    def __init__(self, state: _FakeDatabaseState) -> None:
        self.state = state

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def cursor(self) -> "_FakeCursor":
        return _FakeCursor(self.state)

    def rollback(self) -> None:
        self.state.rollbacks += 1


class _FakeCursor:
    def __init__(self, state: _FakeDatabaseState) -> None:
        self.state = state
        self._fetchone: tuple[object, ...] | None = None
        self._fetchall: list[tuple[object, ...]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        compact_query = " ".join(query.split())
        if compact_query.startswith("CREATE EXTENSION"):
            if not self.state.extension_available:
                raise RuntimeError("vector extension is unavailable")
            return
        if "FROM information_schema.columns" in compact_query:
            self._fetchone = (
                (
                    "vector"
                    if self.state.embedding_storage is EmbeddingStorage.PGVECTOR
                    else "jsonb",
                )
                if self.state.embedding_storage is not None
                else None
            )
            return
        if compact_query.startswith("CREATE TABLE IF NOT EXISTS documents"):
            return
        if compact_query.startswith("CREATE TABLE IF NOT EXISTS document_vectors"):
            self.state.embedding_storage = (
                EmbeddingStorage.PGVECTOR
                if "embedding vector(" in query
                else EmbeddingStorage.JSON
            )
            return
        if compact_query.startswith("CREATE INDEX"):
            self.state.index_created = True
            return
        if compact_query.startswith("INSERT INTO documents"):
            assert params is not None
            document_id = len(self.state.documents) + 1
            self.state.documents.append(
                {
                    "id": document_id,
                    "source_path": params[0],
                    "content": params[1],
                    "metadata": json.loads(str(params[2])),
                }
            )
            self._fetchone = (document_id,)
            return
        if compact_query.startswith("INSERT INTO document_vectors"):
            assert params is not None
            embedding = (
                params[1]
                if self.state.embedding_storage is EmbeddingStorage.PGVECTOR
                else json.loads(str(params[1]))
            )
            self.state.vectors.append(
                {
                    "document_id": params[0],
                    "embedding": embedding,
                    "model_name": params[2],
                }
            )
            return
        if compact_query.startswith("SELECT d.id, d.content, dv.embedding"):
            rows = []
            for document, vector in zip(self.state.documents, self.state.vectors):
                rows.append(
                    (
                        document["id"],
                        document["content"],
                        vector["embedding"],
                        document["metadata"],
                        document["source_path"],
                    )
                )
            self._fetchall = rows
            return
        if compact_query.startswith("SELECT COUNT(*) FROM documents"):
            self._fetchone = (len(self.state.documents),)
            return
        raise AssertionError(f"Unexpected SQL: {compact_query}")

    def fetchone(self) -> tuple[object, ...] | None:
        return self._fetchone

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._fetchall
