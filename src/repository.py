from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.documents import TextDocument


def _vector_literal(values: Sequence[float]) -> str:
    if not values:
        raise ValueError("Embedding vector cannot be empty")
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _parse_vector_literal(value: str) -> list[float]:
    return [float(item) for item in value.strip("[]").split(",") if item]


def _embedding_json(values: Sequence[float]) -> str:
    if not values:
        raise ValueError("Embedding vector cannot be empty")
    return json.dumps([float(value) for value in values])


def _parse_json_embedding(value: Any) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    return [float(item) for item in value]


def _document_filename(source_path: Any, metadata: Any | None = None) -> str:
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if isinstance(metadata, dict) and metadata.get("filename"):
        return str(metadata["filename"])

    normalized_path = str(source_path).replace("\\", "/")
    filename = normalized_path.rsplit("/", maxsplit=1)[-1]
    return filename or f"Document {source_path}"


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")
    if not left:
        raise ValueError("Vectors cannot be empty")

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


@dataclass(frozen=True)
class SearchResult:
    document_id: int
    text: str
    similarity: float
    filename: str


@dataclass(frozen=True)
class VectorRecord:
    document_id: int
    text: str
    embedding: list[float]
    filename: str


class EmbeddingStorage(Enum):
    PGVECTOR = "pgvector"
    JSON = "json"


class PostgresDocumentRepository:
    def __init__(self, database_url: str, embedding_dimension: int) -> None:
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than zero")

        self.database_url = database_url
        self.embedding_dimension = embedding_dimension
        self._embedding_storage: EmbeddingStorage | None = None

    def initialize_schema(self) -> None:
        import psycopg

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                embedding_storage = self._enable_pgvector(cursor, connection)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id BIGSERIAL PRIMARY KEY,
                        source_path TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                existing_embedding_storage = self._get_existing_embedding_storage(cursor)
                if existing_embedding_storage is not None:
                    self._embedding_storage = existing_embedding_storage
                    if self._embedding_storage is EmbeddingStorage.PGVECTOR:
                        self._create_vector_index(cursor)
                    return

                embedding_column_type = (
                    f"vector({self.embedding_dimension})"
                    if embedding_storage is EmbeddingStorage.PGVECTOR
                    else "JSONB"
                )
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS document_vectors (
                        document_id BIGINT PRIMARY KEY
                            REFERENCES documents(id) ON DELETE CASCADE,
                        embedding {embedding_column_type} NOT NULL,
                        model_name TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                self._embedding_storage = embedding_storage
                if self._embedding_storage is EmbeddingStorage.PGVECTOR:
                    self._create_vector_index(cursor)

    def save(
        self, document: TextDocument, embedding: Sequence[float], model_name: str
    ) -> int:
        import psycopg

        if len(embedding) != self.embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.embedding_dimension}, got {len(embedding)}"
            )

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO documents (source_path, content, metadata)
                    VALUES (%s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (
                        document.source_path,
                        document.content,
                        json.dumps(document.metadata),
                    ),
                )
                document_id = cursor.fetchone()[0]
                if self._resolve_embedding_storage() is EmbeddingStorage.PGVECTOR:
                    cursor.execute(
                        """
                        INSERT INTO document_vectors
                            (document_id, embedding, model_name)
                        VALUES (%s, %s::vector, %s)
                        """,
                        (document_id, _vector_literal(embedding), model_name),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO document_vectors
                            (document_id, embedding, model_name)
                        VALUES (%s, %s::jsonb, %s)
                        """,
                        (document_id, _embedding_json(embedding), model_name),
                    )
                return int(document_id)

    def search(self, embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        import psycopg

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if len(embedding) != self.embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.embedding_dimension}, got {len(embedding)}"
            )

        if self._resolve_embedding_storage() is EmbeddingStorage.PGVECTOR:
            with psycopg.connect(self.database_url) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT
                            d.id,
                            d.content,
                            1 - (dv.embedding <=> %s::vector) AS similarity,
                            d.metadata,
                            d.source_path
                        FROM document_vectors dv
                        JOIN documents d ON d.id = dv.document_id
                        ORDER BY dv.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            _vector_literal(embedding),
                            _vector_literal(embedding),
                            top_k,
                        ),
                    )
                    return [
                        SearchResult(
                            document_id=int(row[0]),
                            text=str(row[1]),
                            similarity=float(row[2]),
                            filename=_document_filename(row[4], row[3]),
                        )
                        for row in cursor.fetchall()
                    ]

        return sorted(
            (
                SearchResult(
                    document_id=record.document_id,
                    text=record.text,
                    similarity=_cosine_similarity(record.embedding, embedding),
                    filename=record.filename,
                )
                for record in self.list_vector_records()
            ),
            key=lambda result: result.similarity,
            reverse=True,
        )[:top_k]

    def list_vector_records(self) -> list[VectorRecord]:
        import psycopg

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                if self._resolve_embedding_storage() is EmbeddingStorage.PGVECTOR:
                    cursor.execute(
                        """
                        SELECT d.id, d.content, dv.embedding::text, d.metadata, d.source_path
                        FROM document_vectors dv
                        JOIN documents d ON d.id = dv.document_id
                        ORDER BY d.id
                        """
                    )
                    parse_embedding = _parse_vector_literal
                else:
                    cursor.execute(
                        """
                        SELECT d.id, d.content, dv.embedding, d.metadata, d.source_path
                        FROM document_vectors dv
                        JOIN documents d ON d.id = dv.document_id
                        ORDER BY d.id
                        """
                    )
                    parse_embedding = _parse_json_embedding
                return [
                    VectorRecord(
                        document_id=int(row[0]),
                        text=str(row[1]),
                        embedding=parse_embedding(row[2]),
                        filename=_document_filename(row[4], row[3]),
                    )
                    for row in cursor.fetchall()
                ]

    def get_document_texts(self, document_ids: Sequence[int]) -> dict[int, str]:
        import psycopg

        if not document_ids:
            return {}

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, content
                    FROM documents
                    WHERE id = ANY(%s)
                    """,
                    (list(document_ids),),
                )
                return {int(row[0]): str(row[1]) for row in cursor.fetchall()}

    def count_documents(self) -> int:
        import psycopg

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM documents")
                row = cursor.fetchone()
                return int(row[0]) if row is not None else 0

    def _enable_pgvector(self, cursor: Any, connection: Any) -> EmbeddingStorage:
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception:
            connection.rollback()
            return EmbeddingStorage.JSON
        return EmbeddingStorage.PGVECTOR

    def _resolve_embedding_storage(self) -> EmbeddingStorage:
        if self._embedding_storage is None:
            self.initialize_schema()
        if self._embedding_storage is None:
            raise RuntimeError("Embedding storage was not initialized")
        return self._embedding_storage

    def _get_existing_embedding_storage(
        self, cursor: Any
    ) -> EmbeddingStorage | None:
        cursor.execute(
            """
            SELECT udt_name
            FROM information_schema.columns
            WHERE table_name = 'document_vectors'
                AND column_name = 'embedding'
            """
        )
        row = cursor.fetchone()
        if row is None:
            return None
        if str(row[0]) == "vector":
            return EmbeddingStorage.PGVECTOR
        return EmbeddingStorage.JSON

    def _create_vector_index(self, cursor: Any) -> None:
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS document_vectors_embedding_hnsw_idx
            ON document_vectors
            USING hnsw (embedding vector_cosine_ops)
            """
        )


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    def save(
        self, document: TextDocument, embedding: Sequence[float], model_name: str
    ) -> int:
        document_id = len(self.saved) + 1
        self.saved.append(
            {
                "id": document_id,
                "document": document,
                "embedding": list(embedding),
                "model_name": model_name,
            }
        )
        return document_id

    def search(self, embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        results = [
            SearchResult(
                document_id=record["id"],
                text=record["document"].content,
                similarity=_cosine_similarity(record["embedding"], embedding),
                filename=_document_filename(
                    record["document"].source_path,
                    record["document"].metadata,
                ),
            )
            for record in self.saved
        ]
        return sorted(results, key=lambda result: result.similarity, reverse=True)[
            :top_k
        ]

    def list_vector_records(self) -> list[VectorRecord]:
        return [
            VectorRecord(
                document_id=record["id"],
                text=record["document"].content,
                embedding=list(record["embedding"]),
                filename=_document_filename(
                    record["document"].source_path,
                    record["document"].metadata,
                ),
            )
            for record in self.saved
        ]

    def get_document_texts(self, document_ids: Sequence[int]) -> dict[int, str]:
        requested_ids = set(document_ids)
        return {
            record["id"]: record["document"].content
            for record in self.saved
            if record["id"] in requested_ids
        }

    def count_documents(self) -> int:
        return len(self.saved)
