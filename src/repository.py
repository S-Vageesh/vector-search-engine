from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from src.documents import TextDocument


def _vector_literal(values: Sequence[float]) -> str:
    if not values:
        raise ValueError("Embedding vector cannot be empty")
    return "[" + ",".join(str(float(value)) for value in values) + "]"


class PostgresDocumentRepository:
    def __init__(self, database_url: str, embedding_dimension: int) -> None:
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than zero")

        self.database_url = database_url
        self.embedding_dimension = embedding_dimension

    def initialize_schema(self) -> None:
        import psycopg

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
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
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS document_vectors (
                        document_id BIGINT PRIMARY KEY
                            REFERENCES documents(id) ON DELETE CASCADE,
                        embedding vector({self.embedding_dimension}) NOT NULL,
                        model_name TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS document_vectors_embedding_hnsw_idx
                    ON document_vectors
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )

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
                cursor.execute(
                    """
                    INSERT INTO document_vectors
                        (document_id, embedding, model_name)
                    VALUES (%s, %s::vector, %s)
                    """,
                    (document_id, _vector_literal(embedding), model_name),
                )
                return int(document_id)


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
