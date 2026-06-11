from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.documents import TextDocument, TextFileLoader
from src.embeddings import Embedder, SentenceTransformerEmbedder
from src.repository import PostgresDocumentRepository


class DocumentRepository(Protocol):
    def save(
        self, document: TextDocument, embedding: list[float], model_name: str
    ) -> int:
        ...


class DocumentIngestionPipeline:
    def __init__(
        self,
        loader: TextFileLoader,
        embedder: Embedder,
        repository: DocumentRepository,
    ) -> None:
        self.loader = loader
        self.embedder = embedder
        self.repository = repository

    def ingest_file(self, path: str | Path) -> int:
        document = self.loader.load(path)
        embedding = self.embedder.embed(document.content)
        return self.repository.save(document, embedding, self.embedder.model_name)


def build_postgres_pipeline(
    database_url: str,
    embedding_model_name: str,
    embedding_dimension: int,
) -> DocumentIngestionPipeline:
    repository = PostgresDocumentRepository(database_url, embedding_dimension)
    repository.initialize_schema()
    return DocumentIngestionPipeline(
        loader=TextFileLoader(),
        embedder=SentenceTransformerEmbedder(embedding_model_name),
        repository=repository,
    )
