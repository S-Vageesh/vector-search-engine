from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.embeddings import Embedder
from src.index import HnswSearchRepository
from src.repository import SearchResult


class SearchRepository(Protocol):
    def search(self, embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        ...


class SemanticSearchService:
    def __init__(self, embedder: Embedder, repository: SearchRepository) -> None:
        self.embedder = embedder
        self.repository = repository

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self.embedder.embed(query)
        return self.repository.search(query_embedding, top_k)


def select_search_repository(
    backend: str,
    document_repository,
    embedding_dimension: int,
    hnsw_index_path: str | None = None,
    hnsw_max_elements: int = 100000,
) -> SearchRepository:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "brute_force":
        return document_repository
    if normalized_backend == "hnsw":
        return HnswSearchRepository.load_or_build_from_repository(
            document_repository=document_repository,
            dimension=embedding_dimension,
            index_path=hnsw_index_path,
            max_elements=hnsw_max_elements,
        )

    raise ValueError(f"Unsupported search backend: {backend}")
