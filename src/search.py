from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.embeddings import Embedder
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
