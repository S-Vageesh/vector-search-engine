from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.repository import SearchResult, VectorRecord


class VectorRecordRepository(Protocol):
    def list_vector_records(self) -> list[VectorRecord]:
        ...

    def get_document_texts(self, document_ids: Sequence[int]) -> dict[int, str]:
        ...


@dataclass(frozen=True)
class VectorNeighbor:
    document_id: int
    similarity: float


class HnswVectorIndex:
    def __init__(
        self,
        dimension: int,
        index_path: str | Path | None = None,
        max_elements: int = 1024,
        ef_construction: int = 200,
        m: int = 16,
        ef: int = 50,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        if max_elements <= 0:
            raise ValueError("max_elements must be greater than zero")

        import hnswlib

        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.m = m
        self.ef = ef
        self._index = hnswlib.Index(space="cosine", dim=dimension)
        self._initialized = False
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def build(self, records: Iterable[VectorRecord]) -> None:
        records = list(records)
        self._initialize(max(self.max_elements, len(records), 1))
        if records:
            self._add_many(
                [record.document_id for record in records],
                [record.embedding for record in records],
            )

    def load(self) -> None:
        if self.index_path is None:
            raise ValueError("index_path is required to load an index")
        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)

        self._index.load_index(str(self.index_path), max_elements=self.max_elements)
        self._index.set_ef(self.ef)
        self._initialized = True
        self._count = self._index.get_current_count()

    def load_or_build(self, records: Iterable[VectorRecord]) -> None:
        if self.index_path and self.index_path.exists():
            self.load()
            return

        self.build(records)
        if self.index_path:
            self.save()

    def save(self) -> None:
        if self.index_path is None:
            raise ValueError("index_path is required to save an index")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(self.index_path))

    def add(self, document_id: int, embedding: Sequence[float]) -> None:
        if not self._initialized:
            self._initialize(self.max_elements)
        self._add_many([document_id], [embedding])

    def search(self, embedding: Sequence[float], top_k: int) -> list[VectorNeighbor]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self._validate_embedding(embedding)
        if not self._initialized or self._count == 0:
            return []

        labels, distances = self._index.knn_query(
            [list(embedding)],
            k=min(top_k, self._count),
        )
        return [
            VectorNeighbor(document_id=int(label), similarity=1.0 - float(distance))
            for label, distance in zip(labels[0], distances[0])
        ]

    def _initialize(self, max_elements: int) -> None:
        self.max_elements = max_elements
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=self.ef_construction,
            M=self.m,
        )
        self._index.set_ef(self.ef)
        self._initialized = True
        self._count = 0

    def _add_many(
        self,
        document_ids: Sequence[int],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        for embedding in embeddings:
            self._validate_embedding(embedding)

        required_capacity = self._count + len(document_ids)
        if required_capacity > self.max_elements:
            self.max_elements = max(required_capacity, self.max_elements * 2)
            self._index.resize_index(self.max_elements)

        self._index.add_items([list(embedding) for embedding in embeddings], document_ids)
        self._count += len(document_ids)

    def _validate_embedding(self, embedding: Sequence[float]) -> None:
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {len(embedding)}"
            )


class HnswSearchRepository:
    def __init__(
        self,
        index: HnswVectorIndex,
        document_repository: VectorRecordRepository,
    ) -> None:
        self.index = index
        self.document_repository = document_repository

    @classmethod
    def load_or_build_from_repository(
        cls,
        document_repository: VectorRecordRepository,
        dimension: int,
        index_path: str | Path | None,
        max_elements: int = 1024,
    ) -> "HnswSearchRepository":
        index = HnswVectorIndex(
            dimension=dimension,
            index_path=index_path,
            max_elements=max_elements,
        )
        index.load_or_build(document_repository.list_vector_records())
        return cls(index=index, document_repository=document_repository)

    @classmethod
    def build_from_repository(
        cls,
        document_repository: VectorRecordRepository,
        dimension: int,
        index_path: str | Path | None = None,
    ) -> "HnswSearchRepository":
        index = HnswVectorIndex(dimension=dimension, index_path=index_path)
        index.build(document_repository.list_vector_records())
        if index_path:
            index.save()
        return cls(index=index, document_repository=document_repository)

    def add(self, document_id: int, embedding: Sequence[float]) -> None:
        self.index.add(document_id, embedding)
        if self.index.index_path:
            self.index.save()

    def search(self, embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        neighbors = self.index.search(embedding, top_k)
        texts = self.document_repository.get_document_texts(
            [neighbor.document_id for neighbor in neighbors]
        )
        return [
            SearchResult(
                document_id=neighbor.document_id,
                text=texts.get(neighbor.document_id, ""),
                similarity=neighbor.similarity,
            )
            for neighbor in neighbors
        ]
