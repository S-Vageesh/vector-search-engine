from pathlib import Path

import pytest

from src.documents import TextDocument
from src.index import HnswSearchRepository, HnswVectorIndex
from src.repository import InMemoryDocumentRepository


def _repository_with_vectors() -> InMemoryDocumentRepository:
    repository = InMemoryDocumentRepository()
    repository.save(TextDocument("a.txt", "alpha"), [1.0, 0.0], "test")
    repository.save(TextDocument("b.txt", "beta"), [0.0, 1.0], "test")
    repository.save(TextDocument("c.txt", "alpha nearby"), [0.9, 0.1], "test")
    return repository


def test_hnsw_index_builds_from_stored_embeddings() -> None:
    repository = _repository_with_vectors()
    search_repository = HnswSearchRepository.build_from_repository(
        repository,
        dimension=2,
    )

    results = search_repository.search([1.0, 0.0], top_k=2)

    assert [result.text for result in results] == ["alpha", "alpha nearby"]
    assert results[0].similarity == pytest.approx(1.0)


def test_hnsw_index_supports_incremental_insertion() -> None:
    index = HnswVectorIndex(dimension=2, max_elements=1)
    index.add(1, [1.0, 0.0])
    index.add(2, [0.0, 1.0])

    results = index.search([0.0, 1.0], top_k=1)

    assert results[0].document_id == 2
    assert results[0].similarity == pytest.approx(1.0)


def test_hnsw_index_persists_to_disk(tmp_path: Path) -> None:
    index_path = tmp_path / "documents.hnsw"
    index = HnswVectorIndex(dimension=2, index_path=index_path)
    index.build(_repository_with_vectors().list_vector_records())
    index.save()

    loaded_index = HnswVectorIndex(dimension=2, index_path=index_path)
    loaded_index.load()

    results = loaded_index.search([1.0, 0.0], top_k=1)

    assert index_path.exists()
    assert results[0].document_id == 1


def test_hnsw_index_loads_existing_index_on_startup(tmp_path: Path) -> None:
    index_path = tmp_path / "documents.hnsw"
    repository = _repository_with_vectors()
    first_repository = HnswSearchRepository.load_or_build_from_repository(
        repository,
        dimension=2,
        index_path=index_path,
    )
    first_repository.add(4, [0.95, 0.05])

    second_repository = HnswSearchRepository.load_or_build_from_repository(
        repository,
        dimension=2,
        index_path=index_path,
    )

    results = second_repository.search([0.95, 0.05], top_k=1)

    assert results[0].document_id == 4


def test_hnsw_index_validates_dimensions() -> None:
    index = HnswVectorIndex(dimension=2)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        index.add(1, [1.0, 0.0, 0.0])
