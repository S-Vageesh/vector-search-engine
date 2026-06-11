import pytest

from src.documents import TextDocument
from src.repository import InMemoryDocumentRepository, _vector_literal


def test_vector_literal_formats_pgvector_values() -> None:
    assert _vector_literal([1, 2.5, -3]) == "[1.0,2.5,-3.0]"


def test_vector_literal_rejects_empty_embeddings() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        _vector_literal([])


def test_in_memory_repository_stores_documents_and_vectors_separately() -> None:
    repository = InMemoryDocumentRepository()
    document = TextDocument(source_path="doc.txt", content="body")

    document_id = repository.save(document, [0.4, 0.5], "fake-model")

    assert document_id == 1
    assert repository.saved[0]["document"] == document
    assert repository.saved[0]["embedding"] == [0.4, 0.5]
