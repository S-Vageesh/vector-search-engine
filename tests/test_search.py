import pytest

from src.documents import TextDocument
from src.index import HnswSearchRepository
from src.repository import InMemoryDocumentRepository, _cosine_similarity
from src.search import SemanticSearchService, select_search_repository


class FakeEmbedder:
    model_name = "fake-model"

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings

    def embed(self, text: str) -> list[float]:
        return self.embeddings[text]


def test_cosine_similarity_scores_identical_vectors_highest() -> None:
    assert _cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine_similarity([0, 1], [1, 0]) == pytest.approx(0.0)


def test_repository_returns_top_k_by_cosine_similarity() -> None:
    repository = InMemoryDocumentRepository()
    repository.save(TextDocument("a.txt", "alpha"), [1, 0], "fake-model")
    repository.save(TextDocument("b.txt", "beta"), [0, 1], "fake-model")
    repository.save(TextDocument("c.txt", "close alpha"), [0.9, 0.1], "fake-model")

    results = repository.search([1, 0], top_k=2)

    assert [result.text for result in results] == ["alpha", "close alpha"]
    assert results[0].similarity == pytest.approx(1.0)
    assert results[1].similarity > results[0].similarity - 0.01


def test_semantic_search_embeds_query_and_returns_results() -> None:
    repository = InMemoryDocumentRepository()
    repository.save(TextDocument("a.txt", "matching document"), [1, 0], "fake-model")
    embedder = FakeEmbedder({"find match": [1, 0]})
    service = SemanticSearchService(embedder, repository)

    results = service.search("find match", top_k=1)

    assert len(results) == 1
    assert results[0].text == "matching document"
    assert results[0].similarity == pytest.approx(1.0)


def test_semantic_search_rejects_empty_queries() -> None:
    service = SemanticSearchService(FakeEmbedder({}), InMemoryDocumentRepository())

    with pytest.raises(ValueError, match="query cannot be empty"):
        service.search("   ")


def test_select_search_repository_uses_brute_force_backend() -> None:
    repository = InMemoryDocumentRepository()

    selected = select_search_repository("brute_force", repository, 2)

    assert selected is repository


def test_select_search_repository_uses_hnsw_backend(tmp_path) -> None:
    repository = InMemoryDocumentRepository()
    repository.save(TextDocument("a.txt", "alpha"), [1.0, 0.0], "fake-model")

    selected = select_search_repository(
        "hnsw",
        repository,
        2,
        hnsw_index_path=str(tmp_path / "documents.hnsw"),
    )

    assert isinstance(selected, HnswSearchRepository)
    assert selected.search([1.0, 0.0], top_k=1)[0].text == "alpha"


def test_select_search_repository_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported search backend"):
        select_search_repository("unknown", InMemoryDocumentRepository(), 2)
