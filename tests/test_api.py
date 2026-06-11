from fastapi.testclient import TestClient

from src.api import create_app
from src.repository import SearchResult


class FakeSearchService:
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        assert query == "semantic database"
        assert top_k == 2
        return [
            SearchResult(
                document_id=7,
                text="semantic search document",
                similarity=0.92,
            )
        ]


def test_search_endpoint_returns_document_text_and_similarity() -> None:
    app = create_app(service_factory=lambda: FakeSearchService())
    client = TestClient(app)

    response = client.post(
        "/search",
        json={"query": "semantic database", "top_k": 2},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "document_id": 7,
            "text": "semantic search document",
            "similarity": 0.92,
        }
    ]


def test_search_endpoint_validates_top_k() -> None:
    app = create_app(service_factory=lambda: FakeSearchService())
    client = TestClient(app)

    response = client.post("/search", json={"query": "semantic database", "top_k": 0})

    assert response.status_code == 422
