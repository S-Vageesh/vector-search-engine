from fastapi.testclient import TestClient

from src.api import create_app
from src.documents import TextDocument
from src.repository import SearchResult


class FakeSearchService:
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        assert query == "semantic database"
        assert top_k == 2
        return [
            SearchResult(
                document_id=7,
                filename="notes.txt",
                text="semantic search document",
                similarity=0.92,
            )
        ]

    def count_documents(self) -> int:
        return 12


class FakeIngestionPipeline:
    def __init__(self) -> None:
        self.documents: list[TextDocument] = []

    def ingest_document(self, document: TextDocument) -> int:
        self.documents.append(document)
        return len(self.documents)


def test_search_endpoint_returns_document_text_and_similarity() -> None:
    app = create_app(service_factory=lambda: FakeSearchService())
    client = TestClient(app)

    response = client.post(
        "/search",
        json={"query": "semantic database", "top_k": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["results"] == [
        {
            "document_id": 7,
            "filename": "notes.txt",
            "text": "semantic search document",
            "similarity": 0.92,
        }
    ]
    assert body["document_count"] == 12
    assert body["latency_ms"] >= 0


def test_search_endpoint_validates_top_k() -> None:
    app = create_app(service_factory=lambda: FakeSearchService())
    client = TestClient(app)

    response = client.post("/search", json={"query": "semantic database", "top_k": 0})

    assert response.status_code == 422


def test_upload_endpoint_ingests_single_text_file_from_legacy_file_field() -> None:
    pipeline = FakeIngestionPipeline()
    app = create_app(
        service_factory=lambda: FakeSearchService(),
        ingestion_factory=lambda: pipeline,
    )
    client = TestClient(app)

    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"semantic upload", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "files_processed": [
            {
                "document_id": 1,
                "filename": "notes.txt",
                "size_bytes": 15,
                "error": None,
            }
        ],
        "files_failed": [],
        "total_documents_ingested": 1,
    }
    assert pipeline.documents[0].content == "semantic upload"
    assert pipeline.documents[0].metadata["source"] == "upload"


def test_upload_endpoint_ingests_multiple_text_files() -> None:
    pipeline = FakeIngestionPipeline()
    app = create_app(
        service_factory=lambda: FakeSearchService(),
        ingestion_factory=lambda: pipeline,
    )
    client = TestClient(app)

    response = client.post(
        "/upload",
        files=[
            ("files", ("alpha.txt", b"alpha text", "text/plain")),
            ("files", ("beta.txt", b"beta text", "text/plain")),
        ],
    )

    assert response.status_code == 200
    assert response.json() == {
        "files_processed": [
            {
                "document_id": 1,
                "filename": "alpha.txt",
                "size_bytes": 10,
                "error": None,
            },
            {
                "document_id": 2,
                "filename": "beta.txt",
                "size_bytes": 9,
                "error": None,
            },
        ],
        "files_failed": [],
        "total_documents_ingested": 2,
    }
    assert [document.content for document in pipeline.documents] == [
        "alpha text",
        "beta text",
    ]


def test_upload_endpoint_reports_failed_files_without_rejecting_batch() -> None:
    pipeline = FakeIngestionPipeline()
    app = create_app(
        service_factory=lambda: FakeSearchService(),
        ingestion_factory=lambda: pipeline,
    )
    client = TestClient(app)

    response = client.post(
        "/upload",
        files=[
            ("files", ("notes.txt", b"semantic upload", "text/plain")),
            ("files", ("notes.md", b"semantic upload", "text/markdown")),
        ],
    )

    assert response.status_code == 200
    assert response.json() == {
        "files_processed": [
            {
                "document_id": 1,
                "filename": "notes.txt",
                "size_bytes": 15,
                "error": None,
            }
        ],
        "files_failed": [
            {
                "document_id": None,
                "filename": "notes.md",
                "size_bytes": None,
                "error": "Only .txt files are supported",
            }
        ],
        "total_documents_ingested": 1,
    }
