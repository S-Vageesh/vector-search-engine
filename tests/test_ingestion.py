from pathlib import Path

from src.documents import TextFileLoader
from src.documents import TextDocument
from src.ingestion import DocumentIngestionPipeline
from src.repository import InMemoryDocumentRepository


class FakeEmbedder:
    model_name = "fake-model"

    def embed(self, text: str) -> list[float]:
        assert text == "semantic search"
        return [0.1, 0.2, 0.3]


class FakeVectorIndex:
    def __init__(self) -> None:
        self.insertions: list[tuple[int, list[float]]] = []

    def add(self, document_id: int, embedding: list[float]) -> None:
        self.insertions.append((document_id, embedding))


def test_ingestion_pipeline_embeds_and_stores_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("semantic search", encoding="utf-8")
    repository = InMemoryDocumentRepository()
    pipeline = DocumentIngestionPipeline(
        loader=TextFileLoader(),
        embedder=FakeEmbedder(),
        repository=repository,
    )

    document_id = pipeline.ingest_file(file_path)

    assert document_id == 1
    assert repository.saved[0]["document"].content == "semantic search"
    assert repository.saved[0]["embedding"] == [0.1, 0.2, 0.3]
    assert repository.saved[0]["model_name"] == "fake-model"


def test_ingestion_pipeline_updates_incremental_vector_index(tmp_path: Path) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("semantic search", encoding="utf-8")
    repository = InMemoryDocumentRepository()
    vector_index = FakeVectorIndex()
    pipeline = DocumentIngestionPipeline(
        loader=TextFileLoader(),
        embedder=FakeEmbedder(),
        repository=repository,
        vector_index=vector_index,
    )

    document_id = pipeline.ingest_file(file_path)

    assert document_id == 1
    assert vector_index.insertions == [(1, [0.1, 0.2, 0.3])]


def test_ingestion_pipeline_accepts_document_objects() -> None:
    repository = InMemoryDocumentRepository()
    pipeline = DocumentIngestionPipeline(
        loader=TextFileLoader(),
        embedder=FakeEmbedder(),
        repository=repository,
    )

    document_id = pipeline.ingest_document(
        TextDocument(source_path="upload://document.txt", content="semantic search")
    )

    assert document_id == 1
    assert repository.saved[0]["document"].source_path == "upload://document.txt"
