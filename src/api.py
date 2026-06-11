from __future__ import annotations

from collections.abc import Callable
from pathlib import PurePath
from time import perf_counter

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import Settings
from src.documents import TextDocument, TextFileLoader
from src.embeddings import SentenceTransformerEmbedder
from src.ingestion import DocumentIngestionPipeline
from src.index import HnswSearchRepository
from src.repository import PostgresDocumentRepository
from src.search import SemanticSearchService, select_search_repository


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class SearchResponseItem(BaseModel):
    document_id: int
    filename: str
    text: str
    similarity: float


class SearchResponse(BaseModel):
    results: list[SearchResponseItem]
    document_count: int
    latency_ms: float


class UploadResponse(BaseModel):
    filename: str
    document_id: int | None = None
    size_bytes: int | None = None
    error: str | None = None


class BulkUploadResponse(BaseModel):
    files_processed: list[UploadResponse]
    files_failed: list[UploadResponse]
    total_documents_ingested: int


def build_search_service() -> SemanticSearchService:
    settings = Settings.from_env()
    repository = PostgresDocumentRepository(
        settings.database_url,
        settings.embedding_dimension,
    )
    repository.initialize_schema()
    search_repository = select_search_repository(
        backend=settings.search_backend,
        document_repository=repository,
        embedding_dimension=settings.embedding_dimension,
        hnsw_index_path=settings.hnsw_index_path,
        hnsw_max_elements=settings.hnsw_max_elements,
    )
    return SemanticSearchService(
        embedder=SentenceTransformerEmbedder(settings.embedding_model_name),
        repository=search_repository,
    )


def build_ingestion_pipeline() -> DocumentIngestionPipeline:
    settings = Settings.from_env()
    repository = PostgresDocumentRepository(
        settings.database_url,
        settings.embedding_dimension,
    )
    repository.initialize_schema()
    vector_index = None
    if settings.search_backend.strip().lower() == "hnsw":
        vector_index = HnswSearchRepository.load_or_build_from_repository(
            document_repository=repository,
            dimension=settings.embedding_dimension,
            index_path=settings.hnsw_index_path,
            max_elements=settings.hnsw_max_elements,
        )

    return DocumentIngestionPipeline(
        loader=TextFileLoader(),
        embedder=SentenceTransformerEmbedder(settings.embedding_model_name),
        repository=repository,
        vector_index=vector_index,
    )


def create_app(
    service_factory: Callable[[], SemanticSearchService] | None = None,
    ingestion_factory: Callable[[], DocumentIngestionPipeline] | None = None,
) -> FastAPI:
    app = FastAPI(title="Vector Search Engine")
    get_search_service = service_factory or build_search_service
    get_ingestion_pipeline = ingestion_factory or build_ingestion_pipeline

    @app.post("/search", response_model=SearchResponse)
    def search_documents(
        request: SearchRequest,
        service: SemanticSearchService = Depends(get_search_service),
    ) -> SearchResponse:
        started_at = perf_counter()
        try:
            results = service.search(request.query, request.top_k)
            document_count = service.count_documents()
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return SearchResponse(
            results=[
                SearchResponseItem(
                    document_id=result.document_id,
                    filename=result.filename,
                    text=result.text,
                    similarity=result.similarity,
                )
                for result in results
            ],
            document_count=document_count,
            latency_ms=round((perf_counter() - started_at) * 1000, 2),
        )

    @app.post("/upload", response_model=BulkUploadResponse)
    async def upload_document(
        files: list[UploadFile] | None = File(default=None),
        file: UploadFile | None = File(default=None),
        pipeline: DocumentIngestionPipeline = Depends(get_ingestion_pipeline),
    ) -> BulkUploadResponse:
        upload_files = [*(files or []), *([file] if file is not None else [])]
        if not upload_files:
            raise HTTPException(status_code=400, detail="At least one file is required")

        files_processed: list[UploadResponse] = []
        files_failed: list[UploadResponse] = []
        for upload_file in upload_files:
            result = await _ingest_uploaded_file(upload_file, pipeline)
            if result.error is None:
                files_processed.append(result)
            else:
                files_failed.append(result)

        return BulkUploadResponse(
            files_processed=files_processed,
            files_failed=files_failed,
            total_documents_ingested=len(files_processed),
        )

    return app


async def _ingest_uploaded_file(
    file: UploadFile,
    pipeline: DocumentIngestionPipeline,
) -> UploadResponse:
    filename = PurePath(file.filename or "").name
    if not filename.lower().endswith(".txt"):
        return UploadResponse(
            filename=filename,
            error="Only .txt files are supported",
        )

    raw_content = await file.read()
    try:
        content = raw_content.decode("utf-8")
    except UnicodeDecodeError:
        return UploadResponse(
            filename=filename,
            size_bytes=len(raw_content),
            error="Uploaded text files must be UTF-8 encoded",
        )

    document = TextDocument(
        source_path=f"upload://{filename}",
        content=content,
        metadata={
            "filename": filename,
            "extension": ".txt",
            "size_bytes": len(raw_content),
            "source": "upload",
        },
    )
    try:
        document_id = pipeline.ingest_document(document)
    except ValueError as error:
        return UploadResponse(
            filename=filename,
            size_bytes=len(raw_content),
            error=str(error),
        )

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        size_bytes=len(raw_content),
    )


app = create_app()
