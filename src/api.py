from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import Settings
from src.embeddings import SentenceTransformerEmbedder
from src.repository import PostgresDocumentRepository
from src.search import SemanticSearchService, select_search_repository


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class SearchResponseItem(BaseModel):
    document_id: int
    text: str
    similarity: float


def build_search_service() -> SemanticSearchService:
    settings = Settings.from_env()
    repository = PostgresDocumentRepository(
        settings.database_url,
        settings.embedding_dimension,
    )
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


def create_app(
    service_factory: Callable[[], SemanticSearchService] | None = None,
) -> FastAPI:
    app = FastAPI(title="Vector Search Engine")
    get_search_service = service_factory or build_search_service

    @app.post("/search", response_model=list[SearchResponseItem])
    def search_documents(
        request: SearchRequest,
        service: SemanticSearchService = Depends(get_search_service),
    ) -> list[SearchResponseItem]:
        try:
            results = service.search(request.query, request.top_k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return [
            SearchResponseItem(
                document_id=result.document_id,
                text=result.text,
                similarity=result.similarity,
            )
            for result in results
        ]

    return app


app = create_app()
