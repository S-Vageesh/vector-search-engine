from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

@dataclass(frozen=True)
class Settings:
    database_url: str
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    search_backend: str = "brute_force"
    hnsw_index_path: str = "data/indexes/documents.hnsw"
    hnsw_max_elements: int = 100000

    @classmethod
    def from_env(cls) -> "Settings":
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is required")

        return cls(
            database_url=database_url,
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME", cls.embedding_model_name
            ),
            embedding_dimension=int(
                os.getenv("EMBEDDING_DIMENSION", str(cls.embedding_dimension))
            ),
            search_backend=os.getenv("SEARCH_BACKEND", cls.search_backend),
            hnsw_index_path=os.getenv("HNSW_INDEX_PATH", cls.hnsw_index_path),
            hnsw_max_elements=int(
                os.getenv("HNSW_MAX_ELEMENTS", str(cls.hnsw_max_elements))
            ),
        )
