from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

from src.documents import TextDocument
from src.repository import InMemoryDocumentRepository
from src.vector_index import HnswSearchRepository


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    latency_ms: float
    recall_at_k: float


def _random_unit_vector(dimension: int) -> list[float]:
    values = [random.gauss(0, 1) for _ in range(dimension)]
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values]


def _recall(expected: list[int], actual: list[int], k: int) -> float:
    return len(set(expected[:k]) & set(actual[:k])) / k


def run_benchmark(
    documents: int = 1000,
    queries: int = 100,
    dimension: int = 64,
    top_k: int = 10,
    seed: int = 42,
) -> list[BenchmarkResult]:
    random.seed(seed)
    repository = InMemoryDocumentRepository()
    for index in range(documents):
        repository.save(
            TextDocument(source_path=f"doc-{index}.txt", content=f"document {index}"),
            _random_unit_vector(dimension),
            "benchmark",
        )

    hnsw_repository = HnswSearchRepository.build_from_repository(
        repository,
        dimension=dimension,
    )
    query_vectors = [_random_unit_vector(dimension) for _ in range(queries)]

    brute_force_results = []
    brute_force_start = time.perf_counter()
    for query_vector in query_vectors:
        brute_force_results.append(repository.search(query_vector, top_k))
    brute_force_latency = (time.perf_counter() - brute_force_start) * 1000 / queries

    hnsw_results = []
    hnsw_start = time.perf_counter()
    for query_vector in query_vectors:
        hnsw_results.append(hnsw_repository.search(query_vector, top_k))
    hnsw_latency = (time.perf_counter() - hnsw_start) * 1000 / queries

    recalls = [
        _recall(
            [result.document_id for result in brute_force_result],
            [result.document_id for result in hnsw_result],
            top_k,
        )
        for brute_force_result, hnsw_result in zip(brute_force_results, hnsw_results)
    ]

    return [
        BenchmarkResult(
            name="brute_force_cosine",
            latency_ms=brute_force_latency,
            recall_at_k=1.0,
        ),
        BenchmarkResult(
            name="hnsw",
            latency_ms=hnsw_latency,
            recall_at_k=sum(recalls) / len(recalls),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    results = run_benchmark(
        documents=args.documents,
        queries=args.queries,
        dimension=args.dimension,
        top_k=args.top_k,
    )
    for result in results:
        print(
            f"{result.name}: "
            f"latency_ms={result.latency_ms:.4f}, "
            f"recall@{args.top_k}={result.recall_at_k:.4f}"
        )


if __name__ == "__main__":
    main()
