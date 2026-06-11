from benchmarks.benchmark_hnsw import run_benchmark


def test_benchmark_reports_latency_and_recall() -> None:
    results = run_benchmark(documents=30, queries=5, dimension=8, top_k=3)

    names = {result.name for result in results}
    assert names == {"brute_force_cosine", "hnsw"}
    assert all(result.latency_ms >= 0 for result in results)
    assert all(0 <= result.recall_at_k <= 1 for result in results)
