"""Actual Faiss CPU indexes; separate toy equivalence and measured queries."""
import argparse
import datetime
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from zoneinfo import ZoneInfo

import faiss
import numpy as np


def write(path, data):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.out_dir).resolve()
    fixture = json.loads((directory / "fixture.json").read_text())
    faiss.omp_set_num_threads(1)
    X = np.asarray([doc["vector"] for doc in fixture["inputs"]["documents"]], dtype=np.float32)
    centers = np.asarray(fixture["inputs"]["centroids"], dtype=np.float32)
    quantizer = faiss.IndexFlatL2(2)
    quantizer.add(centers)
    ivf = faiss.IndexIVFFlat(quantizer, 2, 3, faiss.METRIC_L2)
    assert ivf.is_trained, "preset quantizer must already contain all centroids"
    ivf.add(X)
    flat = faiss.IndexFlatL2(2)
    flat.add(X)
    toy = []
    for case in fixture["cases"]:
        q = np.asarray([fixture["inputs"]["queries"][case["query"]]], dtype=np.float32)
        ivf.nprobe = case["nprobe"]
        exact_d, exact_i = flat.search(q, len(X))
        approx_d, approx_i = ivf.search(q, len(X))
        def canonical(ids, distances):
            return sorted([(fixture["inputs"]["documents"][int(i)]["id"], float(d)) for i, d in zip(ids[0], distances[0]) if i >= 0], key=lambda pair: (pair[1], pair[0]))[:3]
        exact, approx = canonical(exact_i, exact_d), canonical(approx_i, approx_d)
        assert [x[0] for x in exact] == case["exact"], case
        assert [x[0] for x in approx] == case["approximate"], case
        np.testing.assert_allclose([x[1] for x in approx], case["distances"], atol=1e-6, rtol=1e-6)
        toy.append({**case, "actual": approx})
    faiss.write_index(ivf, str(directory / "teaching-ivf.index"))

    rng = np.random.default_rng(20261028)
    n, d, nq, k, nlist = 8192, 32, 128, 10, 64
    vectors = rng.standard_normal((n, d), dtype=np.float32)
    queries = rng.standard_normal((nq, d), dtype=np.float32)
    flat = faiss.IndexFlatL2(d)
    flat.add(vectors)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.cp.seed = 73
    index.train(vectors)
    index.add(vectors)
    _, reference = flat.search(queries, k)
    faiss.write_index(flat, str(directory / "flat.index"))
    faiss.write_index(index, str(directory / "ivf.index"))
    np.savez(directory / "data.npz", vectors=vectors, queries=queries)
    records = []
    for name, candidate, nprobe in [("Flat", flat, None)] + [("IVFFlat", index, value) for value in [1, 4, 16, 64]]:
        if nprobe is not None:
            candidate.nprobe = nprobe
        for _ in range(3):
            candidate.search(queries, k)
        times = []
        for _ in range(12):
            before = time.perf_counter_ns()
            _, ids = candidate.search(queries, k)
            times.append((time.perf_counter_ns() - before) / 1e6)
        recall = float(np.mean([len(set(actual) & set(expected)) / k for actual, expected in zip(ids, reference)]))
        if nprobe == nlist or name == "Flat":
            assert recall == 1, "all-list search must recover the exact top-k on this fixed continuous dataset"
        distance_count = None
        if nprobe is not None:
            _, lists = quantizer.search(queries, nprobe)
            sizes = [index.invlists.list_size(i) for i in range(nlist)]
            distance_count = int(sum(sizes[int(list_id)] for row in lists for list_id in row))
        records.append({"index": name, "nprobe": nprobe, "recall_at_10": recall, "query_batch": nq, "median_batch_ms": statistics.median(times), "min_batch_ms": min(times), "max_batch_ms": max(times), "raw_batch_ms": times, "document_distances": distance_count if distance_count is not None else n * nq, "serialized_bytes": (directory / ("flat.index" if name == "Flat" else "ivf.index")).stat().st_size})
    reloaded = faiss.read_index(str(directory / "ivf.index"))
    reloaded.nprobe = nlist
    _, reloaded_ids = reloaded.search(queries, k)
    np.testing.assert_array_equal(reloaded_ids, reference)
    environment = {"python": sys.version, "faiss": faiss.__version__, "numpy": np.__version__, "platform": platform.platform(), "threads": 1, "dtype": "float32", "faiss_compile_options": faiss.get_compile_options()}
    report = {"schema": 1, "kind": "faiss-cpu-index", "success": True, "created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "environment": environment, "teaching_cases": toy,
              "benchmark": {"vectors": n, "dimension": d, "queries": nq, "k": k, "nlist": nlist, "data_seed": 20261028, "kmeans_seed": 73, "warmup": 3, "repetitions": 12, "metric": "squared-L2"}, "measurements": records,
              "scope": "CPU batch query timing, excluding index training/build and disk I/O. Fixed synthetic vectors, no encoder or task-quality measurement. File sizes are serialized indexes, not resident memory."}
    write(directory / "environment.json", environment)
    write(directory / "report.json", report)
    print(json.dumps({"teaching_cases": len(toy), "measurements": records, "report": str(directory / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
