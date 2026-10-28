"""Replay source filters on frozen real embeddings; no encoder or ANN is run."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np


FROZEN = {
    "k": 10,
    "pool_depths": [10, 30, 100, 877],
    "score": "float64 inner product of the saved, already normalized float32 vectors",
    "tie_break": "descending score, then ascending full passage SHA-256",
    "filters": [
        {"id": "all", "title": "全部原始段落", "sources": None},
        {"id": "artifacts", "title": "仅工件与恢复文档", "sources": ["model-artifacts"]},
        {"id": "training", "title": "仅三篇训练与优化文档", "sources": ["training", "training-deep", "optimization-dynamics"]},
        {"id": "small", "title": "仅工件文档的前三段", "sources": ["model-artifacts"], "paragraph_lt": 3},
        {"id": "empty", "title": "不存在的来源（空范围）", "sources": ["__missing_source__"]},
    ],
    "selection": "Explicit synthetic source predicates, shared by all 36 queries, frozen before scoring; exploratory follow-up to already inspected data. No query-specific label-based filter or parameter selection.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def allowed(passage, condition):
    return ((condition["sources"] is None or passage["source_document"] in condition["sources"])
            and ("paragraph_lt" not in condition or passage["paragraph"] < condition["paragraph_lt"]))


def measure(result_indices, reference, eligible_gold, gold_indices):
    denominator = len(reference)
    return {
        "count": len(result_indices),
        "fill": len(result_indices) / denominator if denominator else None,
        "neighbor_recall": len(set(result_indices) & set(reference)) / denominator if denominator else None,
        "evidence_hit": int(bool(set(result_indices) & gold_indices)) if eligible_gold else None,
    }


def run(out, source, source_sha256):
    out.mkdir()
    save(out / "frozen-config.json", FROZEN)
    inputs = out / "inputs"
    inputs.mkdir()
    for name in ["report.json", "passages.json", "queries.json", "embeddings.npz"]:
        shutil.copyfile(source / name, inputs / name)
    if sha(inputs / "report.json") != source_sha256:
        raise ValueError("Original retrieval report identity mismatch")
    report = json.loads((inputs / "report.json").read_text())
    passages = json.loads((inputs / "passages.json").read_text())
    queries = json.loads((inputs / "queries.json").read_text())["queries"]
    if passages != report["passages"] or len(passages) != 877 or len(queries) != 36:
        raise ValueError("Original passage or query order changed")
    for query, recorded in zip(queries, report["queries"], strict=True):
        if any(query[key] != recorded[key] for key in ["id", "query", "gold", "split", "group", "kind"]):
            raise ValueError("Original query/annotation identity mismatch")
    with np.load(inputs / "embeddings.npz", allow_pickle=False) as vectors:
        x, q = vectors["passages"], vectors["queries"]
    if x.shape != (877, 512) or q.shape != (36, 512) or x.dtype != np.float32 or q.dtype != np.float32:
        raise ValueError("Unexpected saved embedding contract")
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError("Non-finite saved embeddings")
    if max(np.max(abs(np.linalg.norm(x, axis=1) - 1)), np.max(abs(np.linalg.norm(q, axis=1) - 1))) > 1e-5:
        raise ValueError("Saved vectors are not normalized")
    scores = q.astype(np.float64) @ x.astype(np.float64).T
    np.save(out / "all-scores.npy", scores, allow_pickle=False)
    ranks = [sorted(range(len(passages)), key=lambda j: (-float(row[j]), passages[j]["id"])) for row in scores]
    np.save(out / "all-ranks.npy", np.asarray(ranks, dtype=np.int32), allow_pickle=False)
    index = {p["id"]: j for j, p in enumerate(passages)}
    if len(index) != len(passages):
        raise ValueError("Duplicate passage identities")
    requests = []
    for query_index, (query, ranking) in enumerate(zip(queries, ranks, strict=True)):
        global_rank = {j: n + 1 for n, j in enumerate(ranking)}
        gold = {index[g["passage_id"]] for g in query["gold"]}
        def records(indices):
            return [{"i": j, "score": float(scores[query_index, j]), "global_rank": global_rank[j]} for j in indices]
        for condition in FROZEN["filters"]:
            permitted = {j for j, p in enumerate(passages) if allowed(p, condition)}
            reference = [j for j in ranking if j in permitted][:FROZEN["k"]]
            eligible_gold = len(gold & permitted)
            post = []
            for depth in FROZEN["pool_depths"]:
                pool = [j for j in ranking[:depth] if j in permitted]
                selected = pool[:FROZEN["k"]]
                if len(set(selected)) != len(selected) or not set(selected) <= permitted:
                    raise AssertionError("Invalid filtered output")
                post.append({"depth": depth, "pool_eligible": len(pool), "results": records(selected),
                             **measure(selected, reference, eligible_gold, gold)})
            if [r["i"] for r in post[-1]["results"]] != reference:
                raise AssertionError("Full-depth post-filter differs from pre-filter reference")
            requests.append({"query_index": query_index, "filter_id": condition["id"], "allowed_count": len(permitted),
                             "eligible_gold_count": eligible_gold, "reference": records(reference),
                             "reference_metrics": measure(reference, reference, eligible_gold, gold), "post": post})
    summaries = []
    for condition in FROZEN["filters"]:
        rows = [r for r in requests if r["filter_id"] == condition["id"]]
        for depth in FROZEN["pool_depths"]:
            selected = [next(p for p in row["post"] if p["depth"] == depth) for row in rows]
            eligible = [(row, p) for row, p in zip(rows, selected, strict=True) if row["eligible_gold_count"]]
            nonempty = [p for p in selected if p["fill"] is not None]
            summaries.append({"filter_id": condition["id"], "depth": depth, "queries": len(rows),
                              "allowed_count": rows[0]["allowed_count"], "evidence_eligible_queries": len(eligible),
                              "shortfall_queries": sum(p["count"] < len(row["reference"]) for row, p in zip(rows, selected, strict=True)),
                              "mean_fill": sum(p["fill"] for p in nonempty) / len(nonempty) if nonempty else None,
                              "evidence_hits": sum(p["evidence_hit"] for _, p in eligible),
                              "evidence_hit_rate": sum(p["evidence_hit"] for _, p in eligible) / len(eligible) if eligible else None})
    result = {"schema": "retrieval-filtering-v1", "status": "passed", "frozen": FROZEN,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(), "device": "cpu"},
              "input_sha256": {p.name: sha(p) for p in sorted(inputs.iterdir())},
              "original_model": report["model"], "original_dataset": report["dataset"],
              "passages": passages, "queries": queries, "requests": requests, "summaries": summaries,
              "correctness": {"full_depth_matches_all_references": True, "request_count": len(requests), "post_filter_runs": len(requests) * len(FROZEN["pool_depths"])},
              "limits": ["Uses saved real embeddings; no new encoding, training, ANN index, vector-database selector, latency or access-control implementation.",
                         "Filters are explicit teaching predicates on source metadata, shared by all queries; labels and queries unchanged.",
                         "Evidence metrics condition on an available designated evidence passage; an unavailable label is not a retrieval failure or a proof that all answers are unsupported.",
                         "One designated evidence passage is not exhaustive relevance annotation; no production retrieval-quality generalization."]}
    save(out / "report.json", result)
    print(json.dumps({"status": "passed", "requests": len(requests), "post_filter_runs": result["correctness"]["post_filter_runs"], "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--retrieval-directory", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    run(args.out_dir, args.retrieval_directory, args.source_sha256)
