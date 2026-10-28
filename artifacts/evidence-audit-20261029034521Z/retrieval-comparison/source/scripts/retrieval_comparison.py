"""Explore candidate union and paired sensitivity using unchanged real retrieval outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from paired_evidence import paired_rows


def digest(file): return hashlib.sha256(Path(file).read_bytes()).hexdigest()
def save(file, data): Path(file).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fuse_lists(lexical, vector, constant=60):
    if constant <= 0 or len(set(lexical)) != len(lexical) or len(set(vector)) != len(vector):
        raise ValueError("Invalid rank lists or RRF constant")
    ranks = [{item: i + 1 for i, item in enumerate(rows)} for rows in (lexical, vector)]
    candidates = set(lexical) | set(vector)
    result = [{"passage_id": item, "bm25_rank": ranks[0].get(item), "flat_rank": ranks[1].get(item),
               "score": sum(1 / (constant + rank[item]) for rank in ranks if item in rank)} for item in candidates]
    return sorted(result, key=lambda row: (-row["score"], row["passage_id"]))


def metrics(ids, gold):
    rank = next((i + 1 for i, item in enumerate(ids[:10]) if item in gold), None)
    return {"hit_at_1": int(rank is not None and rank <= 1), "hit_at_5": int(rank is not None and rank <= 5),
            "hit_at_10": int(rank is not None), "mrr_at_10": 0. if rank is None else 1 / rank}


def run(directory, source_file, expected_sha):
    if digest(source_file) != expected_sha: raise ValueError("Accepted retrieval source changed")
    out = Path(directory); out.mkdir(parents=True, exist_ok=False)
    frozen = {"candidate_depth_per_method": 10, "return_depth": 10, "rrf_constant": 60,
              "missing_rank_contribution": 0, "tie_break": "passage SHA-256 ascending", "repetitions": 2000, "random_seed": 73,
              "analysis_status": "exploratory extension of an already inspected dataset; no fresh held-out confirmation",
              "comparison": "fixed BM25, fixed BGE Flat, and union of their saved top-10 lists; no parameter search",
              "old_reranker": "prior candidate-only RRF used global BM25 ranks; this new truncated-list fusion is a separate method"}
    save(out / "frozen-config.json", frozen)
    shutil.copyfile(source_file, out / "retrieval-source.json"); source = json.loads((out / "retrieval-source.json").read_text())
    observations = []
    for query in source["queries"]:
        lex = [r["passage_id"] for r in query["methods"]["bm25"]["results"]]
        vec = [r["passage_id"] for r in query["methods"]["flat"]["results"]]
        if len(lex) != 10 or len(vec) != 10: raise ValueError("Incomplete saved top-10 inputs")
        gold = {r["passage_id"] for r in query["gold"]}; fused = fuse_lists(lex, vec)
        for row in fused: row["designated_evidence"] = row["passage_id"] in gold
        methods = {name: query["methods"][name]["metrics"] for name in ("bm25", "flat")}
        methods["hybrid"] = metrics([r["passage_id"] for r in fused], gold)
        observations.append({k: query[k] for k in ("id", "split", "group", "kind", "query", "gold") } | {
            "bm25_ids": lex, "flat_ids": vec, "candidates": fused, "candidate_count": len(fused),
            "candidate_coverage": int(bool(gold & (set(lex) | set(vec)))),
            "bm25_only_evidence": int(bool(gold & (set(lex) - set(vec)))),
            "flat_only_evidence": int(bool(gold & (set(vec) - set(lex)))), "methods": methods})
    if {r["group"] for r in observations if r["split"] == "test"} & {r["group"] for r in observations if r["split"] == "validation"}:
        raise ValueError("Question source groups overlap")
    summaries = []; comparisons = []; distributions = {}; exports = {}
    for split in ("test", "validation"):
        for kind in ("all", "keyword", "paraphrase"):
            rows = [r for r in observations if r["split"] == split and (kind == "all" or r["kind"] == kind)]
            summaries.append({"split": split, "kind": kind, "questions": len(rows), "groups": len({r["group"] for r in rows}),
                              "candidate_coverage": sum(r["candidate_coverage"] for r in rows) / len(rows),
                              "mean_candidates": sum(r["candidate_count"] for r in rows) / len(rows),
                              "bm25_only_evidence": sum(r["bm25_only_evidence"] for r in rows),
                              "flat_only_evidence": sum(r["flat_only_evidence"] for r in rows),
                              "coverage_lost_after_ranking": sum(r["candidate_coverage"] - r["methods"]["hybrid"]["hit_at_10"] for r in rows),
                              "methods": {method: {key: sum(r["methods"][method][key] for r in rows) / len(rows)
                                                   for key in ("hit_at_1", "hit_at_5", "hit_at_10", "mrr_at_10")}
                                          for method in ("bm25", "flat", "hybrid")}})
        rows = [r for r in observations if r["split"] == split]
        for old in ("bm25", "flat"):
            for metric in ("hit_at_10", "mrr_at_10"):
                paired = [{"id": r["id"], "group": r["group"], "old": r["methods"][old][metric], "new": r["methods"]["hybrid"][metric]} for r in rows]
                key = f"{split}-{old}-{metric}"
                exports[key] = {"schema": 1, "rows": paired}
                save(out / f"paired-{key}.json", exports[key])
                analyses = {}
                for mode in ("group", "row"):
                    analysis = paired_rows(paired, mode); distributions[f"{key}-{mode}"] = analysis.pop("distribution"); analyses[mode] = analysis
                comparisons.append({"split": split, "baseline": old, "metric": metric, "analyses": analyses})
    save(out / "bootstrap-distributions.json", distributions)
    report = {"schema": "retrieval-comparison-v1", "status": "passed", "frozen": frozen,
              "source": {"report_path": str(source_file), "report_sha256": expected_sha, "model": source["model"], "dataset": source["dataset"]},
              "queries": observations, "summaries": summaries, "comparisons": comparisons, "paired_exports": exports,
              "passages": source["passages"], "limits": ["same repository-authored questions with non-exhaustive designated evidence", "already inspected test questions: exploratory mechanism comparison only",
                  "whole-source-group percentile ranges assume representative independent groups; 12 test groups cannot establish production reliability",
                  "candidate coverage is an upper bound; top-10 ranking can discard covered evidence", "candidate-list sizes are not latency, memory, FLOPs, or equal retrieval cost"]}
    save(out / "report.json", report); return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out-dir", required=True); p.add_argument("--source", required=True); p.add_argument("--source-sha256", required=True)
    a = p.parse_args(); run(a.out_dir, a.source, a.source_sha256)
