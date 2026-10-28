"""Independent scalar inner products and restricted heap selection of saved inputs."""
import hashlib
import heapq
import json
import math
import unittest
from pathlib import Path

import numpy as np


class FilteringEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(Path("src/experiment-sources.json").read_text())
        cls.entry = cls.manifest["experiments"]["retrieval-filtering"]
        cls.root = Path(cls.entry["bundle_root"])
        cls.directory = Path(cls.entry["report"]["path"]).parent
        cls.report = json.loads((cls.directory / "report.json").read_text())
        cls.passages, cls.queries = cls.report["passages"], cls.report["queries"]
        with np.load(cls.directory / "inputs/embeddings.npz", allow_pickle=False) as z:
            cls.x, cls.q = z["passages"].astype(float), z["queries"].astype(float)
        # Each scalar sum uses Python's compensated summation, not matrix multiply.
        x, q = cls.x.tolist(), cls.q.tolist()
        cls.scalar = [[math.fsum(a * b for a, b in zip(query, passage, strict=True)) for passage in x] for query in q]
        cls.order = [sorted(range(len(cls.passages)), key=lambda i: (-scores[i], cls.passages[i]["id"])) for scores in cls.scalar]
        cls.allowed = {
            "all": set(range(877)),
            "artifacts": {i for i, p in enumerate(cls.passages) if p["source_document"] == "model-artifacts"},
            "training": {i for i, p in enumerate(cls.passages) if p["source_document"] in {"training", "training-deep", "optimization-dynamics"}},
            "small": {i for i, p in enumerate(cls.passages) if p["source_document"] == "model-artifacts" and p["paragraph"] in {0, 1, 2}},
            "empty": set(),
        }

    def test_input_lineage_and_captured_executed_sources(self):
        original = Path(self.manifest["experiments"]["retrieval"]["report"]["path"]).parent
        for name, digest in self.report["input_sha256"].items():
            data = (self.directory / "inputs" / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
            self.assertEqual(data, (original / name).read_bytes())
        environment = json.loads((self.root / "run-environment.json").read_text())
        for file, digest in environment["source_sha256"].items():
            self.assertEqual(hashlib.sha256(Path(file).read_bytes()).hexdigest(), digest)
            self.assertEqual((self.root / "source" / file).read_bytes(), Path(file).read_bytes())
        self.assertEqual(self.report["input_sha256"]["report.json"], self.manifest["experiments"]["retrieval"]["report"]["sha256"])
        self.assertEqual(json.loads((self.directory / "frozen-config.json").read_text()), self.report["frozen"])

    def test_every_saved_inner_product_matches_independent_scalar_summation(self):
        saved = np.load(self.directory / "all-scores.npy", allow_pickle=False)
        self.assertEqual(saved.shape, (36, 877))
        self.assertEqual(saved.dtype, np.float64)
        error = float(np.max(abs(saved - self.scalar)))
        self.assertLess(error, 1e-12)
        print(f"Independent scalar-dot maximum error: {error:.3g}")

    def test_all_references_and_720_post_filter_outputs_match_independent_reconstruction(self):
        np.testing.assert_array_equal(np.load(self.directory / "all-ranks.npy", allow_pickle=False), self.order)
        self.assertEqual(len(self.report["requests"]), 180)
        seen = set()
        for request in self.report["requests"]:
            qi, scope = request["query_index"], request["filter_id"]
            self.assertNotIn((qi, scope), seen)
            seen.add((qi, scope))
            permitted, scores, order = self.allowed[scope], self.scalar[qi], self.order[qi]
            key = lambda i: (-scores[i], self.passages[i]["id"])
            reference = heapq.nsmallest(10, permitted, key=key)
            self.assertEqual([row["i"] for row in request["reference"]], reference)
            self.assertEqual(request["allowed_count"], len(permitted))
            self.assertEqual(len(request["post"]), 4)
            for post in request["post"]:
                pool = [i for i in order[:post["depth"]] if i in permitted]
                actual = [row["i"] for row in post["results"]]
                self.assertEqual(actual, pool[:10])
                self.assertEqual(post["pool_eligible"], len(pool))
                self.assertEqual(len(set(actual)), len(actual))
                self.assertTrue(set(actual) <= permitted)
                for row in post["results"] + request["reference"]:
                    self.assertAlmostEqual(row["score"], scores[row["i"]], delta=1e-12)
                    self.assertEqual(row["global_rank"], order.index(row["i"]) + 1)

    def test_evidence_and_neighbor_denominators_use_the_same_filtered_range(self):
        for request in self.report["requests"]:
            permitted = self.allowed[request["filter_id"]]
            gold = {g["passage_id"] for g in self.queries[request["query_index"]]["gold"]}
            eligible = gold & {self.passages[i]["id"] for i in permitted}
            target = min(10, len(permitted))
            self.assertEqual(request["eligible_gold_count"], len(eligible))
            ref = {r["i"] for r in request["reference"]}
            for measured, results in [(request["reference_metrics"], request["reference"])] + [(p, p["results"]) for p in request["post"]]:
                ids = {r["i"] for r in results}
                self.assertEqual(measured["count"], len(ids))
                self.assertEqual(measured["fill"], len(ids) / target if target else None)
                self.assertEqual(measured["neighbor_recall"], len(ids & ref) / target if target else None)
                expected = int(bool({self.passages[i]["id"] for i in ids} & gold)) if eligible else None
                self.assertEqual(measured["evidence_hit"], expected)

    def test_summary_eligibility_and_shortfall_counts_recompute_from_all_requests(self):
        for summary in self.report["summaries"]:
            rows = [r for r in self.report["requests"] if r["filter_id"] == summary["filter_id"]]
            posts = [next(p for p in r["post"] if p["depth"] == summary["depth"]) for r in rows]
            hits = [p["evidence_hit"] for p in posts if p["evidence_hit"] is not None]
            fill = [p["fill"] for p in posts if p["fill"] is not None]
            self.assertEqual(summary["queries"], 36)
            self.assertEqual(summary["evidence_eligible_queries"], len(hits))
            self.assertEqual(summary["evidence_hits"], sum(hits))
            self.assertEqual(summary["evidence_hit_rate"], sum(hits) / len(hits) if hits else None)
            self.assertAlmostEqual(summary["mean_fill"], sum(fill) / len(fill)) if fill else self.assertIsNone(summary["mean_fill"])
            self.assertEqual(summary["shortfall_queries"], sum(p["count"] < len(r["reference"]) for r, p in zip(rows, posts, strict=True)))

    def test_observed_controls_distinguish_missing_pool_entries_small_ranges_and_excluded_labels(self):
        for request in self.report["requests"]:
            target = min(10, request["allowed_count"])
            full = request["post"][-1]
            self.assertEqual(full["results"], request["reference"])
            self.assertEqual(full["count"], target)
            self.assertEqual([p["count"] for p in request["post"]], sorted(p["count"] for p in request["post"]))
            if request["filter_id"] == "small":
                self.assertEqual(target, 3)
                self.assertEqual(full["neighbor_recall"], 1)
                self.assertIsNone(full["evidence_hit"])
            if request["filter_id"] == "empty":
                self.assertEqual(target, 0)
                self.assertTrue(all(p["fill"] is None and p["neighbor_recall"] is None for p in request["post"]))
        example = next(r for r in self.report["requests"] if r["filter_id"] == "artifacts" and self.queries[r["query_index"]]["id"] == "resume-keyword")
        self.assertEqual([p["count"] for p in example["post"]], [3, 7, 10, 10])
        self.assertTrue(all(p["evidence_hit"] == 1 for p in example["post"]))
        self.assertTrue(any(r["eligible_gold_count"] and r["reference_metrics"]["evidence_hit"] == 0 for r in self.report["requests"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
