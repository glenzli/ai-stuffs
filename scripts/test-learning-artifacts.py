"""Reload the saved learning checkpoints and recompute every displayed split."""
import hashlib
import json
from pathlib import Path
import unittest

import torch
import learning_core as core

ROOT = Path(__file__).resolve().parents[1]


class SavedLearningAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.configure_cpu(1)
        manifest = json.loads((ROOT / "src/experiment-sources.json").read_text())
        cls.entry = manifest["experiments"]["learning"]
        cls.path = ROOT / cls.entry["report"]["path"]
        cls.report = json.loads(cls.path.read_text())
        cls.splits = json.loads((cls.path.parent / "splits.json").read_text())

    def test_saved_files_sources_and_frozen_budget(self):
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.entry["report"]["sha256"])
        for file in self.report["provenance"]["files"]:
            self.assertEqual(hashlib.sha256((ROOT / file["path"]).read_bytes()).hexdigest(), file["sha256"], file["path"])
        for source in self.report["provenance"]["sources"]:
            self.assertEqual(hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(), source["sha256"])
        self.assertEqual(core.split_digest(self.splits), self.report["task"]["split_sha256"])
        self.assertEqual(self.report["frozen"]["seeds"], [11, 29, 47])
        self.assertEqual(self.report["frozen"]["training"]["steps"], 1200)
        self.assertEqual(self.report["frozen"]["training"]["checkpoint_selection"], "final fixed-budget step; no selection on test or length")

    def test_all_initial_and_final_weights_reproduce_every_metric_and_prediction(self):
        for run in self.report["runs"]:
            for phase, checkpoint in [("initial", "initial"), ("learned", "final")]:
                saved = torch.load(ROOT / run["checkpoints"][checkpoint], map_location="cpu", weights_only=True)
                model = core.make_model(seed=run["seed"], **saved["model_config"])
                model.load_state_dict(saved["model"])
                for split, records in self.splits.items():
                    with self.subTest(seed=run["seed"], phase=phase, split=split):
                        self.assertEqual(core.evaluate_records(model, records), run[phase][split])

    def test_saved_new_process_resume_proofs_cover_full_state_and_trace(self):
        for run in self.report["runs"]:
            resume = json.loads((self.path.parent / f"seed-{run['seed']}-resume.json").read_text())
            self.assertTrue(resume["fresh_process"])
            self.assertTrue(resume["passed"])
            self.assertEqual((resume["from_step"], resume["to_step"]), (600, 1200))
            self.assertEqual(set(resume["comparisons"]), {"model", "optimizer", "stream", "rng", "progress", "trace", "evaluation"})
            for check in resume["comparisons"].values():
                self.assertTrue(check["equal"])
                self.assertEqual(check["max_error"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
