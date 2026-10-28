"""Read-only audit of saved real runs; no training or replay process is started."""
import json
import os
from pathlib import Path
import unittest

import torch

from failure_replay import CASES, VARIANTS, digest, file_digest, recompute_loss, scalar, tensor_summary

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = Path(os.environ.get("FAILURE_REPORT_DIR", ROOT / "artifacts/failure-lab-2026-10-28T03-56-27-850Z"))


class SavedActualFailureAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((DIRECTORY / "report.json").read_text(),
                                parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
        cls.checkpoint = torch.load(DIRECTORY / "pre-step.pt", map_location="cpu", weights_only=False)

    def test_provenance_checks_and_pending_step(self):
        r = self.report
        self.assertEqual(r["schema"], "failure-replay-v1"); self.assertEqual(r["status"], "passed")
        self.assertEqual(len(r["checks"]), 115); self.assertTrue(all(row["passed"] for row in r["checks"]))
        self.assertEqual(file_digest(DIRECTORY / "pre-step.pt"), r["provenance"]["checkpoint_sha256"])
        self.assertEqual(digest(self.checkpoint["state"]), r["provenance"]["pre_step_digest"])
        self.assertEqual(self.checkpoint["state"]["counters"]["updates"], 8)
        self.assertEqual(r["provenance"]["saved_after_updates"], 8); self.assertEqual(r["provenance"]["pending_step"], 9)

    def test_loss_and_18_real_traces_from_saved_tensors(self):
        pids = []
        for case in self.report["cases"]:
            for selection in ("target", "control"):
                for variant in VARIANTS:
                    with self.subTest(case=case["id"], selection=selection, variant=variant):
                        run = case["runs"][f"{selection}-{variant}"]
                        raw = torch.load(DIRECTORY / run["artifacts"]["tensors"], map_location="cpu", weights_only=False)
                        pids.append(run["pid"])
                        self.assertEqual(run["loaded_pre_step_digest"], self.checkpoint["state_digest"])
                        loss = recompute_loss(raw["logits"], raw["labels"], run["shared_logit_shift"], case["id"] == "overflow" and run["fault_enabled"])
                        self.assertEqual(scalar(loss.item())["status"], run["loss"]["status"])
                        if bool(torch.isfinite(loss)): self.assertAlmostEqual(loss.item(), run["loss"]["value"], places=5)
                        for event in run["events"]: self.assertEqual(tensor_summary(raw["events"][event["key"]]), event["stats"])
        self.assertEqual(len(pids), 18); self.assertEqual(len(set(pids)), 18)

    def test_repairs_and_controls_are_binary_state_and_tensor_equal(self):
        for case in CASES:
            for selection in ("target", "control"):
                baseline = DIRECTORY / case / f"{selection}-baseline"
                baseline_state = torch.load(baseline / "final-state.pt", map_location="cpu", weights_only=False)
                baseline_tensors = torch.load(baseline / "tensors.pt", map_location="cpu", weights_only=False)
                for variant in (("repaired",) if selection == "target" else ("faulty", "repaired")):
                    with self.subTest(case=case, selection=selection, variant=variant):
                        output = DIRECTORY / case / f"{selection}-{variant}"
                        actual_state = torch.load(output / "final-state.pt", map_location="cpu", weights_only=False)
                        actual_tensors = torch.load(output / "tensors.pt", map_location="cpu", weights_only=False)
                        self.assertEqual(digest(actual_state), digest(baseline_state))
                        self.assertEqual(digest(actual_tensors), digest(baseline_tensors))

    def test_three_fault_types_and_probe_scope(self):
        by_id = {item["id"]: item for item in self.report["cases"]}
        overflow = by_id["overflow"]["runs"]["target-faulty"]
        self.assertEqual(overflow["first_forward_nonfinite"]["name"], "loss.exp")
        self.assertEqual(overflow["first_forward_nonfinite"]["stats"]["positive_infinity_count"], 768)
        self.assertEqual(overflow["loss"], {"value": None, "status": "nan"})
        high = by_id["high-lr"]["runs"]["target-faulty"]
        self.assertTrue(high["all_finite"]); self.assertTrue(high["update"]["above_threshold"])
        self.assertFalse(by_id["high-lr"]["anomaly_probe"]["optimizer_update_executed"])
        labels = by_id["labels"]["runs"]["target-faulty"]
        self.assertTrue(labels["all_finite"]); self.assertEqual(labels["supervised_count"], 44)
        self.assertEqual(labels["labels_changed"], 36)
        for gold_row, actual_row in zip(labels["batch"]["labels"], labels["actual_labels"]):
            for gold, actual in zip(gold_row, actual_row): self.assertEqual(actual, 4 + (gold - 4 + 1) % 4 if gold >= 4 else gold)


if __name__ == "__main__":
    unittest.main(verbosity=2)
