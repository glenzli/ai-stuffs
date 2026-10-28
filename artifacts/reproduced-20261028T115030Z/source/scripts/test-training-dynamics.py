"""Lightweight independent checks; never reruns width/depth/batch training."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mup
import numpy as np
import torch
from torch.nn import functional as F

import learning_core as core
import training_dynamics as lab


class DynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lab.configure_runtime()

    def test_population_and_exact_sampling_against_vector_enumeration(self):
        gradients = np.array([[1., 0.], [0., 2.], [3., -1.]])
        stats = lab.population_statistics(gradients)
        np.testing.assert_allclose(stats["mean_norm_sq"], 17 / 9)
        np.testing.assert_allclose(stats["covariance_trace"], 28 / 9)
        for size in (1, 2, 3):
            for scheme in ("with-replacement", "without-replacement", "repeated", "correlated"):
                with self.subTest(size=size, scheme=scheme):
                    if scheme == "without-replacement":
                        draws = list(itertools.combinations(range(3), size))
                    elif scheme == "repeated":
                        draws = [(i,) * size for i in range(3)]
                    else:
                        draws = list(itertools.product(range(3), repeat=size))
                    weights = [1 / len(draws)] * len(draws)
                    if scheme == "correlated":
                        weights = [weight / 2 for weight in weights] + [1 / 6] * 3
                        draws += [(i,) * size for i in range(3)]
                    mean = gradients.mean(0)
                    exact = sum(weight * np.linalg.norm(gradients[list(draw)].mean(0) - mean) ** 2
                                for weight, draw in zip(weights, draws))
                    actual = lab.enumerate_batch_distribution(stats["gram"], size, scheme=scheme)
                    self.assertAlmostEqual(actual["enumerated_trace"], exact, places=13)
                    self.assertAlmostEqual(actual["formula_trace"], exact, places=13)
                    self.assertAlmostEqual(actual["probability_sum"], 1, places=13)
        self.assertEqual(lab.batch_variance(0., 1, 1, False), 0)
        self.assertEqual(lab.batch_variance(7., 3, 3, False), 0)
        self.assertAlmostEqual(lab.batch_variance(7., 3, 8, True), 7 / 8)
        with self.assertRaises(ValueError):
            lab.batch_variance(1., 3, 4, False)
        with self.assertRaises(ValueError):
            lab.enumerate_batch_distribution(stats["gram"], 12, cap=100)

    def test_psd_and_near_zero_mean(self):
        for invalid in ([[1., 2.], [2., 1.]], [[-1.]], [[1., 1.], [0., 1.]], [[float("nan")]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                lab.gram_statistics(invalid)
        exact = lab.population_statistics([[1., -2.], [-1., 2.]])
        self.assertEqual(exact["noise_ratio_state"], "undefined-zero-mean")
        self.assertIsNone(exact["noise_ratio"])
        near = lab.population_statistics([[1.], [-1. + 2e-12]])
        self.assertEqual(near["noise_ratio_state"], "ill-conditioned-mean")
        self.assertIsNone(near["noise_ratio"])
        gram = lab.gram_statistics(np.array([[1.], [-1. + 2e-12]]) @ np.array([[1., -1. + 2e-12]]))
        self.assertIsNone(gram["noise_ratio"])
        self.assertEqual(lab.population_statistics([[0., 0.]])["covariance_trace"], 0)

    def test_author_parameter_groups_readout_and_base_equivalence(self):
        sp = lab.make_width_model(16, "sp", 11)
        mu = lab.make_width_model(16, "mup", 11)
        self.assertEqual(lab.tensor_state_error(sp.state_dict(), mu.state_dict()), 0)
        for width in (32, 128):
            model = lab.make_width_model(width, "mup", 11)
            optimizer = lab.make_width_optimizer(model, .03)
            ledger = lab.parameter_ledger(model, optimizer, .03)
            for parameter in ledger:
                multiplier = width / 16
                expected_lr = .03 / multiplier if parameter["name"] == "hidden.weight" else .03
                self.assertAlmostEqual(parameter["actual_lr"], expected_lr)
                self.assertEqual(parameter["weight_decay"], 0)
                self.assertEqual(parameter["infinite_dimensions"], 2 if parameter["name"] == "hidden.weight"
                                 else 0 if parameter["name"] == "readout.bias" else 1)
            with torch.no_grad():
                model.readout.bias.fill_(.7)
            x = torch.ones(2, width)
            expected = F.linear(x / (width / 16), model.readout.weight, model.readout.bias)
            torch.testing.assert_close(model.readout(x), expected, rtol=0, atol=0)

    def test_checkpoint_shape_restore_never_rescales_twice(self):
        model = lab.make_width_model(32, "mup", 29)
        saved = {key: value.clone() for key, value in model.state_dict().items()}
        base, delta = lab.WidthMLP(16, "mup"), lab.WidthMLP(32, "mup")
        mup.set_base_shapes(model, base, delta=delta, rescale_params=False)
        self.assertEqual(lab.tensor_state_error(model.state_dict(), saved), 0)
        with self.assertRaises(RuntimeError):
            mup.set_base_shapes(model, base, delta=delta)

    def test_shared_decoder_sequence_objective_with_unequal_lengths(self):
        splits = lab.decoder_splits()
        records = [splits["train"][0], splits["length"][0]]
        model = core.make_model(seed=11, **{**lab.DECODER_CONFIG, "width": 8, "heads": 2, "depth": 1}).double()
        batch = core.collate_records(records)
        separate = [core.forward_loss(model, core.collate_records([row]), reduction="sequence") for row in records]
        combined = core.forward_loss(model, batch, reduction="sequence")
        torch.testing.assert_close(combined, torch.stack(separate).mean(), rtol=1e-13, atol=1e-13)
        measured = lab.sequence_gradients(model, records)
        self.assertLess(measured["sequence_gradient_mean_error"], 1e-12)
        self.assertTrue(model.training)
        self.assertEqual(measured["gradients"].shape, (2, core.parameter_count(model)))

    def test_token_stream_scheduler_and_rng_detector(self):
        records = lab.decoder_splits()["train"]
        indices, state = lab.make_record_stream(records, 11, 128)
        again, again_state = lab.make_record_stream(records, 11, 128)
        self.assertEqual(indices, again)
        torch.testing.assert_close(state, again_state, rtol=0, atol=0)
        for size in lab.BATCH_SIZES:
            consumed = [i for offset in range(0, 128, size) for i in indices[offset:offset + size]]
            self.assertEqual(consumed, indices)
            self.assertEqual(sum(core.collate_records([records[i] for i in indices[offset:offset + size]])["input_tokens"]
                                 for offset in range(0, 128, size)), 1024)
        self.assertEqual(lab.token_learning_rate(.03, 0, 32768), 0)
        self.assertAlmostEqual(lab.token_learning_rate(.03, 512, 32768), .015)
        self.assertAlmostEqual(lab.token_learning_rate(.03, 1024, 32768), .03)
        self.assertAlmostEqual(lab.token_learning_rate(.03, 32768, 32768), .003)
        before = core.capture_rng_state()
        torch.rand(1)
        self.assertGreater(lab.tensor_state_error(core.capture_rng_state(), before), 0)
        core.restore_rng_state(before)
        self.assertEqual(lab.tensor_state_error(core.capture_rng_state(), before), 0)

    def test_refuses_existing_report_and_directory(self):
        with tempfile.TemporaryDirectory(prefix="dynamics-test-") as temp:
            target = Path(temp) / "report.json"
            lab.write_json(target, {"keep": True})
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                lab.write_json(target, {"keep": False})
            child = subprocess.run([sys.executable, str(Path(lab.__file__)), "--out-dir", temp,
                                    "--reanalyze", str(target)], capture_output=True, text=True)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn("FileExistsError", child.stderr)
            self.assertEqual(target.read_bytes(), original)

    def test_saved_reanalysis_and_original_provenance(self):
        source = lab.ROOT / "artifacts/dynamics-lab-20261028-01/report.json"
        report = json.loads((lab.ROOT / "artifacts/dynamics-reanalysis-20261028-02/report.json").read_text())
        checks = json.loads((lab.ROOT / "artifacts/dynamics-reanalysis-20261028-02/reanalysis-checks.json").read_text())
        self.assertEqual(lab.digest_bytes(source.read_bytes()), "587611efd52b00d9b0209af948bc72a54eb92075eb3aebea6e2d14bf9bfff645")
        self.assertTrue(report["provenance"]["no_full_retraining"])
        self.assertNotEqual(report["provenance"]["source_sha256"]["training_dynamics.py"],
                            report["provenance"]["analysis_source_sha256"]["training_dynamics.py"])
        self.assertEqual(report["provenance"]["analysis_source_sha256"]["training_dynamics.py"],
                         lab.digest_bytes(Path(lab.__file__).read_bytes()))
        self.assertTrue(all(checks["unchanged_training_measurements"].values()))
        self.assertEqual(checks["saved_fixed_theta_gradient_max_error"], 0)
        self.assertEqual(checks["saved_gradient_gram_max_error"], 0)
        self.assertEqual(checks["resume"]["max_errors"], dict.fromkeys(("model", "optimizer", "next_steps", "evaluation", "rng"), 0))
        self.assertTrue(checks["resume"]["rng_reference_kind"].startswith("derived"))
        grid = checks["common_token_grid"]
        self.assertEqual(grid, list(range(0, 32769, 4096)))
        for run in report["batch"]["runs"]:
            self.assertEqual([point["consumed_input_tokens"] for point in run["curve"]], grid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
