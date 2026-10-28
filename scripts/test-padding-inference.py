"""Independent dense softmax/full recomputation of saved padded and cached paths."""
import hashlib
import json
import math
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

import learning_core as symbolic
import text_training_core as core
import padding_inference_lab as lab


@torch.no_grad()
def dense(model, ids, positions, valid, ignore_key_mask=False):
    """No SDPA, adapter or DecoderBlock.forward calls; explicitly form attention."""
    hidden = F.embedding(ids, model.embedding.weight) + model.positions[positions]
    length, width = ids.shape[1], model.config["width"]
    allowed = torch.tril(torch.ones(length, length, dtype=torch.bool))[None, None]
    if not ignore_key_mask:
        allowed = allowed & valid[:, None, None, :]
    nonempty = allowed.any(-1, keepdim=True)
    caches = []
    for block in model.blocks:
        assert block.norm_kind == "pre" and block.residual_scale == 1
        normalized = F.layer_norm(hidden, (width,), block.norm1.weight, block.norm1.bias, block.norm1.eps)
        a = block.attention
        q, k, v = F.linear(normalized, a.qkv.weight, a.qkv.bias).reshape(ids.shape[0], length, 3, a.heads, a.head_width).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(a.head_width)
        scores = scores.masked_fill(~allowed, -float("inf"))
        # PAD query rows with no allowed key are unused; match this CPU kernel's zero output.
        weights = torch.softmax(torch.where(nonempty, scores, 0), -1) * nonempty
        context = (weights @ v).transpose(1, 2).reshape(ids.shape[0], length, width)
        hidden = hidden + F.linear(context, a.output.weight, a.output.bias)
        normalized = F.layer_norm(hidden, (width,), block.norm2.weight, block.norm2.bias, block.norm2.eps)
        hidden = hidden + F.linear(F.gelu(F.linear(normalized, block.mlp[0].weight, block.mlp[0].bias)), block.mlp[2].weight, block.mlp[2].bias)
        caches.append((k, v))
    hidden = F.layer_norm(hidden, (width,), model.norm.weight, model.norm.bias, model.norm.eps)
    return F.linear(hidden, model.head.weight), caches


def expected_positions(valid, width, method):
    result = torch.zeros_like(valid, dtype=torch.long)
    for row in range(valid.shape[0]):
        count = 0
        for col in range(valid.shape[1]):
            if valid[row, col]:
                count += 1
            result[row, col] = col if method == "physical" or (method == "cache-physical" and col >= width) else max(count - 1, 0)
    return result


class PaddedInferenceEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        symbolic.configure_cpu(1)
        cls.manifest = json.loads(Path("src/experiment-sources.json").read_text())
        cls.entry = cls.manifest["experiments"]["padding-inference"]
        cls.root = Path(cls.entry["bundle_root"])
        cls.directory = Path(cls.entry["report"]["path"]).parent
        cls.report = json.loads((cls.directory / "report.json").read_text())
        cls.references = torch.load(cls.directory / "references.pt", weights_only=True)
        saved = torch.load(cls.directory / "inputs/seed-11-final.pt", weights_only=True)
        cls.model = core.make_model(saved["seed"], saved["config"]).double().eval()
        cls.model.load_state_dict(saved["model"])
        cls.trials = {row["tensor_file"]: torch.load(cls.directory / row["tensor_file"], weights_only=True) for row in cls.report["trials"]}

    def test_inputs_executed_sources_and_unchanged_full_model_identity(self):
        original = Path(self.manifest["experiments"]["practice"]["report"]["path"]).parent / "text"
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for name, sha in self.report["input_sha256"].items():
            self.assertEqual(digest(self.directory / "inputs" / name), sha)
            self.assertEqual((self.directory / "inputs" / name).read_bytes(), (original / name).read_bytes())
        environment = json.loads((self.root / "run-environment.json").read_text())
        for file, sha in environment["source_sha256"].items():
            self.assertEqual(digest(file), sha)
            self.assertEqual(digest(self.root / "source" / file), sha)
        self.assertEqual(digest(self.directory / "references.pt"), self.report["references_sha256"])
        for row in self.report["trials"]:
            self.assertEqual(digest(self.directory / row["tensor_file"]), row["tensor_sha256"])
        self.assertEqual(self.report["initial_weight_sha256"], self.report["final_weight_sha256"])
        self.assertEqual(lab.weight_sha(self.model), self.report["initial_weight_sha256"])
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 298176)

    def test_all_45_original_unpadded_references_match_independent_dense_softmax(self):
        for saved in self.report["inputs"]:
            for sample, source in enumerate(saved["records"]):
                for event, reference in enumerate(self.references[saved["case"]][sample]):
                    ids = torch.tensor([source["prefix"] + source["continuation"][:event]])
                    logits, caches = dense(self.model, ids, torch.arange(ids.shape[1])[None], torch.ones_like(ids, dtype=torch.bool))
                    torch.testing.assert_close(logits[0], reference["logits"], rtol=0, atol=2e-11)
                    for actual, expected in zip(caches, reference["layers"], strict=True):
                        for left, right in zip(actual, expected, strict=True):
                            torch.testing.assert_close(left, right, rtol=0, atol=2e-11)

    def test_all_180_cached_events_match_dense_full_recomputation_with_historical_positions(self):
        for description in self.report["trials"]:
            trial, method = self.trials[description["tensor_file"]], description["method"]
            width = trial["input_ids"].shape[1]
            full_ids, full_valid = trial["input_ids"], trial["valid"]
            for event, recorded in enumerate(trial["events"]):
                if event:
                    full_ids = torch.cat((full_ids, recorded["input_ids"]), 1)
                    full_valid = torch.cat((full_valid, torch.ones_like(recorded["input_ids"], dtype=torch.bool)), 1)
                positions = expected_positions(full_valid, width, method)
                torch.testing.assert_close(recorded["positions"], positions[:, :width] if event == 0 else positions[:, -1:], rtol=0, atol=0)
                logits, caches = dense(self.model, full_ids, positions, full_valid, method == "no-key-mask")
                if not event:
                    torch.testing.assert_close(logits, trial["prefill_logits"], rtol=0, atol=2e-11)
                    if method == "last-column":
                        selected = torch.full((3,), width - 1)
                    elif method == "length-index":
                        selected = full_valid.sum(1) - 1
                    else:
                        selected = torch.tensor([int(v.nonzero()[-1]) for v in full_valid])
                    expected_next = logits[torch.arange(3), selected]
                else:
                    selected = torch.zeros(3, dtype=torch.long)
                    expected_next = logits[:, -1]
                torch.testing.assert_close(recorded["selected_indices"], selected, rtol=0, atol=0)
                torch.testing.assert_close(recorded["next_logits"], expected_next, rtol=0, atol=2e-11)
                # Earlier actual cache outputs are attested by hashes of their final prefix.
                digests = []
                for actual, final in zip(caches, trial["final_cache"]["layers"], strict=True):
                    pair = []
                    for value, end in zip(actual, final, strict=True):
                        prefix = end[:, :, :full_ids.shape[1]]
                        torch.testing.assert_close(value, prefix, rtol=0, atol=2e-11)
                        pair.append(hashlib.sha256(prefix.contiguous().numpy().tobytes()).hexdigest())
                    digests.append(pair)
                self.assertEqual(digests, recorded["cache_sha256"])
            torch.testing.assert_close(full_valid, trial["final_cache"]["valid"], rtol=0, atol=0)

    def test_all_540_error_fields_and_physical_logical_ledgers_recompute_from_raw_tensors(self):
        lookup = {(r["case"], r["side"], r["method"]): self.trials[r["tensor_file"]] for r in self.report["trials"]}
        for row in self.report["rows"]:
            trial = lookup[row["case"], row["side"], row["method"]]
            sample, event = row["sample"], row["event"]
            recorded, reference = trial["events"][event], self.references[row["case"]][sample][event]
            width, slots = trial["input_ids"].shape[1], recorded["physical_slots"]
            valid = trial["final_cache"]["valid"][sample, :slots]
            indices = valid.nonzero().flatten()
            queries = trial["prefill_logits"][sample][trial["valid"][sample]] if not event else recorded["next_logits"][sample:sample + 1]
            expected = reference["logits"] if not event else reference["logits"][-1:]
            errors = {"valid_logits_max_error": float((queries - expected).abs().max()),
                      "next_logits_max_error": float((recorded["next_logits"][sample] - reference["logits"][-1]).abs().max()),
                      "kv_max_error": max(float((actual[sample:sample + 1].index_select(2, indices) - baseline).abs().max()) for layer, ref in zip(trial["final_cache"]["layers"], reference["layers"], strict=True) for actual, baseline in zip(layer, ref, strict=True))}
            for key, value in errors.items():
                self.assertAlmostEqual(row[key], value, delta=1e-12)
            self.assertEqual(row["within_tolerance"], max(errors.values()) <= self.report["frozen"]["tolerance"])
            self.assertEqual(row["valid_cache_slots"], len(indices))
            self.assertEqual(row["physical_cache_slots"], width + event)
            physical_offset = width + event - 1 if event else 0
            self.assertEqual(row["selected_column"], physical_offset + int(recorded["selected_indices"][sample]))
            self.assertEqual(row["last_valid_column"], int(indices[-1]))
            self.assertEqual(row["last_valid_position_id"], int(recorded["positions"][sample, int(recorded["valid"][sample].nonzero()[-1])]))
            self.assertEqual(row["expected_logical_position"], len(indices) - 1)
            self.assertEqual(row["actual_argmax"], int(recorded["next_logits"][sample].argmax()))
            self.assertEqual(row["reference_argmax"], int(reference["logits"][-1].argmax()))

    def test_observed_controls_preserve_prefill_readout_and_continuation_as_distinct_failures(self):
        rows = self.report["rows"]
        self.assertEqual(len(rows), 540)
        self.assertTrue(all(r["within_tolerance"] for r in rows if r["method"] == "correct" or r["case"] == "balanced"))
        find = lambda side, method, event: next(r for r in rows if r["case"] == "ragged" and r["side"] == side and r["method"] == method and r["sample"] == 0 and r["event"] == event)
        for side, method in [("right", "last-column"), ("left", "length-index")]:
            row = find(side, method, 0)
            self.assertLess(row["valid_logits_max_error"], 2e-10)
            self.assertLess(row["kv_max_error"], 2e-10)
            self.assertGreater(row["next_logits_max_error"], 1)
            self.assertNotEqual(row["selected_column"], row["last_valid_column"])
        for method in ["no-key-mask", "physical", "cache-physical"]:
            self.assertTrue(find("right", method, 0)["within_tolerance"])
            self.assertFalse(find("right", method, 1)["within_tolerance"])
        self.assertTrue(all(r["within_tolerance"] for r in rows if r["method"] == "last-column" and r["side"] == "left"))
        self.assertTrue(all(r["within_tolerance"] for r in rows if r["method"] == "length-index" and r["side"] == "right"))
        self.assertTrue(any(not r["within_tolerance"] and r["actual_argmax"] == r["reference_argmax"] for r in rows))

    def test_empty_prompt_context_limit_and_malformed_cache_are_rejected(self):
        for prompts, side in [([], "left"), ([[], [2]], "right"), ([[2]], "unknown")]:
            with self.assertRaises(ValueError):
                lab.make_batch(prompts, side)
        with self.assertRaises(ValueError):
            lab.selection_indices(torch.zeros(1, 2, dtype=torch.bool), "correct")
        with self.assertRaises(ValueError):
            lab.forward_adapter(self.model, torch.ones(1, 97, dtype=torch.long), torch.ones(1, 97, dtype=torch.bool))
        with self.assertRaises(ValueError):
            lab.forward_adapter(self.model, torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool), past={"valid": torch.ones(2, 1, dtype=torch.bool), "layers": ()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
