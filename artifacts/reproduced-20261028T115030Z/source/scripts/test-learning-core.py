#!/usr/bin/env python3
"""Meaningful task/decoder/generation/checkpoint tests; no model downloads."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import learning_core as core

spec = importlib.util.spec_from_file_location("learning_runner", Path(__file__).with_name("learning-run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.configure_cpu(1)
        cls.splits = core.make_task_splits()

    def small_model(self, **config):
        return core.make_model(seed=11, width=16, depth=2, heads=4, dropout=0, **config)

    def test_group_disjointness_and_reversal_partner_presence(self):
        core.assert_split_integrity(self.splits)
        self.assertEqual(self.splits, core.make_task_splits())
        self.assertNotEqual(core.split_digest(self.splits), core.split_digest(core.make_task_splits(data_seed=12)))
        for rows in self.splits.values():
            by_content = {tuple(r["content"]): r for r in rows}
            for row in rows:
                partner = by_content[tuple(row["target"])]
                self.assertEqual(partner["group_id"], row["group_id"])
        bad = copy.deepcopy(self.splits)
        bad["test"].append(bad["train"][0])
        with self.assertRaises(ValueError):
            core.assert_split_integrity(bad)

    def test_supervision_shift_and_padding(self):
        rows = [self.splits["train"][0], next(r for r in self.splits["train"] if r["length"] == 5)]
        batch = core.collate_records(rows)
        for i, row in enumerate(rows):
            n = row["length"]
            self.assertEqual(batch["input_ids"][i, :n+2].tolist(), [core.BOS] + row["content"] + [core.SEP])
            self.assertEqual(batch["labels"][i, :n+1].tolist(), [core.IGNORE_INDEX]*(n+1))
            self.assertEqual(batch["labels"][i, n+1:2*n+2].tolist(), row["target"] + [core.EOS])
            self.assertEqual(int(batch["attention_mask"][i].sum()), 2*n+2)
            self.assertTrue(bool((batch["labels"][i, 2*n+2:] == core.IGNORE_INDEX).all()))
        self.assertEqual(batch["supervised_tokens"], sum(r["length"]+1 for r in rows))

    def test_causality_and_padding_do_not_change_supervised_logits(self):
        model = self.small_model().eval()
        batch = core.collate_records([self.splits["train"][0]])
        ids, mask = batch["input_ids"], batch["attention_mask"]
        original = model(ids, mask)
        changed = ids.clone(); changed[:, -1] = core.CONTENT_IDS[(int(ids[0, -1])-3) % 4]
        self.assertTrue(torch.equal(model(changed, mask)[:, :-1], original[:, :-1]))
        extended = F.pad(ids, (0, 4), value=core.PAD)
        extended_mask = F.pad(mask, (0, 4), value=False)
        extended[:, -3:] = torch.tensor([[4, 5, 6]])
        torch.testing.assert_close(model(extended, extended_mask)[:, :ids.shape[1]], original, atol=1e-6, rtol=1e-6)

    def test_loss_denominators_and_all_ignore_skip(self):
        rows = [self.splits["train"][0], next(r for r in self.splits["train"] if r["length"] == 5)]
        batch = core.collate_records(rows)
        model = self.small_model()
        logits = model(batch["input_ids"], batch["attention_mask"])
        raw = F.cross_entropy(logits.reshape(-1, core.VOCAB_SIZE), batch["labels"].reshape(-1), ignore_index=-100, reduction="none").view_as(batch["labels"])
        counts = (batch["labels"] != -100).sum(1)
        torch.testing.assert_close(core.loss_from_logits(logits, batch["labels"], "token"), raw.sum()/counts.sum())
        torch.testing.assert_close(core.loss_from_logits(logits, batch["labels"], "sequence"), (raw.sum(1)/counts).mean())
        batch["labels"].fill_(-100)
        with self.assertRaisesRegex(ValueError, "skip"):
            core.forward_loss(model, batch)

    def test_parameter_gradient_against_directional_finite_difference(self):
        model = self.small_model().double()
        batch = core.collate_records(self.splits["train"][:3])
        loss = core.forward_loss(model, batch); loss.backward()
        parameter = model.blocks[0].attention.qkv.weight
        direction = torch.linspace(-1, 1, parameter.numel(), dtype=torch.float64).view_as(parameter)
        direction /= direction.norm()
        analytic = float((parameter.grad*direction).sum())
        saved = parameter.detach().clone(); eps = 1e-5
        with torch.no_grad():
            parameter.copy_(saved + eps*direction)
            plus = float(core.forward_loss(model, batch))
            parameter.copy_(saved - eps*direction)
            minus = float(core.forward_loss(model, batch))
            parameter.copy_(saved)
        self.assertAlmostEqual(analytic, (plus-minus)/(2*eps), delta=1e-8)

    def test_full_cache_chunk_and_norm_equality(self):
        for norm in ("pre", "post", "none"):
            model = self.small_model(norm=norm).eval()
            batch = core.collate_records([self.splits["train"][0], next(r for r in self.splits["train"] if r["length"] == 4)])
            ids, mask = batch["input_ids"], batch["attention_mask"]
            full = model(ids, mask)
            for chunk in (1, 2, 3, 64):
                cache, parts = None, []
                for offset in range(0, ids.shape[1], chunk):
                    logits, cache = model.forward_cached(ids[:, offset:offset+chunk], mask[:, offset:offset+chunk], cache)
                    parts.append(logits)
                torch.testing.assert_close(torch.cat(parts, 1), full, atol=2e-6, rtol=2e-6)
                self.assertEqual(cache.length, ids.shape[1])
            model.train()
            with self.assertRaises(ValueError):
                model.forward_cached(ids)

    def test_generation_pad_is_real_position_and_cache_trajectory_matches(self):
        model = self.small_model().eval()
        observed = []
        def force_pad_then_symbol_then_eos(module, inputs, outputs):
            observed.append(inputs[0].detach().clone())
            if len(observed) == 1:
                symbol = core.PAD
            elif len(observed) == 2:
                symbol = 4
            else:
                symbol = core.EOS
            logits = torch.full_like(outputs, -20); logits[..., symbol] = 20
            return logits
        # Force at the actual output projection; preserve the real model and
        # attention computations. Check hidden state after PAD in both paths.
        hook = model.lm_head.register_forward_hook(force_pad_then_symbol_then_eos)
        record = self.splits["train"][0]
        cached = core.generate_records(model, [record], max_new_tokens=8)
        cached_states = [x[:, -1].clone() for x in observed]
        observed.clear()
        full = core.generate_records(model, [record], max_new_tokens=8, use_cache=False)
        full_states = [x[:, -1].clone() for x in observed]
        hook.remove()
        self.assertEqual(cached[0]["predicted"], [core.PAD, 4, core.EOS])
        self.assertEqual(full[0]["predicted"], cached[0]["predicted"])
        for a, b in zip(cached_states, full_states):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        prompt = torch.tensor([[core.BOS] + record["content"] + [core.SEP]])
        _, cache = model.forward_cached(prompt, torch.ones_like(prompt, dtype=torch.bool))
        with_pad, advanced = model.forward_cached(torch.tensor([[core.PAD]]), torch.tensor([[True]]), cache)
        ignored_pad, excluded = model.forward_cached(torch.tensor([[core.PAD]]), torch.tensor([[False]]), cache)
        self.assertEqual(int(advanced.attention_mask.sum()), prompt.shape[1]+1)
        self.assertEqual(int(excluded.attention_mask.sum()), prompt.shape[1])
        self.assertGreater(float((with_pad-ignored_pad).detach().abs().max()), 1e-5)

    def test_eos_extra_missing_and_incomplete_score(self):
        target = [4, 5]
        exact = core.score_generation([4, 5, core.EOS], target, max_new_tokens=8)
        self.assertTrue(exact["exact_match"]); self.assertEqual(exact["token_accuracy"], 1)
        self.assertEqual(core.score_generation([4, core.EOS], target)["missing_tokens"], 1)
        extra = core.score_generation([4, 5, 6, core.EOS], target)
        self.assertEqual(extra["extra_tokens"], 1); self.assertEqual(extra["token_accuracy"], .5)
        self.assertFalse(core.score_generation([4, 5], target)["exact_match"])
        self.assertEqual(core.score_generation([4, 5], target)["termination"], "incomplete")
        self.assertEqual(core.score_generation([4]*8, target, max_new_tokens=8)["termination"], "cap")
        self.assertTrue(core.score_generation([core.EOS], target)["early_eos"])
        with self.assertRaises(ValueError):
            core.score_generation([4, core.EOS, 5], target)

    def test_generation_common_cap_and_no_gold_length_stop(self):
        model = self.small_model()
        with torch.no_grad():
            model.lm_head.weight.zero_()
        rows = core.generate_records(model, self.splits["train"][:3], max_new_tokens=9)
        self.assertTrue(all(len(r["predicted"]) == 9 and r["termination"] == "cap" for r in rows))
        self.assertTrue(all(r["extra_tokens"] > 0 and not r["exact_match"] for r in rows))

    def test_evaluation_restores_rng_and_each_module_mode(self):
        model = core.make_model(width=16, depth=1, heads=4, dropout=.2)
        model.train(); model.blocks[0].mlp.eval()
        modes = [m.training for m in model.modules()]
        core.seed_all(23); before = runner.serializable_rng()
        result = core.evaluate_records(model, self.splits["validation"][:4])
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(runner.tree_difference(before, runner.serializable_rng())[0])
        self.assertIn("teacher_forced_predicted", result["predictions"][0])
        self.assertIn("predicted", result["predictions"][0])

    def test_model_construction_rng_isolation_and_baseline_separation(self):
        core.seed_all(22); before = runner.serializable_rng(); self.small_model()
        self.assertTrue(runner.tree_difference(before, runner.serializable_rng())[0])
        rows = self.splits["test"]
        oracle = core.baseline_predictions(rows, "oracle")
        identity = core.baseline_predictions(rows, "identity")
        self.assertEqual(oracle["metrics"]["exact_match"], 1)
        self.assertIsNone(oracle["metrics"]["teacher_forced_loss"])
        self.assertLess(identity["metrics"]["exact_match"], 1)
        counts = {token: sum(row["target"].count(token) for row in self.splits["train"]) for token in core.CONTENT_IDS}
        selected = core.most_common_training_symbol(self.splits["train"])
        self.assertEqual(counts[selected], max(counts.values()))

    def test_checkpoint_optimizer_rng_cursor_trajectory_roundtrip(self):
        rows = self.splits["train"][:17]
        core.seed_all(77)
        model = core.make_model(width=16, depth=1, heads=4, dropout=.25)
        training = {"lr": .003, "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01, "clip_norm": 1.0}
        frozen = {"training": training}
        optimizer = runner.new_optimizer(model, training)
        stream = core.RecordBatchStream(rows, batch_size=7, seed=53)
        progress = {"step": 0, "examples": 0, "input_tokens": 0, "supervised_tokens": 0}
        for _ in range(3):
            runner.train_step(model, optimizer, stream, progress, training)
        with tempfile.TemporaryDirectory(prefix="learning-checkpoint-test-") as tmp:
            path = Path(tmp)/"checkpoint.pt"
            runner.save_checkpoint(path, model, optimizer, stream, 11, progress, frozen)
            expected_trace = [runner.train_step(model, optimizer, stream, progress, training) for _ in range(4)]
            expected = copy.deepcopy(runner.checkpoint_value(model, optimizer, stream, 11, progress, frozen))
            restored, opt, order, cursor, _ = runner.restore_checkpoint(path, rows)
            actual_trace = [runner.train_step(restored, opt, order, cursor, training) for _ in range(4)]
            actual = runner.checkpoint_value(restored, opt, order, 11, cursor, frozen)
            for key in ("model", "optimizer", "rng", "stream", "progress"):
                self.assertTrue(runner.tree_difference(actual[key], expected[key])[0], key)
            self.assertEqual(actual_trace, expected_trace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
