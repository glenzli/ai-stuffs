#!/usr/bin/env python3
"""Independent custom CE API/autograd tests; no dependency downloads or timings."""
import gc
import math
import unittest

import torch

from chunked_ce_reference import (
    DTYPE, SavedLedger, audit_custom_allocation, chunked_linear_cross_entropy,
    configure_runtime, dense_linear_cross_entropy,
)


class ChunkedCETest(unittest.TestCase):
    @staticmethod
    def tensors(n=5, v=7, d=3, bias=True):
        g = torch.Generator(device="cpu").manual_seed(20261028)
        H = (torch.randn(n, d, generator=g, dtype=DTYPE) * .3).requires_grad_()
        W = (torch.randn(v, d, generator=g, dtype=DTYPE) * .4).requires_grad_()
        b = (torch.randn(v, generator=g, dtype=DTYPE) * .2).requires_grad_() if bias else None
        labels = torch.arange(n, dtype=torch.long) % v
        if n > 2:
            labels[2] = -100
        return H, W, b, labels

    def assertGradients(self, tensors, labels=None, mask=None, upstream=1., **options):
        H, W, b, base_labels = tensors
        labels = base_labels if labels is None else labels
        leaves = [H, W] + ([b] if b is not None else [])
        actual_loss = chunked_linear_cross_entropy(H, W, labels, bias=b, mask=mask, **options)
        actual = torch.autograd.grad(actual_loss, leaves, grad_outputs=actual_loss.new_tensor(upstream))
        reference_loss = dense_linear_cross_entropy(H, W, labels, bias=b, mask=mask, **options)
        expected = torch.autograd.grad(reference_loss, leaves, grad_outputs=reference_loss.new_tensor(upstream))
        torch.testing.assert_close(actual_loss, reference_loss, atol=2e-10, rtol=2e-9)
        for name, a, e in zip(("dH", "dW", "db"), actual, expected):
            with self.subTest(variable=name):
                torch.testing.assert_close(a, e, atol=2e-10, rtol=2e-9)
        return actual_loss, actual

    def test_all_upstream_scalars_each_gradient_independently(self):
        for bias in (False, True):
            for upstream in (0., .375, -2.25, 3.5):
                with self.subTest(bias=bias, upstream=upstream):
                    self.assertGradients(self.tensors(bias=bias), upstream=upstream, token_block=3, vocab_block=2)

    def test_tails_masks_and_target_in_every_block(self):
        for target in range(7):
            tensors = self.tensors()
            tensors[3][3] = target
            mask = torch.tensor([True, False, True, True, True])
            for t, b in ((1, 1), (2, 3), (4, 2), (5, 9), (8, 7)):
                with self.subTest(target=target, token=t, vocab=b):
                    self.assertGradients(tensors, mask=mask, upstream=-2.25, token_block=t, vocab_block=b)

    def test_zero_valid_positions_and_zero_n(self):
        for n, mode in ((5, "mask"), (5, "ignore"), (0, "empty")):
            tensors = self.tensors(n=n)
            labels = torch.full((n,), -100, dtype=torch.long) if mode == "ignore" else tensors[3]
            mask = torch.zeros(n, dtype=torch.bool) if mode == "mask" else None
            loss, gradients = self.assertGradients(tensors, labels=labels, mask=mask, upstream=-3.5, token_block=3, vocab_block=2)
            self.assertEqual(loss.item(), 0)
            for gradient in gradients:
                self.assertTrue(bool((gradient == 0).all()))

    def test_single_class_and_custom_ignore_index(self):
        tensors = self.tensors(n=4, v=1)
        labels = torch.tensor([0, -7, 0, 0], dtype=torch.long)
        loss, grads = self.assertGradients(tensors, labels=labels, token_block=3, vocab_block=5, ignore_index=-7, upstream=-2.25)
        self.assertAlmostEqual(loss.item(), 0, places=12)
        for value in grads:
            torch.testing.assert_close(value, torch.zeros_like(value), atol=2e-12, rtol=0)

    def test_stable_common_shifts_in_argument_and_bias(self):
        for shift in (1e4, -1e4, 1e6, -1e6):
            tensors = self.tensors()
            _, before = self.assertGradients(tensors, token_block=3, vocab_block=2, upstream=-2.25)
            _, after = self.assertGradients(tensors, token_block=3, vocab_block=2, upstream=-2.25, score_shift=shift)
            for a, b in zip(before, after):
                torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-9)
            with torch.no_grad():
                tensors[2].add_(shift)
            self.assertGradients(tensors, token_block=3, vocab_block=2, upstream=.375)

    def test_noncontiguous_inputs_and_bias(self):
        g = torch.Generator().manual_seed(51)
        H = torch.randn(3, 5, generator=g, dtype=DTYPE).T.requires_grad_()
        W = torch.randn(3, 7, generator=g, dtype=DTYPE).T.requires_grad_()
        bias = torch.randn(14, generator=g, dtype=DTYPE)[::2].requires_grad_()
        self.assertFalse(H.is_contiguous()); self.assertFalse(W.is_contiguous()); self.assertFalse(bias.is_contiguous())
        self.assertGradients((H, W, bias, torch.tensor([0, 6, -100, 2, 4])), token_block=3, vocab_block=2, upstream=-2.25)

    def test_autograd_gradcheck_and_higher_order_scope(self):
        H, W, b, labels = self.tensors(n=3, v=4, d=2)
        function = lambda h, w, bias: chunked_linear_cross_entropy(h, w, labels, bias=bias, token_block=2, vocab_block=3)
        self.assertTrue(torch.autograd.gradcheck(function, (H, W, b), eps=1e-6, atol=1e-7, rtol=1e-5))
        first = torch.autograd.grad(function(H, W, b), H, create_graph=True)[0]
        self.assertFalse(first.requires_grad)
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(first.sum(), H)

    def test_multiple_losses_sum_shared_output_gradients(self):
        H, W, b, labels = self.tensors()
        second = torch.tensor([6, 1, -100, 5, 2])
        one = chunked_linear_cross_entropy(H, W, labels, bias=b, token_block=2, vocab_block=3)
        two = chunked_linear_cross_entropy(H, W, second, bias=b, token_block=3, vocab_block=2)
        actual = torch.autograd.grad(.375 * one - 2.25 * two, (H, W, b))
        ref = .375 * dense_linear_cross_entropy(H, W, labels, bias=b) - 2.25 * dense_linear_cross_entropy(H, W, second, bias=b)
        expected = torch.autograd.grad(ref, (H, W, b))
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=2e-11, rtol=2e-10)

    def test_frozen_classifier_still_returns_hidden_gradient(self):
        H, W, b, labels = self.tensors()
        W.requires_grad_(False); b.requires_grad_(False)
        actual = torch.autograd.grad(chunked_linear_cross_entropy(H, W, labels, bias=b), H)[0]
        reference = torch.autograd.grad(dense_linear_cross_entropy(H, W, labels, bias=b), H)[0]
        torch.testing.assert_close(actual, reference, atol=2e-11, rtol=2e-10)

    def test_saved_inputs_cannot_be_mutated_between_stages(self):
        H, W, b, labels = self.tensors()
        loss = chunked_linear_cross_entropy(H, W, labels, bias=b)
        with torch.no_grad():
            W[0, 0] += .1
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            loss.backward()

    def test_hooks_deduplicate_shared_backing_storages(self):
        ledger = SavedLedger()
        backing = torch.arange(30, dtype=DTYPE).reshape(10, 3).requires_grad_()
        h, w = backing[:3], backing[3:]
        labels = torch.tensor([0, 2, 6])
        with torch.autograd.graph.saved_tensors_hooks(ledger.pack, ledger.unpack):
            loss = chunked_linear_cross_entropy(h, w, labels, token_block=2, vocab_block=2)
            self.assertEqual(ledger.peak, backing.untyped_storage().nbytes() + 3 * 8 + 2 * 3 * 8)
            # H and W are separate saves but both reference one backing allocation.
            self.assertEqual(ledger.requests[0]["storage_ordinal"], ledger.requests[1]["storage_ordinal"])
            self.assertEqual(ledger.peak_unique, 4)
            loss.backward()
        del loss
        gc.collect()
        self.assertEqual(ledger.live, 0)

    def test_independent_dispatch_and_hook_allocation_audit(self):
        H, W, b, labels = self.tensors()
        inputs = {"H": H.tolist(), "W": W.tolist(), "bias": b.tolist(), "labels": labels.tolist(), "mask": [True]*5, "ignoreIndex": -100}
        for t, v in ((2, 2), (3, 3), (5, 2), (2, 7)):
            audit = audit_custom_allocation(inputs, {"tokenBlock": t, "vocabBlock": v})
            self.assertTrue(audit["no_full_score_allocation"])
            self.assertTrue(audit["no_score_saved_for_backward"])
            self.assertTrue(audit["all_tile_ops_no_grad"])
            self.assertEqual(audit["saved_tensor_storage"]["save_requests"], 6)
        # Full-sized choices are allowed and deliberately excluded from the claim.
        edge = audit_custom_allocation(inputs, {"tokenBlock": 5, "vocabBlock": 7})
        self.assertFalse(edge["no_full_score_allocation"])

    def test_invalid_arguments_and_masked_invalid_label(self):
        H, W, b, labels = self.tensors()
        bad_labels = labels.clone(); bad_labels[1] = 99
        with self.assertRaises(ValueError):
            chunked_linear_cross_entropy(H, W, bad_labels, mask=torch.zeros(5, dtype=torch.bool))
        for options in ({"token_block": 0}, {"vocab_block": -2}, {"score_shift": math.inf}, {"fault": "unknown"}):
            with self.assertRaises(ValueError):
                chunked_linear_cross_entropy(H, W, labels, bias=b, **options)
        with self.assertRaises(ValueError):
            chunked_linear_cross_entropy(H.float(), W.float(), labels)
        with self.assertRaises(ValueError):
            chunked_linear_cross_entropy(H, W, labels, mask=torch.ones(5))
        with torch.no_grad():
            H[0, 0] = math.nan
        with self.assertRaises(ValueError):
            chunked_linear_cross_entropy(H, W, labels)


if __name__ == "__main__":
    configure_runtime()
    unittest.main(verbosity=2)
