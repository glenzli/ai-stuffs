#!/usr/bin/env python3
"""Independent exact CPU linear CE; no GPU/CCE dependency or downloaded model.

The custom Function explicitly runs both stages under no_grad. Only H, W,
optional bias, effective labels, and two N-vectors survive the forward pass.
The score tile is reused for exp and dZ. Full-sized block choices are permitted
as an edge reference; the no-full-score audit applies only to sub-full choices.
Higher-order derivatives, tied lookup weights, class weights, smoothing and
per-position upstream tensors are outside this scalar-mean demonstration.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_runtime():
    if torch.__version__.split("+")[0] != "2.9.0" or np.__version__ != "2.3.4":
        raise RuntimeError("Reuse the pinned .venv-framework: PyTorch 2.9.0 / NumPy 2.3.4.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def validate(H, W, labels, bias, mask, token_block, vocab_block, ignore_index, score_shift):
    if H.device.type != "cpu" or W.device.type != "cpu" or H.dtype != DTYPE or W.dtype != DTYPE:
        raise ValueError("This independent reference requires CPU float64 H/W.")
    if H.ndim != 2 or W.ndim != 2 or not W.shape[0] or not W.shape[1] or H.shape[1] != W.shape[1]:
        raise ValueError("Expected H[N,D] and W[V,D] with V,D>0.")
    n, v = H.shape[0], W.shape[0]
    if labels.device.type != "cpu" or labels.dtype != torch.long or labels.shape != (n,):
        raise ValueError("Expected CPU int64 labels[N].")
    if bias is not None and (bias.device.type != "cpu" or bias.dtype != DTYPE or bias.shape != (v,)):
        raise ValueError("Expected optional CPU float64 bias[V].")
    if mask is not None and (mask.device.type != "cpu" or mask.dtype != torch.bool or mask.shape != (n,)):
        raise ValueError("Expected optional CPU bool supervision mask[N].")
    if any(isinstance(b, bool) or not isinstance(b, int) or b < 1 for b in (token_block, vocab_block)):
        raise ValueError("Block sizes must be positive integers.")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise ValueError("ignore_index must be an integer.")
    if not isinstance(score_shift, (int, float)) or not math.isfinite(score_shift) or abs(score_shift) > 1e9:
        raise ValueError("score_shift must be finite and at most 1e9 in absolute value.")
    if not bool(((labels == ignore_index) | ((labels >= 0) & (labels < v))).all()):
        raise ValueError("Labels outside the vocabulary are invalid even at masked positions.")
    for value in (H, W, bias):
        if value is not None and not bool(torch.isfinite(value).all()):
            raise ValueError("All input scores/parameters must be finite.")


class ChunkedLinearCE(torch.autograd.Function):
    """Exact first derivative of the scalar valid-position mean, including dLoss."""

    @staticmethod
    def forward(ctx, H, W, bias_tensor, effective_labels, token_block, vocab_block, ignore_index, score_shift, fault):
        with torch.no_grad():
            n, v = H.shape[0], W.shape[0]
            count = int((effective_labels != ignore_index).sum())
            row_max = H.new_zeros(n if count else 0)
            row_logsum = H.new_zeros(n if count else 0)
            loss_sum = H.new_zeros(())
            if count:
                for t_start in range(0, n, token_block):
                    t_end = min(n, t_start + token_block)
                    h = H[t_start:t_end]
                    y = effective_labels[t_start:t_end]
                    active = y != ignore_index
                    if not bool(active.any()):
                        continue
                    m = H.new_full((t_end - t_start,), -math.inf)
                    denom = H.new_zeros(t_end - t_start)
                    target = H.new_zeros(t_end - t_start)
                    for v_start in range(0, v, vocab_block):
                        v_end = min(v, v_start + vocab_block)
                        z = h @ W[v_start:v_end].T
                        if bias_tensor.numel():
                            z.add_(bias_tensor[v_start:v_end])
                        if score_shift:
                            z.add_(score_shift)
                        if not bool(torch.isfinite(z).all()):
                            raise FloatingPointError("Nonfinite score tile; stable LSE requires finite logits.")
                        hit = active & (y >= v_start) & (y < v_end)
                        hit_rows = hit.nonzero(as_tuple=True)[0]
                        if hit_rows.numel():
                            target[hit_rows] = z[hit_rows, y[hit_rows] - v_start]
                        next_max = torch.maximum(m, z.max(dim=1).values)
                        correction = (m - next_max).exp()
                        # Overwrite the score tile: no second tile-sized exp allocation.
                        z.sub_(next_max[:, None]).exp_()
                        denom.mul_(correction).add_(z.sum(dim=1))
                        m = next_max
                        del z
                    logsum = denom.log()
                    row_max[t_start:t_end] = m
                    row_logsum[t_start:t_end] = logsum
                    # Separate m and logsum avoids cancellation of m+logsum-target.
                    loss_sum.add_(((m - target) + logsum)[active].sum())
            ctx.save_for_backward(H, W, bias_tensor, effective_labels, row_max, row_logsum)
            ctx.token_block, ctx.vocab_block = token_block, vocab_block
            ctx.ignore_index, ctx.score_shift, ctx.count, ctx.fault = ignore_index, score_shift, count, fault
            return loss_sum / count if count else loss_sum

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        with torch.no_grad():
            H, W, bias_tensor, labels, row_max, row_logsum = ctx.saved_tensors
            n, v = H.shape[0], W.shape[0]
            dH, dW = torch.zeros_like(H), torch.zeros_like(W)
            db = torch.zeros_like(bias_tensor) if bias_tensor.numel() else None
            if ctx.count:
                factor = d_loss / ctx.count
                for t_start in range(0, n, ctx.token_block):
                    t_end = min(n, t_start + ctx.token_block)
                    h, y = H[t_start:t_end], labels[t_start:t_end]
                    active = y != ctx.ignore_index
                    if not bool(active.any()):
                        continue
                    for v_start in range(0, v, ctx.vocab_block):
                        v_end = min(v, v_start + ctx.vocab_block)
                        w = W[v_start:v_end]
                        z = h @ w.T
                        if bias_tensor.numel():
                            z.add_(bias_tensor[v_start:v_end])
                        if ctx.score_shift:
                            z.add_(ctx.score_shift)
                        if ctx.fault == "local-softmax":
                            m = z.max(dim=1).values
                            z.sub_(m[:, None]).exp_()
                            z.div_(z.sum(dim=1, keepdim=True))
                        else:
                            z.sub_(row_max[t_start:t_end, None])
                            z.sub_(row_logsum[t_start:t_end, None]).exp_()
                        hit = active & (y >= v_start) & (y < v_end)
                        hit_rows = hit.nonzero(as_tuple=True)[0]
                        if hit_rows.numel():
                            z[hit_rows, y[hit_rows] - v_start] -= 1
                        # z is now dZ. Mask rows BEFORE the global valid-position mean.
                        z.mul_(active[:, None]).mul_(factor)
                        contribution = z @ w
                        if ctx.fault == "overwrite-dh":
                            dH[t_start:t_end].copy_(contribution)
                        else:
                            dH[t_start:t_end].add_(contribution)
                        dW[v_start:v_end].add_(z.T @ h)
                        if db is not None:
                            db[v_start:v_end].add_(z.sum(dim=0))
                        del z, contribution
            return dH, dW, db, None, None, None, None, None, None


def chunked_linear_cross_entropy(H, W, labels, bias=None, mask=None, token_block=2, vocab_block=3,
                                 ignore_index=-100, score_shift=0., fault="none"):
    """Return a scalar mean or a differentiable zero sentinel when no label is valid.

    Callers must skip optimizer updates for the zero-valid case. No label shift is
    performed here: labels and H positions already describe the same prediction.
    H/W may be views; their saved storage references include the backing storage.
    """
    validate(H, W, labels, bias, mask, token_block, vocab_block, ignore_index, score_shift)
    if fault not in ("none", "local-softmax", "overwrite-dh"):
        raise ValueError("Unknown fault.")
    effective = labels.clone() if mask is None else torch.where(mask, labels, ignore_index)
    bias_tensor = H.new_empty(0) if bias is None else bias
    return ChunkedLinearCE.apply(H, W, bias_tensor, effective, token_block, vocab_block, ignore_index, float(score_shift), fault)


def dense_linear_cross_entropy(H, W, labels, bias=None, mask=None, token_block=2, vocab_block=3,
                               ignore_index=-100, score_shift=0., **_):
    """Independent dense PyTorch autograd oracle with the same empty sentinel."""
    validate(H, W, labels, bias, mask, token_block, vocab_block, ignore_index, score_shift)
    effective = labels if mask is None else torch.where(mask, labels, ignore_index)
    count = int((effective != ignore_index).sum())
    if not count:
        return (H.sum() + W.sum() + (bias.sum() if bias is not None else 0)) * 0
    logits = H @ W.T
    if bias is not None:
        logits = logits + bias
    if score_shift:
        logits = logits + score_shift
    return F.cross_entropy(logits, effective, ignore_index=ignore_index, reduction="sum") / count


def input_tensors(inputs, requires_grad=True):
    return (torch.tensor(inputs["H"], dtype=DTYPE).reshape(len(inputs["H"]), len(inputs["W"][0])).requires_grad_(requires_grad),
            torch.tensor(inputs["W"], dtype=DTYPE).requires_grad_(requires_grad),
            torch.tensor(inputs["bias"], dtype=DTYPE).requires_grad_(requires_grad) if inputs.get("bias") is not None else None,
            torch.tensor(inputs["labels"], dtype=torch.long), torch.tensor(inputs["mask"], dtype=torch.bool))


def run_case(inputs, options, chunked=True, upstream=None):
    H, W, bias, labels, mask = input_tensors(inputs)
    method = chunked_linear_cross_entropy if chunked else dense_linear_cross_entropy
    loss = method(H, W, labels, bias=bias, mask=mask, token_block=options["tokenBlock"], vocab_block=options["vocabBlock"],
                  ignore_index=inputs["ignoreIndex"], score_shift=options.get("scoreShift", 0), fault=options.get("fault", "none"))
    u = options.get("upstream", 1) if upstream is None else upstream
    loss.backward(torch.tensor(float(u), dtype=DTYPE))
    return {"loss": float(loss.detach()), "dH": H.grad.detach().tolist(), "dW": W.grad.detach().tolist(),
            "db": bias.grad.detach().tolist() if bias is not None else None}


def max_error(left, right):
    if left is None or right is None:
        if left is not right:
            raise AssertionError("Optional gradients differ.")
        return 0.
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} != {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.


class Checks:
    def __init__(self):
        self.rows = []

    def compare(self, name, left, right, atol=2e-9, rtol=2e-8):
        error = max_error(left, right)
        if left is not None and not np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False):
            raise AssertionError(f"{name}: max error {error:.4g}")
        self.rows.append({"name": name, "max_error": error, "atol": atol, "rtol": rtol})
        return error


class SavedPayload:
    def __init__(self, tensor, ledger):
        self.tensor, self.ledger = tensor.detach(), ledger
        storage = tensor.untyped_storage()
        # cdata identifies the storage object, even for empty storages or views.
        self.key = (str(tensor.device), storage._cdata, storage.nbytes())
        ledger.counts[self.key] = ledger.counts.get(self.key, 0) + 1
        if ledger.counts[self.key] == 1:
            ledger.live += storage.nbytes()
        live_keys = sum(1 for key, count in ledger.counts.items() if count and key[2])
        ledger.peak = max(ledger.peak, ledger.live)
        ledger.peak_unique = max(ledger.peak_unique, live_keys)
        ledger.requests.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel(),
                                "storage_bytes": storage.nbytes(), "requires_grad": tensor.requires_grad,
                                "storage_ordinal": ledger.ordinal(self.key)})

    def __del__(self):
        self.ledger.counts[self.key] -= 1
        if not self.ledger.counts[self.key]:
            self.ledger.live -= self.key[2]


class SavedLedger:
    """References retained for backward, NOT allocator/reserved/RSS memory."""
    def __init__(self):
        self.counts, self.keys, self.requests = {}, {}, []
        self.live, self.peak, self.peak_unique = 0, 0, 0

    def ordinal(self, key):
        if key not in self.keys:
            self.keys[key] = len(self.keys)
        return self.keys[key]

    def pack(self, tensor):
        return SavedPayload(tensor, self)

    @staticmethod
    def unpack(payload):
        return payload.tensor

    def summary(self, full_shape):
        return {"peak_referenced_bytes": self.peak, "peak_unique_storages": self.peak_unique,
                "save_requests": len(self.requests), "live_bytes_after_backward": self.live,
                "full_score_shape_save_requests": sum(row["shape"] == list(full_shape) for row in self.requests),
                "saved_tensors": self.requests,
                "includes": "Unique backing storages referenced by tensors saved for backward, including H/W/bias and labels. Views share a storage and count once.",
                "excludes": "Unsaved tiles and GEMM temporaries, gradient buffers, allocator allocated/reserved peaks, process RSS and GPU memory."}


class ScoreDispatchAudit(TorchDispatchMode):
    """Observe score mm/exp shapes in one custom run; no timing uses this mode."""
    def __init__(self, n, v, d, token_block, vocab_block):
        super().__init__()
        self.n, self.v, self.d = n, v, d
        self.token_block, self.vocab_block = token_block, vocab_block
        self.phase, self.rows = "forward", []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)
        if name.startswith("aten.mm") or name.startswith("aten.exp"):
            tensors = [x for x in (out if isinstance(out, (tuple, list)) else [out]) if isinstance(x, torch.Tensor)]
            for tensor in tensors:
                self.rows.append({"phase": self.phase, "operation": name, "shape": list(tensor.shape),
                                  "grad_enabled": torch.is_grad_enabled(), "requires_grad": tensor.requires_grad})
        return out


def audit_custom_allocation(inputs, options):
    H, W, bias, labels, mask = input_tensors(inputs)
    n, v, d = H.shape[0], W.shape[0], H.shape[1]
    ledger = SavedLedger()
    audit = ScoreDispatchAudit(n, v, d, options["tokenBlock"], options["vocabBlock"])
    with torch.autograd.graph.saved_tensors_hooks(ledger.pack, ledger.unpack), audit:
        loss = chunked_linear_cross_entropy(H, W, labels, bias=bias, mask=mask, token_block=options["tokenBlock"], vocab_block=options["vocabBlock"], ignore_index=inputs["ignoreIndex"])
        audit.phase = "backward"
        loss.backward(torch.tensor(-2.25, dtype=DTYPE))
    del loss
    gc.collect()
    assert ledger.live == 0
    full = [n, v]
    sub_full = min(n, options["tokenBlock"]) * min(v, options["vocabBlock"]) < n * v
    # The tested d differs from N/V to disambiguate score and gradient mm shapes.
    score_mms = [r for r in audit.rows if r["operation"].startswith("aten.mm") and r["shape"][0] <= min(n, options["tokenBlock"]) and r["shape"][1] <= min(v, options["vocabBlock"]) and r["shape"][1] != d]
    if sub_full:
        assert all(r["shape"] != full for r in audit.rows), audit.rows
        assert not any(r["shape"] == full for r in ledger.requests)
    assert all(not r["grad_enabled"] and not r["requires_grad"] for r in audit.rows)
    allowed = [[n, d], [v, d], [v] if bias is not None else [0], [n], [n], [n]]
    assert [r["shape"] for r in ledger.requests] == allowed, ledger.requests
    return {"shape": [n, v, d], "blocks": [options["tokenBlock"], options["vocabBlock"]], "scope": "sub-full score tiles only" if sub_full else "full-sized single tile allowed edge case",
            "no_full_score_allocation": sub_full and not any(r["shape"] == full for r in audit.rows),
            "no_score_saved_for_backward": all(r["shape"] != full for r in ledger.requests),
            "all_tile_ops_no_grad": True, "observed_operations": audit.rows, "saved_tensor_storage": ledger.summary(full),
            "score_mm_observations": len(score_mms)}


def finite_differences(inputs, options, checks, epsilon=1e-5):
    tensors = input_tensors(inputs, requires_grad=False)
    H, W, bias, labels, mask = tensors
    actual = run_case(inputs, options)
    largest, elements = 0., 0
    for name, value in (("H", H), ("W", W), ("b", bias)):
        if value is None:
            continue
        analytic = actual["db" if name == "b" else "d" + name]
        numerical = torch.zeros_like(value)
        for index in np.ndindex(tuple(value.shape)):
            original = value[index].item()
            value[index] = original + epsilon
            plus = float(dense_linear_cross_entropy(H, W, labels, bias=bias, mask=mask))
            value[index] = original - epsilon
            minus = float(dense_linear_cross_entropy(H, W, labels, bias=bias, mask=mask))
            value[index] = original
            numerical[index] = options.get("upstream", 1) * (plus - minus) / (2 * epsilon)
            elements += 1
        largest = max(largest, checks.compare("finite-difference/" + name, analytic, numerical.tolist(), atol=2e-9, rtol=2e-7))
    return {"epsilon": epsilon, "elements": elements, "upstream": options.get("upstream", 1), "max_error": largest,
            "oracle": "Central differences of the independent dense PyTorch scalar loss; H/W/b perturbed independently."}


def correctness(fixture):
    checks = Checks()
    dense_error = js_error = 0.
    case_rows, faults = [], []
    for case in fixture["cases"]:
        inputs, options = fixture["inputs"][case["input"]], case["options"]
        actual = run_case(inputs, options)
        dense = run_case(inputs, options, chunked=False)
        errors = {}
        for key in ("loss", "dH", "dW", "db"):
            error = checks.compare(case["id"] + "/dense/" + key, actual[key], dense[key])
            dense_error = max(dense_error, error)
            js_error = max(js_error, checks.compare(case["id"] + "/js/" + key, actual[key], case["expected"][key]))
            errors[key] = error
        case_rows.append({"id": case["id"], "options": options, "reference_errors": errors})
    for case in fixture["fault_cases"]:
        inputs, options = fixture["inputs"][case["input"]], case["options"]
        actual, dense = run_case(inputs, options), run_case(inputs, options, chunked=False)
        errors = {key: max_error(actual[key], dense[key]) for key in ("loss", "dH", "dW", "db")}
        for key in ("loss", "dH", "dW", "db"):
            js_error = max(js_error, checks.compare(case["id"] + "/fault-js/" + key, actual[key], case["expected"][key]))
        checks.compare(case["id"] + "/unchanged-loss", actual["loss"], dense["loss"])
        if options["fault"] == "overwrite-dh":
            checks.compare(case["id"] + "/unaffected-dW", actual["dW"], dense["dW"])
            checks.compare(case["id"] + "/unaffected-db", actual["db"], dense["db"])
        if case["detectable"]:
            assert errors["dH"] > 1e-3, (case["id"], errors)
            if options["fault"] == "local-softmax":
                assert errors["dW"] > 1e-3 and errors["db"] > 1e-3
        else:
            for key in ("dH", "dW", "db"):
                checks.compare(case["id"] + "/hidden-fault/" + key, actual[key], dense[key])
        faults.append({"id": case["id"], "options": options, "detectable": case["detectable"], "errors": errors})
    finite = [finite_differences(fixture["inputs"][key], {"tokenBlock": 3, "vocabBlock": 2, "upstream": -2.25}, checks) for key in ("base-bias", "mixed-no-bias")]
    audits = [audit_custom_allocation(fixture["inputs"]["base-bias"], {"tokenBlock": t, "vocabBlock": b}) for t, b in ((2, 2), (3, 3), (5, 2), (2, 7))]
    return {"schema": "output-lab-correctness-v1", "status": "passed", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "cases": len(case_rows), "comparisons": len(checks.rows), "max_dense_error": dense_error, "max_js_error": js_error,
            "finite_difference_max_error": max(r["max_error"] for r in finite), "finite_differences": finite,
            "fault_cases": len(faults), "fault_results": faults, "case_results": case_rows,
            "no_full_score_allocation": all(a["no_full_score_allocation"] for a in audits),
            "no_score_saved_for_backward": all(a["no_score_saved_for_backward"] for a in audits),
            "allocation_audit_scope": "Dispatch/hook audits of [N,V,D]=[5,7,3], blocks [2,2], [3,3], [5,2], [2,7]. Full-size [5,7] edge cases in the correctness sweep may form one full-sized tile and are excluded from this claim.",
            "allocation_audits": audits, "checks": checks.rows}


def logical_counts(n, v, d, token_block, vocab_block, chunked, valid_count):
    return {"unit": "logical elements; categories are not summed into an allocator/RSS peak", "dense_logits_elements": n * v,
            "maximum_score_tile_elements": min(n, token_block) * min(v, vocab_block) if chunked else n * v,
            "shared_input_parameter_elements": n * d + v * d + v,
            "gradient_buffer_elements": n * d + v * d + v,
            "saved_row_statistics_elements": 2 * n if chunked else 0,
            "effective_labels_elements": n, "score_forward_elements": n * v,
            "score_recomputed_elements": n * v if chunked else 0,
            "notes": "Counts refer to this implementation with all benchmark rows valid. The same score tile is overwritten for exp/p/dZ; temporary row vectors and dH/dW GEMM results also exist. H/W/bias/labels and dH/dW/db are separate persistent categories. No teaching snapshots are created."}


def benchmark_inputs(n, v, d, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(n, d, generator=generator, dtype=DTYPE) * .3,
            torch.randn(v, d, generator=generator, dtype=DTYPE) / math.sqrt(d),
            torch.randn(v, generator=generator, dtype=DTYPE) * .05,
            torch.randint(v, (n,), generator=generator, dtype=torch.long))


def measure(name, method, fixed, blocks, warmup, repeats, errors):
    H = fixed[0].clone().requires_grad_(); W = fixed[1].clone().requires_grad_(); bias = fixed[2].clone().requires_grad_()
    labels = fixed[3]
    n, d = H.shape; v = W.shape[0]
    implementation = chunked_linear_cross_entropy if method == "chunked" else dense_linear_cross_entropy

    def operation():
        H.grad = W.grad = bias.grad = None
        loss = implementation(H, W, labels, bias=bias, token_block=blocks[0], vocab_block=blocks[1])
        loss.backward()
        return loss.detach()

    for _ in range(warmup):
        operation()
    samples = []
    # No profiler, hooks, dispatch instrumentation, tensor cloning, I/O or RNG inside the timed region.
    # Gradient reset, argument validation, forward, and backward are part of each operation.
    for _ in range(repeats):
        start = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    ledger = SavedLedger()
    with torch.autograd.graph.saved_tensors_hooks(ledger.pack, ledger.unpack):
        operation()
    gc.collect()
    assert ledger.live == 0
    summary = ledger.summary((n, v))
    if method == "chunked":
        assert summary["full_score_shape_save_requests"] == 0
        assert len(summary["saved_tensors"]) == 6
        expected_bytes = (n * d + v * d + v + 2 * n) * 8 + n * 8
        assert summary["peak_referenced_bytes"] == expected_bytes, summary
    return {"name": name, "method": method, "shape": [n, v, d], "blocks": list(blocks) if method == "chunked" else None,
            "valid_positions": n, "seed": 20261028 + n, "warmup_runs": warmup, "repetitions": repeats,
            "wall_ms": {"median": statistics.median(samples), "min": min(samples), "max": max(samples), "samples": samples},
            "saved_tensor_storage": summary, "logical": logical_counts(n, v, d, *blocks, method == "chunked", n),
            "reference_errors": errors,
            "timing_scope": "CPU forward+backward, same fixed H/W/bias/labels; includes input validation and resetting .grad=None. Excludes preallocated-input cloning, RNG, file I/O, hooks, dispatch audits and profiler. Hook sampling uses a separate call."}


def benchmarks(warmup=3, repeats=12):
    rows = []
    for n, v, d, configs in ((64, 257, 32, ((8, 32), (16, 64), (64, 64))),
                              (128, 1021, 64, ((16, 64), (32, 256), (128, 256)))):
        fixed = benchmark_inputs(n, v, d, 20261028 + n)
        inputs = {"H": fixed[0].tolist(), "W": fixed[1].tolist(), "bias": fixed[2].tolist(), "labels": fixed[3].tolist(), "mask": [True] * n, "ignoreIndex": -100}
        dense = run_case(inputs, {"tokenBlock": n, "vocabBlock": v}, chunked=False)
        rows.append(measure(f"dense-{n}x{v}x{d}", "dense", fixed, (n, v), warmup, repeats, {k: 0. for k in ("loss", "dH", "dW", "db")}))
        for t, b in configs:
            actual = run_case(inputs, {"tokenBlock": t, "vocabBlock": b})
            errors = {key: max_error(actual[key], dense[key]) for key in ("loss", "dH", "dW", "db")}
            assert all(error < 2e-9 for error in errors.values()), errors
            rows.append(measure(f"chunked-{n}x{v}x{d}-t{t}-v{b}", "chunked", fixed, (t, b), warmup, repeats, errors))
    return rows


def runtime_environment():
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "dtype": "float64", "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def source_metadata():
    directory = ROOT / "artifacts/output-source-audit-2026-10-28"
    specs = (("apple-cce-readme.md", "https://raw.githubusercontent.com/apple/ml-cross-entropy/main/README.md", "README directly fetched and read: intro, requirements, Basic usage, implementation table with cce_exact, Generalized Usage including bias. No repository redirect claim; no fixed git commit available. Source bytes pinned by SHA256."),
             ("apple-research.html", "https://machinelearning.apple.com/research/cut-your-losses", "Read title, author list, ICLR / February 2025 metadata, abstract and paper-specific GPU claims. Those performance values are not used as CPU measurements."),
             ("paper.html", "https://arxiv.org/html/2411.09009v2", "Read title/author/version metadata (10 March 2025), Abstract, sections 3 and 4.1-4.3, 5.2 Gradient Filtering, appendix B Removing Ignored Tokens and appendix E Floating Point Addition. Other performance tables were not independently reproduced."))
    rows = []
    for name, url, read_scope in specs:
        path = directory / name
        rows.append({"url": url, "accessed": "2026-10-28", "read_scope": read_scope, "snapshot_path": str(path.relative_to(ROOT)), "sha256": sha256(path) if path.exists() else None})
    rows.append({"url": "https://docs.pytorch.org/docs/2.9/autograd.html", "accessed": "2026-10-28", "read_scope": "Installed PyTorch 2.9.0 docstrings read for saved_tensors_hooks, Function.forward/backward/save_for_backward and F.cross_entropy. Corresponding official URL supplied for integration; online docs body not independently fetched."})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--correctness-only", action="store_true")
    mode.add_argument("--timing-only", action="store_true")
    parser.add_argument("--correctness-report", type=Path)
    parser.add_argument("--contention-note", default="No independently monitored exclusive-CPU guarantee; one torch thread. Coordinate timing with other lab runs.")
    args = parser.parse_args()
    configure_runtime()
    directory = args.out_dir.resolve()
    fixture_path = directory / "reference.json"
    fixture = json.loads(fixture_path.read_text())
    if args.timing_only:
        if args.correctness_report is None:
            parser.error("--timing-only requires --correctness-report from a completed independent correctness run.")
        checked = json.loads(args.correctness_report.read_text())
        if checked.get("status") != "passed" or checked.get("fixture_sha256") != sha256(fixture_path) or checked.get("python_sha256") != sha256(__file__):
            raise RuntimeError("Timing-only requires a passing correctness file for these exact fixture and Python source bytes.")
        check_origin = str(args.correctness_report.resolve().relative_to(ROOT))
    else:
        checked = correctness(fixture)
        checked["fixture_sha256"], checked["python_sha256"] = sha256(fixture_path), sha256(__file__)
        checked["runtime"] = runtime_environment()
        write_json(directory / "correctness.json", checked)
        check_origin = str((directory / "correctness.json").relative_to(ROOT))
        print(f'Independent correctness passed: {checked["cases"]} cases / {checked["comparisons"]} comparisons; dense {checked["max_dense_error"]:.3e}, JS {checked["max_js_error"]:.3e}, FD {checked["finite_difference_max_error"]:.3e}.', flush=True)
    environment = {"runtime": runtime_environment(), "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}, "dependencies_added": False, "no_pretrained_downloads": True}
    write_json(directory / "environment.json", environment)
    if args.correctness_only:
        print(f"Correctness only; no timing samples. {directory}/correctness.json", flush=True)
        return
    rows = benchmarks()
    report = {"schema": "output-lab-v1", "status": "passed", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "runtime": runtime_environment(),
              "provenance": {"run_id": directory.name, "report_path": str((directory / "report.json").relative_to(ROOT)), "fixture_sha256": sha256(fixture_path), "python_sha256": sha256(__file__), "correctness_report_path": check_origin,
                             "benchmark_seed_formula": "20261028 + N", "generator": "torch.Generator(device='cpu')", "no_downloaded_models": True},
              "correctness": {key: checked[key] for key in ("cases", "comparisons", "max_dense_error", "max_js_error", "finite_difference_max_error", "fault_cases", "no_full_score_allocation", "no_score_saved_for_backward", "allocation_audit_scope")},
              "measurements": rows, "sources": source_metadata(), "contention_note": args.contention_note,
              "scope": "Independent small CPU H[N,D] @ W[V,D].T + optional bias, scalar valid-position mean, exact full first derivatives. No gradient filtering, tied input lookup, class weights, smoothing, higher derivatives, GPU kernels, GPU performance or actual Apple CCE package.",
              "zero_valid_policy": "Differentiable zero sentinel; mean loss is undefined and the caller must skip optimizer update. Dense reference uses sum/count for nonempty and the same explicit sentinel for empty; no NaN mean is hidden.",
              "measurement_scope": "Single-thread CPU Python/PyTorch float64 forward+backward. Each method has 3 warmups and 12 independent time samples without hooks/profiler/audit. Hooks run separately and count unique backing-storage references retained for backward; they are not allocator peaks or process RSS. Logical score tile, saved statistics, shared parameters and gradients are separate categories. No GPU speed claim. Full-sized single blocks in correctness are edge references; no-full-score observation applies to sub-full audited and benchmarked blocks."}
    write_json(directory / "report.json", report)
    for row in rows:
        print(f'{row["name"]}: {row["wall_ms"]["median"]:.3f} ms median, {row["saved_tensor_storage"]["peak_referenced_bytes"]} saved-reference bytes', flush=True)
    print(f"Output lab passed. Report: {directory}/report.json", flush=True)


if __name__ == "__main__":
    main()
