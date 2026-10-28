"""Actual faults and deterministic replays of the shared learning-core-v1.

Every number in the report is computed from this run's tensors. Binary tensors
are kept for independent loss re-computation; JSON never contains NaN/Infinity.
This module does not train on import. See docs/training-failures-implementation.md.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from learning_core import (
    CORE_API_VERSION, CONTENT_IDS, EOS, IGNORE_INDEX, VOCAB, VOCAB_SIZE,
    RecordBatchStream, capture_rng_state, collate_records, configure_cpu,
    environment_metadata, forward_loss, loss_from_logits, make_model,
    make_task_splits, model_config, parameter_count, restore_rng_state,
    seed_all, split_digest,
)

SCHEMA = "failure-replay-v1"
ROOT = Path(__file__).resolve().parents[1]
CASES = ("overflow", "high-lr", "labels")
VARIANTS = ("baseline", "faulty", "repaired")
DEFAULT_CONFIG = {
    "model": {"width": 32, "depth": 2, "heads": 4, "dropout": 0.1,
              "norm": "pre", "residual_scale": 1.0, "ff_multiplier": 2,
              "max_seq_len": 64, "norm_eps": 1e-5},
    "seed": 11, "data_seed": 20261028, "stream_seed": 97,
    "batch_size": 8, "warmup_updates": 8,
    "optimizer": {"lr": 0.003, "betas": (0.9, 0.999), "eps": 1e-8,
                  "weight_decay": 0.01, "foreach": False},
    "faults": {"shared_logit_shift": 128.0, "high_lr": 0.6,
               "relative_update_threshold": 0.5},
    "reduction": "token", "threads": 1,
}


def scalar(value) -> dict:
    value = float(value)
    status = ("finite" if math.isfinite(value) else "nan" if math.isnan(value)
              else "positive_infinity" if value > 0 else "negative_infinity")
    return {"value": value if status == "finite" else None, "status": status}


def tensor_summary(tensor: Tensor) -> dict:
    """RMS/L2 are accumulated in float64, rather than squaring in float32."""
    value = tensor.detach().cpu()
    finite = torch.isfinite(value)
    result = {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
              "numel": value.numel(), "finite_count": int(finite.sum()),
              "nan_count": int(torch.isnan(value).sum()),
              "positive_infinity_count": int(torch.isposinf(value).sum()),
              "negative_infinity_count": int(torch.isneginf(value).sum()),
              "finite": bool(finite.all()), "rms": None, "max_abs": None, "l2": None}
    if result["finite"]:
        double = value.double()
        result.update(rms=float(double.square().mean().sqrt()) if value.numel() else 0.0,
                      max_abs=float(double.abs().max()) if value.numel() else 0.0,
                      l2=float(double.square().sum().sqrt()))
    return result


def json_value(value):
    if isinstance(value, Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return scalar(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    # Exclusive creation also protects reports inside a partially completed run.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_value(value), handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def digest(value) -> str:
    """Content hash for tensors, optimizer, all RNGs and cursor, not pickle bytes."""
    hasher = hashlib.sha256()

    def visit(item):
        if isinstance(item, Tensor):
            data = item.detach().cpu().contiguous()
            hasher.update(f"tensor:{data.dtype}:{list(data.shape)}:".encode())
            hasher.update(data.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            hasher.update(f"ndarray:{item.dtype}:{list(item.shape)}:".encode())
            hasher.update(item.tobytes())
        elif isinstance(item, dict):
            hasher.update(b"dict{")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key); visit(item[key])
            hasher.update(b"}")
        elif isinstance(item, (tuple, list)):
            hasher.update((type(item).__name__ + "[").encode())
            for child in item:
                visit(child)
            hasher.update(b"]")
        else:
            hasher.update(f"{type(item).__name__}:{repr(item)};".encode())

    visit(value)
    return hasher.hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_difference(actual: Tensor, expected: Tensor) -> dict:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"equal": False, "finite": False, "max_abs": None, "l2": None,
                "relative_l2": None, "reason": "shape-or-dtype"}
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    both_finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    # A repeated NaN trajectory may match its masks, but is never called finite.
    same_masks = all(torch.equal(fn(actual), fn(expected)) for fn in
                     (torch.isnan, torch.isposinf, torch.isneginf))
    positions = torch.isfinite(actual) & torch.isfinite(expected)
    equal = same_masks and torch.equal(actual[positions], expected[positions])
    if not both_finite:
        return {"equal": equal, "finite": False, "max_abs": None, "l2": None,
                "relative_l2": None, "reason": "nonfinite"}
    delta = actual.double() - expected.double()
    l2 = float(delta.square().sum().sqrt())
    denominator = float(expected.double().square().sum().sqrt())
    return {"equal": equal, "finite": True,
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
            "l2": l2, "relative_l2": l2 / max(denominator, 1e-12), "reason": None}


def first_nonfinite(events: list[dict], phase=None, kind=None, kinds=None):
    return next((event for event in events if not event["stats"]["finite"]
                 and (phase is None or event["phase"] == phase)
                 and (kind is None or event["kind"] == kind)
                 and (kinds is None or event["kind"] in kinds)), None)


def flatten_output(value, suffix=""):
    if isinstance(value, Tensor):
        yield suffix, value
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from flatten_output(child, f"{suffix}[{index}]")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_output(child, f"{suffix}.{key}")


class TensorTrace:
    """Ordered observations; hooks never change tensors or gradients.

    A module output can alias another module's output. It is observed at both
    boundaries, but receives only one tensor-gradient hook. Parameter hooks run
    when autograd computes their gradient, not in named_parameters() order.
    """
    def __init__(self):
        self.events, self.tensors, self.handles = [], {}, []
        self.phase = "step"
        self.occurrences, self.hooked = {}, set()
        self.live_tensors = []  # Keep identities alive until hooks are removed.

    def record(self, name: str, value: Tensor, kind: str, gradient=False):
        occurrence_key = (self.phase, kind, name)
        occurrence = self.occurrences.get(occurrence_key, 0)
        self.occurrences[occurrence_key] = occurrence + 1
        key = f"{self.phase}|{kind}|{name}|{occurrence}"
        tensor = value.detach().cpu().clone()
        self.tensors[key] = tensor
        event = {"index": len(self.events), "key": key, "phase": self.phase,
                 "kind": kind, "name": name, "occurrence": occurrence,
                 "stats": tensor_summary(tensor)}
        self.events.append(event)
        if gradient and value.requires_grad and id(value) not in self.hooked:
            self.hooked.add(id(value)); self.live_tensors.append(value)
            source_phase = self.phase

            def observe_gradient(grad, name=name, phase=source_phase):
                previous = self.phase
                self.phase = phase
                self.record(f"d({name})", grad, "gradient")
                self.phase = previous
                return None

            self.handles.append(value.register_hook(observe_gradient))
        return value

    def attach(self, model):
        for name, module in model.named_modules():
            def observe(module, args, output, name=name or "model"):
                for suffix, tensor in flatten_output(output):
                    self.record(f"{name}.output{suffix}", tensor, "activation", gradient=True)
                return None

            self.handles.append(module.register_forward_hook(observe))
        for name, parameter in model.named_parameters():
            def observe_parameter(grad, name=name):
                self.record(f"parameter:{name}.grad", grad, "parameter-gradient")
                return None

            self.handles.append(parameter.register_hook(observe_parameter))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear(); self.live_tensors.clear(); self.hooked.clear()


def rotate_labels(labels: Tensor) -> Tensor:
    """Wrong *finite* targets, with exactly the same supervision/EOS mask.

    A->B->C->D->A at content targets. EOS and -100 positions are unchanged.
    The batch inputs and teacher-forced history remain the correct record.
    """
    faulty = labels.clone()
    content = (labels >= CONTENT_IDS[0]) & (labels <= CONTENT_IDS[-1])
    faulty[content] = CONTENT_IDS[0] + (labels[content] - CONTENT_IDS[0] + 1) % len(CONTENT_IDS)
    return faulty


def instrumented_loss(logits: Tensor, labels: Tensor, trace: TensorTrace,
                      shift: float = 0.0, naive: bool = False) -> Tensor:
    """Shared stimulus, same mathematical CE. Explicit exp is monitored first."""
    if not bool((labels != IGNORE_INDEX).any()):
        raise ValueError("no supervised tokens: caller must skip the update")
    shifted = trace.record("loss.shifted_logits", logits + shift, "loss-op", gradient=True)
    if naive:
        exponentials = trace.record("loss.exp", torch.exp(shifted), "loss-op", gradient=True)
        denominator = trace.record("loss.exp_sum", exponentials.sum(dim=-1, keepdim=True),
                                   "loss-op", gradient=True)
        probabilities = trace.record("loss.probabilities", exponentials / denominator,
                                     "loss-op", gradient=True)
        log_probabilities = trace.record("loss.log_probabilities", torch.log(probabilities),
                                         "loss-op", gradient=True)
    else:
        log_probabilities = trace.record("loss.log_probabilities", F.log_softmax(shifted, dim=-1),
                                         "loss-op", gradient=True)
    valid = labels != IGNORE_INDEX
    selected = trace.record("loss.target_log_probabilities", log_probabilities[valid].gather(
        -1, labels[valid].unsqueeze(-1)).squeeze(-1), "loss-op", gradient=True)
    return trace.record("loss.mean_nll", -selected.mean(), "loss-op", gradient=True)


def recompute_loss(logits: Tensor, labels: Tensor, shift=0.0, naive=False) -> Tensor:
    """Independent expression, executed from saved actual logits/labels."""
    valid = labels != IGNORE_INDEX
    if not bool(valid.any()):
        raise ValueError("no supervised tokens")
    values = logits.detach() + shift
    if naive:
        exp = values.exp()
        log_p = (exp / exp.sum(dim=-1, keepdim=True)).log()
        return -log_p[valid].gather(-1, labels[valid, None]).mean()
    return F.cross_entropy(values.reshape(-1, values.shape[-1]), labels.reshape(-1),
                           ignore_index=IGNORE_INDEX, reduction="sum") / valid.sum()


def new_optimizer(model, config):
    return torch.optim.AdamW(model.parameters(), **config)


def capture_state(model, optimizer, stream, counters, config) -> dict:
    return {"model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "rng": capture_rng_state(), "stream": copy.deepcopy(stream.state_dict()),
            "counters": copy.deepcopy(counters), "config": copy.deepcopy(config),
            "modes": {name: module.training for name, module in model.named_modules()}}


def restore_checkpoint(checkpoint):
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("core_api") != CORE_API_VERSION:
        raise ValueError("checkpoint schema/core API differs")
    if digest(checkpoint["state"]) != checkpoint["state_digest"]:
        raise ValueError("checkpoint state digest does not match its saved payload")
    config = checkpoint["state"]["config"]
    model = make_model(seed=config["seed"], **config["model"])
    optimizer = new_optimizer(model, config["optimizer"])
    model.load_state_dict(checkpoint["state"]["model"])
    optimizer.load_state_dict(checkpoint["state"]["optimizer"])
    stream = RecordBatchStream(checkpoint["train_records"], config["batch_size"], config["stream_seed"])
    stream.load_state_dict(checkpoint["state"]["stream"])
    modes = checkpoint["state"]["modes"]
    for name, module in model.named_modules():
        module.training = modes[name]
    restore_rng_state(checkpoint["state"]["rng"])
    counters = copy.deepcopy(checkpoint["state"]["counters"])
    loaded = capture_state(model, optimizer, stream, counters, config)
    if digest(loaded) != digest(checkpoint["state"]):
        raise AssertionError("loaded pre-step snapshot differs before fault injection")
    return model, optimizer, stream, counters


def batch_view(batch):
    return {key: json_value(batch[key]) for key in
            ("input_ids", "labels", "attention_mask", "record_ids", "group_ids",
             "records", "supervised_tokens", "supervised_per_record", "input_tokens")}


def perform_replay(checkpoint: dict, case: str, variant: str, selection="target"):
    if case not in CASES or variant not in VARIANTS or selection not in {"target", "control"}:
        raise ValueError("unsupported replay configuration")
    model, optimizer, stream, counters = restore_checkpoint(checkpoint)
    config = checkpoint["state"]["config"]
    pre_step_digest = digest(capture_state(model, optimizer, stream, counters, config))
    selected = stream.next_records()
    if selection == "control":
        # An explicitly separate normal batch, from the same saved data order.
        selected = stream.next_records()
    batch = collate_records(selected)
    following = collate_records(stream.next_records())
    target_ids = checkpoint["target_record_ids"]
    triggered = batch["record_ids"] == target_ids
    enabled = triggered and variant == "faulty"
    # All three target replays share this finite offset, including the baseline.
    shift = config["faults"]["shared_logit_shift"] if case == "overflow" and triggered else 0.0
    labels = rotate_labels(batch["labels"]) if enabled and case == "labels" else batch["labels"].clone()
    trace = TensorTrace()
    for key in ("input_ids", "attention_mask"):
        trace.record(f"batch.{key}", batch[key], "input")
    trace.record("batch.labels", labels, "input")
    trace.record("batch.supervised_count", (labels != IGNORE_INDEX).sum(), "input")
    trace.attach(model)
    optimizer.zero_grad(set_to_none=True)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    try:
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = instrumented_loss(logits, labels, trace, shift, enabled and case == "overflow")
        observed = loss.detach().clone()
        # This check is against actual detached tensors, including the faulty objective.
        recomputed = recompute_loss(logits, labels, shift, enabled and case == "overflow")
        correct_label_loss = recompute_loss(logits, batch["labels"], shift, False)
        unshifted_core_loss = loss_from_logits(logits.detach(), batch["labels"])
        if scalar(observed.item())["status"] != scalar(recomputed.item())["status"]:
            raise AssertionError("actual and independently recomputed loss statuses differ")
        if bool(torch.isfinite(observed)) and not torch.allclose(observed, recomputed, atol=2e-6, rtol=2e-6):
            raise AssertionError("loss differs from actual tensor re-computation")
        loss.backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        gradient_vector = torch.cat([value.reshape(-1) for value in gradients.values()])
        gradient_stats = tensor_summary(gradient_vector)
        lr = config["faults"]["high_lr"] if enabled and case == "high-lr" else config["optimizer"]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        # Numerical safety is common to all variants, rather than a hidden repair.
        updated = bool(torch.isfinite(loss)) and gradient_stats["finite"]
        if updated:
            optimizer.step()
            counters["updates"] += 1
        counters["attempted_steps"] += 1
        counters["input_tokens"] += batch["input_tokens"]
        counters["supervised_tokens"] += batch["supervised_tokens"]
        parameter_rows, deltas, after = [], {}, {}
        trace.phase = "post-update"
        for name, parameter in model.named_parameters():
            value = parameter.detach().clone()
            delta = value - before[name]
            after[name], deltas[name] = value, delta
            stats, old_stats, delta_stats = tensor_summary(value), tensor_summary(before[name]), tensor_summary(delta)
            denominator = max(old_stats["l2"], 1e-12) if old_stats["finite"] else None
            relative = delta_stats["l2"] / denominator if delta_stats["finite"] and denominator else None
            trace.record(f"parameter:{name}", value, "parameter")
            trace.record(f"delta:{name}", delta, "parameter-update")
            parameter_rows.append({"name": name, "before": old_stats, "after": stats,
                                   "gradient": tensor_summary(gradients[name]) if name in gradients else None,
                                   "delta": delta_stats, "relative_update": relative})
        total_before = tensor_summary(torch.cat([value.reshape(-1) for value in before.values()]))
        total_delta = tensor_summary(torch.cat([value.reshape(-1) for value in deltas.values()]))
        relative_update = total_delta["l2"] / max(total_before["l2"], 1e-12)
        # A real next forward at updated parameters, with the actual next RNG state.
        trace.phase = "next-forward"
        with torch.no_grad():
            trace.record("batch.input_ids", following["input_ids"], "input")
            next_logits = model(following["input_ids"], attention_mask=following["attention_mask"])
            next_loss = loss_from_logits(next_logits, following["labels"])
            trace.record("loss.mean_nll", next_loss, "loss-op")
        final_state = capture_state(model, optimizer, stream, counters, config)
        loss_checks = {"recomputed_from_saved_logits": scalar(recomputed.item()),
                       "recomputed_same_status": scalar(observed.item())["status"] == scalar(recomputed.item())["status"],
                       "finite_recomputed_abs_error": float(abs(observed.item() - recomputed.item()))
                       if bool(torch.isfinite(observed) and torch.isfinite(recomputed)) else None,
                       "correct_labels_stable_loss": scalar(correct_label_loss.item()),
                       "unshifted_core_loss": scalar(unshifted_core_loss.item()),
                       "stable_shift_abs_error_vs_core": float(abs(correct_label_loss.item() - unshifted_core_loss.item())),
                       "wrong_label_loss_minus_correct": float(observed.item() - correct_label_loss.item())
                       if bool(torch.isfinite(observed)) else None}
        result = {"schema": SCHEMA, "case": case, "variant": variant, "selection": selection,
                  "pid": os.getpid(), "triggered": triggered, "fault_enabled": enabled,
                  "loaded_pre_step_digest": pre_step_digest,
                  "loaded_components": {key: digest(value) for key, value in checkpoint["state"].items()},
                  "counters_before": checkpoint["state"]["counters"], "counters_after": counters,
                  "cursor_before": {key: checkpoint["state"]["stream"][key] for key in ("cursor", "epoch")},
                  "cursor_after": {key: final_state["stream"][key] for key in ("cursor", "epoch")},
                  "batch": batch_view(batch), "actual_labels": labels.tolist(),
                  "labels_changed": int((labels != batch["labels"]).sum()),
                  "supervised_count": int((labels != IGNORE_INDEX).sum()),
                  "next_batch": batch_view(following), "lr": lr, "shared_logit_shift": shift,
                  "loss": scalar(observed.item()), "loss_checks": loss_checks,
                  "gradient": gradient_stats, "parameters": parameter_rows,
                  "update": {"applied": updated, "l2": total_delta["l2"],
                             "relative_l2": relative_update,
                             "threshold": config["faults"]["relative_update_threshold"],
                             "above_threshold": relative_update > config["faults"]["relative_update_threshold"],
                             "threshold_kind": "declared diagnostic heuristic, not a universal stability bound"},
                  "next_loss": scalar(next_loss.item()),
                  "first_forward_nonfinite": first_nonfinite(trace.events, phase="step",
                      kinds={"input", "activation", "loss-op"}),
                  "first_backward_nonfinite": next((event for event in trace.events
                      if event["kind"] in {"gradient", "parameter-gradient"} and not event["stats"]["finite"]), None),
                  "events": trace.events, "final_state_digest": digest(final_state),
                  "final_components": {key: digest(value) for key, value in final_state.items()},
                  "all_finite": all(event["stats"]["finite"] for event in trace.events),
                  "gradient_order": [event["key"] for event in trace.events
                                     if event["kind"] in {"gradient", "parameter-gradient"}]}
        tensors = {"events": trace.tensors, "logits": logits.detach().clone(), "labels": labels,
                   "correct_labels": batch["labels"], "loss": observed,
                   "next_logits": next_logits.detach().clone(), "gradients": gradients,
                   "parameters_before": before, "parameters_after": after, "deltas": deltas}
        return result, tensors, final_state
    finally:
        trace.close()


def compare_replays(actual: dict, actual_tensors: dict, expected: dict, expected_tensors: dict):
    compared, unmatched = [], []
    for event in actual["events"]:
        key = event["key"]
        if key not in expected_tensors["events"]:
            unmatched.append(key)
            continue
        difference = tensor_difference(actual_tensors["events"][key], expected_tensors["events"][key])
        compared.append({"index": event["index"], "key": key, "phase": event["phase"],
                         "kind": event["kind"], "name": event["name"], **difference})
    first = next((row for row in compared if not row["equal"]), None)
    next_divergence = next((row for row in compared if row["phase"] == "next-forward"
                            and row["kind"] == "activation" and not row["equal"]), None)
    gradient = tensor_difference(torch.cat([g.reshape(-1) for g in actual_tensors["gradients"].values()]),
                                 torch.cat([g.reshape(-1) for g in expected_tensors["gradients"].values()]))
    return {"first_divergence": first, "first_next_forward_divergence": next_divergence,
            "gradients": gradient,
            "parameters_after": tensor_difference(torch.cat([g.reshape(-1) for g in actual_tensors["parameters_after"].values()]),
                torch.cat([g.reshape(-1) for g in expected_tensors["parameters_after"].values()])),
            "next_logits": tensor_difference(actual_tensors["next_logits"], expected_tensors["next_logits"]),
            "all_compared_tensors_equal": all(row["equal"] for row in compared),
            "identical_event_keys": [event["key"] for event in actual["events"]] ==
                                    [event["key"] for event in expected["events"]],
            "unmatched_event_keys": unmatched,
            "final_state_equal": actual["final_state_digest"] == expected["final_state_digest"],
            "event_differences": compared}


def save_replay(directory: Path, checkpoint_path: Path, case, variant, selection):
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result, tensors, state = perform_replay(checkpoint, case, variant, selection)
    result["checkpoint_sha256"] = file_digest(checkpoint_path)
    result["runtime"] = environment_metadata()
    write_json(directory / "trace.json", result)
    torch.save(tensors, directory / "tensors.pt")
    torch.save(state, directory / "final-state.pt")
    return result


def anomaly_probe(checkpoint, case):
    """A separate replay: debugger may stop early, ordinary trace stays complete."""
    model, _, stream, _ = restore_checkpoint(checkpoint)
    config = checkpoint["state"]["config"]
    batch = collate_records(stream.next_records())
    labels = rotate_labels(batch["labels"]) if case == "labels" else batch["labels"]
    trace = TensorTrace()
    captured = io.StringIO()
    failure = None
    from contextlib import redirect_stderr
    with warnings.catch_warnings(record=True) as notices, redirect_stderr(captured):
        warnings.simplefilter("always")
        try:
            with torch.autograd.detect_anomaly(check_nan=True):
                logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = instrumented_loss(logits, labels, trace,
                    config["faults"]["shared_logit_shift"] if case == "overflow" else 0.0,
                    case == "overflow")
                loss.backward()
        except RuntimeError as error:
            failure = str(error)
        finally:
            trace.close()
    return {"raised": failure is not None, "error": failure,
            "warnings": [str(item.message) for item in notices],
            "stderr": captured.getvalue(),
            "phase": "pre-update-forward-and-backward",
            "optimizer_update_executed": False, "next_forward_executed": False,
            "scope": ("pre-update backward only; high-LR optimizer action NOT executed in this probe"
                      if case == "high-lr" else
                      "separate pre-update backward debugger; does not test label semantics")}


def prepare_checkpoint(config: dict):
    seed_all(config["seed"])
    splits = make_task_splits(data_seed=config["data_seed"])
    model = make_model(seed=config["seed"], **config["model"])
    optimizer = new_optimizer(model, config["optimizer"])
    stream = RecordBatchStream(splits["train"], config["batch_size"], config["stream_seed"])
    counters = {"updates": 0, "attempted_steps": 0, "input_tokens": 0, "supervised_tokens": 0}
    history = []
    model.train()
    for index in range(config["warmup_updates"]):
        batch = collate_records(stream.next_records())
        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss(model, batch)
        loss.backward(); optimizer.step()
        counters["updates"] += 1; counters["attempted_steps"] += 1
        counters["input_tokens"] += batch["input_tokens"]
        counters["supervised_tokens"] += batch["supervised_tokens"]
        history.append({"step": index + 1, "loss": scalar(loss.detach().item()),
                        "record_ids": batch["record_ids"], "input_tokens": batch["input_tokens"],
                        "supervised_tokens": batch["supervised_tokens"], "lr": config["optimizer"]["lr"]})
    state = capture_state(model, optimizer, stream, counters, config)
    target = stream.next_records()
    stream.load_state_dict(state["stream"])
    return {"schema": SCHEMA, "state": state, "state_digest": digest(state),
            "core_api": CORE_API_VERSION, "train_records": splits["train"],
            "split_digest": split_digest(splits), "history": history,
            "target_record_ids": [row["id"] for row in target],
            "parameter_count": parameter_count(model), "model_config": model_config(model)}


def primary_references():
    # These are installed, versioned PyTorch docs, read in this implementation.
    entries = (
        ("torch-autograd-2.9", "PyTorch 2.9 — Autograd / anomaly detection",
         "https://docs.pytorch.org/docs/2.9/autograd.html", torch.autograd.detect_anomaly.__doc__,
         "forward traceback for a failing backward; check_nan raises on backward NaN; debugging overhead"),
        ("torch-hooks-2.9", "PyTorch 2.9 — Module / Tensor hooks",
         "https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html", torch.nn.Module.register_forward_hook.__doc__ + torch.Tensor.register_hook.__doc__,
         "module hooks run after forward; tensor hooks receive computed gradients; hooks may return None"),
        ("torch-logsoftmax-2.9", "PyTorch 2.9 — log_softmax / cross_entropy",
         "https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.log_softmax.html", F.log_softmax.__doc__ + F.cross_entropy.__doc__,
         "stable log_softmax formulation; logits inputs; ignore_index and reduction denominator"),
        ("torch-randomness-2.9", "PyTorch 2.9 — Reproducibility / CPU RNG",
         "https://docs.pytorch.org/docs/2.9/notes/randomness.html", torch.use_deterministic_algorithms.__doc__ + torch.random.get_rng_state.__doc__,
         "same software/hardware deterministic algorithms; flag alone insufficient; saved RNG is CPU-only"),
        ("torch-adamw-2.9", "PyTorch 2.9 — AdamW",
         "https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html", torch.optim.AdamW.__doc__,
         "LR multiplies decoupled decay and moment-normalized update; optimizer state contains moments/step"),
    )
    return [{"id": key, "title": title, "url": url, "version": "2.9.0",
             "actually_read": "installed PyTorch 2.9.0 docstrings, not an unread web page",
             "read_scope": scope, "docstring_sha256": hashlib.sha256(doc.encode()).hexdigest()}
            for key, title, url, doc, scope in entries]


def compact_report(report: dict, report_sha256: str) -> dict:
    """Projection of a *saved* actual report; no recomputed/synthetic metrics.

    Keep actual layer boundaries, loss ops, parameter-gradient callbacks and the
    measured first findings. Omitted events remain available in the full traces.
    Original indices are retained; the compact trace never renumbers execution.
    """
    compact = copy.deepcopy(report)
    compact["view"] = "compact-saved-report"
    compact["runtime"].pop("torch_build", None)
    compact["provenance"]["report_sha256"] = report_sha256
    compact["provenance"]["snapshot_kind"] = "deterministic projection of saved report.json"
    compact["provenance"]["saved_after_updates"] = report["config"]["warmup_updates"]
    compact["provenance"]["pending_step"] = report["config"]["warmup_updates"] + 1
    boundary_names = {"token_embedding.output", "embedding_dropout.output", "final_norm.output", "lm_head.output"}
    for item in compact["cases"]:
        for key, run in item["runs"].items():
            comparisons = item["comparisons"]["faulty_vs_baseline"]
            findings = [run["first_forward_nonfinite"], run["first_backward_nonfinite"],
                        comparisons["first_divergence"], comparisons["first_next_forward_divergence"]]
            finding_keys = {finding["key"] for finding in findings if finding}

            def keep(event):
                return (event["key"] in finding_keys or not event["stats"]["finite"] or
                        event["kind"] in {"input", "loss-op"} or
                        (key.startswith("target") and event["kind"] == "parameter-gradient") or
                        (key.startswith("target") and event["kind"] == "gradient" and event["name"].startswith("d(loss.")) or
                        (event["kind"] == "activation" and
                         (event["name"] in boundary_names or
                          (event["name"].startswith("blocks.") and event["name"].endswith(".output[0]")))))

            events = run["events"]
            run["events"] = [event for event in events if keep(event)]
            run["event_count_saved"] = len(events)
            run["events_complete"] = False
            for remove in ("gradient_order", "runtime", "loaded_components", "final_components", "pid"):
                run.pop(remove, None)
            # Parameter summaries are useful for the target. Normal controls
            # still retain their measured global update/gradient and proof.
            if key.startswith("control"):
                run["parameters"] = []
            else:
                run["parameters"] = [{"name": row["name"], "gradient": {
                    field: row["gradient"][field] for field in ("finite", "l2")} if row["gradient"] else None,
                    "delta": {field: row["delta"][field] for field in ("finite", "l2")},
                    "relative_update": row["relative_update"], "after": {"finite": row["after"]["finite"]}}
                    for row in run["parameters"]]
            run["batch"].pop("records", None)
            run["next_batch"] = {name: value for name, value in run["next_batch"].items()
                                 if name in {"record_ids", "group_ids", "supervised_tokens", "input_tokens"}}
        for key, comparison in item["comparisons"].items():
            relevant = {event["key"] for event in item["runs"][
                "target-faulty" if key == "faulty_vs_baseline" else
                "target-repaired" if key == "repaired_vs_baseline" else
                "control-faulty" if key == "control_faulty_vs_baseline" else "control-repaired"]["events"]}
            comparison["event_differences"] = [row for row in comparison["event_differences"] if row["key"] in relevant] if key == "faulty_vs_baseline" else []
        item["anomaly_probe"].pop("warnings", None)
        item["anomaly_probe"].pop("stderr", None)
    return compact


def export_snapshot(report_path: Path, output: Path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA or report.get("status") != "passed":
        raise ValueError("snapshot requires a saved, passed actual failure report")
    snapshot = compact_report(report, file_digest(report_path))
    text = ("// Generated from saved actual CPU runs; regenerate via failure_replay.py --snapshot-from.\n"
            "// Full report provenance and SHA-256 are included below.\n"
            "export const trainingFailuresSnapshot = " + json.dumps(
                json_value(snapshot), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + ";\n")
    output.write_text(text, encoding="utf-8")
    print(f"saved actual snapshot: {output} ({len(text.encode('utf-8'))} bytes)")


def run_lab(directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    configure_cpu(threads=1, deterministic=True)
    if torch.__version__ != "2.9.0" or np.__version__ != "2.3.4":
        raise RuntimeError("Use the pinned .venv-framework: PyTorch 2.9.0 / NumPy 2.3.4")
    config = copy.deepcopy(DEFAULT_CONFIG)
    checkpoint = prepare_checkpoint(config)
    checkpoint_path = directory / "pre-step.pt"
    torch.save(checkpoint, checkpoint_path)
    write_json(directory / "checkpoint-manifest.json", {
        "schema": SCHEMA, "sha256": file_digest(checkpoint_path),
        "state_digest": checkpoint["state_digest"], "model_config": checkpoint["model_config"],
        "parameter_count": checkpoint["parameter_count"], "config": config,
        "counters": checkpoint["state"]["counters"], "target_record_ids": checkpoint["target_record_ids"],
        "stream": {"cursor": checkpoint["state"]["stream"]["cursor"],
                   "epoch": checkpoint["state"]["stream"]["epoch"],
                   "order": checkpoint["state"]["stream"]["order"],
                   "record_ids": checkpoint["state"]["stream"]["record_ids"]},
        "component_digests": {key: digest(value) for key, value in checkpoint["state"].items()}})
    checks, cases = [], []

    def check(name, condition, detail=None):
        row = {"name": name, "passed": bool(condition), "detail": detail}
        checks.append(row)
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    for case in CASES:
        runs, raw = {}, {}
        for selection in ("target", "control"):
            for variant in VARIANTS:
                key = f"{selection}-{variant}"
                subdir = directory / case / key
                command = [sys.executable, str(Path(__file__).resolve()), "--replay", str(checkpoint_path),
                           "--case", case, "--variant", variant, "--selection", selection,
                           "--out-dir", str(subdir)]
                child = subprocess.run(command, check=True, capture_output=True, text=True)
                runs[key] = json.loads((subdir / "trace.json").read_text())
                raw[key] = torch.load(subdir / "tensors.pt", map_location="cpu", weights_only=False)
                check(f"{case}/{key}/fresh-process", runs[key]["pid"] != os.getpid())
                check(f"{case}/{key}/identical-loaded-snapshot", runs[key]["loaded_pre_step_digest"] == checkpoint["state_digest"])
                check(f"{case}/{key}/loss-from-actual-tensors", runs[key]["loss_checks"]["recomputed_same_status"] and
                      (runs[key]["loss_checks"]["finite_recomputed_abs_error"] is None or
                       runs[key]["loss_checks"]["finite_recomputed_abs_error"] < 1e-5))
                # Re-read saved binary logits; this is not a copied report scalar.
                again = recompute_loss(raw[key]["logits"], raw[key]["labels"], runs[key]["shared_logit_shift"],
                                       case == "overflow" and runs[key]["fault_enabled"])
                check(f"{case}/{key}/saved-loss-status", scalar(again.item())["status"] == runs[key]["loss"]["status"])
                check(f"{case}/{key}/saved-loss-value", not bool(torch.isfinite(again)) or
                      abs(again.item() - runs[key]["loss"]["value"]) < 1e-5)
                runs[key]["artifacts"] = {"trace": f"{case}/{key}/trace.json", "tensors": f"{case}/{key}/tensors.pt",
                                          "final_state": f"{case}/{key}/final-state.pt"}
        bad = compare_replays(runs["target-faulty"], raw["target-faulty"], runs["target-baseline"], raw["target-baseline"])
        repair = compare_replays(runs["target-repaired"], raw["target-repaired"], runs["target-baseline"], raw["target-baseline"])
        control_bad = compare_replays(runs["control-faulty"], raw["control-faulty"], runs["control-baseline"], raw["control-baseline"])
        control_repair = compare_replays(runs["control-repaired"], raw["control-repaired"], runs["control-baseline"], raw["control-baseline"])
        check(f"{case}/repair-all-tensors-and-state", repair["all_compared_tensors_equal"] and
              repair["identical_event_keys"] and repair["final_state_equal"])
        check(f"{case}/normal-batch-control", control_bad["all_compared_tensors_equal"] and
              control_bad["final_state_equal"] and control_repair["all_compared_tensors_equal"] and control_repair["final_state_equal"])
        check(f"{case}/fault-is-observed", bad["first_divergence"] is not None)
        if case == "overflow":
            first = runs["target-faulty"]["first_forward_nonfinite"]
            check("overflow/first-invalid-is-actual-exp", first is not None and first["name"] == "loss.exp" and
                  first["stats"]["positive_infinity_count"] > 0)
            check("overflow/pre-exp-logits-are-finite", tensor_summary(raw["target-faulty"]["logits"])["finite"])
            check("overflow/shared-offset", len({runs[f"target-{variant}"]["shared_logit_shift"] for variant in VARIANTS}) == 1)
            check("overflow/stable-shift-preserves-core-target", runs["target-baseline"]["loss_checks"]["stable_shift_abs_error_vs_core"] < 2e-5)
            check("overflow/nonfinite-update-is-skipped", not runs["target-faulty"]["update"]["applied"])
        elif case == "high-lr":
            check("high-lr/finite-forward-gradients-and-parameters", runs["target-faulty"]["all_finite"])
            check("high-lr/same-loss-and-gradients-before-update", bad["gradients"]["equal"] and
                  runs["target-faulty"]["loss"] == runs["target-baseline"]["loss"])
            check("high-lr/declared-update-heuristic", runs["target-faulty"]["update"]["above_threshold"] and
                  not runs["target-baseline"]["update"]["above_threshold"])
            check("high-lr/actual-next-forward-deviation", bad["first_next_forward_divergence"] is not None and
                  bad["next_logits"]["finite"] and bad["next_logits"]["max_abs"] > 0)
        else:
            check("labels/all-tensors-finite", runs["target-faulty"]["all_finite"])
            check("labels/same-supervision-mask-and-count", runs["target-faulty"]["supervised_count"] ==
                  runs["target-baseline"]["supervised_count"] and torch.equal(
                      raw["target-faulty"]["labels"] == IGNORE_INDEX, raw["target-baseline"]["labels"] == IGNORE_INDEX))
            check("labels/actual-label-and-gradient-divergence", runs["target-faulty"]["labels_changed"] > 0 and
                  not bad["gradients"]["equal"] and bad["gradients"]["finite"])
            check("labels/pre-loss-model-logits-identical", torch.equal(raw["target-faulty"]["logits"], raw["target-baseline"]["logits"]))
        probe = anomaly_probe(checkpoint, case)
        write_json(directory / case / "anomaly-probe.json", probe)
        check(f"{case}/anomaly-scope", probe["raised"] == (case == "overflow"))
        cases.append({"id": case, "runs": runs,
                      "comparisons": {"faulty_vs_baseline": bad, "repaired_vs_baseline": repair,
                                      "control_faulty_vs_baseline": control_bad, "control_repaired_vs_baseline": control_repair},
                      "anomaly_probe": probe})
        print(f"{case}: actual fault, fixed-snapshot repair and normal control passed", flush=True)
    report = {"schema": SCHEMA, "status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
              "runtime": environment_metadata(), "config": config,
              "scope": {"task": "shared symbolic reversal decoder; 8 warmup updates, one fault step and next forward",
                        "hardware": "single-thread CPU float32; no pretrained downloads or GPU",
                        "replay": "same saved pre-step state, new Python process per case/variant/batch",
                        "first_tensor": "earliest observed module output, explicit loss operation or tensor-gradient callback",
                        "limits": ["module boundaries do not expose all SDPA/GELU internal kernel tensors",
                                   "post-update parameters are observed in named-parameter order, not optimizer internal op order",
                                   "relative update > 0.5 is a declared local diagnostic heuristic",
                                   "finite tensors do not establish task/label semantics",
                                   "deterministic equality is checked on this software and CPU, not across hardware or releases",
                                   "trace hooks are diagnostic overhead; no performance claims"]},
              "provenance": {"report_path": str(directory / "report.json"), "checkpoint": "pre-step.pt",
                             "checkpoint_sha256": file_digest(checkpoint_path), "pre_step_digest": checkpoint["state_digest"],
                             "core_api": CORE_API_VERSION, "split_sha256": checkpoint["split_digest"],
                             "sources": {str(path.relative_to(ROOT)): file_digest(path) for path in
                                         (ROOT / "scripts/learning_core.py", Path(__file__).resolve())},
                             "command": [sys.executable, "scripts/failure_replay.py", "--out-dir", str(directory)],
                             "parameter_count": checkpoint["parameter_count"],
                             "saved_after_updates": config["warmup_updates"], "pending_step": config["warmup_updates"] + 1},
              "history": checkpoint["history"], "checks": checks, "cases": cases,
              "references": primary_references()}
    write_json(directory / "environment.json", report["runtime"])
    write_json(directory / "report.json", report)
    write_json(directory / "snapshot.json", compact_report(report, file_digest(directory / "report.json")))
    docs = {"detect_anomaly": torch.autograd.detect_anomaly.__doc__,
            "forward_hook": torch.nn.Module.register_forward_hook.__doc__,
            "tensor_hook": torch.Tensor.register_hook.__doc__,
            "log_softmax": F.log_softmax.__doc__, "cross_entropy": F.cross_entropy.__doc__,
            "deterministic_algorithms": torch.use_deterministic_algorithms.__doc__,
            "get_rng_state": torch.random.get_rng_state.__doc__, "AdamW": torch.optim.AdamW.__doc__}
    with (directory / "primary-installed-docstrings.txt").open("x", encoding="utf-8") as handle:
        for name, text in docs.items():
            handle.write(f"PyTorch {torch.__version__} / {name}\n{text}\n\n")
    print(f"{len(checks)} checks passed; report: {directory / 'report.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--snapshot-from", type=Path)
    parser.add_argument("--snapshot-output", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--selection", choices=("target", "control"), default="target")
    args = parser.parse_args()
    if args.snapshot_from:
        if not args.snapshot_output or args.replay or args.out_dir:
            parser.error("--snapshot-from requires --snapshot-output, without --out-dir/--replay")
        export_snapshot(args.snapshot_from.resolve(), args.snapshot_output.resolve())
        return
    if args.out_dir is None:
        parser.error("--out-dir is required for actual runs/replays")
    configure_cpu(threads=1, deterministic=True)
    if args.replay:
        if not args.case or not args.variant:
            parser.error("--replay requires --case and --variant")
        save_replay(args.out_dir.resolve(), args.replay.resolve(), args.case, args.variant, args.selection)
    else:
        run_lab(args.out_dir.resolve())


if __name__ == "__main__":
    main()
