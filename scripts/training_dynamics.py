"""Saved CPU experiments for initialization/width and sequence gradient noise.

The decoder experiments import learning_core.  The fixed-depth MLP is deliberately
separate: its input, hidden, readout and optimizer rules use the author mup 1.0.0
implementation.  This module never downloads a model and never reuses an output
directory.  All plotted coordinates and training curves are observed tensors.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata as metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "training-dynamics-v1"
SEEDS = (11, 29, 47)
MUP_VERSION = "1.0.0"
BASE_WIDTH = 16
INPUT_DIM = 8
OUTPUT_DIM = 3
WIDTHS = (16, 32, 64)
HELD_WIDTH = 128
WIDTH_LRS = (0.0003, 0.001, 0.003, 0.01, 0.03)
WIDTH_STEPS = 80
ADAM_CONFIG = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
DECODER_CONFIG = {"width": 32, "depth": 2, "heads": 4, "dropout": 0.0,
                  "norm": "pre", "residual_scale": 1.0, "ff_multiplier": 4,
                  "max_seq_len": 64, "norm_eps": 1e-5}
BATCH_SIZES = (4, 16, 32)
BATCH_LRS = (0.001, 0.003)
BATCH_RECORD_BUDGET = 4096
INPUT_TOKENS_PER_RECORD = 8
SUPERVISED_TOKENS_PER_RECORD = 4


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(data) -> str:
    return digest_bytes(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                  allow_nan=False).encode())


def write_json(path: Path, data) -> None:
    with path.open("x", encoding="utf8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def tensor_statistics(tensor: torch.Tensor) -> dict:
    value = tensor.detach().to(torch.float64)
    if not torch.isfinite(value).all():
        raise FloatingPointError("The experiment encountered a non-finite tensor")
    return {"mean_abs": value.abs().mean().item(),
            "rms": value.square().mean().sqrt().item(),
            "max_abs": value.abs().max().item(),
            "mean": value.mean().item(),
            "variance_population": value.var(unbiased=False).item(),
            "elements": value.numel()}


def configure_runtime() -> dict:
    if torch.__version__.split("+")[0] != "2.9.0" or np.__version__ != "2.3.4":
        raise RuntimeError("Use the shared PyTorch 2.9.0 / NumPy 2.3.4 environment")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(20261028)
    np.random.seed(20261028)
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "mup": metadata.version("mup"),
            "pyyaml": metadata.version("PyYAML"), "device": "cpu",
            "width_dtype": "float32", "noise_dtype": "float64",
            "decoder_dtype": "float32", "platform": platform.platform(),
            "machine": platform.machine(), "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "no_pretrained_downloads": True}


def mup_source_metadata() -> dict:
    import mup
    if metadata.version("mup") != MUP_VERSION:
        raise RuntimeError("This implementation was audited against mup==1.0.0")
    directory = Path(mup.__file__).parent
    return {"version": MUP_VERSION, "tag": "v1.0.0",
            "author_url": "https://github.com/microsoft/mup",
            "package_url": "https://pypi.org/pypi/mup/1.0.0/json",
            "sdist_sha256": "9639e3d19f90e754f985ed444542ed2f8a049f3c0488fcb6efe150f30922cf74",
            "installed_source_sha256": {
                name: digest_bytes((directory / name).read_bytes())
                for name in ("layer.py", "optim.py", "shape.py", "init.py", "infshape.py")},
            "read_scope": ["installed layer/optim/shape/init/infshape source",
                           "author README usage, limitations and coordinate checks",
                           "Tensor Programs V Appendix B Tables 8/9 and D.1/D.2",
                           "Xiong et al. sections 3/4, LayerNorm architecture and initialization assumptions",
                           "McCandlish et al. section 2.2-2.4 equations 2.2/2.8/2.9 and appendix A.1"]}


def population_statistics(gradients, *, relative_mean_tolerance=1e-10) -> dict:
    """Finite population, denominator N, sequence-average gradient objective."""
    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[0] < 1 or gradients.shape[1] < 1:
        raise ValueError("Provide a nonempty N by P gradient matrix")
    if not np.isfinite(gradients).all():
        raise ValueError("Gradients must be finite")
    if not math.isfinite(relative_mean_tolerance) or relative_mean_tolerance < 0:
        raise ValueError("Mean tolerance must be finite and nonnegative")
    gram = gradients @ gradients.T
    stats = gram_statistics(gram)
    # The measured gradient matrix is available: use its mean and centered
    # gradients directly instead of subtracting nearly equal Gram moments.
    mean = gradients.mean(axis=0)
    centered = gradients - mean
    mean_norm_sq = float(mean @ mean)
    trace = float(np.einsum("ij,ij->", centered, centered) / len(gradients))
    mean_norm = math.sqrt(mean_norm_sq)
    second_moment = float(np.einsum("ij,ij->", gradients, gradients) / len(gradients))
    threshold = max(1e-15, relative_mean_tolerance * math.sqrt(second_moment))
    state = "undefined-zero-mean" if mean_norm == 0 else (
        "ill-conditioned-mean" if mean_norm <= threshold else "defined")
    stats.update({"mean_norm_sq": mean_norm_sq, "mean_norm": mean_norm,
                  "second_moment": second_moment, "covariance_trace": trace,
                  "mean_norm_threshold": threshold, "noise_ratio_state": state,
                  "noise_ratio": trace / mean_norm_sq if state == "defined" else None,
                  "statistics_method": "direct mean gradient and centered squared norm, float64",
                  "gram_consistency": {"mean_norm_sq_abs_error": abs(mean_norm_sq - float(gram.mean())),
                                       "covariance_trace_abs_error": abs(trace - (float(np.trace(gram) / len(gram)) - float(gram.mean())))}})
    return stats


def gram_statistics(gram, *, relative_mean_tolerance=1e-7) -> dict:
    gram = np.asarray(gram, dtype=np.float64)
    if (gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not len(gram)
            or not np.isfinite(gram).all()):
        raise ValueError("Provide a finite symmetric population Gram matrix")
    symmetry_tolerance = 1e-10 * max(1e-30, float(np.max(np.abs(gram))))
    if not np.allclose(gram, gram.T, atol=symmetry_tolerance, rtol=0):
        raise ValueError("Population Gram matrix is not symmetric")
    if not math.isfinite(relative_mean_tolerance) or relative_mean_tolerance < 0:
        raise ValueError("Mean tolerance must be finite and nonnegative")
    n = len(gram)
    gram = (gram + gram.T) / 2
    eigenvalues = np.linalg.eigvalsh(gram)
    psd_tolerance = 1e-10 * max(1e-30, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -psd_tolerance:
        raise ValueError("Gradient Gram matrix must be positive semidefinite")
    mean_norm_sq = max(0.0, float(gram.mean()))
    second_moment = max(0.0, float(np.trace(gram) / n))
    trace = max(0.0, second_moment - mean_norm_sq)
    mean_norm = math.sqrt(mean_norm_sq)
    reference_norm = math.sqrt(second_moment)
    # Scale relative to the gradients, with an absolute zero safeguard. Never
    # serialize an Infinity or call the zero/zero case a noise-free optimum.
    threshold = max(1e-15, relative_mean_tolerance * reference_norm)
    state = "undefined-zero-mean" if mean_norm == 0 else (
        "ill-conditioned-mean" if mean_norm <= threshold else "defined")
    return {"population_size": n, "covariance_denominator": n,
            "gram": gram.tolist(), "mean_norm": mean_norm,
            "mean_norm_sq": mean_norm_sq, "second_moment": second_moment,
            "covariance_trace": trace,
            "mean_norm_threshold": threshold,
            "noise_ratio_state": state,
            "noise_ratio": trace / mean_norm_sq if state == "defined" else None,
            "statistics_method": "Gram moments; relative mean threshold 1e-7 guards cancellation"}


def batch_variance(trace: float, n: int, batch_size: int, replacement: bool) -> float:
    if (not math.isfinite(trace) or trace < 0 or not isinstance(n, int) or n < 1
            or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("Invalid finite population or batch size")
    if replacement:
        return trace / batch_size
    if batch_size > n:
        raise ValueError("Without-replacement B cannot exceed N")
    if n == 1:
        return 0.0  # The singleton population has exactly zero covariance.
    return trace * (n - batch_size) / (batch_size * (n - 1))


def enumerate_batch_distribution(gram, batch_size, *, scheme="with-replacement",
                                 correlation=0.5, cap=100000) -> dict:
    """Enumerate index draws, evaluate norms via the saved gradient Gram matrix.

    Correlated law: with probability rho repeat one uniformly selected index in
    every position; otherwise draw B indices independently. All marginals remain
    uniform, so Cov(g_j,g_k)=rho*Sigma_N for j!=k.
    """
    gram = np.asarray(gram, dtype=np.float64)
    stats = gram_statistics(gram)
    n = len(gram)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("B must be a positive integer")
    if scheme not in ("with-replacement", "without-replacement", "repeated", "correlated"):
        raise ValueError("Unknown sampling scheme")
    if not 0 <= correlation <= 1:
        raise ValueError("Correlation must lie in [0,1]")
    if scheme == "without-replacement":
        if batch_size > n:
            raise ValueError("Cannot select B distinct indices from N")
        count = math.comb(n, batch_size)
        draws = itertools.combinations(range(n), batch_size)
    elif scheme == "repeated":
        count = n
        draws = ((i,) * batch_size for i in range(n))
    else:
        count = n ** batch_size
        draws = itertools.product(range(n), repeat=batch_size)
    if count > cap:
        raise ValueError("Enumeration cap exceeded")
    indices = list(draws)
    weights = np.full(count, 1.0 / count)
    if scheme == "correlated":
        weights *= 1 - correlation
        indices += [(i,) * batch_size for i in range(n)]
        weights = np.concatenate([weights, np.full(n, correlation / n)])
    coefficients = np.zeros((len(indices), n), dtype=np.float64)
    for row, draw in enumerate(indices):
        for index in draw:
            coefficients[row, index] += 1 / batch_size
    center = coefficients - 1 / n
    norms_sq = np.maximum(0, np.einsum("bi,ij,bj->b", coefficients, gram, coefficients))
    distances_sq = np.maximum(0, np.einsum("bi,ij,bj->b", center, gram, center))
    inner = coefficients @ gram.mean(axis=1)
    denominator = np.sqrt(norms_sq * stats["mean_norm_sq"])
    cosines = np.divide(inner, denominator, out=np.zeros_like(inner), where=denominator > 1e-15)
    valid_cos = (denominator > 1e-15) & (stats["noise_ratio_state"] == "defined")
    exact_trace = float(weights @ distances_sq)
    trace = stats["covariance_trace"]
    formula = (batch_variance(trace, n, batch_size, False) if scheme == "without-replacement"
               else trace if scheme == "repeated"
               else trace * (correlation + (1 - correlation) / batch_size)
               if scheme == "correlated" else trace / batch_size)
    rows = [{"indices": list(draw), "probability": float(weights[i]),
             "gradient_norm": math.sqrt(float(norms_sq[i])),
             "distance_sq": float(distances_sq[i]),
             "cosine_to_mean": float(np.clip(cosines[i], -1, 1)) if valid_cos[i] else None}
            for i, draw in enumerate(indices)]
    return {"batch_size": batch_size, "scheme": scheme,
            "correlation": correlation if scheme == "correlated" else None,
            "enumerated_outcomes": len(rows), "probability_sum": float(weights.sum()),
            "enumerated_trace": exact_trace, "formula_trace": formula,
            "iid_formula_trace": trace / batch_size,
            "absolute_error": abs(exact_trace - formula),
            "norm_min": math.sqrt(float(norms_sq.min())),
            "norm_max": math.sqrt(float(norms_sq.max())),
            "mean_gradient_norm": float(weights @ np.sqrt(norms_sq)),
            "undefined_cosine_outcomes": int((~valid_cos).sum()),
            "rows": rows}


class WidthMLP(nn.Module):
    """8 -> width -> width -> 3. Width varies; two hidden layers stay fixed."""
    def __init__(self, width: int, parameterization: str):
        super().__init__()
        import mup
        self.width = width
        self.parameterization = parameterization
        self.input = nn.Linear(INPUT_DIM, width)
        self.hidden = nn.Linear(width, width)
        readout = mup.MuReadout if parameterization == "mup" else nn.Linear
        self.readout = readout(width, OUTPUT_DIM)

    def forward_coordinates(self, x):
        input_pre = self.input(x)
        input_act = F.relu(input_pre)
        hidden_pre = self.hidden(input_act)
        hidden_act = F.relu(hidden_pre)
        output = self.readout(hidden_act)
        return output, {"input_pre": input_pre, "input_relu": input_act,
                        "hidden_pre": hidden_pre, "hidden_relu": hidden_act,
                        "readout_logits": output}

    def forward(self, x):
        return self.forward_coordinates(x)[0]


def make_width_model(width: int, parameterization: str, seed: int) -> WidthMLP:
    import mup
    if parameterization not in ("sp", "mup") or width < 1:
        raise ValueError("Unsupported width/parameterization")
    if metadata.version("mup") != MUP_VERSION:
        raise RuntimeError("Install mup==1.0.0")
    # Shape-only templates never consume the actual model's initialization RNG.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        base, delta = WidthMLP(BASE_WIDTH, "mup"), WidthMLP(BASE_WIDTH * 2, "mup")
    torch.manual_seed(seed)
    model = WidthMLP(width, parameterization)
    if parameterization == "mup":
        mup.set_base_shapes(model, base, delta=delta)
    # Author initializers take constants at the BASE width, then use infshape.
    # The finite input and output dimensions never change with width.
    torch.manual_seed(seed + 100000)
    if parameterization == "mup":
        mup.init.normal_(model.input.weight, std=math.sqrt(2 / INPUT_DIM))
        mup.init.normal_(model.hidden.weight, std=math.sqrt(2 / BASE_WIDTH))
        mup.init.normal_(model.readout.weight, std=math.sqrt(1 / BASE_WIDTH))
    else:
        nn.init.normal_(model.input.weight, std=math.sqrt(2 / INPUT_DIM))
        nn.init.normal_(model.hidden.weight, std=math.sqrt(2 / width))
        nn.init.normal_(model.readout.weight, std=math.sqrt(1 / width))
    for name, p in model.named_parameters():
        if name.endswith("bias"):
            nn.init.zeros_(p)
    return model


def make_width_optimizer(model, lr):
    import mup
    implementation = mup.MuAdam if model.parameterization == "mup" else torch.optim.Adam
    return implementation(model.parameters(), lr=lr, **ADAM_CONFIG)


def parameter_ledger(model, optimizer, base_lr) -> list[dict]:
    group_map = {id(p): (i, group) for i, group in enumerate(optimizer.param_groups)
                 for p in group["params"]}
    width, mup_model = model.width, model.parameterization == "mup"
    multiplier = width / BASE_WIDTH
    result = []
    for name, p in model.named_parameters():
        group_index, group = group_map[id(p)]
        role = name.split(".")[0]
        expected_var = (0 if name.endswith("bias") else
                        2 / INPUT_DIM if role == "input" else
                        2 / width if role == "hidden" else
                        1 / BASE_WIDTH if mup_model else 1 / width)
        stats = tensor_statistics(p)
        result.append({"name": name, "role": role, "shape": list(p.shape),
                       "infinite_dimensions": p.infshape.ninf() if mup_model else None,
                       "width_multiplier": p.infshape.width_mult() if mup_model else 1,
                       "optimizer_group": group_index, "base_lr": base_lr,
                       "actual_lr": group["lr"], "weight_decay": group["weight_decay"],
                       "expected_initial_variance": expected_var,
                       "observed_initial_variance": stats["variance_population"],
                       "observed_initial_rms": stats["rms"],
                       "readout_input_factor": 1 / multiplier if mup_model and role == "readout" else 1,
                       "bias_forward_factor": 1})
    return result


def width_task(seed=20261028) -> dict:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(512, INPUT_DIM, generator=generator)
    teacher_a = torch.randn(INPUT_DIM, 12, generator=generator) / math.sqrt(INPUT_DIM)
    teacher_b = torch.randn(12, OUTPUT_DIM, generator=generator) / math.sqrt(12)
    y = (F.relu(x @ teacher_a) @ teacher_b).argmax(-1)
    # Selection uses validation. Independent test is only opened after LR choice.
    return {"train": (x[:256], y[:256]), "validation": (x[256:384], y[256:384]),
            "test": (x[384:], y[384:]), "teacher_a": teacher_a,
            "teacher_b": teacher_b, "seed": seed,
            "description": "synthetic 8D Gaussian input; fixed 8->12->3 ReLU teacher argmax labels"}


@torch.no_grad()
def width_evaluation(model, records) -> dict:
    x, y = records
    logits = model(x)
    return {"loss": F.cross_entropy(logits, y).item(),
            "accuracy": logits.argmax(-1).eq(y).float().mean().item(), "examples": len(y)}


@torch.no_grad()
def width_coordinates(model, x, step, initial=None) -> dict:
    _, tensors = model.forward_coordinates(x)
    rows = []
    for name, tensor in tensors.items():
        row = {"layer": name, **tensor_statistics(tensor)}
        row["displacement_rms"] = (tensor - initial[name]).square().mean().sqrt().item() if initial else 0
        rows.append(row)
    return {"after_updates": step, "rows": rows}


def train_width_run(task, width, parameterization, seed, lr, directory,
                    steps=WIDTH_STEPS, role="selection-sweep") -> tuple[dict, WidthMLP]:
    started = time.perf_counter()
    model = make_width_model(width, parameterization, seed)
    optimizer = make_width_optimizer(model, lr)
    run_id = f"{parameterization}-w{width}-seed{seed}-lr{lr:g}"
    run_dir = directory / run_id
    run_dir.mkdir()
    x, y = task["train"]
    generator = torch.Generator().manual_seed(seed + 70000)
    indices = torch.randint(len(y), (steps, 32), generator=generator)
    ledger = parameter_ledger(model, optimizer, lr)
    with torch.no_grad():
        initial_coords = {name: tensor.detach().clone() for name, tensor
                          in model.forward_coordinates(task["validation"][0][:16])[1].items()}
    coordinates = [width_coordinates(model, task["validation"][0][:16], 0, initial_coords)]
    curve = [{"updates": 0, "consumed_examples": 0, "validation": width_evaluation(model, task["validation"]),
              "minibatch_loss_before_update": None}]
    initial_state = copy.deepcopy(model.state_dict())
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_indices = indices[step - 1]
        loss = F.cross_entropy(model(x[batch_indices]), y[batch_indices])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite MLP loss in {run_id}, step {step}")
        loss.backward()
        optimizer.step()
        if step in (1, 4, steps):
            coordinates.append(width_coordinates(model, task["validation"][0][:16], step, initial_coords))
        # Save EVERY actual minibatch value, no fitted/formula-generated curve.
        curve.append({"updates": step, "consumed_examples": step * 32,
                      "minibatch_loss_before_update": loss.item(),
                      "validation": width_evaluation(model, task["validation"])
                      if step % 20 == 0 or step == steps else None})
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "initial_model": initial_state, "indices": indices,
                  "torch_rng": torch.get_rng_state(), "seed": seed, "width": width,
                  "parameterization": parameterization, "base_width": BASE_WIDTH,
                  "lr": lr, "steps": steps, "consumed_examples": steps * 32,
                  "initialization": ("mup.init.normal_ with fixed base-width standard deviations; biases zero"
                                     if parameterization == "mup" else
                                     "torch.nn.init.normal_ with actual-width fan-in; biases zero"),
                  "role": role,
                  "data_seed": task["seed"], "data_cursor": steps,
                  "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    row = {"run_id": run_id, "width": width, "seed": seed, "parameterization": parameterization,
           "base_lr": lr, "updates": steps, "consumed_examples": steps * 32,
           "parameter_count": sum(p.numel() for p in model.parameters()),
           "batch_size": 32, "role": role, "scheduler": "constant LR, update-based Adam states",
           "initialization": checkpoint["initialization"],
           "optimizer": "MuAdam" if parameterization == "mup" else "Adam",
           "weight_decay": 0, "ledger": ledger, "coordinates": coordinates,
           "curve": curve, "final_validation": width_evaluation(model, task["validation"]),
           "wall_seconds": time.perf_counter() - started,
           "wall_seconds_scope": "elapsed run time including construction, coordinate probes, validation and checkpoint save; not pure throughput",
           "checkpoint": str((run_dir / "checkpoint.pt").relative_to(directory.parent)),
           "data_stream_sha256": digest_bytes(indices.numpy().tobytes()),
           "parameter_rms_displacement": {name: (p.detach() - initial_state[name]).square().mean().sqrt().item()
                                          for name, p in model.named_parameters()}}
    return row, model


def run_width_experiment(directory: Path, *, steps=WIDTH_STEPS) -> dict:
    directory.mkdir()
    task = width_task()
    torch.save(task, directory / "task.pt")
    runs, models = [], {}
    for parameterization in ("sp", "mup"):
        for width in WIDTHS:
            for lr in WIDTH_LRS:
                for seed in SEEDS:
                    row, model = train_width_run(task, width, parameterization, seed, lr, directory, steps,
                                                 role="selection-sweep" if width == BASE_WIDTH else "diagnostic-sweep")
                    runs.append(row)
                    models[row["run_id"]] = model
        print(f"width: completed {parameterization} selection widths {WIDTHS}", flush=True)
    selection = []
    selected = {}
    for parameterization in ("sp", "mup"):
        scores = [{"base_lr": lr,
                   "mean_validation_loss": float(np.mean([row["final_validation"]["loss"] for row in runs
                                                           if row["width"] == BASE_WIDTH
                                                           and row["parameterization"] == parameterization
                                                           and row["base_lr"] == lr])),
                   "seeds": list(SEEDS)} for lr in WIDTH_LRS]
        chosen = min(scores, key=lambda value: value["mean_validation_loss"])["base_lr"]
        selected[parameterization] = chosen
        selection.append({"parameterization": parameterization, "selected_lr": chosen,
                          "selection_width": BASE_WIDTH, "scores": scores,
                          "criterion": "minimum mean final validation CE across three seeds at width 16 only",
                          "held_width_used": False,
                          "grid_boundary": chosen in (min(WIDTH_LRS), max(WIDTH_LRS)),
                          "interpretation": "best among tested LR grid; an optimum may lie beyond its boundary"})
    # The choice is now frozen. No held-width candidate sweep, and no test score
    # exists above this point. Larger selection widths serve as diagnostics only.
    selected_runs = []
    for row in runs:
        if row["base_lr"] == selected[row["parameterization"]]:
            row["selected_for_transfer"] = True
            row["test"] = width_evaluation(models[row["run_id"]], task["test"])
            selected_runs.append(row)
    for parameterization in ("sp", "mup"):
        for seed in SEEDS:
            row, model = train_width_run(task, HELD_WIDTH, parameterization, seed,
                                         selected[parameterization], directory, steps,
                                         role="held-out-transfer")
            row["selected_for_transfer"] = True
            row["test"] = width_evaluation(model, task["test"])
            selected_runs.append(row)
    # Final per-run JSON is written once, after selection and held-out testing.
    # Every duplicate appearance in aggregate selected_runs refers to this row.
    all_rows = {row["run_id"]: row for row in runs + selected_runs}
    for row in all_rows.values():
        write_json(directory / row["run_id"] / "run.json", row)
    write_json(directory / "selected-runs.json", selected_runs)
    return {"task": {"description": task["description"], "data_seed": task["seed"],
                     "split_sizes": {key: len(task[key][1]) for key in ("train", "validation", "test")},
                     "task_file": "width/task.pt", "input_dimensions": INPUT_DIM,
                     "output_dimensions": OUTPUT_DIM},
            "hidden_depth": 2, "base_width": BASE_WIDTH, "delta_width": BASE_WIDTH * 2,
            "sweep_widths": list(WIDTHS), "held_width": HELD_WIDTH,
            "seeds": list(SEEDS), "lr_grid": list(WIDTH_LRS), "steps": steps,
            "optimizer_config": {**ADAM_CONFIG, "clip": None},
            "readout_initialization": "nonzero Gaussian; initial mup output may shrink with width",
            "coordinate_definition": "mean(abs(x)), RMS and displacement RMS on the SAME 16 validation inputs",
            "coordinate_steps": [0, 1, 4, steps], "selection": selection,
            "runs": runs, "selected_runs": selected_runs,
            "scope": "finite-width synthetic classification experiment; coordinate stability is not a theorem of LR transfer"}


def decoder_splits():
    import learning_core as core
    splits = core.make_task_splits(train_lengths=(3,), length_lengths=(5,),
                                   max_groups_per_length=40, length_groups_per_length=8)
    core.assert_split_integrity(splits)
    return splits


def tensor_state_error(actual, expected):
    if isinstance(actual, np.ndarray):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return math.inf
        return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.size else 0
    if isinstance(actual, torch.Tensor):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return math.inf
        if actual.numel() == 0:
            return 0.0
        return (actual.to(torch.float64) - expected.to(torch.float64)).abs().max().item()
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            return math.inf
        return max((tensor_state_error(actual[k], expected[k]) for k in actual), default=0)
    if isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            return math.inf
        return max((tensor_state_error(a, b) for a, b in zip(actual, expected)), default=0)
    if isinstance(actual, (int, float)):
        return abs(actual - expected)
    return 0.0 if actual == expected else math.inf


def probe_decoder(model, batch, *, after_updates: int) -> dict:
    """Layer hooks observe the SAME core model. No synthetic scale curves.

    Gradients are of the declared scalar sequence-mean loss. Activation summaries
    include valid input positions only; the selected records are equal length.
    Parameter and hidden gradients have separate names and per-element RMS.
    """
    import learning_core as core
    modules = [("embedding", model.embedding_dropout)]
    for i, block in enumerate(model.blocks):
        modules.extend([(f"block{i}.attention", block.attention),
                        (f"block{i}.mlp", block.mlp), (f"block{i}.output", block)])
    modules += [("final_norm", model.final_norm), ("readout_logits", model.lm_head)]
    stored, hooks = {}, []
    def remember(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            tensor.retain_grad()
            stored[name] = tensor
        return hook
    modes = {module: module.training for module in model.modules()}
    model.eval()  # dropout=0, but state is still restored explicitly.
    model.zero_grad(set_to_none=True)
    try:
        for name, module in modules:
            hooks.append(module.register_forward_hook(remember(name)))
        loss = core.forward_loss(model, batch, reduction="sequence")
        loss.backward()
        valid = batch["attention_mask"]
        rows = []
        for name, _module in modules:
            tensor = stored[name]
            rows.append({"layer": name, "activation": tensor_statistics(tensor[valid]),
                         "gradient": tensor_statistics(tensor.grad[valid]),
                         "shape": list(tensor.shape)})
        parameters = [{"name": name, "gradient": tensor_statistics(p.grad) if p.grad is not None else None,
                       "parameter": tensor_statistics(p)} for name, p in model.named_parameters()]
        return {"after_updates": after_updates, "loss": loss.item(), "rows": rows,
                "parameters": parameters,
                "gradient_definition": "d(sequence-mean CE)/d(parameter or hidden); RMS per tensor element"}
    finally:
        for handle in hooks:
            handle.remove()
        for module, mode in modes.items():
            module.training = mode
        model.zero_grad(set_to_none=True)


def run_depth_experiment(directory: Path) -> dict:
    import learning_core as core
    directory.mkdir()
    splits = decoder_splits()
    records = splits["train"][:8]
    batch = core.collate_records(records)
    write_json(directory / "records.json", records)
    runs = []
    for depth in (1, 4, 8):
        for norm in ("pre", "post", "none"):
            for residual_rule in ("one", "inverse-sqrt-depth"):
                scale = 1.0 if residual_rule == "one" else 1 / math.sqrt(2 * depth)
                for seed in SEEDS:
                    config = {**DECODER_CONFIG, "depth": depth, "norm": norm,
                              "residual_scale": scale}
                    model = core.make_model(seed=seed, **config)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, **ADAM_CONFIG)
                    initial = copy.deepcopy(model.state_dict())
                    before = probe_decoder(model, batch, after_updates=0)
                    losses = []
                    for _ in range(4):
                        optimizer.zero_grad(set_to_none=True)
                        loss = core.forward_loss(model, batch, reduction="sequence")
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                    after = probe_decoder(model, batch, after_updates=4)
                    run_id = f"depth{depth}-{norm}-{residual_rule}-seed{seed}"
                    torch.save({"model": model.state_dict(), "initial_model": initial,
                                "optimizer": optimizer.state_dict(), "config": config,
                                "records": records, "seed": seed, "updates": 4,
                                "rng": core.capture_rng_state(), "losses": losses,
                                "objective": "sequence-mean CE", "data_cursor": 4 * len(records)},
                               directory / (run_id + ".pt"))
                    runs.append({"run_id": run_id, "seed": seed, "depth": depth,
                                 "norm": norm, "residual_rule": residual_rule,
                                 "residual_scale": scale, "config": config,
                                 "parameter_count": core.parameter_count(model),
                                 "base_lr": 0.003, "optimizer": "Adam", "weight_decay": 0,
                                 "minibatch_losses": losses, "probes": [before, after],
                                 "parameter_displacement_rms": {
                                     name: (p.detach() - initial[name]).square().mean().sqrt().item()
                                     for name, p in model.named_parameters()}})
    write_json(directory / "runs.json", runs)
    print(f"depth: saved {len(runs)} actual decoder probes", flush=True)
    return {"core_api": core.CORE_API_VERSION, "width": DECODER_CONFIG["width"],
            "heads": DECODER_CONFIG["heads"], "depths": [1, 4, 8],
            "norms": ["pre", "post", "none"], "seeds": list(SEEDS),
            "residual_rules": ["one", "inverse-sqrt-depth"],
            "fixed_record_ids": batch["record_ids"], "input_tokens_per_probe": batch["input_tokens"],
            "supervised_tokens_per_probe": batch["supervised_tokens"],
            "objective": "sequence-mean CE; every record has 4 supervised positions",
            "initialization": "core standard Gaussian std=0.02 weights, zero biases; fixed sinusoidal position encoding",
            "norm_definition": "LayerNorm affine gamma=1 beta=0, eps=1e-5; Pre-LN includes final LayerNorm",
            "scope": "width fixed; residual rule means alpha=1 or 1/sqrt(2*depth); 4 updates on the same eight training sequences",
            "paper_scope": "Xiong theorem uses single-head, dxd matrices, zero Q/K and Gaussian inputs; this core differs and does not test a universal no-warmup claim",
            "runs": runs}


def sequence_gradients(model, records) -> dict:
    import learning_core as core
    modes = {module: module.training for module in model.modules()}
    parameters = list(model.parameters())
    model.eval()
    gradients, losses = [], []
    try:
        for record in records:
            batch = core.collate_records([record])
            loss = core.forward_loss(model, batch, reduction="sequence")
            gradient = torch.autograd.grad(loss, parameters)
            gradients.append(torch.cat([g.detach().reshape(-1) for g in gradient]).to(torch.float64))
            losses.append(loss.item())
        values = torch.stack(gradients)
        population_batch = core.collate_records(records)
        full_loss = core.forward_loss(model, population_batch, reduction="sequence")
        full_grad = torch.cat([g.reshape(-1) for g in torch.autograd.grad(full_loss, parameters)])
        error = (values.mean(0) - full_grad).abs().max().item()
        return {"gradients": values, "losses": losses, "sequence_mean_loss": full_loss.item(),
                "sequence_gradient_mean_error": error,
                "parameter_names": [name for name, _ in model.named_parameters()]}
    finally:
        for module, mode in modes.items():
            module.training = mode


def run_noise_experiment(directory: Path) -> dict:
    import learning_core as core
    directory.mkdir()
    splits = decoder_splits()
    # Six equal-length records, one per reversal-equivalence group. A complete
    # sequence is the unit; correlated token positions are never iid samples.
    records, seen_groups = [], set()
    for record in splits["train"]:
        if record["group_id"] not in seen_groups:
            records.append(record)
            seen_groups.add(record["group_id"])
        if len(records) == 6:
            break
    model = core.make_model(seed=11, **DECODER_CONFIG).double()
    values = sequence_gradients(model, records)
    gradients = values["gradients"].numpy()
    stats = population_statistics(gradients)
    enumerations = []
    for batch_size in (1, 2, 3, 4):
        for scheme in ("with-replacement", "without-replacement", "repeated", "correlated"):
            enumerations.append(enumerate_batch_distribution(stats["gram"], batch_size, scheme=scheme))
    enumerations.append(enumerate_batch_distribution(stats["gram"], 6, scheme="without-replacement"))
    # Repeated population entries are still iid if one really draws indices iid.
    # Their own population covariance changes; repetition within a batch instead
    # has nonzero cross-covariance and is a different law.
    duplicated = np.repeat(gradients[:3], 2, axis=0)
    duplicate_stats = population_statistics(duplicated)
    duplicate_law = enumerate_batch_distribution(duplicate_stats["gram"], 3)
    centered_stats = population_statistics(gradients - gradients.mean(axis=0))
    centered_law = enumerate_batch_distribution(centered_stats["gram"], 3)
    torch.save({"model": model.state_dict(), "config": DECODER_CONFIG, "records": records,
                "gradients": values["gradients"], "parameter_names": values["parameter_names"],
                "seed": 11, "rng": core.capture_rng_state(), "objective": "sequence-mean CE"},
               directory / "fixed-theta-gradients.pt")
    write_json(directory / "records.json", records)
    write_json(directory / "enumerations.json", enumerations)
    max_error = max(row["absolute_error"] for row in enumerations + [duplicate_law])
    if max_error > 1e-10 or values["sequence_gradient_mean_error"] > 1e-10:
        raise AssertionError("Exact finite-population gradient checks failed")
    print(f"noise: {len(enumerations)} exact sampling laws, max trace error {max_error:.2e}", flush=True)
    return {"core_api": core.CORE_API_VERSION, "config": DECODER_CONFIG, "seed": 11,
            "parameter_count": core.parameter_count(model), "records": records,
            "record_ids": [r["id"] for r in records], "losses": values["losses"],
            "objective": "Each sequence: mean CE over reversed answer and EOS (4 labels); population: mean over 6 sequences",
            "sequence_gradient_mean_error": values["sequence_gradient_mean_error"],
            "population": stats, "enumerations": enumerations,
            "max_formula_error": max_error,
            "duplicate_population": {"description": "first three gradients duplicated twice; iid index sampling still obeys this NEW covariance / B",
                                     "population": duplicate_stats, "enumeration": {k: v for k, v in duplicate_law.items() if k != "rows"}},
            "centered_counterexample": {"description": "algebraic counterexample: subtract the measured population mean from each saved gradient; not a new decoder objective or training measurement",
                                        "population": centered_stats,
                                        "enumeration": {k: v for k, v in centered_law.items() if k != "rows"}},
            "correlation_definition": "rho=0.5 mixture: iid batch with prob 0.5, repeated uniform record in every slot with prob 0.5",
            "scope": "Fixed initial decoder theta, no optimizer, no dropout, CPU float64; exact finite population, no Hessian or optimal-batch estimator"}


def token_learning_rate(base_lr, consumed_input_tokens, budget_input_tokens,
                        warmup_input_tokens=1024):
    if (not math.isfinite(base_lr) or base_lr <= 0 or consumed_input_tokens < 0
            or budget_input_tokens <= warmup_input_tokens or warmup_input_tokens < 1
            or consumed_input_tokens > budget_input_tokens):
        raise ValueError("Invalid token scheduler configuration")
    if consumed_input_tokens < warmup_input_tokens:
        return base_lr * consumed_input_tokens / warmup_input_tokens
    fraction = (consumed_input_tokens - warmup_input_tokens) / (budget_input_tokens - warmup_input_tokens)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * fraction)))


def make_record_stream(records, seed, count):
    generator = torch.Generator().manual_seed(seed + 50000)
    stream = []
    while len(stream) < count:
        stream.extend(torch.randperm(len(records), generator=generator).tolist())
    return stream[:count], generator.get_state()


def advance_batch(model, optimizer, records, indices, cursor, batch_size, base_lr,
                  record_budget=BATCH_RECORD_BUDGET):
    import learning_core as core
    selected = [records[i] for i in indices[cursor:cursor + batch_size]]
    if len(selected) != batch_size:
        raise ValueError("Token budget must divide exactly into complete batches")
    batch = core.collate_records(selected)
    if (batch["input_tokens"] != batch_size * INPUT_TOKENS_PER_RECORD
            or batch["supervised_tokens"] != batch_size * SUPERVISED_TOKENS_PER_RECORD):
        raise AssertionError("Equal-length token accounting changed")
    # Scheduler samples the token endpoint of this actual batch. Larger B has
    # fewer optimizer state advances and a coarser integration of the SAME law.
    next_cursor = cursor + batch_size
    consumed = next_cursor * INPUT_TOKENS_PER_RECORD
    lr = token_learning_rate(base_lr, consumed, record_budget * INPUT_TOKENS_PER_RECORD)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss = core.forward_loss(model, batch, reduction="sequence")
    loss.backward()
    gradient_norm = math.sqrt(sum(p.grad.detach().double().square().sum().item()
                                  for p in model.parameters() if p.grad is not None))
    if not math.isfinite(gradient_norm) or not torch.isfinite(loss):
        raise FloatingPointError("Batch training became nonfinite")
    optimizer.step()
    return {"cursor": next_cursor, "consumed_input_tokens": consumed,
            "consumed_supervised_tokens": next_cursor * SUPERVISED_TOKENS_PER_RECORD,
            "lr": lr, "sequence_mean_loss_before_update": loss.item(),
            "gradient_norm": gradient_norm, "record_ids": batch["record_ids"]}


def teacher_evaluation(model, records):
    import learning_core as core
    with core.evaluation_mode(model):
        batch = core.collate_records(records)
        return {"teacher_forced_loss": core.forward_loss(model, batch, reduction="token").item(),
                "supervised_tokens": batch["supervised_tokens"]}


def sampled_noise_at_theta(model, records):
    values = sequence_gradients(model, records)
    stats = population_statistics(values["gradients"].numpy())
    return {key: value for key, value in stats.items() if key != "gram"}


def batch_checkpoint(model, optimizer, records, splits, indices, stream_rng, cursor,
                     batch_size, base_lr, seed, record_budget, steps):
    import learning_core as core
    return {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
            "config": DECODER_CONFIG, "rng": core.capture_rng_state(), "records": records,
            "splits": splits, "indices": indices, "stream_generator_state": stream_rng,
            "cursor": cursor, "batch_size": batch_size, "base_lr": base_lr, "seed": seed,
            "record_budget": record_budget, "steps": steps,
            "consumed_input_tokens": cursor * INPUT_TOKENS_PER_RECORD,
            "consumed_supervised_tokens": cursor * SUPERVISED_TOKENS_PER_RECORD,
            "objective": "sequence-mean CE", "scheduler": "input-token warmup then cosine to 0.1, batch endpoint"}


def resume_batch_check(directory):
    """New Python process checks the actual saved mid-run state, next data and eval."""
    import learning_core as core
    configure_runtime()
    checkpoint = torch.load(directory / "mid-checkpoint.pt", map_location="cpu", weights_only=False)
    expected = torch.load(directory / "resume-expected.pt", map_location="cpu", weights_only=False)
    model = core.make_model(seed=checkpoint["seed"], **checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=checkpoint["base_lr"], **ADAM_CONFIG)
    optimizer.load_state_dict(checkpoint["optimizer"])
    core.restore_rng_state(checkpoint["rng"])
    cursor = checkpoint["cursor"]
    steps = []
    for _ in range(len(expected["next_steps"])):
        row = advance_batch(model, optimizer, checkpoint["records"], checkpoint["indices"], cursor,
                            checkpoint["batch_size"], checkpoint["base_lr"], checkpoint["record_budget"])
        cursor = row["cursor"]
        steps.append(row)
    actual_evaluation = core.evaluate_records(model, checkpoint["splits"]["validation"])
    errors = {"model": tensor_state_error(model.state_dict(), expected["model"]),
              "optimizer": tensor_state_error(optimizer.state_dict(), expected["optimizer"]),
              "next_steps": tensor_state_error(steps, expected["next_steps"]),
              "evaluation": tensor_state_error(actual_evaluation, expected["evaluation"]),
              "rng": tensor_state_error(core.capture_rng_state(), expected["rng"])}
    if any(value != 0 for value in errors.values()):
        raise AssertionError(f"New-process resume mismatch: {errors}")
    write_json(directory / "resume-check.json", {"status": "passed", "fresh_python_process": True,
                                                "resume_steps": len(steps), "max_errors": errors,
                                                "cursor_before": checkpoint["cursor"], "cursor_after": cursor})


def reanalyze_saved_report(source_report: Path, out: Path, runtime: dict) -> dict:
    """Recompute diagnostics from saved tensors; preserve every training curve.

    This does not call any of the full width/depth/batch experiment runners.
    The only updates replayed are the four saved resume-reference steps.
    """
    import learning_core as core
    source_bytes = source_report.read_bytes()
    original = json.loads(source_bytes)
    if original.get("schema") != SCHEMA or original.get("status") != "complete":
        raise ValueError("Reanalysis requires a complete original dynamics report")
    original_hash = digest_bytes(source_bytes)
    expected_original_hash = "587611efd52b00d9b0209af948bc72a54eb92075eb3aebea6e2d14bf9bfff645"
    if source_report.parent.name == "dynamics-lab-20261028-01" and original_hash != expected_original_hash:
        raise ValueError("Original measurement report hash changed; refusing reanalysis")
    analysis_sources = {name: digest_bytes((ROOT / "scripts" / name).read_bytes())
                        for name in ("training_dynamics.py", "learning_core.py",
                                     "run-dynamics-lab.mjs", "dynamics-requirements.txt")}
    if analysis_sources["learning_core.py"] != original["provenance"]["source_sha256"]["learning_core.py"]:
        raise ValueError("Shared core source differs from original measurement; inspect before reanalysis")
    report = copy.deepcopy(original)
    source_root = source_report.parent
    artifacts = []
    def saved(path):
        data = path.read_bytes()
        artifacts.append({"path": str(path.resolve()), "sha256": digest_bytes(data), "bytes": len(data)})
        return torch.load(path, map_location="cpu", weights_only=False)
    # Directly center the ACTUAL saved fixed-theta gradient matrix.
    fixed = saved(source_root / "noise" / "fixed-theta-gradients.pt")
    matrix = fixed["gradients"].double().numpy()
    population = population_statistics(matrix)
    gram_max_error = float(np.max(np.abs(np.asarray(original["noise"]["population"]["gram"]) - np.asarray(population["gram"]))))
    if gram_max_error > 1e-10:
        raise AssertionError("Saved gradient matrix does not match original population Gram")
    report["noise"]["population"] = population
    for row in report["noise"]["enumerations"]:
        new = enumerate_batch_distribution(population["gram"], row["batch_size"], scheme=row["scheme"])
        if abs(new["enumerated_trace"] - row["enumerated_trace"]) > 1e-10:
            raise AssertionError("Original exact noise measurements changed")
        row.update(new)
    duplicate = population_statistics(np.repeat(matrix[:3], 2, axis=0))
    duplicate_law = enumerate_batch_distribution(duplicate["gram"], 3)
    report["noise"]["duplicate_population"] = {
        "description": "first three SAVED gradients duplicated twice; iid index law with the new population",
        "population": duplicate, "enumeration": {k: v for k, v in duplicate_law.items() if k != "rows"}}
    centered = population_statistics(matrix - matrix.mean(axis=0))
    centered_law = enumerate_batch_distribution(centered["gram"], 3)
    report["noise"]["centered_counterexample"] = {
        "description": "reanalysis-only algebraic counterexample: subtract saved population mean; not a new decoder objective or training run",
        "population": centered, "enumeration": {k: v for k, v in centered_law.items() if k != "rows"}}
    report["noise"]["max_formula_error"] = max(row["absolute_error"] for row in report["noise"]["enumerations"])
    fixed_model = core.make_model(seed=fixed["seed"], **fixed["config"]).double()
    fixed_model.load_state_dict(fixed["model"])
    remeasured = sequence_gradients(fixed_model, fixed["records"])
    gradient_recheck_error = (remeasured["gradients"] - fixed["gradients"]).abs().max().item()
    if gradient_recheck_error > 1e-12:
        raise AssertionError("Current core does not reproduce saved fixed-theta gradients")
    report["noise"]["sequence_gradient_mean_error"] = remeasured["sequence_gradient_mean_error"]
    # Width/depth measurements and curves are reused verbatim. Only selection
    # scope and the grid boundary metadata are clarified.
    for row in report["width"]["selection"]:
        row["grid_boundary"] = row["selected_lr"] in (min(report["width"]["lr_grid"]), max(report["width"]["lr_grid"]))
        row["interpretation"] = "best among the originally tested grid; the continuous optimum may lie beyond the boundary; no new LR candidate was trained"
    for row in report["width"]["runs"] + report["width"]["selected_runs"]:
        row["wall_seconds_scope"] = "ORIGINAL elapsed run time, includes validation/probes/save; not throughput; no new timing measurement"
    batch_rechecks = []
    common_grid = None
    for run in report["batch"]["runs"]:
        checkpoint = saved(source_root / "batch" / run["run_id"] / "checkpoint.pt")
        ids = {row["id"]: row for row in checkpoint["records"]}
        records = [ids[record_id] for record_id in run["noise_record_ids"]]
        final_model = core.make_model(seed=run["seed"], **checkpoint["config"])
        final_model.load_state_dict(checkpoint["model"])
        final_stats = sampled_noise_at_theta(final_model, records)
        initial_stats = sampled_noise_at_theta(core.make_model(seed=run["seed"], **checkpoint["config"]), records)
        residuals = {label: abs(run[label]["covariance_trace"] - stat["covariance_trace"])
                     for label, stat in (("initial_noise", initial_stats), ("final_noise", final_stats))}
        if max(residuals.values()) > 1e-8:
            raise AssertionError("Saved batch checkpoint noise changed beyond roundoff")
        run["initial_noise"], run["final_noise"] = initial_stats, final_stats
        grid = [point["consumed_input_tokens"] for point in run["curve"]]
        if common_grid is not None and grid != common_grid:
            raise AssertionError("Batch/seed curves do not share the same token grid")
        common_grid = grid
        run["wall_seconds_scope"] = "ORIGINAL elapsed experiment including evaluation/probes/save/optional child process; concurrent work may interfere; no new throughput timing"
        batch_rechecks.append({"run_id": run["run_id"], "centered_trace_abs_residuals": residuals,
                               "checkpoint_cursor": checkpoint["cursor"], "common_token_grid": True})
    # Original expected checkpoint did not store RNG. Measure the four-step
    # continuous replay RNG now, compare all existing reference fields, then
    # compare an independently launched process against that measured RNG.
    # Keep the original expected bytes and their hash untouched.
    resume_root = out / "resume-recheck"
    resume_root.mkdir()
    resume_run = next(run for run in report["batch"]["runs"] if run.get("resume"))
    old_resume_root = source_root / "batch" / resume_run["run_id"]
    mid = saved(old_resume_root / "mid-checkpoint.pt")
    original_expected = saved(old_resume_root / "resume-expected.pt")
    (resume_root / "mid-checkpoint.pt").write_bytes((old_resume_root / "mid-checkpoint.pt").read_bytes())
    replay_model = core.make_model(seed=mid["seed"], **mid["config"])
    replay_model.load_state_dict(mid["model"])
    replay_optimizer = torch.optim.Adam(replay_model.parameters(), lr=mid["base_lr"], **ADAM_CONFIG)
    replay_optimizer.load_state_dict(mid["optimizer"])
    core.restore_rng_state(mid["rng"])
    replay_cursor, replay_steps = mid["cursor"], []
    for _ in range(len(original_expected["next_steps"])):
        step = advance_batch(replay_model, replay_optimizer, mid["records"], mid["indices"],
                             replay_cursor, mid["batch_size"], mid["base_lr"], mid["record_budget"])
        replay_cursor = step["cursor"]
        replay_steps.append(step)
    replay = {"model": replay_model.state_dict(), "optimizer": replay_optimizer.state_dict(),
              "next_steps": replay_steps,
              "evaluation": core.evaluate_records(replay_model, mid["splits"]["validation"])}
    replay_errors = {key: tensor_state_error(replay[key], original_expected[key]) for key in replay}
    if any(value != 0 for value in replay_errors.values()):
        raise AssertionError(f"Continuous reanalysis replay differs from original reference: {replay_errors}")
    replay_rng = core.capture_rng_state()
    replay_rng_error = tensor_state_error(replay_rng, mid["rng"])
    if replay_rng_error != 0:
        raise AssertionError("Zero-dropout saved-index replay unexpectedly consumed global RNG")
    expected = copy.deepcopy(original_expected)
    expected["rng"] = copy.deepcopy(replay_rng)
    torch.save(expected, resume_root / "resume-expected.pt")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--resume-check", str(resume_root)],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    resume_result = json.loads((resume_root / "resume-check.json").read_text())
    resume_result["rng_reference_kind"] = "derived-from-saved-mid-checkpoint-replay"
    resume_result["rng_reference_description"] = "derived by actual four-step continuous replay; terminal RNG measured during reanalysis, not saved by the original run; fresh process compared against this newly measured reference"
    resume_result["continuous_replay_max_errors"] = replay_errors
    resume_result["continuous_replay_rng_vs_mid_max_error"] = replay_rng_error
    resume_result["original_resume_result"] = original["batch"]["runs"][report["batch"]["runs"].index(resume_run)]["resume"]
    resume_run["resume"] = resume_result
    # Assert every plotted training measurement is byte-for-byte equal in the
    # canonical JSON representation. Diagnostics may have new numeric precision.
    unchanged = {
        "width_curves": all(a["curve"] == b["curve"] for a, b in zip(original["width"]["runs"], report["width"]["runs"])),
        "width_coordinates": all(a["coordinates"] == b["coordinates"] for a, b in zip(original["width"]["selected_runs"], report["width"]["selected_runs"])),
        "width_test_results": all(a["test"] == b["test"] for a, b in zip(original["width"]["selected_runs"], report["width"]["selected_runs"])),
        "depth_probes": original["depth"] == report["depth"],
        "batch_curves": all(a["curve"] == b["curve"] for a, b in zip(original["batch"]["runs"], report["batch"]["runs"])),
        "batch_evaluation": all(a["initial_evaluation"] == b["initial_evaluation"] and a["final_evaluation"] == b["final_evaluation"]
                                for a, b in zip(original["batch"]["runs"], report["batch"]["runs"])),
        "batch_optimizer_step_logs": all(a["optimizer_steps"] == b["optimizer_steps"] for a, b in zip(original["batch"]["runs"], report["batch"]["runs"]))}
    if not all(unchanged.values()):
        raise AssertionError("Reanalysis modified an original training measurement")
    analysis_dir = out / "analysis-source"
    analysis_dir.mkdir()
    for name in analysis_sources:
        (analysis_dir / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    original_provenance = original["provenance"]
    report["provenance"] = {
        "run_id": out.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "report_path": str(out / "report.json"),
        "source_sha256": original_provenance["source_sha256"],
        "primary_sources": original_provenance["primary_sources"],
        "kind": "saved-measurement-reanalysis", "no_full_retraining": True,
        "original_measurement": {**original_provenance, "report_sha256": original_hash},
        "analysis_source_sha256": analysis_sources,
        "analysis_runtime": runtime,
        "change_scope": ["PSD validation and direct centering from saved gradients",
                         "relative zero-mean tolerance in Gram-only diagnostics",
                         "explicit LR-grid boundary and original elapsed-time metadata",
                         "saved checkpoint noise recheck; four-step continuous/fresh-process replay and measured RNG recheck",
                         "compact page extraction; all original learning/coordinate measurements unchanged"],
        "training_unchanged_basis": "original core hash equals current core; no full training runner invoked; curves/evaluations/coordinate/depth tensors and optimizer logs explicitly compared unchanged; saved checkpoint replay remains equal"}
    report["scope"] = "transparent reanalysis of existing width/depth/batch measurements; no full retraining"
    report["wall_seconds_scope"] = "ORIGINAL elapsed complete experiment, not throughput; reanalysis time recorded separately"
    checks = {"original_report_sha256": original_hash, "saved_gradient_gram_max_error": gram_max_error,
              "saved_fixed_theta_gradient_max_error": gradient_recheck_error,
              "gram_psd_and_direct_centering": True, "common_token_grid": common_grid,
              "unchanged_training_measurements": unchanged, "batch_noise_rechecks": batch_rechecks,
              "resume": resume_result, "read_artifacts": artifacts}
    write_json(out / "reanalysis-checks.json", checks)
    write_json(out / "analysis-environment.json", runtime)
    return report


def run_batch_training(splits, seed, batch_size, base_lr, directory,
                       *, record_budget=BATCH_RECORD_BUDGET, resume=False):
    import learning_core as core
    wall_start = time.perf_counter()
    records = splits["train"]
    run_id = f"batch{batch_size}-seed{seed}-lr{base_lr:g}"
    run_dir = directory / run_id
    run_dir.mkdir()
    indices, stream_rng = make_record_stream(records, seed, record_budget)
    stream_ids = [records[index]["id"] for index in indices]
    write_json(run_dir / "consumed-record-stream.json", stream_ids)
    model = core.make_model(seed=seed, **DECODER_CONFIG)
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, **ADAM_CONFIG)
    core.seed_all(seed)
    noise_records = records[::2][:6]
    initial_noise = sampled_noise_at_theta(model, noise_records)
    curve = [{"consumed_input_tokens": 0, "consumed_supervised_tokens": 0, "updates": 0,
              "train": teacher_evaluation(model, records),
              "validation": teacher_evaluation(model, splits["validation"])}]
    initial_eval = {key: core.evaluate_records(model, rows) for key, rows in splits.items()}
    cursor, steps, mid = 0, [], None
    while cursor < record_budget:
        row = advance_batch(model, optimizer, records, indices, cursor, batch_size, base_lr, record_budget)
        cursor = row["cursor"]
        row["updates"] = len(steps) + 1
        steps.append(row)
        if cursor % 512 == 0 or cursor == record_budget:
            curve.append({"consumed_input_tokens": cursor * INPUT_TOKENS_PER_RECORD,
                          "consumed_supervised_tokens": cursor * SUPERVISED_TOKENS_PER_RECORD,
                          "updates": len(steps), "train": teacher_evaluation(model, records),
                          "validation": teacher_evaluation(model, splits["validation"])})
        if cursor == record_budget // 2:
            mid = batch_checkpoint(model, optimizer, records, splits, indices, stream_rng, cursor,
                                   batch_size, base_lr, seed, record_budget, len(steps))
            torch.save(mid, run_dir / "mid-checkpoint.pt")
        if resume and mid and cursor == record_budget // 2 + 4 * batch_size:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "next_steps": [{k: v for k, v in item.items() if k != "updates"} for item in steps[-4:]],
                        "evaluation": core.evaluate_records(model, splits["validation"]),
                        "rng": core.capture_rng_state()},
                       run_dir / "resume-expected.pt")
    final_eval = {key: core.evaluate_records(model, rows) for key, rows in splits.items()}
    final_noise = sampled_noise_at_theta(model, noise_records)
    torch.save(batch_checkpoint(model, optimizer, records, splits, indices, stream_rng, cursor,
                                batch_size, base_lr, seed, record_budget, len(steps)),
               run_dir / "checkpoint.pt")
    # Compare ordinary and incremental core logits on the same final model.
    model.eval()
    probe = core.collate_records(splits["validation"][:4])
    with torch.no_grad():
        dense = model(probe["input_ids"], attention_mask=probe["attention_mask"])
        cache, parts = None, []
        for i in range(probe["input_ids"].shape[1]):
            logits, cache = model.forward_cached(probe["input_ids"][:, i:i+1],
                                                 attention_mask=probe["attention_mask"][:, i:i+1], cache=cache)
            parts.append(logits)
        cache_error = (dense - torch.cat(parts, 1)).abs().max().item()
    model.train()
    if cache_error > 1e-5:
        raise AssertionError("Core full/cached logits differ")
    resume_result = None
    if resume:
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--resume-check", str(run_dir)],
                               cwd=ROOT, check=True, text=True, capture_output=True)
        resume_result = json.loads((run_dir / "resume-check.json").read_text())
    row = {"run_id": run_id, "seed": seed, "batch_size": batch_size, "base_lr": base_lr,
           "config": DECODER_CONFIG, "parameter_count": core.parameter_count(model),
           "optimizer": "Adam", "weight_decay": 0, "clip": None, "dropout": 0,
           "updates": len(steps), "consumed_records": cursor,
           "consumed_input_tokens": cursor * INPUT_TOKENS_PER_RECORD,
           "consumed_supervised_tokens": cursor * SUPERVISED_TOKENS_PER_RECORD,
           "objective": "sequence-mean CE; equal length means token mean is identical",
           "scheduler": {"unit": "input_tokens", "warmup_tokens": 1024,
                         "budget_tokens": record_budget * INPUT_TOKENS_PER_RECORD,
                         "final_fraction": 0.1, "lr_sampling": "actual batch endpoint"},
           "data_stream_sha256": digest_json(stream_ids), "evaluation_split_sha256": core.split_digest(splits),
           "initial_model_sha256": digest_bytes(b"".join(initial_state[k].numpy().tobytes() for k in sorted(initial_state))),
           "curve": curve, "optimizer_steps": steps,
           "initial_evaluation": initial_eval, "final_evaluation": final_eval,
           "noise_record_ids": [r["id"] for r in noise_records],
           "initial_noise": initial_noise, "final_noise": final_noise,
           "cached_logits_max_error": cache_error, "resume": resume_result,
           "wall_seconds": time.perf_counter() - wall_start,
           "wall_seconds_scope": "elapsed experiment including initial/final evaluation, gradient probes, saves and optional resume subprocess; concurrent CPU work may interfere; not throughput",
           "checkpoint": str((run_dir / "checkpoint.pt").relative_to(directory.parent))}
    write_json(run_dir / "run.json", row)
    return row


def run_batch_experiment(directory: Path) -> dict:
    import learning_core as core
    directory.mkdir()
    splits = decoder_splits()
    write_json(directory / "splits.json", splits)
    runs = []
    for seed in SEEDS:
        for lr in BATCH_LRS:
            for batch_size in BATCH_SIZES:
                runs.append(run_batch_training(splits, seed, batch_size, lr, directory,
                                                resume=(seed == SEEDS[0] and lr == BATCH_LRS[0]
                                                        and batch_size == BATCH_SIZES[1])))
        print(f"batch: completed seed {seed}; equal consumed input tokens", flush=True)
    checks = []
    for seed in SEEDS:
        rows = [row for row in runs if row["seed"] == seed]
        valid = all(len({row[key] for row in rows}) == 1 for key in
                    ("data_stream_sha256", "evaluation_split_sha256", "initial_model_sha256",
                     "consumed_records", "consumed_input_tokens", "consumed_supervised_tokens"))
        if not valid:
            raise AssertionError("Batch comparison changed stream, initial state or token budget")
        checks.append({"seed": seed, "same_consumed_stream_and_budget": True,
                       "data_stream_sha256": rows[0]["data_stream_sha256"]})
    write_json(directory / "checks.json", checks)
    return {"core_api": core.CORE_API_VERSION, "config": DECODER_CONFIG,
            "seeds": list(SEEDS), "batch_sizes": list(BATCH_SIZES), "lr_grid": list(BATCH_LRS),
            "record_budget": BATCH_RECORD_BUDGET, "input_tokens_per_record": INPUT_TOKENS_PER_RECORD,
            "supervised_tokens_per_record": SUPERVISED_TOKENS_PER_RECORD,
            "budget_input_tokens": BATCH_RECORD_BUDGET * INPUT_TOKENS_PER_RECORD,
            "budget_supervised_tokens": BATCH_RECORD_BUDGET * SUPERVISED_TOKENS_PER_RECORD,
            "split_counts": core.assert_split_integrity(splits),
            "split_sha256": core.split_digest(splits), "checks": checks,
            "scheduler_definition": "LR(t)=base*t/1024 for t<1024; then base*(0.1+0.45*(1+cos(pi*(t-1024)/(budget-1024)))); t is consumed nonpadding input tokens at the batch endpoint",
            "scope": "18 actual Adam runs, width32/depth2 shared reversal decoder; same stream per seed, 4096 records, no batch-optimal LR claim",
            "runs": runs}


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--out-dir", type=Path)
    value.add_argument("--only", choices=("all", "width"), default="all")
    value.add_argument("--resume-check", type=Path)
    value.add_argument("--reanalyze", type=Path, help="Reuse an existing report; no full training")
    return value


def main():
    args = parser().parse_args()
    if args.resume_check:
        resume_batch_check(args.resume_check)
        return
    out = args.out_dir or ROOT / "artifacts" / ("dynamics-lab-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = configure_runtime()
    started = time.perf_counter()
    if args.reanalyze:
        report = reanalyze_saved_report(args.reanalyze.resolve(), out, runtime)
        report["reanalysis_seconds"] = time.perf_counter() - started
        write_json(out / "report.json", report)
        print(f"Saved transparent reanalysis without full retraining: {out}", flush=True)
        return
    sources = mup_source_metadata()
    source_names = ["training_dynamics.py", "learning_core.py", "run-dynamics-lab.mjs",
                    "dynamics-requirements.txt"]
    source_hashes = {name: digest_bytes((ROOT / "scripts" / name).read_bytes())
                     for name in source_names if (ROOT / "scripts" / name).is_file()}
    report = {"schema": SCHEMA, "status": "running", "runtime": runtime,
              "provenance": {"run_id": out.name, "created_at": datetime.now(timezone.utc).isoformat(),
                             "command": [sys.executable, *sys.argv], "source_sha256": source_hashes,
                             "report_path": str(out / "report.json"),
                             "primary_sources": sources}, "scope": args.only}
    write_json(out / "environment.json", runtime)
    report["width"] = run_width_experiment(out / "width")
    if args.only == "all":
        report["depth"] = run_depth_experiment(out / "depth")
        report["noise"] = run_noise_experiment(out / "noise")
        report["batch"] = run_batch_experiment(out / "batch")
    report["status"] = "complete"
    report["wall_seconds"] = time.perf_counter() - started
    report["wall_seconds_scope"] = "elapsed complete experiment including evaluations, probes, saves and resume subprocess; not a controlled throughput benchmark"
    write_json(out / "report.json", report)
    print(f"Saved actual training-dynamics experiments: {out}", flush=True)


if __name__ == "__main__":
    main()
