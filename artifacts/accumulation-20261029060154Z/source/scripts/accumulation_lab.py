"""One fixed Transformer update: large batch, uneven supervision, accumulation and real CPU DDP."""
import argparse
import copy
from contextlib import nullcontext
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import learning_core as symbolic
import text_training_core as core

TOLERANCE = 2e-10
CASES = [
    {"id": "ragged", "title": "不等量监督", "keep": [4, 8, 40, 96], "chunks": [[0, 1], [2], [3]], "planned": 3, "wrong": "micro-means", "ddp": True},
    {"id": "balanced", "title": "等量监督的对照", "keep": [16, 16, 16, 16], "chunks": [[0], [1], [2], [3]], "planned": 4, "wrong": "micro-means", "ddp": False},
    {"id": "tail", "title": "只剩三个微批的尾窗口", "keep": [16, 16, 16], "chunks": [[0], [1], [2]], "planned": 4, "wrong": "planned-divisor", "ddp": False},
    {"id": "empty", "title": "整个窗口没有监督", "keep": [0, 0, 0, 0], "chunks": [[0], [1], [2], [3]], "planned": 4, "wrong": "zero-step", "ddp": False},
    {"id": "rank-empty", "title": "一个副本没有监督", "keep": [0, 8, 0, 96], "chunks": [[0], [1], [2], [3]], "planned": 4, "wrong": None, "ddp": True},
]


def file_sha(file): return hashlib.sha256(Path(file).read_bytes()).hexdigest()
def save(file, data): Path(file).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def flat_model(model): return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().numpy().copy()
def flat_gradient(model): return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()]).cpu().numpy().copy()
def array_sha(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def state_sha(value):
    def encode(x):
        if isinstance(x, torch.Tensor):
            a = x.detach().contiguous().cpu().numpy()
            return {"dtype": str(a.dtype), "shape": list(a.shape), "sha256": array_sha(a)}
        if isinstance(x, dict): return {str(k): encode(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [encode(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def masked_records(base, case):
    records = []
    for row, keep in zip(base, case["keep"]):
        if keep < 0 or keep > len(row["target"]): raise ValueError("Invalid supervised suffix")
        target = [-100] * (len(row["target"]) - keep) + (row["target"][-keep:] if keep else [])
        records.append({**row, "target": target})
    return records


def new_model_optimizer(saved, kind):
    model = core.make_model(saved["seed"], saved["config"]).double().eval(); model.load_state_dict(saved["model"])
    if kind == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01, foreach=False)
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    elif kind == "sgd": optimizer = torch.optim.SGD(model.parameters(), lr=.01, weight_decay=.01, foreach=False)
    else: raise ValueError("Unknown optimizer")
    return model, optimizer


def optimizer_step_number(optimizer):
    values = [float(v["step"]) for v in optimizer.state.values() if "step" in v]
    return max(values) if values else None


def loss_sum(model, records):
    batch = core.collate(records); logits = model(batch["input_ids"])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1), ignore_index=-100, reduction="sum")


def persist_gradient(directory, name, gradient):
    file = Path(directory) / "gradients" / (name + ".npy"); file.parent.mkdir(exist_ok=True)
    if file.exists():
        if not np.array_equal(np.load(file, allow_pickle=False), gradient): raise ValueError("Optimizer changed a pre-update gradient")
    else: np.save(file, gradient, allow_pickle=False)
    return {"file": "gradients/" + file.name, "sha256": file_sha(file), "values": int(gradient.size), "dtype": str(gradient.dtype)}


def single_trial(saved, records, case, optimizer_kind, method):
    model, optimizer = new_model_optimizer(saved, optimizer_kind); initial = flat_model(model)
    before = state_sha(optimizer.state_dict()); step_before = optimizer_step_number(optimizer)
    count = sum(y != -100 for r in records for y in r["target"]); optimizer.zero_grad(set_to_none=True)
    chunks = [list(range(len(records)))] if method == "reference" else case["chunks"]
    objective = None; backward_calls = 0; scales = []
    update = count > 0 or method == "zero-step"
    if update:
        objective = 0.
        for indexes in chunks:
            batch = [records[i] for i in indexes]; n = sum(y != -100 for r in batch for y in r["target"])
            if method in ("reference", "global-count"): scale = 1 / count
            elif method == "micro-means": scale = 1 / (n * len(chunks)) if n else 0.
            elif method == "planned-divisor": scale = 1 / (n * case["planned"]) if n else 0.
            elif method == "zero-step": scale = 1.
            else: raise ValueError("Unknown accumulation method")
            value = loss_sum(model, batch) * scale; objective += float(value.detach()); value.backward(); backward_calls += 1; scales.append(scale)
    gradient = flat_gradient(model)
    if update: optimizer.step()
    parameters = flat_model(model)
    row = {"case": case["id"], "optimizer": optimizer_kind, "execution": "single", "method": method,
           "global_labels": count, "micro_counts": [sum(y != -100 for i in chunk for y in records[i]["target"]) for chunk in chunks],
           "loss_scales": scales, "scaled_loss": objective, "global_mean_defined": count > 0, "backward_calls": backward_calls,
           "optimizer_step": update, "adam_step_before": step_before, "adam_step_after": optimizer_step_number(optimizer),
           "initial_state_sha256": before, "final_state_sha256": state_sha(optimizer.state_dict()), "parameters_sha256": array_sha(parameters),
           "parameter_move_max": float(np.max(np.abs(parameters - initial)))}
    return row, gradient, parameters, copy.deepcopy(optimizer.state_dict())


def compare(row, gradient, parameters, state, reference):
    _, g, p, s = reference
    row.update({"gradient_max_error": float(np.max(np.abs(gradient - g))), "parameter_max_error": float(np.max(np.abs(parameters - p))),
                "optimizer_state_max_error": core.state_error(state, s), "gradient_l2": float(np.linalg.norm(gradient)),
                "gradient_l2_ratio": float(np.linalg.norm(gradient) / np.linalg.norm(g)) if np.linalg.norm(g) else None})
    return row


def ddp_worker(rank, directory):
    directory = Path(directory); symbolic.configure_cpu(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", init_method=(directory / "gloo-rendezvous").resolve().as_uri(), rank=rank, world_size=2, timeout=datetime.timedelta(seconds=45))
    try:
        data = json.loads((directory / "inputs.json").read_text()); saved = torch.load(directory / "inputs/seed-11-final.pt", weights_only=True, map_location="cpu")
        rows = []
        for case in CASES:
            if not case["ddp"]: continue
            records = masked_records(data["records"], case); local = [records[i] for i in range(rank, len(records), 2)]
            n_local = sum(y != -100 for r in local for y in r["target"])
            for kind in ("adamw", "sgd"):
                for method in ("ddp-global", "ddp-local-mean"):
                    model, optimizer = new_model_optimizer(saved, kind); ddp = DistributedDataParallel(model)
                    counter = torch.tensor(n_local, dtype=torch.int64); dist.all_reduce(counter); n_global = int(counter)
                    before = state_sha(optimizer.state_dict()); step_before = optimizer_step_number(optimizer); optimizer.zero_grad(set_to_none=True)
                    scale = 2 / n_global if method == "ddp-global" else (1 / n_local if n_local else 0.)
                    value_sum = 0.; contexts = []
                    for index, record in enumerate(local):
                        synchronize = index == len(local) - 1
                        with nullcontext() if synchronize else ddp.no_sync():
                            value = loss_sum(ddp, [record]) * scale; value_sum += float(value.detach()); value.backward()
                        contexts.append("synchronize" if synchronize else "no_sync: forward and backward")
                    grad = flat_gradient(model); optimizer.step(); params = flat_model(model)
                    row = {"case": case["id"], "optimizer": kind, "execution": "ddp", "method": method, "rank": rank,
                           "local_labels": n_local, "global_labels": n_global, "micro_counts": [sum(y != -100 for y in r["target"]) for r in local],
                           "loss_scales": [scale] * len(local), "local_scaled_loss": value_sum, "global_mean_defined": True, "contexts": contexts,
                           "backward_calls": len(local), "optimizer_step": True, "adam_step_before": step_before, "adam_step_after": optimizer_step_number(optimizer),
                           "initial_state_sha256": before, "final_state_sha256": state_sha(optimizer.state_dict()), "parameters_sha256": array_sha(params)}
                    row["gradient"] = persist_gradient(directory, f"rank{rank}-{case['id']}-{method}", grad)
                    # Both ranks attest their actual outputs. Identical bytes need only one saved copy.
                    digests = [None, None]
                    dist.all_gather_object(digests, {"rank": rank, "gradient": array_sha(grad), "parameters": row["parameters_sha256"], "optimizer": row["final_state_sha256"]})
                    if any(any(peer[key] != digests[0][key] for key in ("gradient", "parameters", "optimizer")) for peer in digests):
                        raise ValueError("Replica digests differ")
                    file = directory / f"rank0-{case['id']}-{kind}-{method}.pt"
                    if rank == 0: torch.save({"parameters": torch.from_numpy(params), "optimizer": optimizer.state_dict()}, file)
                    dist.barrier()
                    row["replica_digests"] = digests; row["update_file"] = file.name; row["update_sha256"] = file_sha(file); rows.append(row)
        save(directory / f"rank-{rank}.json", {"rank": rank, "world_size": 2, "backend": "gloo", "pid": os.getpid(), "runtime": symbolic.environment_metadata(), "rows": rows})
    finally: dist.destroy_process_group()


def run(directory, practice_directory):
    symbolic.configure_cpu(1); out = Path(directory); out.mkdir(parents=True, exist_ok=False); inputs = out / "inputs"; inputs.mkdir()
    text = Path(practice_directory) / "text"; report = json.loads((text / "report.json").read_text())
    original = next(r for r in report["runs"] if r["seed"] == 11)
    if file_sha(text / "seed-11-final.pt") != original["checkpoint_sha256"]: raise ValueError("Checkpoint identity changed")
    frozen = {"model": report["model"], "parameter_count": report["parameter_count"], "checkpoint": "unchanged seed 11 final step 800",
              "dtype": "float64; original float32 model and AdamW state cast consistently", "device": "cpu", "threads": 1, "cases": CASES,
              "supervision": "keep the declared suffix of each saved shifted target; all other targets are -100; not a chat template or SFT quality evaluation",
              "objective": "sum of cross-entropies / total nonignored labels in this update", "ddp": "two CPU Gloo ranks; default DDP gradient average; no_sync includes forward and backward",
              "adamw": "restore the same saved optimizer state at step 800, lr=.002 and weight_decay=.01", "sgd": "fresh SGD, lr=.01, weight_decay=.01, momentum=0",
              "empty_policy": "global N=0: mean undefined, skip backward and optimizer update; a zero-step negative control is separately retained",
              "tolerance": TOLERANCE, "excluded": ["dropout", "gradient clipping", "AMP", "GPU/NCCL", "performance or learning-quality claims"]}
    save(out / "frozen-config.json", frozen)
    for name in ("seed-11-final.pt", "tokenizer.json", "windows.json", "report.json"): shutil.copyfile(text / name, inputs / name)
    windows = json.loads((inputs / "windows.json").read_text()); base = [copy.deepcopy(r) for r in windows["train"] if len(r["target"]) == 96][:4]
    if len(base) != 4: raise ValueError("Need four full saved windows")
    tokenizer = json.loads((inputs / "tokenizer.json").read_text())
    save(out / "inputs.json", {"records": base, "previews": [core.decode(r["input"], tokenizer) for r in base], "source": {name: file_sha(inputs / name) for name in ("seed-11-final.pt", "tokenizer.json", "windows.json", "report.json")}})
    saved = torch.load(inputs / "seed-11-final.pt", weights_only=True, map_location="cpu")
    rows = []; references = {}
    for case in CASES:
        records = masked_records(base, case)
        for kind in ("adamw", "sgd"):
            reference = single_trial(saved, records, case, kind, "reference"); references[(case["id"], kind)] = reference
            ref = compare(reference[0], *reference[1:], reference)
            ref["gradient"] = persist_gradient(out, f"single-{case['id']}-reference", reference[1]); rows.append(ref)
            if case["id"] == "rank-empty": continue
            for method in ("global-count", case["wrong"]):
                result = single_trial(saved, records, case, kind, method); row = compare(result[0], *result[1:], reference)
                row["gradient"] = persist_gradient(out, f"single-{case['id']}-{method}", result[1]); rows.append(row)
    mp.spawn(ddp_worker, args=(str(out.resolve()),), nprocs=2, join=True)
    rank_results = [json.loads((out / f"rank-{r}.json").read_text()) for r in (0, 1)]
    cross_rank = []
    for row in rank_results[0]["rows"]:
        other = next(r for r in rank_results[1]["rows"] if (r["case"], r["optimizer"], r["method"]) == (row["case"], row["optimizer"], row["method"]))
        grad = np.load(out / row["gradient"]["file"], allow_pickle=False); other_grad = np.load(out / other["gradient"]["file"], allow_pickle=False)
        actual = torch.load(out / row["update_file"], weights_only=True, map_location="cpu"); peer = torch.load(out / other["update_file"], weights_only=True, map_location="cpu")
        compare(row, grad, actual["parameters"].numpy(), actual["optimizer"], references[(row["case"], row["optimizer"])])
        row["peer_gradient_max_error"] = float(np.max(np.abs(grad - other_grad))); row["peer_parameter_max_error"] = float((actual["parameters"] - peer["parameters"]).abs().max())
        row["peer_optimizer_state_max_error"] = core.state_error(actual["optimizer"], peer["optimizer"]); rows.append(row)
        cross_rank.append({"case": row["case"], "optimizer": row["optimizer"], "method": row["method"], "gradient": row["peer_gradient_max_error"], "parameters": row["peer_parameter_max_error"], "optimizer": row["peer_optimizer_state_max_error"]})
    correct = [r for r in rows if r["method"] in ("global-count", "ddp-global")]
    if any(max(r["gradient_max_error"], r["parameter_max_error"], r["optimizer_state_max_error"]) > TOLERANCE for r in correct): raise ValueError("A correct normalization path differs from the reference")
    if any(max(r["gradient"], r["parameters"], r["optimizer"]) > TOLERANCE for r in cross_rank): raise ValueError("DDP replicas diverged")
    # These algebraic boundaries are frozen before execution; unfavorable controls remain in the report.
    for kind in ("adamw", "sgd"):
        pick = lambda case, method: next(r for r in rows if r["case"] == case and r["optimizer"] == kind and r["method"] == method)
        if pick("balanced", "micro-means")["gradient_max_error"] > TOLERANCE: raise ValueError("Balanced control did not agree")
        if abs(pick("tail", "planned-divisor")["gradient_l2_ratio"] - .75) > 1e-10: raise ValueError("Tail divisor ratio differs")
        if abs(pick("rank-empty", "ddp-local-mean")["gradient_l2_ratio"] - .5) > 1e-10: raise ValueError("Empty-rank local mean ratio differs")
        if pick("empty", "global-count")["initial_state_sha256"] != pick("empty", "global-count")["final_state_sha256"]: raise ValueError("Empty update advanced optimizer state")
    output = {"schema": "accumulation-equivalence-v1", "status": "passed", "frozen": frozen, "runtime": symbolic.environment_metadata(),
              "inputs": json.loads((out / "inputs.json").read_text()), "rows": rows, "distributed": {"backend": "gloo", "world_size": 2, "ranks": [{k: v for k, v in r.items() if k != "rows"} for r in rank_results], "cross_rank": cross_rank},
              "correct_paths_passed": True, "limits": ["one frozen CPU Transformer update, not a training-quality or GPU benchmark", "mask suffixes explicitly isolate label counts, not assistant-template correctness",
                  "float64 and fixed dropout-free forward; numerical equivalence uses tolerance rather than bit identity", "correctness status applies to declared correct paths; incorrect normalization and zero-step controls are preserved"]}
    save(out / "report.json", output); return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out-dir", required=True); parser.add_argument("--practice-directory", required=True)
    args = parser.parse_args(); run(args.out_dir, args.practice_directory)
