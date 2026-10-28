"""Independent CPU references for context, RL, quantization, statistics and IVF.

Autograd differentiates dense/block objectives independently of the JavaScript
derivatives. The two-process experiment uses actual point-to-point Gloo messages.
"""
import argparse
import datetime
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
from importlib.metadata import distributions
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DTYPE = torch.float64


def tensor(values, grad=False):
    return torch.tensor(values, dtype=DTYPE, requires_grad=grad)


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def write(path, data):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(plain(data), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


class Checks:
    def __init__(self):
        self.rows = []

    def compare(self, name, actual, expected):
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().numpy()
        if actual is None or expected is None:
            assert actual is None and expected is None, name
            error = 0.
        else:
            a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
            np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-8, err_msg=name)
            error = float(np.max(np.abs(a - b))) if a.size else 0.
        self.rows.append({"name": name, "max_error": error})


def ranges_for(n, split):
    return [(0, n)] if split == n else [(0, split), (split, n)]


def context_autograd(inputs, options):
    n = len(inputs["Q"])
    Q, K, V, G = (tensor(inputs[name], name != "G") for name in ["Q", "K", "V", "G"])
    ranges = ranges_for(n, options["split"])
    owners = [next(r for r, (s, e) in enumerate(ranges) if s <= i < e) for i in range(n)]
    blocks = [(s, min(s + options["block"], end)) for start, end in ranges for s in range(start, end, options["block"])]
    outputs, losses = [], []
    for i in range(n):
        rank = owners[i]
        indices = []
        for j in range(n):
            visible = options["mask"] == "full" or (j - ranges[owners[j]][0] <= i - ranges[rank][0] if options["fault"] == "local-position" else j <= i)
            if visible and not (options["mask"] == "empty-first" and i == 0):
                indices.append(j)
        if not indices:
            output = (Q[i] + K[i] + V[i]) * 0
        else:
            parts = [[j for j in indices if s <= j < e] for s, e in blocks] if options["fault"] == "local-softmax" else [indices]
            chunks = []
            for part in parts:
                if not part:
                    continue
                keys = torch.stack([K[j].detach() if options["fault"] == "drop-remote" and owners[j] != rank else K[j] for j in part])
                values = torch.stack([V[j].detach() if options["fault"] == "drop-remote" and owners[j] != rank else V[j] for j in part])
                probabilities = (Q[i] @ keys.T / math.sqrt(Q.shape[1])).softmax(dim=-1)
                chunks.append(probabilities @ values)
            output = torch.stack(chunks).mean(dim=0)
        outputs.append(output)
        denominator = len(ranges) * (ranges[rank][1] - ranges[rank][0]) if options["fault"] == "local-mean" else n
        losses.append((output * G[i]).sum() / denominator)
    loss = torch.stack(losses).sum()
    grads = torch.autograd.grad(loss, [Q, K, V])
    return {"O": torch.stack(outputs), "loss": loss, **dict(zip(["dQ", "dK", "dV"], grads))}


def stream_kv(rank, ranges, local_kv, block):
    """Both ranks traverse owners in the same order; messages contain one tile."""
    for owner, (start, end) in enumerate(ranges):
        for first in range(start, end, block):
            last = min(end, first + block)
            if rank == owner:
                tile = local_kv[first - start:last - start].contiguous()
                dist.send(tile, 1 - rank)
            else:
                tile = torch.empty((last - first, 4), dtype=DTYPE)
                dist.recv(tile, owner)
            yield owner, first, last, tile


def gloo_worker(rank, directory, rendezvous):
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=60))
    try:
        fixture = json.loads((Path(directory) / "reference.json").read_text())
        inputs = fixture["contextInputs"]
        results = []
        for split in [1, 2, 4]:
            ranges = ranges_for(5, split)
            start, end = ranges[rank]
            peer_start, peer_end = ranges[1 - rank]
            Q, K, V, G = [tensor(inputs[name][start:end]) for name in ["Q", "K", "V", "G"]]
            local_kv = torch.cat([K, V], dim=1)
            for mask in ["causal", "full", "empty-first"]:
                for fault in ["none", "drop-remote"]:
                    for block in [1, 3]:
                        maximum = torch.full((end - start,), -torch.inf, dtype=DTYPE)
                        denominator, weighted = torch.zeros(end - start, dtype=DTYPE), torch.zeros_like(Q)
                        messages = []
                        for owner, first, last, tile in stream_kv(rank, ranges, local_kv, block):
                            messages.append({"phase": "forward", "owner": owner, "start": first, "end": last, "elements": tile.numel()})
                            for row, i in enumerate(range(start, end)):
                                indices = [j - first for j in range(first, last) if (mask == "full" or j <= i) and not (mask == "empty-first" and i == 0)]
                                if not indices:
                                    continue
                                scores = Q[row] @ tile[indices, :2].T / math.sqrt(2)
                                new_maximum = max(float(maximum[row]), float(scores.max()))
                                correction = math.exp(float(maximum[row]) - new_maximum)
                                p = torch.exp(scores - new_maximum)
                                denominator[row] = denominator[row] * correction + p.sum()
                                weighted[row] = weighted[row] * correction + p @ tile[indices, 2:]
                                maximum[row] = new_maximum
                        output = weighted / denominator.clamp_min(torch.finfo(DTYPE).tiny)[:, None]
                        lse = maximum + denominator.log()
                        del tile
                        dQ, owned = torch.zeros_like(Q), torch.zeros((end - start, 4), dtype=DTYPE)
                        for owner, first, last, tile in stream_kv(rank, ranges, local_kv, block):
                            messages.append({"phase": "backward-replay", "owner": owner, "start": first, "end": last, "elements": tile.numel()})
                            contribution = torch.zeros_like(tile)
                            for row, i in enumerate(range(start, end)):
                                if not torch.isfinite(lse[row]):
                                    continue
                                upstream, baseline = G[row] / 5, (G[row] / 5) @ output[row]
                                for j in range(first, last):
                                    if mask != "full" and j > i:
                                        continue
                                    index = j - first
                                    probability = torch.exp(Q[row] @ tile[index, :2] / math.sqrt(2) - lse[row])
                                    ds = probability * (upstream @ tile[index, 2:] - baseline) / math.sqrt(2)
                                    dQ[row] += ds * tile[index, :2]
                                    contribution[index, :2] += ds * Q[row]
                                    contribution[index, 2:] += probability * upstream
                            if rank == owner:
                                remote = torch.empty_like(contribution)
                                dist.recv(remote, 1 - rank)
                                owned[first - start:last - start] += contribution + (remote if fault == "none" else torch.zeros_like(remote))
                            else:
                                # Send even in the fault case so both ranks keep the same protocol.
                                dist.send(contribution, owner)
                            messages.append({"phase": "gradient-return", "owner": owner, "start": first, "end": last, "elements": contribution.numel()})
                        results.append({"options": {"split": split, "block": block, "mask": mask, "fault": fault}, "start": start, "end": end, "O": output, "dQ": dQ, "dK": owned[:, :2], "dV": owned[:, 2:], "send_elements": 8 * (end - start) + 4 * (peer_end - peer_start), "messages": messages})
        write(Path(directory) / f"rank-{rank}.json", {"rank": rank, "backend": "gloo", "results": results})
    finally:
        dist.destroy_process_group()


def policy_tokens(inputs):
    tokens = []
    for sample, actions in enumerate(inputs["responses"]):
        tokens.append((sample, -1, "prompt", 1, [1, -.5]))
        tokens.extend((sample, position, "response", action, [1 + sample * .1, .2 + position * .35 - sample * .1]) for position, action in enumerate(actions))
        tokens.append((sample, len(actions), "padding", 2, [0, 1]))
    return tokens


def policy_reference(inputs, options):
    weights, behavior, reference = [tensor(inputs[name]) for name in ["weights", "weights", "reference"]]
    tokens = policy_tokens(inputs)
    X = tensor([row[4] for row in tokens])
    actions = torch.tensor([row[3] for row in tokens], dtype=torch.int64)
    rewards = tensor(options["rewards"])
    advantages = (rewards - rewards.mean()) / rewards.std(correction=0) if rewards.max() != rewards.min() else torch.zeros_like(rewards)
    active = [(kind == "response" and position < options["limit"]) or (options["fault"] == "include-controls" and kind != "response") for sample, position, kind, action, x in tokens]
    lengths = [sum(enabled and tokens[i][0] == sample for i, enabled in enumerate(active)) for sample in range(4)]
    count, nonempty = sum(active), sum(length > 0 for length in lengths)
    normal = options["normalization"]
    if options["fault"] == "wrong-denominator":
        normal = "sequence" if normal == "token" else "token"
    factors = tensor([0. if not active[i] else 1. / count if normal == "token" else 1. / (nonempty * lengths[row[0]]) for i, row in enumerate(tokens)])
    expanded_advantage = torch.stack([advantages[row[0]] for row in tokens])
    history = []
    for _ in range(options["steps"]):
        w = weights.detach().clone().requires_grad_(True)
        old = weights.detach() if options["fault"] == "refresh-old" else behavior
        current_logp, old_logp, reference_logp = [torch.log_softmax(X @ v, dim=-1) for v in [w, old, reference]]
        probabilities = current_logp.exp()
        index = torch.arange(len(tokens))
        denominator = current_logp[index, actions] if options["fault"] == "old-graph" else old_logp[index, actions].detach()
        ratio = (current_logp[index, actions] - denominator).exp()
        surrogate = torch.minimum(ratio * expanded_advantage, ratio.clamp(.8, 1.2) * expanded_advantage)
        kl = (probabilities * (current_logp - reference_logp.detach())).sum(dim=-1)
        loss = (factors * (-surrogate + options["beta"] * kl)).sum()
        gradient, = torch.autograd.grad(loss, [w])
        updated = w.detach() - .2 * gradient
        history.append({"weights": w, "oldWeights": old, "updated": updated, "gradient": gradient, "loss": loss if count else None, "kl": (factors * kl).sum(), "count": count, "ratios": ratio, "probabilities": probabilities})
        weights = updated
    return history


def expectation_reference(inputs):
    W = tensor(inputs["weights"], True)
    root = (tensor([1, .2]) @ W).softmax(dim=-1)
    rewards = tensor([[1, -.3, .2], [0, .8, -.1], [-.2, .1, 1.2]])
    expectation = sum(root[a] * ((tensor([1, .3 + .4 * a]) @ W).softmax(dim=-1) * rewards[a]).sum() for a in range(3))
    gradient, = torch.autograd.grad(expectation, [W])
    return expectation, gradient


def round_away(values):
    return np.sign(values) * np.floor(np.abs(values) + .5)


def calibrate(values):
    low, high = min(0., float(np.min(values))), max(0., float(np.max(values)))
    scale = (high - low) / 255 if low != high else 1.
    return scale, np.clip(round_away(-low / scale), 0, 255)


def quantization_reference(inputs, options):
    X, W, bias, calibration = [np.asarray(inputs[name], dtype=np.float64) for name in ["X", "W", "bias", "calibrationX"]]
    floating = X @ W + bias
    if options["mode"] == "w4a16":
        scales = np.max(np.abs(W), axis=0) / 7
        scales = np.where(scales == 0, 1., scales)
        qW = np.clip(round_away(W / scales), -7, 7).astype(np.int64)
        restored = X.astype(np.float16).astype(np.float64) @ (qW * scales) + bias
        return {"floating": floating, "restored": restored, "actual": restored, "output": restored, "qW": qW}
    sx, zx = calibrate(calibration)
    sw, zw = np.array([calibrate(W[:, j]) for j in range(2)]).T
    sy, zy = calibrate(calibration @ W + bias)
    qX = np.clip(round_away(X / sx) + zx, 0, 255).astype(np.int64)
    qW = np.clip(round_away(W / sw) + zw, 0, 255).astype(np.int64)
    qBias = round_away(bias / (sx * sw)).astype(np.int64)
    centeredX, centeredW = qX - int(zx), qW - zw.astype(np.int64)
    C = centeredX @ centeredW
    restored = (centeredX * sx) @ (centeredW * sw) + qBias * (sx * sw)
    actualC = qX @ qW if options["fault"] == "zero-point" else C
    actualScale = np.full(2, sx * sw[0]) if options["fault"] == "scale-axis" else sx * sw
    actual = actualC * actualScale + qBias if options["fault"] == "bias-unit" else (actualC + qBias) * actualScale
    output = (np.clip(round_away(actual / sy) + zy, 0, 255) - zy) * sy
    return {"floating": floating, "restored": restored, "actual": actual, "output": output, "qW": qW, "qX": qX, "qBias": qBias}


def lcg(seed):
    state = seed
    while True:
        state = (1664525 * state + 1013904223) & 0xffffffff
        yield state / 4294967296


def statistics_reference(data, options):
    rows = [row for row in data["rows"] if row["seed"] == options["seed"]]
    groups = {}
    for row in rows:
        key = row["group"] if options["mode"] == "group" else row["id"]
        groups.setdefault(key, []).append(row["new"] - row["old"])
    units = list(groups.values())
    random = lcg(options["randomSeed"])
    distribution = []
    for _ in range(options["repetitions"]):
        selected = [units[int(next(random) * len(units))] for _ in units]
        distribution.append(sum(sum(group) for group in selected) / sum(len(group) for group in selected))
    distribution.sort()
    total = sum(sum(unit) for unit in units)
    sums = np.array([sum(sign * sum(unit) for sign, unit in zip(signs, units)) for signs in itertools.product([-1, 1], repeat=len(units))])
    p = float(np.mean(np.abs(sums) >= abs(total) - 1e-12))
    return {"delta": total / len(rows), "interval": np.quantile(distribution, [.025, .975], method="linear"), "distribution": distribution, "p": p}


def selection_reference(candidates):
    validation, test = lcg(91), lcg(1398)
    chosen, independent = [], []
    for _ in range(300):
        pairs = [(sum(next(validation) < .6 for _ in range(40)) / 40, sum(next(test) < .6 for _ in range(40)) / 40) for _ in range(candidates)]
        selected = max(range(candidates), key=lambda i: pairs[i][0])
        chosen.append(pairs[selected][0])
        independent.append(pairs[selected][1])
    return np.mean(chosen), np.mean(independent)


def ann_reference(inputs, options):
    docs = inputs["documents"]
    X = np.asarray([doc["vector"] for doc in docs], dtype=np.float64)
    C = np.asarray(inputs["centroids"], dtype=np.float64)
    query = np.asarray(inputs["queries"][options["query"]], dtype=np.float64)
    R = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    if options["version"] == "rebuilt":
        X, C = X @ R, C @ R
    q = query @ R if options["version"] != "aligned" else query
    assignments = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1), axis=1)
    probes = np.argsort(((C - q) ** 2).sum(axis=-1), kind="stable")[:options["nprobe"]]
    valid = [i for i, doc in enumerate(docs) if options["filter"] == "all" or doc["group"] == options["filter"]]
    candidates = [i for i in valid if assignments[i] in probes]
    distances = ((X - q) ** 2).sum(axis=-1)
    order = lambda indices, values: sorted(indices, key=lambda i: (values[i], docs[i]["id"]))[:options["k"]]
    exact, approx = order(valid, distances), order(candidates, distances)
    original = np.asarray([doc["vector"] for doc in docs])
    oracle = order(valid, ((original - query) ** 2).sum(axis=-1))
    overlap = lambda actual, expected: len(set(actual) & set(expected)) / len(expected) if expected else None
    relevant = [i for i in valid if docs[i]["relevant"]] if options["query"] == "center" else []
    return {"exact": [docs[i]["id"] for i in exact], "approximate": [docs[i]["id"] for i in approx], "distances": [distances[i] for i in approx], "annRecall": overlap(approx, exact), "oracleRecall": overlap(approx, oracle), "taskRecall": overlap(approx, relevant), "compatible": options["version"] != "stale"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.out_dir).resolve()
    fixture = json.loads((directory / "reference.json").read_text())
    torch.set_num_threads(1)
    checks = Checks()
    for i, case in enumerate(fixture["context"]):
        actual = context_autograd(fixture["contextInputs"], case["options"])
        for name in ["O", "loss", "dQ", "dK", "dV"]:
            checks.compare(f"context/{i}/{name}", actual[name], case[name])
    print(f"Context autograd: {len(fixture['context'])} configurations passed.", flush=True)
    for i, case in enumerate(fixture["policy"]):
        actual = policy_reference(fixture["policyInputs"], case["options"])
        for step, (got, expected) in enumerate(zip(actual, case["history"])):
            for name in got:
                checks.compare(f"policy/{i}/step-{step}/{name}", got[name], expected[name])
    expectation, gradient = expectation_reference(fixture["policyInputs"])
    checks.compare("expectation/value", expectation, fixture["expectation"]["expectation"])
    checks.compare("expectation/gradient", gradient, fixture["expectation"]["gradient"])
    print(f"Policy autograd: {len(fixture['policy'])} four-step configurations passed.", flush=True)
    for i, case in enumerate(fixture["quantization"]):
        for name, actual in quantization_reference(case["inputs"], case["options"]).items():
            checks.compare(f"quantization/{i}/{name}", actual, case[name])
    for i, case in enumerate(fixture["statistics"]):
        actual = statistics_reference(case["data"], case["options"])
        for name in ["delta", "interval", "distribution"]:
            checks.compare(f"statistics/{i}/{name}", actual[name], case[name])
        checks.compare(f"statistics/{i}/p", actual["p"], case["permutation"]["p"])
    for case in fixture["selection"]:
        actual = selection_reference(case["candidates"])
        checks.compare(f"selection/{case['candidates']}", actual, [case["selected"], case["independent"]])
    for i, case in enumerate(fixture["ann"]):
        actual = ann_reference(fixture["annInputs"], case["options"])
        for name in ["exact", "approximate", "compatible"]:
            assert actual[name] == case[name], f"ann/{i}/{name}"
        for name in ["distances", "annRecall", "oracleRecall", "taskRecall"]:
            checks.compare(f"ann/{i}/{name}", actual[name], case[name])
    print("Integer arithmetic, paired statistics, selection and IVF references passed.", flush=True)
    mp.spawn(gloo_worker, args=(str(directory), (directory / "gloo-rendezvous").as_uri()), nprocs=2, join=True)
    fixture_map = {json.dumps(case["options"], sort_keys=True): case for case in fixture["context"]}
    for rank in range(2):
        result = json.loads((directory / f"rank-{rank}.json").read_text())
        for i, actual in enumerate(result["results"]):
            expected = fixture_map[json.dumps(actual["options"], sort_keys=True)]
            for name in ["O", "dQ", "dK", "dV"]:
                checks.compare(f"gloo/rank-{rank}/{i}/{name}", actual[name], expected[name][actual["start"]:actual["end"]])
    environment = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "dtype": "float64", "threads": 1, "packages": {item.metadata["Name"]: item.version for item in distributions()}}
    write(directory / "environment.json", environment)
    report = {"schema": 1, "kind": "decision-chains", "success": True, "created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "environment": environment,
              "configurations": {name: len(fixture[name]) for name in ["context", "policy", "quantization", "statistics", "selection", "ann"]}, "gloo_per_rank": 36,
              "comparisons": len(checks.rows), "max_error": max(row["max_error"] for row in checks.rows), "checks": checks.rows,
              "scope": "Independent CPU correctness; two local Gloo processes. No CUDA/NCCL, low-bit kernels, language-model training or throughput claims."}
    write(directory / "report.json", report)
    print(f"Passed {report['comparisons']} comparisons; max absolute error {report['max_error']:.3e}. Report: {directory / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
