"""Independent PyTorch CPU checks, Gloo clipping and measured cost reports.

Timing, saved-tensor storage references and algorithm element counts are separate
measurements. The tiled Python implementation is not a FlashAttention GPU kernel.
"""
import argparse
import contextlib
import gc
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import datetime
from importlib.metadata import distributions
from zoneinfo import ZoneInfo

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

DTYPE = torch.float64


def t(value, grad=False):
    return torch.tensor(value, dtype=DTYPE, requires_grad=grad)


def plain(value):
    if isinstance(value, torch.Tensor):
        return plain(value.detach().tolist())
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write(path, data):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(plain(data), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


class Checks:
    def __init__(self):
        self.rows = []

    def compare(self, name, actual, expected, tolerance=1e-9):
        a, b = torch.as_tensor(actual, dtype=DTYPE).detach(), torch.as_tensor(expected, dtype=DTYPE).detach()
        torch.testing.assert_close(a, b, atol=tolerance, rtol=1e-8, msg=name)
        self.rows.append({"name": name, "max_error": float((a - b).abs().max()) if a.numel() else 0})


def allowed(qids, kids, mask):
    result = torch.ones((len(qids), len(kids)), dtype=torch.bool) if mask == "full" else kids[None, :] <= qids[:, None]
    return result & (qids[:, None] != 0) if mask == "empty-first" else result


def dense_attention(Q, K, V, mask):
    scores = Q @ K.T / math.sqrt(Q.shape[1])
    valid = allowed(torch.arange(len(Q)), torch.arange(len(K)), mask)
    has_key = valid.any(dim=-1, keepdim=True)
    safe = valid | ~has_key
    P = scores.masked_fill(~safe, -torch.inf).softmax(dim=-1) * has_key
    return P @ V


def tiled_forward(Q, K, V, qb, kb, mask):
    n, d = Q.shape
    O, lse = torch.zeros_like(Q), torch.full((n,), -torch.inf, dtype=DTYPE)
    for qs in range(0, n, qb):
        qe = min(qs + qb, n)
        m = torch.full((qe - qs,), -torch.inf, dtype=DTYPE)
        denominator, weighted = torch.zeros_like(m), torch.zeros((qe - qs, d), dtype=DTYPE)
        for ks in range(0, n, kb):
            ke = min(ks + kb, n)
            valid = allowed(torch.arange(qs, qe), torch.arange(ks, ke), mask)
            scores = (Q[qs:qe] @ K[ks:ke].T / math.sqrt(d)).masked_fill(~valid, -torch.inf)
            new_m = torch.maximum(m, scores.max(dim=-1).values)
            safe_m = torch.where(torch.isfinite(new_m), new_m, torch.zeros_like(new_m))
            correction = torch.exp(m - safe_m)
            exp = torch.exp(scores - safe_m[:, None])
            denominator = correction * denominator + exp.sum(dim=-1)
            weighted = correction[:, None] * weighted + exp @ V[ks:ke]
            m = new_m
        O[qs:qe] = weighted / denominator.clamp_min(torch.finfo(DTYPE).tiny)[:, None]
        lse[qs:qe] = m + denominator.log()
    return O, lse


def tiled_backward(Q, K, V, O, lse, G, qb, kb, mask, fault="none"):
    n, d = Q.shape
    dQ, dK, dV = torch.zeros_like(Q), torch.zeros_like(K), torch.zeros_like(V)
    D = (O * G).sum(dim=-1)
    for qs in range(0, n, qb):
        qe = min(n, qs + qb)
        for ks in range(0, n, kb):
            ke = min(n, ks + kb)
            valid = allowed(torch.arange(qs, qe), torch.arange(ks, ke), mask)
            scores = (Q[qs:qe] @ K[ks:ke].T / math.sqrt(d)).masked_fill(~valid, -torch.inf)
            if fault == "local-softmax":
                has_key = valid.any(dim=-1, keepdim=True)
                P = scores.masked_fill(~(valid | ~has_key), -torch.inf)
                P = torch.where(has_key, P, torch.zeros_like(P)).softmax(dim=-1) * has_key
            else:
                safe_lse = torch.where(torch.isfinite(lse[qs:qe]), lse[qs:qe], torch.zeros_like(lse[qs:qe]))
                P = torch.exp(scores - safe_lse[:, None])
            dScore = P * (G[qs:qe] @ V[ks:ke].T - D[qs:qe, None]) / math.sqrt(d)
            dQ[qs:qe] += dScore @ K[ks:ke]
            dk, dv = dScore.T @ Q[qs:qe], P.T @ G[qs:qe]
            if fault == "overwrite-kv":
                dK[ks:ke], dV[ks:ke] = dk, dv
            else:
                dK[ks:ke] += dk
                dV[ks:ke] += dv
    return dQ, dK, dV


class TiledAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, qb, kb, mask):
        O, lse = tiled_forward(Q, K, V, qb, kb, mask)
        ctx.save_for_backward(Q, K, V, O, lse)
        ctx.options = qb, kb, mask
        return O

    @staticmethod
    def backward(ctx, G):
        Q, K, V, O, lse = ctx.saved_tensors
        return (*tiled_backward(Q, K, V, O, lse, G, *ctx.options), None, None, None)


def check_attention(fixture, checks):
    for index, case in enumerate(fixture["attention"]):
        inputs = {name: t(value, name != "G") for name, value in fixture["attentionInputs"].items()}
        Q, K, V, G = (inputs[name] for name in ["Q", "K", "V", "G"])
        options = case["options"]
        reference = dense_attention(Q, K, V, options["mask"])
        (reference * G).sum().backward()
        with torch.no_grad():
            O, lse = tiled_forward(Q, K, V, options["queryBlock"], options["keyBlock"], options["mask"])
            gradients = tiled_backward(Q, K, V, O, lse, G, options["queryBlock"], options["keyBlock"], options["mask"], options["fault"])
        prefix = f"attention/{index}"
        checks.compare(prefix + "/forward", O, reference)
        checks.compare(prefix + "/js-output", O, case["O"])
        for name, source, gradient in zip(["dQ", "dK", "dV"], [Q, K, V], gradients):
            checks.compare(prefix + "/js-" + name, gradient, case[name])
            if options["fault"] == "none":
                checks.compare(prefix + "/autograd-" + name, gradient, source.grad)


def lcg_mask(shape, dropout, state):
    values = []
    for _ in range(math.prod(shape)):
        if not dropout:
            values.append(1.)
        else:
            state[0] = (1664525 * state[0] + 1013904223) & 0xffffffff
            values.append((1 / (1 - dropout)) if state[0] / 4294967296 >= dropout else 0.)
    return t(values).reshape(shape)


def encode(X, W1, W2, dropout, state=None):
    H = (X @ W1).tanh()
    Z = H * lcg_mask(H.shape, dropout, state) if state is not None else F.dropout(H, p=dropout, training=True)
    return Z @ W2


def cache_run(inputs, chunk, dropout, cached, restore=True, native_rng=False, verify_replay=True):
    W1, W2 = inputs["W1"].detach().requires_grad_(), inputs["W2"].detach().requires_grad_()
    count = len(inputs["query"])
    batches = [(role, start, inputs[role][start:start + chunk]) for role in ["query", "document"] for start in range(0, count, chunk)]
    state = None if native_rng else [31]
    if native_rng:
        torch.manual_seed(31)
    states, features = [], []
    with torch.profiler.record_function("cache_first_encoding" if cached else "retained_encoding"):
        with torch.no_grad() if cached else contextlib.nullcontext():
            for _, _, X in batches:
                states.append(torch.get_rng_state() if native_rng else state[0])
                features.append(encode(X, W1, W2, dropout, state))
    after_first = torch.get_rng_state() if native_rng else state[0]
    leaves = [value.detach().requires_grad_() for value in features] if cached else features
    split = sum(role == "query" for role, _, _ in batches)
    Q, D = torch.cat(leaves[:split]), torch.cat(leaves[split:])
    with torch.profiler.record_function("global_contrastive_loss_and_backward"):
        loss = F.cross_entropy(Q @ D.T, torch.arange(count))
        loss.backward()
    replay_error = 0.
    if cached:
        for (_, _, X), leaf, saved in zip(batches, leaves, states):
            if restore:
                if native_rng:
                    torch.set_rng_state(saved)
                else:
                    state[0] = saved
            with torch.profiler.record_function("cache_replay_and_parameter_vjp"):
                replay = encode(X, W1, W2, dropout, state)
                if verify_replay:
                    replay_error = max(replay_error, float((replay.detach() - leaf.detach()).abs().max()))
                replay.backward(leaf.grad)
        if restore and native_rng:
            torch.set_rng_state(after_first)
        elif restore:
            state[0] = after_first
    return {"loss": loss.detach(), "W1": W1.grad.detach(), "W2": W2.grad.detach(), "replay_error": replay_error, "rng_after": torch.get_rng_state() if native_rng else state[0]}


def check_cache(fixture, checks):
    inputs = {name: t(value) for name, value in fixture["cacheInputs"].items() if name != "seed"}
    for index, case in enumerate(fixture["cache"]):
        options = case["options"]
        reference = cache_run(inputs, options["chunk"], options["dropout"], False)
        actual = cache_run(inputs, options["chunk"], options["dropout"], True, options["fault"] == "none")
        checks.compare(f"cache/{index}/loss", actual["loss"], case["loss"])
        checks.compare(f"cache/{index}/replay", actual["replay_error"], case["replayError"])
        for name in ["W1", "W2"]:
            checks.compare(f"cache/{index}/js-{name}", actual[name], case["gradients"][name])
            checks.compare(f"cache/{index}/reference-{name}", reference[name], case["reference"][name])
            if options["fault"] == "none":
                checks.compare(f"cache/{index}/same-{name}", actual[name], reference[name])
                checks.compare(f"cache/{index}/sgd-{name}", inputs[name] - .1 * actual[name], inputs[name] - .1 * reference[name])
        if options["fault"] == "none":
            assert actual["rng_after"] == reference["rng_after"]
    # Also exercise real torch dropout and torch RNG; compare the same chunked
    # forward trajectory, not arbitrary batch shapes with unrelated masks.
    native = []
    for chunk in [1, 2, 3, 4]:
        reference = cache_run(inputs, chunk, .25, False, native_rng=True)
        for restore in [True, False]:
            actual = cache_run(inputs, chunk, .25, True, restore, True)
            error = max(float((actual[name] - reference[name]).abs().max()) for name in ["W1", "W2"])
            if restore:
                for name in ["W1", "W2"]:
                    checks.compare(f"native-dropout/{chunk}/{name}", actual[name], reference[name])
                assert torch.equal(actual["rng_after"], reference["rng_after"])
            else:
                assert error > 1e-6
            native.append({"chunk": chunk, "restore_rng": restore, "gradient_error": error})
    return native


def clipping_rows(scenario):
    if scenario == "mixed": return [[("a", 3.), ("r", 12.)], [("b", 4.), ("r", 12.)]]
    if scenario == "replicated": return [[("a", 3.), ("b", 4.)], [("a", 3.), ("b", 4.)]]
    if scenario == "zero": return [[("a", 0.)], [("b", 0.)]]
    if scenario == "overflow": return [[("a", 3.)], [("b", math.inf)]]
    return [[("a", 3.)], [("b", 4.)]]


def clipping_worker(rank, directory):
    torch.set_num_threads(1)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo" if platform.system() == "Linux" else "lo0"
    dist.init_process_group("gloo", init_method="file://" + str(Path(directory) / "clipping.store"), rank=rank, world_size=2, timeout=datetime.timedelta(seconds=40))
    results = []
    try:
        for scenario in ["shards", "mixed", "replicated", "zero", "overflow"]:
            layout = clipping_rows(scenario)
            owners = {}
            for owner, rows in enumerate(layout):
                for key, _ in rows:
                    owners.setdefault(key, owner)
            rows = layout[rank]
            for mode in ["global", "local", "duplicate", "local-skip"]:
                for threshold in [0, 1, 20]:
                    gradient = t([value * 8 for _, value in rows]) / 8
                    bad = torch.tensor(int(not bool(gradient.isfinite().all())))
                    global_bad = bad.clone()
                    with torch.profiler.record_function("finite_flag_all_reduce"):
                        dist.all_reduce(global_bad, op=dist.ReduceOp.MAX)
                    local_square = gradient.square().sum()
                    owned = t([owners[key] == rank for key, _ in rows])
                    owned_square = gradient[owned.bool()].square().sum()
                    total = owned_square.clone()
                    physical = local_square.clone()
                    dist.all_reduce(total)
                    dist.all_reduce(physical)
                    skip = bool(bad) if mode == "local-skip" else bool(global_bad)
                    norm = local_square.sqrt() if mode in ["local", "local-skip"] else (physical if mode == "duplicate" else total).sqrt()
                    if skip:
                        coefficient, output, updated = None, None, torch.ones_like(gradient)
                    else:
                        coefficient = 1. if norm == 0 else min(1., threshold / float(norm))
                        output = gradient * coefficient
                        updated = 1 - .1 * output
                    results.append({"scenario": scenario, "mode": mode, "threshold": threshold, "rank": rank, "skip": skip, "coefficient": coefficient, "output": output, "updated": updated})
        write(Path(directory) / f"clipping-rank-{rank}.json", results)
    finally:
        dist.destroy_process_group()


def check_clipping(fixture, checks, directory):
    mp.spawn(clipping_worker, args=(str(directory),), nprocs=2, join=True)
    for rank in range(2):
        actual = json.loads((directory / f"clipping-rank-{rank}.json").read_text())
        for index, (row, case) in enumerate(zip(actual, fixture["clipping"], strict=True)):
            expected = case["ranks"][rank]
            assert row["skip"] == expected["skip"]
            checks.compare(f"gloo-clip/{rank}/{index}/update", row["updated"], [entry["parameterAfter"] for entry in expected["output"]])
            if not row["skip"]:
                checks.compare(f"gloo-clip/{rank}/{index}/gradient", row["output"], [entry["gradient"] for entry in expected["output"]])


class SavedPayload:
    def __init__(self, tensor, ledger):
        self.tensor, self.ledger = tensor.detach(), ledger
        storage = tensor.untyped_storage()
        self.key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        ledger.counts[self.key] = ledger.counts.get(self.key, 0) + 1
        if ledger.counts[self.key] == 1:
            ledger.live += storage.nbytes()
        ledger.peak = max(ledger.peak, ledger.live)
        ledger.requests += 1

    def __del__(self):
        ledger = self.ledger
        ledger.counts[self.key] -= 1
        if ledger.counts[self.key] == 0:
            ledger.live -= self.key[2]


class SavedLedger:
    def __init__(self):
        self.counts, self.live, self.peak, self.requests = {}, 0, 0, 0

    def pack(self, value): return SavedPayload(value, self)

    @staticmethod
    def unpack(value): return value.tensor


def measure(name, operation, directory, logical, reference_error):
    # Timing excludes profiling/hooks and is kept separate from instrumentation.
    for _ in range(3): operation()
    timings = []
    for _ in range(12):
        start = time.perf_counter_ns(); operation(); timings.append((time.perf_counter_ns() - start) / 1e6)
    ledger = SavedLedger()
    with torch.autograd.graph.saved_tensors_hooks(ledger.pack, ledger.unpack): operation()
    gc.collect()
    assert ledger.live == 0, (name, "saved tensor references leaked", ledger.live)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as profile:
        with torch.profiler.record_function(name): operation()
    trace_path = directory / f"{name}-trace.json"
    profile.export_chrome_trace(str(trace_path))
    top = sorted(profile.key_averages(), key=lambda event: event.self_cpu_time_total, reverse=True)[:12]
    summary = [{"operation": event.key, "self_cpu_us": event.self_cpu_time_total, "calls": event.count, "self_cpu_memory_bytes": event.self_cpu_memory_usage} for event in top]
    return {"name": name, "reference_error": reference_error, "warmup_runs": 3, "repetitions": len(timings), "wall_ms": {"median": statistics.median(timings), "min": min(timings), "max": max(timings), "samples": timings}, "saved_tensor_storage": {"peak_referenced_bytes": ledger.peak, "save_requests": ledger.requests, "includes": "all unique storages referenced by saved tensors, including inputs and parameters", "excludes": "unsaved temporaries, allocator reserved memory and process RSS"}, "logical": logical, "trace_file": trace_path.name, "profiler_top_self_cpu": summary}


def benchmarks(directory, checks):
    torch.manual_seed(88)
    n, d, block = 64, 16, 16
    fixed = [torch.randn(n, d, dtype=DTYPE) for _ in range(4)]

    def attention_operation(tiled):
        Q, K, V = [value.detach().requires_grad_() for value in fixed[:3]]
        with torch.profiler.record_function("attention_forward"):
            output = TiledAttention.apply(Q, K, V, block, block, "causal") if tiled else dense_attention(Q, K, V, "causal")
        with torch.profiler.record_function("attention_backward"):
            (output * fixed[3]).sum().backward()
        return output.detach(), [value.grad for value in [Q, K, V]]

    full, tiled = attention_operation(False), attention_operation(True)
    checks.compare("benchmark-attention/output", tiled[0], full[0])
    for index in range(3): checks.compare(f"benchmark-attention/gradient-{index}", tiled[1][index], full[1][index])
    rows = []
    for is_tiled in [False, True]:
        rows.append(measure("attention-tiled" if is_tiled else "attention-dense", lambda: attention_operation(is_tiled), directory,
            {"shape": [n, d], "block": block, "probability_or_lse_elements": n if is_tiled else n * n, "tile_probability_capacity": block * block if is_tiled else n * n, "valid_qk_forward": n * (n + 1) // 2, "valid_qk_recomputed": n * (n + 1) // 2 if is_tiled else 0, "shared_qkv_output_elements": 4 * n * d},
            max(float((tiled[1][i] - full[1][i]).abs().max()) for i in range(3)) if is_tiled else 0.))
    count, width, hidden, out, chunk = 32, 16, 64, 8, 8
    inputs = {"query": torch.randn(count, width, dtype=DTYPE), "document": torch.randn(count, width, dtype=DTYPE), "W1": torch.randn(width, hidden, dtype=DTYPE) * .15, "W2": torch.randn(hidden, out, dtype=DTYPE) * .15}
    full = cache_run(inputs, chunk, .25, False, native_rng=True)
    cached = cache_run(inputs, chunk, .25, True, native_rng=True)
    for name in ["loss", "W1", "W2"]: checks.compare("benchmark-cache/" + name, cached[name], full[name])
    for use_cache in [False, True]:
        rows.append(measure("contrastive-cached" if use_cache else "contrastive-retained", lambda: cache_run(inputs, chunk, .25, use_cache, native_rng=True, verify_replay=False), directory,
            {"pairs": count, "input_width": width, "hidden_width": hidden, "representation_width": out, "chunk": chunk, "hidden_triplet_elements": (chunk if use_cache else 2 * count) * hidden * 3, "representation_elements": 2 * count * out, "representation_gradient_elements": 2 * count * out if use_cache else 0, "score_matrix_elements": count * count, "encoder_forward_rows": (4 if use_cache else 2) * count},
            max(float((cached[name] - full[name]).abs().max()) for name in ["W1", "W2"]) if use_cache else 0.))

    gradient = torch.arange(1, 4097, dtype=DTYPE) / 4096

    def clipping_operation():
        shards = gradient.chunk(2)
        norm = torch.stack([shard.square().sum() for shard in shards]).sum().sqrt()
        coefficient = min(1., 1. / float(norm))
        return torch.cat([shard * coefficient for shard in shards])

    checks.compare("benchmark-clipping/norm", clipping_operation().norm(), 1.)
    expected_clip = gradient / gradient.norm()
    rows.append(measure("clipping-local-accounting", clipping_operation, directory, {"unique_gradient_elements": 4096, "logical_shards": 2, "scalar_reduction_inputs": 2, "communication_in_timing": False, "note": "local arithmetic timing; real Gloo correctness is checked separately"}, float((clipping_operation() - expected_clip).abs().max())))
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out-dir", type=Path, required=True); args = parser.parse_args()
    if torch.__version__.split("+")[0] != "2.9.0": raise RuntimeError("Install the pinned scripts/framework-requirements.txt environment.")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    fixture = json.loads((args.out_dir / "reference.json").read_text()); checks = Checks()
    check_attention(fixture, checks); native = check_cache(fixture, checks)
    print("Attention and cache references passed; checking actual Gloo clipping.", flush=True)
    check_clipping(fixture, checks, args.out_dir)
    measurements = benchmarks(args.out_dir, checks)
    report = {"schema": "memory-lab-v1", "status": "passed", "created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu", "dtype": "float64", "platform": platform.platform(), "threads": 1}, "tolerance": {"atol": 1e-9, "rtol": 1e-8}, "cases": {"attention": len(fixture["attention"]), "cache": len(fixture["cache"]), "clipping": len(fixture["clipping"]), "gloo_ranks": 2}, "comparisons": len(checks.rows), "max_error": max(row["max_error"] for row in checks.rows), "native_dropout": native, "measurements": measurements, "checks": checks.rows,
        "measurement_scope": "CPU Python implementations. No GPU speedup claim. Wall timing excludes profiler and hooks; saved tensor reference bytes are not allocator peak memory. Clipping timing excludes Gloo; Gloo is verified separately."}
    write(args.out_dir / "report.json", report)
    write(args.out_dir / "environment.json", {item.metadata["Name"]: item.version for item in distributions()})
    print(f'Passed {len(checks.rows)} comparisons; max error {report["max_error"]:.3e}.', flush=True)
    for row in measurements: print(row["name"], f'{row["wall_ms"]["median"]:.3f} ms median', row["saved_tensor_storage"]["peak_referenced_bytes"], "saved-reference peak bytes")


if __name__ == "__main__": main()
