"""Independent CPU float64 references, real DDP collectives and process resume.

The JSON inputs carry JS reference results. Every actual derivative is computed
by PyTorch autograd. No JS derivative is used to construct a PyTorch result.
"""
import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.multiprocessing as mp
import torch.nn.functional as F

DTYPE = torch.float64
ATOL, RTOL = 1e-9, 1e-8
SAMPLES = [([[.5, -.2], [.1, .7]], 1), ([[-.3, .6], [.8, .2]], 2), ([[.2, .9], [-.4, .3]], 0), ([[.7, .1], [.3, -.6]], 1)]


def tensor(value, grad=False):
    return torch.tensor(value, dtype=DTYPE, requires_grad=grad)


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def write_json(path, data):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(plain(data), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


class Checks:
    def __init__(self):
        self.rows = []

    def compare(self, name, actual, expected, exact=False):
        a, e = torch.as_tensor(actual, dtype=DTYPE).detach(), torch.as_tensor(expected, dtype=DTYPE).detach()
        torch.testing.assert_close(a, e, atol=0 if exact else ATOL, rtol=0 if exact else RTOL, msg=name)
        error = float((a - e).abs().max()) if a.numel() else 0
        self.rows.append({"name": name, "max_absolute_error": error, "exact": exact})

    def gradients(self, prefix, parameters, expected):
        for key, parameter in parameters.items():
            assert parameter.grad is not None, (prefix, key, "missing gradient")
            self.compare(f"{prefix}/{key}", parameter.grad, expected[key])


def block_forward(parameters, X, target):
    def norm(value, scale):
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-5) * scale
    u1 = norm(X, parameters["norm1"])
    Q, K, V = (u1 @ parameters[name] for name in ["q", "k", "v"])
    scores = Q @ K.T / math.sqrt(2)
    mask = torch.ones_like(scores, dtype=torch.bool).tril()
    P = scores.masked_fill(~mask, -torch.inf).softmax(dim=-1)
    h1 = X + (P @ V) @ parameters["o"]
    u2 = norm(h1, parameters["norm2"])
    h2 = h1 + (F.silu(u2 @ parameters["gate"]) * (u2 @ parameters["up"])) @ parameters["down"]
    logits = h2 @ parameters["out"]
    return logits, F.cross_entropy(logits[1:2], torch.tensor([target]))


def check_block(fixtures, checks):
    for case in fixtures["block"]:
        parameters = {key: tensor(value, True) for key, value in case["parameters"].items()}
        logits, loss = block_forward(parameters, parameters["X"], case["target"])
        loss.backward()
        prefix = f'block/target-{case["target"]}'
        checks.compare(prefix + "/logits", logits, case["expected"]["forward"]["logits"])
        checks.compare(prefix + "/loss", loss, case["expected"]["forward"]["loss"])
        checks.gradients(prefix, parameters, case["expected"]["gradients"])


def render_tokens(records, options):
    result = []
    for record in records:
        last = max(i for i, message in enumerate(record["messages"]) if message["role"] == "assistant")
        tokens = [{"id": 1, "supervised": False}]
        for index, message in enumerate(record["messages"]):
            is_assistant = message["role"] == "assistant"
            tokens.append({"id": 3 if is_assistant else 2, "supervised": False})
            supervised = is_assistant and (options["supervise"] == "all" or index == last)
            tokens.extend({"id": value, "supervised": supervised} for value in message["ids"])
            if is_assistant:
                tokens.append({"id": 4, "supervised": supervised})
        result.append([dict(token, record=record["id"], position=i, valid=True, key=f'{record["id"]}:{i}') for i, token in enumerate(tokens[:options["limit"]])])
    return result


def packing_forward(parameters, records, mode):
    shift = 2 if mode == "double-shift" else 1
    if mode == "padded":
        width = max(map(len, records))
        # Real batch dimension, independently masked and right padded.
        groups = [record + [dict(id=0, valid=False, position=i, record=record[0]["record"], supervised=False, key="pad") for i in range(len(record), width)] for record in records]
    elif mode == "separate":
        groups = records
    else:
        groups = [[token for record in records for token in record]]
    batches = [groups] if mode == "padded" else [[group] for group in groups]
    all_outputs, labels_out, losses = [], [], []
    for batch in batches:
        ids = torch.tensor([[row["id"] for row in group] for group in batch])
        valid = torch.tensor([[row["valid"] for row in group] for group in batch])
        positions = tensor([[row["position"] for row in group] for group in batch])
        X = parameters["embedding"][ids] + positions.unsqueeze(-1) * tensor([.04, -.03])
        X = X * valid.unsqueeze(-1)
        Q, K, V = (X @ parameters[name] for name in ["q", "k", "v"])
        scores = Q @ K.transpose(-1, -2) / math.sqrt(2)
        allowed = torch.tensor([[[row["valid"] and key["valid"] and j <= i and (mode == "leaky" or row["record"] == key["record"]) for j, key in enumerate(group)] for i, row in enumerate(group)] for group in batch])
        # Invalid query rows avoid all -inf before being explicitly zeroed.
        safe_mask = allowed | (~valid).unsqueeze(-1)
        P = scores.masked_fill(~safe_mask, -torch.inf).softmax(dim=-1) * valid.unsqueeze(-1)
        logits = (X + P @ V) @ parameters["out"]
        labels = []
        for group in batch:
            current = []
            for i, row in enumerate(group):
                target = group[i + shift] if i + shift < len(group) else None
                current.append(target["id"] if row["valid"] and target and target["valid"] and target["record"] == row["record"] and target["supervised"] else -100)
            labels.append(current)
        label_tensor = torch.tensor(labels)
        losses.append(F.cross_entropy(logits.reshape(-1, 12), label_tensor.reshape(-1), reduction="sum", ignore_index=-100))
        for b, group in enumerate(batch):
            for i, row in enumerate(group):
                if row["valid"]:
                    all_outputs.append(logits[b, i])
                    labels_out.append(None if labels[b][i] == -100 else labels[b][i])
    count = sum(token["supervised"] for record in records for token in record[shift:])
    loss = sum(losses) / count if count else None
    return torch.stack(all_outputs), labels_out, count, loss


def check_packing(fixtures, checks):
    for index, case in enumerate(fixtures["packing"]):
        parameters = {key: tensor(value, True) for key, value in fixtures["packingParameters"].items()}
        records = render_tokens(case["records"], case["options"])
        logits, labels, count, loss = packing_forward(parameters, records, case["options"]["mode"])
        expected = case["expected"]
        prefix = f'packing/{index}/{case["options"]["mode"]}'
        assert labels == [row["label"] for row in expected["outputs"]], prefix + "/labels"
        assert count == expected["denominator"], prefix + "/denominator"
        checks.compare(prefix + "/logits", logits, [row["logits"] for row in expected["outputs"]])
        if loss is None:
            assert expected["loss"] is None and expected["gradients"] is None
        else:
            loss.backward()
            checks.compare(prefix + "/loss", loss, expected["loss"])
            checks.gradients(prefix, parameters, expected["gradients"])
            optimizer = torch.optim.SGD(parameters.values(), lr=.01)
            optimizer.step()
            for key, parameter in parameters.items():
                expected_updated = tensor(fixtures["packingParameters"][key]) - .01 * tensor(expected["gradients"][key])
                checks.compare(prefix + "/updated-" + key, parameter, expected_updated)
        if loss is None:
            # No optimizer step, even when an optimizer could apply decay.
            for key, parameter in parameters.items():
                checks.compare(prefix + "/skipped-" + key, parameter, fixtures["packingParameters"][key], exact=True)


def quantize(value):
    maximum = value.abs().max()
    scale = maximum / 7 if maximum > 0 else tensor(1.)
    return value.sign() * torch.floor(value.abs() / scale + .5).clamp(0, 7) * scale


def check_lora(fixtures, checks):
    for index, case in enumerate(fixtures["lora"]):
        options, expected = case["options"], case["expected"]
        W = tensor([[1, .2], [-.3, .7]])
        W0 = quantize(W) if options["precision"] == "int4" else W
        A = torch.nn.Parameter(tensor([[.5, -.25]] if options["initialization"] == "zero-b" else [[0, 0]]))
        B = torch.nn.Parameter(tensor([[0], [0]]))
        X, target = tensor([[2], [-1]], True), tensor([[0], [1]])
        optimizer = torch.optim.SGD([A, B], lr=.1)

        def forward():
            base = W0 @ X
            if options["graph"] == "detach-base":
                base = base.detach()
            output = base + B @ (A @ X)
            if options["graph"] == "detach-all":
                output = output.detach()
            return output, .5 * (output - target).square().sum()

        for _ in range(options["steps"]):
            optimizer.zero_grad(set_to_none=True)
            X.grad = None
            _, loss = forward()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        X.grad = None
        output, loss = forward()
        if loss.requires_grad:
            loss.backward()
        prefix = f'lora/{index}/{options["graph"]}'
        checks.compare(prefix + "/output", output, expected["output"])
        checks.compare(prefix + "/loss", loss, expected["loss"])
        checks.compare(prefix + "/A", A, expected["A"])
        checks.compare(prefix + "/B", B, expected["B"])
        assert W.grad is None and W0.grad is None
        if expected["gradients"] is None:
            assert A.grad is None and B.grad is None and X.grad is None
        else:
            checks.gradients(prefix, {"A": A, "B": B, "X": X}, expected["gradients"])
        merged = W0 + B @ A
        checks.compare(prefix + "/merged", merged, expected["merged"])
        checks.compare(prefix + "/merged-output", merged @ X, expected["mergedOutput"])
        checks.compare(prefix + "/requantized-output", quantize(merged) @ X, expected["requantizedOutput"])


def check_contrastive(fixtures, checks):
    for case in fixtures["contrastive"]:
        q, d = tensor(case["query"], True), tensor(case["document"], True)
        losses = []
        for i in range(2):
            if case["mode"] == "micro":
                logits = (q[i] * d[i] / case["temperature"]).reshape(1, 1)
                target = 0
            else:
                candidates = torch.stack([d[j].detach() if case["mode"] == "detached" and i != j else d[j] for j in range(2)])
                logits, target = (q[i] * candidates / case["temperature"]).reshape(1, 2), i
            losses.append(F.cross_entropy(logits, torch.tensor([target])))
        loss = sum(losses) / 2
        loss.backward()
        prefix = f'contrastive/{case["mode"]}/{case["temperature"]}'
        checks.compare(prefix + "/loss", loss, case["loss"])
        checks.compare(prefix + "/query-gradient", q.grad, case["queryGradients"])
        checks.compare(prefix + "/document-gradient", d.grad, case["documentGradients"])


class PairEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Parameter(tensor(1.))
        self.wd = torch.nn.Parameter(tensor(1.))

    def forward(self, rank):
        return self.wq * tensor([1., 2.][rank]), self.wd * tensor([1., -1.][rank])


def distributed_worker(rank, mode, rendezvous, output_directory):
    torch.set_num_threads(1)
    # Only loopback is used. The file store is created in this run's directory.
    os.environ["GLOO_SOCKET_IFNAME"] = "lo" if platform.system() == "Linux" else "lo0"
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=35))
    try:
        model = torch.nn.parallel.DistributedDataParallel(PairEncoder())
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        query, document = model(rank)
        query.retain_grad()
        document.retain_grad()
        if mode == "global":
            candidates = torch.stack(dist_nn.all_gather(document))
            label = rank
        elif mode == "detached":
            gathered = [torch.zeros_like(document) for _ in range(2)]
            dist.all_gather(gathered, document.detach())
            gathered[rank] = document
            candidates, label = torch.stack(gathered), rank
        else:
            candidates, label = document.reshape(1), 0
        loss = F.cross_entropy((query * candidates).reshape(1, -1), torch.tensor([label]))
        loss.backward()
        gradients = {"query": float(model.module.wq.grad), "document": float(model.module.wd.grad)}
        optimizer.step()
        row = {"rank": rank, "mode": mode, "local_loss": float(loss.detach()), "query_representation_grad": float(query.grad), "document_representation_grad": float(document.grad), "parameter_gradients": gradients, "parameters_after": {"query": float(model.module.wq.detach()), "document": float(model.module.wd.detach())}}
        write_json(Path(output_directory) / f"gloo-{mode}-rank-{rank}.json", row)
    finally:
        dist.destroy_process_group()


def check_distributed(fixtures, checks, directory):
    results = []
    for mode in ["global", "detached", "micro"]:
        rendezvous = str((directory / f"gloo-{mode}.store").resolve())
        mp.spawn(distributed_worker, args=(mode, rendezvous, str(directory)), nprocs=2, join=True)
        reference = next(case for case in fixtures["contrastive"] if case["mode"] == mode and case["temperature"] == 1)
        for rank in range(2):
            row = json.loads((directory / f"gloo-{mode}-rank-{rank}.json").read_text())
            prefix = f"gloo/{mode}/rank-{rank}"
            checks.compare(prefix + "/local-loss", row["local_loss"], reference["localLosses"][rank])
            checks.compare(prefix + "/query-representation", row["query_representation_grad"], 2 * reference["queryGradients"][rank])
            checks.compare(prefix + "/document-representation", row["document_representation_grad"], 2 * reference["documentGradients"][rank])
            for key in ["query", "document"]:
                checks.compare(prefix + "/parameter-" + key, row["parameter_gradients"][key], reference["parameterGradients"][key])
                checks.compare(prefix + "/updated-" + key, row["parameters_after"][key], 1 - .1 * reference["parameterGradients"][key])
            results.append(row)
    return results


def make_training(initial):
    parameters = torch.nn.ParameterDict({key: torch.nn.Parameter(tensor(value)) for key, value in initial["parameters"].items()})
    optimizer = torch.optim.AdamW(parameters.parameters(), lr=.015, betas=(.9, .99), eps=1e-8, weight_decay=.01, foreach=False)
    state = {key: initial[key] for key in ["rng", "cursor", "scheduler", "step"]}
    return parameters, optimizer, state


def advance_training(parameters, optimizer, state):
    raw, target = SAMPLES[state["cursor"] % len(SAMPLES)]
    inputs = []
    for row in raw:
        current = []
        for value in row:
            state["rng"] = (1664525 * state["rng"] + 1013904223) & 0xffffffff
            current.append(value + .05 * (2 * state["rng"] / 4294967296 - 1))
        inputs.append(current)
    lr = .015 / (1 + .15 * state["scheduler"])
    optimizer.param_groups[0]["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    _, loss = block_forward(parameters, tensor(inputs), target)
    loss.backward()
    gradients = {key: value.grad.detach().clone() for key, value in parameters.items()}
    optimizer.step()
    for key in ["cursor", "scheduler", "step"]:
        state[key] += 1
    return {"step": state["step"], "loss": float(loss.detach()), "input": inputs, "lr": lr, "parameters": plain(dict(parameters.items())), "gradients": plain(gradients), "state": copy.deepcopy(state)}


def check_training(fixtures, checks, directory):
    initial = fixtures["training"]["initial"]
    parameters, optimizer, state = make_training(initial)
    baseline = []
    for index in range(8):
        row = advance_training(parameters, optimizer, state)
        baseline.append(row)
        expected = fixtures["training"]["trajectory"][index]
        checks.compare(f"training/{index}/input", row["input"], expected["input"], exact=True)
        checks.compare(f"training/{index}/loss", row["loss"], expected["loss"])
        for key in parameters:
            checks.compare(f"training/{index}/gradient-{key}", row["gradients"][key], expected["gradients"][key])
            checks.compare(f"training/{index}/parameter-{key}", row["parameters"][key], expected["parameters"][key])
        if index == 2:
            torch.save({"parameters": parameters.state_dict(), "optimizer": optimizer.state_dict(), "state": copy.deepcopy(state)}, directory / "checkpoint.pt")
    write_json(directory / "torch-uninterrupted.json", baseline)
    # Actual new interpreter, loading only this run's locally created checkpoint.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--fixtures", str(directory / "reference.json"), "--out-dir", str(directory), "--resume"], check=True, timeout=45)
    resumed = json.loads((directory / "torch-resumed.json").read_text())
    for index, row in enumerate(resumed, 3):
        reference = baseline[index]
        checks.compare(f"resume/{index}/loss", row["loss"], reference["loss"], exact=True)
        for key in parameters:
            checks.compare(f"resume/{index}/{key}", row["parameters"][key], reference["parameters"][key], exact=True)
        assert row["state"] == reference["state"]
    checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu")
    # One omitted-state counterexample: identical weights, fresh Adam history.
    omitted, omitted_optimizer, omitted_state = make_training(initial)
    omitted.load_state_dict(checkpoint["parameters"])
    omitted_state.update(checkpoint["state"])
    omitted_row = advance_training(omitted, omitted_optimizer, omitted_state)
    difference = max(float((tensor(omitted_row["parameters"][key]) - tensor(baseline[3]["parameters"][key])).abs().max()) for key in omitted)
    assert difference > 1e-5
    write_json(directory / "torch-omitted-adam.json", {"first_resumed_step": omitted_row, "max_parameter_difference": difference})
    return {"steps": 8, "checkpoint_after": 3, "complete_resume_exact": True, "omitted_adam_first_difference": difference}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if torch.__version__.split("+")[0] != "2.9.0":
        raise RuntimeError("This reference is pinned to torch==2.9.0; install scripts/framework-requirements.txt.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fixtures = json.loads(args.fixtures.read_text())
    assert fixtures["schema"] == 1
    if args.resume:
        parameters, optimizer, state = make_training(fixtures["training"]["initial"])
        saved = torch.load(args.out_dir / "checkpoint.pt", weights_only=True, map_location="cpu")
        parameters.load_state_dict(saved["parameters"])
        optimizer.load_state_dict(saved["optimizer"])
        state.update(saved["state"])
        write_json(args.out_dir / "torch-resumed.json", [advance_training(parameters, optimizer, state) for _ in range(8 - state["step"])])
        return
    checks = Checks()
    check_block(fixtures, checks)
    check_packing(fixtures, checks)
    check_lora(fixtures, checks)
    check_contrastive(fixtures, checks)
    print("Autograd references passed; running optimizer/resume and two-process Gloo checks.", flush=True)
    training = check_training(fixtures, checks, args.out_dir)
    distributed = check_distributed(fixtures, checks, args.out_dir)
    report = {"status": "passed", "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(), "device": "cpu", "dtype": "float64", "threads": torch.get_num_threads()}, "tolerance": {"absolute": ATOL, "relative": RTOL}, "case_counts": {"block": 3, "packing": len(fixtures["packing"]), "lora": len(fixtures["lora"]), "contrastive": len(fixtures["contrastive"]), "gloo_modes": 3, "gloo_processes_per_mode": 2}, "comparisons": len(checks.rows), "max_absolute_error": max(row["max_absolute_error"] for row in checks.rows), "training": training, "distributed": distributed, "checks": checks.rows}
    write_json(args.out_dir / "report.json", report)
    from importlib.metadata import distributions
    write_json(args.out_dir / "environment.json", {package.metadata["Name"]: package.version for package in distributions()})
    print(f'Passed {len(checks.rows)} numerical comparisons; maximum absolute error {report["max_absolute_error"]:.3e}.', flush=True)


if __name__ == "__main__":
    main()
