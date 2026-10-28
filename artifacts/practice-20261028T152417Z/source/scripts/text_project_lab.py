"""Actual bounded natural-language training, replay and full-step measurements."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.nn import functional as F
import learning_core as symbolic
import text_training_core as core
from chunked_ce_reference import chunked_linear_cross_entropy, SavedLedger

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 29, 47)
TRAINING = {"updates": 800, "batch_size": 8, "context": 96, "lr": .002, "weight_decay": .01, "clip_norm": 1.,
            "eval_every": 100, "dropout": 0, "selection": "final fixed budget; no early stopping, test selection or after-test tuning"}


def save(path, value):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")


def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01, foreach=False)


def step(model, optimizer, records, stream):
    batch = core.collate([records[i] for i in stream.next_indices(8)])
    optimizer.zero_grad(set_to_none=True); value = core.loss(model, batch); value.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(value) or not torch.isfinite(norm): raise FloatingPointError("nonfinite text update")
    optimizer.step()
    return {"loss": value.item(), "tokens": batch["tokens"], "records": batch["record_ids"], "gradient_norm": float(norm)}


def checkpoint(model, optimizer, stream, seed, update):
    return {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
            "stream": stream.state_dict(), "torch_rng": torch.get_rng_state(), "config": model.config, "seed": seed, "update": update}


def resume_check(directory):
    directory = Path(directory); symbolic.configure_cpu(1)
    expected = torch.load(directory / "resume-expected.pt", map_location="cpu", weights_only=True)
    saved = torch.load(directory / "seed-11-midpoint.pt", map_location="cpu", weights_only=True)
    records = json.loads((directory / "windows.json").read_text())["train"]
    model = core.make_model(saved["seed"], saved["config"]); model.load_state_dict(saved["model"])
    optimizer = new_optimizer(model); optimizer.load_state_dict(saved["optimizer"])
    stream = core.WindowStream(len(records), 0); stream.load_state_dict(saved["stream"])
    torch.set_rng_state(saved["torch_rng"])
    trace = [step(model, optimizer, records, stream) for _ in range(16)]
    actual = checkpoint(model, optimizer, stream, 11, 416)
    errors = {key: core.state_error(actual[key], expected["checkpoint"][key]) for key in ("model", "optimizer", "stream", "torch_rng")}
    errors["trace"] = core.state_error(trace, expected["trace"])
    result = {"fresh_python_process": True, "from_update": 400, "to_update": 416, "errors": errors, "passed": all(x == 0 for x in errors.values()),
              "scope": "CPU single process, dropout=0; sampler has its own Python Random state; no asynchronous data workers or GPU RNG"}
    save(directory / "resume-check.json", result)
    if not result["passed"]: raise AssertionError(result)


def run_text(directory, corpus_path):
    directory = Path(directory).resolve(); corpus_path = Path(corpus_path).resolve(); directory.mkdir(parents=True, exist_ok=False)
    symbolic.configure_cpu(1)
    corpus = json.loads(Path(corpus_path).read_text())
    save(directory / "frozen-config.json", {"seeds": SEEDS, "training": TRAINING, "bpe_merges": 128, "prompts": ["模型训练", "梯度", "数据"],
                                            "corpus_sha256": file_hash(corpus_path), "no_downloads": True})
    tokenizer = core.train_tokenizer([d for d in corpus["documents"] if d["split"] == "train"])
    windows = core.make_windows(corpus, tokenizer)
    assert all(windows.values())
    assert all(core.decode(core.encode(p["text"], tokenizer), tokenizer) == p["text"] for d in corpus["documents"] for p in d["paragraphs"])
    save(directory / "tokenizer.json", tokenizer); save(directory / "windows.json", windows)
    config = {"vocab_size": tokenizer["vocab_size"], "width": 96, "depth": 2, "heads": 4, "context": 96}
    # Every split is evaluated in full; only train/validation enter the learning curve.
    runs = []
    for seed in SEEDS:
        symbolic.seed_all(seed); model = core.make_model(seed, config); optimizer = new_optimizer(model)
        stream = core.WindowStream(len(windows["train"]), seed + 1000)
        initial = checkpoint(model, optimizer, stream, seed, 0)
        torch.save(initial, directory / f"seed-{seed}-initial.pt")
        curve = [{"update": 0, "supervised_tokens": 0, "train": core.evaluate(model, windows["train"]), "validation": core.evaluate(model, windows["validation"])}]
        trace, tokens = [], 0
        started = time.perf_counter()
        for update in range(1, 801):
            row = step(model, optimizer, windows["train"], stream); tokens += row["tokens"]; trace.append({"update": update, **row})
            if update == 400: torch.save(checkpoint(model, optimizer, stream, seed, update), directory / f"seed-{seed}-midpoint.pt")
            if seed == 11 and update == 416:
                torch.save({"checkpoint": checkpoint(model, optimizer, stream, seed, update), "trace": [{k: v for k, v in r.items() if k != "update"} for r in trace[-16:]]}, directory / "resume-expected.pt")
            if update % 100 == 0:
                curve.append({"update": update, "supervised_tokens": tokens, "train": core.evaluate(model, windows["train"]), "validation": core.evaluate(model, windows["validation"])})
                print(f"text seed={seed} step={update} train={curve[-1]['train']['ce_nats_per_token']:.4f} val={curve[-1]['validation']['ce_nats_per_token']:.4f}", flush=True)
        seconds = time.perf_counter() - started
        final = checkpoint(model, optimizer, stream, seed, 800)
        path = directory / f"seed-{seed}-final.pt"; torch.save(final, path)
        initial_model = core.make_model(seed, config); initial_model.load_state_dict(initial["model"])
        evaluations = {phase: {split: core.evaluate(m, records) for split, records in windows.items()} for phase, m in (("initial", initial_model), ("learned", model))}
        generations = {phase: [core.generate(m, prompt, tokenizer) for prompt in ("模型训练", "梯度", "数据")] for phase, m in (("initial", initial_model), ("learned", model))}
        save(directory / f"seed-{seed}-trace.json", trace)
        runs.append({"seed": seed, "curve": curve, "evaluation": evaluations, "generations": generations,
                     "updates": 800, "supervised_tokens": tokens, "elapsed_seconds_including_eval": seconds,
                     "checkpoint": str(path.relative_to(ROOT)), "checkpoint_sha256": file_hash(path)})
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--resume-check", str(directory)], check=True, cwd=ROOT)
    documents = {split: [d["id"] for d in corpus["documents"] if d["split"] == split] for split in windows}
    example = next(d["paragraphs"][0]["text"] for d in corpus["documents"] if d["split"] == "train" and d["paragraphs"])[:45]
    report = {"schema": "natural-text-project-v1", "corpus": {"path": str(Path(corpus_path).relative_to(ROOT)), "sha256": file_hash(corpus_path),
              "scope": corpus["scope"], "documents": documents, "paragraphs": {s: sum(len(d["paragraphs"]) for d in corpus["documents"] if d["split"] == s) for s in windows},
              "windows": {s: len(r) for s, r in windows.items()}, "exact_paragraph_overlap": 0, "near_duplicate_audit": "not established; narrow-domain shared vocabulary and semantic overlap remain"},
              "tokenizer": {"vocab_size": tokenizer["vocab_size"], "merges": len(tokenizer["merges"]), "training_documents": tokenizer["training_documents"],
              "roundtrip_all_paragraphs": True, "example": {"text": example, "tokens": core.encode(example, tokenizer), "decoded": core.decode(core.encode(example, tokenizer), tokenizer)},
              "path": str((directory / "tokenizer.json").relative_to(ROOT))}, "model": config, "parameter_count": sum(p.numel() for p in model.parameters()),
              "training": TRAINING, "runs": runs, "resume": json.loads((directory / "resume-check.json").read_text()),
              "limits": ["real natural-language paragraphs from this Wiki; NOT TinyStories, a broad web corpus or instruction tuning",
                         "document IDs and exact paragraph hashes are disjoint; semantic and near-duplicate independence is not established",
                         "no trained tokenizer outside the training split; loss units only compare with this saved tokenizer",
                         "teacher-forced CE/accuracy and 48-token greedy samples do not demonstrate factual reliability, instruction following or general Chinese fluency",
                         "invalid UTF-8 replacements and emitted non-EOS special tokens are retained and reported"]}
    save(directory / "report.json", report); return report


def flatten(parameters): return torch.cat([p.detach().reshape(-1).double() for p in parameters])


def run_reproduction(directory, text_directory):
    directory = Path(directory); directory.mkdir()
    text_directory = Path(text_directory)
    saved = torch.load(text_directory / "seed-11-initial.pt", weights_only=True)
    records = json.loads((text_directory / "windows.json").read_text())["train"]
    stream = core.WindowStream(len(records), 1111); indices = [stream.next_indices(2) for _ in range(16)]
    save(directory / "frozen-config.json", {"steps": 16, "seed": 11, "batch": 2, "input_limit": 48, "same_initial_weights": True,
                                           "cases": ["repeat-float32", "cast-float64", "changed-data-order"], "historical_cause": "unresolved; old complete checkpoint/trace artifacts unavailable"})
    traces = {}
    for case in ("baseline", "repeat-float32", "cast-float64", "changed-data-order"):
        model = core.make_model(11, saved["config"]); model.load_state_dict(saved["model"])
        if case == "cast-float64": model.double()
        optimizer = new_optimizer(model); trace = []
        for i, selected in enumerate(indices):
            if case == "changed-data-order": selected = indices[(i + 1) % len(indices)]
            subset = [{**records[j], "input": records[j]["input"][:48], "target": records[j]["target"][:48]} for j in selected]
            batch = core.collate(subset); optimizer.zero_grad(set_to_none=True)
            embedding = model.embedding(batch["input_ids"]) + model.positions[:batch["input_ids"].shape[1]]
            hidden = model.hidden(batch["input_ids"]); logits = model.head(hidden)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1), ignore_index=-100)
            loss.backward(); gradients = flatten([p.grad for p in model.parameters()]); optimizer.step()
            trace.append({"input_ids": batch["input_ids"].clone(), "embedding": embedding.detach().clone(), "hidden": hidden.detach().clone(),
                          "logits": logits.detach().clone(), "loss": loss.detach().clone(), "gradient": gradients, "parameters": flatten(model.parameters())})
        traces[case] = trace
    baseline = traces["baseline"]; cases = []
    for name, trace in traces.items():
        if name == "baseline": continue
        rows, first = [], None
        for i, (left, right) in enumerate(zip(baseline, trace)):
            errors = {key: core.state_error(left[key], right[key]) for key in left}
            if first is None:
                for key, error in errors.items():
                    if error != 0: first = {"update": i + 1, "stage": key, "max_absolute_difference": error}; break
            rows.append({"update": i + 1, "errors": errors})
        cases.append({"case": name, "first_difference": first, "rows": rows, "same_inputs": name != "changed-data-order", "same_initial_values": True})
    assert cases[0]["first_difference"] is None
    torch.save({"first_update": {name: trace[0] for name, trace in traces.items()}, "indices": indices, "initial": saved}, directory / "comparison-tensors.pt")
    report = {"schema": "reproduction-differences-v1", "cases": cases, "matched_repeat_exact": True,
              "historical_difference": {"status": "unresolved", "reason": "old page snapshot and new run differ, but old complete checkpoints/traces are unavailable; these controlled probes do not establish that historical cause"},
              "scope": "same CPU and installed PyTorch, 16 updates; float64 cast and changed data order are intentional new interventions, not different OS/GPU reproductions",
              "levels": ["byte identity via hashes", "same-environment numerical replay", "cross-environment numerical agreement under declared tolerance", "multi-seed capability statistics"]}
    save(directory / "report.json", report); return report


def profile_step(model, optimizer, records, method):
    times = {}; start = time.perf_counter_ns(); optimizer.zero_grad(set_to_none=True); batch = core.collate(records)
    now = time.perf_counter_ns(); times["prepare_and_zero_grad"] = (now - start) / 1e6
    hidden = model.hidden(batch["input_ids"]); after = time.perf_counter_ns(); times["model_forward"] = (after - now) / 1e6
    if method == "dense": value = F.cross_entropy(model.head(hidden).reshape(-1, model.config["vocab_size"]), batch["labels"].reshape(-1), ignore_index=-100)
    else: value = chunked_linear_cross_entropy(hidden.reshape(-1, hidden.shape[-1]), model.head.weight, batch["labels"].reshape(-1), token_block=64, vocab_block=128)
    now = time.perf_counter_ns(); times["output_and_loss"] = (now - after) / 1e6
    value.backward(); after = time.perf_counter_ns(); times["backward"] = (after - now) / 1e6
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); now = time.perf_counter_ns(); times["clip_and_update"] = (now - after) / 1e6
    times["total"] = (now - start) / 1e6
    return times, value.item(), batch["tokens"]


def summarize_times(values): return {"median": statistics.median(values), "min": min(values), "max": max(values), "samples": values}


def run_profile(directory, text_directory):
    directory = Path(directory); directory.mkdir(); text_directory = Path(text_directory)
    saved = torch.load(text_directory / "seed-11-final.pt", weights_only=True)
    windows = json.loads((text_directory / "windows.json").read_text())
    stream = core.WindowStream(len(windows["train"]), 9911)
    batches = [[windows["train"][j] for j in stream.next_indices(4)] for _ in range(15)]
    frozen = {"device": "cpu", "dtype": "float64", "warmups": 3, "samples": 12, "batch_size": 4, "context": 96,
              "methods": ["dense", "chunked-exact"], "token_block": 64, "vocab_block": 128,
              "timing_order": "alternate method order each iteration; distinct models/optimizers with identical initial values and batch sequence",
              "timing_scope": "pretokenized records collate + zero grad + hidden forward + output/loss + backward + clipping + AdamW; no disk input, tokenization, profiler or saved-tensor hooks in timed runs",
              "parameter_tolerance": 1e-6, "loss_tolerance": 1e-8}
    save(directory / "frozen-config.json", frozen)
    models, optimizers, measurements = {}, {}, {key: [] for key in frozen["methods"]}
    for key in frozen["methods"]:
        models[key] = core.make_model(11, saved["config"]).double(); models[key].load_state_dict(saved["model"])
        optimizers[key] = new_optimizer(models[key])
    losses = {key: [] for key in models}
    for i, batch in enumerate(batches):
        for key in (frozen["methods"] if i % 2 == 0 else list(reversed(frozen["methods"]))):
            times, loss, count = profile_step(models[key], optimizers[key], batch, key)
            losses[key].append(loss)
            if i >= 3: measurements[key].append({"iteration": i - 2, "times_ms": times, "loss": loss, "supervised_tokens": count})
    parameter_error = core.state_error(models["dense"].state_dict(), models["chunked-exact"].state_dict())
    loss_error = max(abs(a - b) for a, b in zip(*losses.values()))
    assert parameter_error <= frozen["parameter_tolerance"] and loss_error <= frozen["loss_tolerance"]
    rows = []
    for key, model in models.items():
        batch = core.collate(batches[-1]); ledger = SavedLedger(); model.zero_grad(set_to_none=True)
        with torch.autograd.graph.saved_tensors_hooks(ledger.pack, ledger.unpack):
            hidden = model.hidden(batch["input_ids"])
            if key == "dense": loss = F.cross_entropy(model.head(hidden).reshape(-1, model.config["vocab_size"]), batch["labels"].reshape(-1), ignore_index=-100)
            else: loss = chunked_linear_cross_entropy(hidden.reshape(-1, hidden.shape[-1]), model.head.weight, batch["labels"].reshape(-1), token_block=64, vocab_block=128)
            loss.backward()
        storage = ledger.summary((batch["input_ids"].numel(), model.config["vocab_size"])); storage.pop("saved_tensors")
        stages = {stage: summarize_times([r["times_ms"][stage] for r in measurements[key]]) for stage in measurements[key][0]["times_ms"]}
        rows.append({"method": key, "stages_ms": stages, "saved_storage": storage,
                     "supervised_tokens_per_second": sum(r["supervised_tokens"] for r in measurements[key]) / (sum(r["times_ms"]["total"] for r in measurements[key]) / 1000)})
    extras = {}
    for task in ("evaluate", "checkpoint_write"):
        values = []
        for i in range(3):
            start = time.perf_counter_ns()
            if task == "evaluate": core.evaluate(models["dense"], windows["validation"])
            else: torch.save({"model": models["dense"].state_dict(), "optimizer": optimizers["dense"].state_dict()}, directory / f"profile-checkpoint-{i}.pt")
            values.append((time.perf_counter_ns() - start) / 1e6)
        extras[task] = summarize_times(values)
    save(directory / "raw-step-times.json", measurements)
    report = {"schema": "full-step-profile-v1", "frozen": frozen, "rows": rows, "separate_overheads_ms": extras,
              "numerical_comparison": {"loss_max_error": loss_error, "final_parameter_max_error": parameter_error, "passed": True},
              "limits": ["CPU double precision timing, not GPU or the float32 training throughput", "saved storage references are not allocator/RSS/physical memory peaks",
                         "stage medians need not sum to the total median; raw paired per-step samples do sum", "evaluation and writes measured separately; write timing does not guarantee fsync durability", "no full model retraining quality claim from changing the CE implementation"]}
    save(directory / "report.json", report); return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out-dir"); parser.add_argument("--corpus"); parser.add_argument("--resume-check")
    args = parser.parse_args(); symbolic.configure_cpu(1)
    if args.resume_check: resume_check(args.resume_check)
    else:
        if not args.out_dir or not args.corpus: parser.error("--out-dir and --corpus required")
        root = Path(args.out_dir); root.mkdir(parents=True, exist_ok=True)
        run_text(root / "text", args.corpus)
        run_reproduction(root / "reproduction", root / "text")
        run_profile(root / "profile", root / "text")
