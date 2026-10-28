"""Same trained text Transformer: padding, logit selection and fixed-token K/V replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

import learning_core as symbolic
import text_training_core as core


FROZEN = {
    "tolerance": 2e-10,
    "suffix_steps": 4,
    "sides": ["left", "right"],
    "cases": [
        {"id": "ragged", "title": "变长前缀", "lengths": [7, 19, 41], "extra_pad": 0},
        {"id": "balanced", "title": "等长、无填充的对照", "lengths": [19, 19, 19], "extra_pad": 0},
        {"id": "extra", "title": "变长前缀、额外三个填充槽", "lengths": [7, 19, 41], "extra_pad": 3},
    ],
    "methods": [
        {"id": "correct", "title": "逻辑位置、有效 key、最后有效位置"},
        {"id": "physical", "title": "故障：物理列号充当位置"},
        {"id": "last-column", "title": "故障：总取张量最后一列"},
        {"id": "length-index", "title": "故障：总取有效长度减一"},
        {"id": "no-key-mask", "title": "故障：不屏蔽填充 key"},
        {"id": "cache-physical", "title": "故障：仅续算位置取缓存槽数"},
    ],
    "input_selection": "First three saved training windows, fixed prefix lengths and next four original input tokens; no sampling or selection by outcomes.",
    "reference": "Original TextDecoder.forward on each unpadded prefix at every event, with hooks recording actual per-layer K/V.",
    "position_encoding": "Saved fixed sinusoidal buffer, float32 values cast to float64; this complete model does not use RoPE.",
    "generation": "Four teacher-forced original continuation inputs. Compare next-position logits; no free-running generation, EOS stopping, context cropping or padding-token emission experiment.",
}


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cache_sha(layers, length):
    return [[tensor_sha(value[:, :, :length]) for value in layer] for layer in layers]


def weight_sha(model):
    return {name: tensor_sha(value) for name, value in model.state_dict().items()}


def make_batch(prefixes, side, extra_pad=0):
    if side not in FROZEN["sides"] or not prefixes or any(not row for row in prefixes):
        raise ValueError("Nonempty prompts and a known padding side are required")
    if type(extra_pad) is not int or extra_pad < 0:
        raise ValueError("Invalid extra padding count")
    width = max(map(len, prefixes)) + extra_pad
    ids = torch.full((len(prefixes), width), core.PAD, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(prefixes):
        start = width - len(row) if side == "left" else 0
        ids[i, start:start + len(row)] = torch.tensor(row, dtype=torch.long)
        valid[i, start:start + len(row)] = True
    return ids, valid


def forward_adapter(model, ids, valid, method="correct", past=None):
    if method not in {row["id"] for row in FROZEN["methods"]}:
        raise ValueError("Unknown inference method")
    if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[1] < 1 or valid.shape != ids.shape or valid.dtype != torch.bool:
        raise ValueError("Expected int64 IDs and bool validity with equal [batch,time] shape")
    previous = 0 if past is None else past["valid"].shape[1]
    if past is not None and (past["valid"].shape[0] != ids.shape[0] or len(past["layers"]) != len(model.blocks)):
        raise ValueError("Cache batch or layer count changed")
    complete = valid if past is None else torch.cat((past["valid"], valid), dim=1)
    if not complete.any(dim=1).all() or complete.shape[1] > model.config["context"]:
        raise ValueError("Empty prompt or physical context limit exceeded")
    positions = (complete.long().cumsum(1)[:, previous:] - 1).clamp_min(0)
    if method == "physical" or (method == "cache-physical" and past is not None):
        positions = torch.arange(previous, complete.shape[1])[None].expand_as(ids)
    key_valid = torch.ones_like(complete) if method == "no-key-mask" else complete
    causal = torch.arange(complete.shape[1])[None] <= torch.arange(previous, complete.shape[1])[:, None]
    allowed = causal[None, None] & key_valid[:, None, None, :]
    hidden = model.embedding(ids) + model.positions[positions]
    layers = []
    for i, block in enumerate(model.blocks):
        hidden, kv = block(hidden, allowed, None if past is None else past["layers"][i])
        layers.append(kv)
    return model.head(model.norm(hidden)), {"layers": tuple(layers), "valid": complete}, positions


def selection_indices(valid, method):
    if not valid.any(dim=1).all():
        raise ValueError("Cannot read next logits from an empty input row")
    if method == "last-column":
        return torch.full((valid.shape[0],), valid.shape[1] - 1, dtype=torch.long)
    if method == "length-index":
        return valid.long().sum(1) - 1
    columns = torch.arange(valid.shape[1])[None].expand_as(valid)
    return torch.where(valid, columns, -1).amax(1)


@torch.no_grad()
def original_reference(model, tokens):
    captures = []
    handles = [block.register_forward_hook(lambda _module, _args, output: captures.append(tuple(v.detach().clone() for v in output[1]))) for block in model.blocks]
    try:
        logits = model(torch.tensor([tokens], dtype=torch.long)).detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    return {"logits": logits[0], "layers": tuple(captures)}


def kv_error(layers, valid, sample, reference):
    indices = valid[sample].nonzero().flatten()
    return max(float((actual[sample:sample + 1].index_select(2, indices) - expected).abs().max())
               for layer, baseline in zip(layers, reference["layers"], strict=True)
               for actual, expected in zip(layer, baseline, strict=True))


@torch.no_grad()
def run(out, text_directory):
    out.mkdir()
    save(out / "frozen-config.json", FROZEN)
    inputs = out / "inputs"
    inputs.mkdir()
    for name in ["seed-11-final.pt", "tokenizer.json", "windows.json", "report.json"]:
        shutil.copyfile(text_directory / name, inputs / name)
    saved = torch.load(inputs / "seed-11-final.pt", weights_only=True)
    model = core.make_model(saved["seed"], saved["config"]).double().eval()
    model.load_state_dict(saved["model"])
    original_weights = weight_sha(model)
    records = json.loads((inputs / "windows.json").read_text())["train"][:3]
    if len(records) != 3 or any(len(r["input"]) != 96 or core.PAD in r["input"] for r in records):
        raise ValueError("Unexpected frozen text inputs")
    input_rows, references, report_rows, trials = [], {}, [], []
    (out / "tensors").mkdir()
    for case in FROZEN["cases"]:
        prefixes = [row["input"][:n] for row, n in zip(records, case["lengths"], strict=True)]
        suffixes = [row["input"][n:n + FROZEN["suffix_steps"]] for row, n in zip(records, case["lengths"], strict=True)]
        input_rows.append({"case": case["id"], "records": [{"id": row["id"], "prefix": prefix, "continuation": suffix} for row, prefix, suffix in zip(records, prefixes, suffixes, strict=True)]})
        refs = [[original_reference(model, prefix + suffix[:event]) for event in range(FROZEN["suffix_steps"] + 1)] for prefix, suffix in zip(prefixes, suffixes, strict=True)]
        references[case["id"]] = refs
        for side in FROZEN["sides"]:
            ids, valid = make_batch(prefixes, side, case["extra_pad"])
            for definition in FROZEN["methods"]:
                method, past, events, rows = definition["id"], None, [], []
                all_prefill_logits = None
                for event in range(FROZEN["suffix_steps"] + 1):
                    current_ids = ids if event == 0 else torch.tensor([[suffix[event - 1]] for suffix in suffixes])
                    current_valid = valid if event == 0 else torch.ones_like(current_ids, dtype=torch.bool)
                    previous_layers = None if past is None else past["layers"]
                    logits, past, positions = forward_adapter(model, current_ids, current_valid, method, past)
                    if event == 0:
                        all_prefill_logits = logits.clone()
                    else:
                        for before, after in zip(previous_layers, past["layers"], strict=True):
                            for a, b in zip(before, after, strict=True):
                                if not torch.equal(a, b[:, :, :a.shape[2]]):
                                    raise AssertionError("A cached prefix was rewritten")
                    selected = selection_indices(current_valid, method)
                    next_logits = logits[torch.arange(ids.shape[0]), selected]
                    physical_before = 0 if event == 0 else ids.shape[1] + event - 1
                    events.append({"input_ids": current_ids, "valid": current_valid, "positions": positions, "selected_indices": selected,
                                   "next_logits": next_logits.clone(), "physical_slots": past["valid"].shape[1], "cache_sha256": cache_sha(past["layers"], past["valid"].shape[1])})
                    for sample, length in enumerate(case["lengths"]):
                        reference = refs[sample][event]
                        real_logits = logits[sample][current_valid[sample]]
                        expected_logits = reference["logits"] if event == 0 else reference["logits"][-1:]
                        columns = current_valid[sample].nonzero().flatten()
                        row = {"case": case["id"], "side": side, "method": method, "sample": sample, "event": event,
                               "physical_cache_slots": past["valid"].shape[1], "valid_cache_slots": int(past["valid"][sample].sum()),
                               "last_valid_column": physical_before + int(columns[-1]), "selected_column": physical_before + int(selected[sample]),
                               "expected_logical_position": length + event - 1, "last_valid_position_id": int(positions[sample, columns[-1]]),
                               "valid_logits_max_error": float((real_logits - expected_logits).abs().max()),
                               "next_logits_max_error": float((next_logits[sample] - reference["logits"][-1]).abs().max()),
                               "kv_max_error": kv_error(past["layers"], past["valid"], sample, reference),
                               "reference_argmax": int(reference["logits"][-1].argmax()), "actual_argmax": int(next_logits[sample].argmax())}
                        row["within_tolerance"] = max(row[k] for k in ["valid_logits_max_error", "next_logits_max_error", "kv_max_error"]) <= FROZEN["tolerance"]
                        rows.append(row)
                for event in events:
                    if event["cache_sha256"] != cache_sha(past["layers"], event["physical_slots"]):
                        raise AssertionError("Final cache cannot reconstruct an earlier actual cache")
                name = f"{case['id']}-{side}-{method}.pt"
                torch.save({"input_ids": ids, "valid": valid, "prefill_logits": all_prefill_logits,
                            "events": events, "final_cache": past}, out / "tensors" / name)
                trials.append({"case": case["id"], "side": side, "method": method, "tensor_file": "tensors/" + name,
                               "tensor_sha256": file_sha(out / "tensors" / name), "cached_prefixes_unchanged": True})
                report_rows.extend(rows)
    torch.save(references, out / "references.pt")
    save(out / "input-sequences.json", input_rows)
    empty_rejected = False
    try:
        make_batch([[], [2]], "left")
    except ValueError:
        empty_rejected = True
    if not empty_rejected or weight_sha(model) != original_weights or not all(row["within_tolerance"] for row in report_rows if row["method"] == "correct"):
        raise AssertionError("Correct padded paths, empty-input guard or unchanged weights failed")
    report = {"schema": "padding-inference-v1", "status": "passed", "frozen": FROZEN,
              "runtime": symbolic.environment_metadata(), "model": model.config, "parameter_count": sum(p.numel() for p in model.parameters()),
              "input_sha256": {p.name: file_sha(p) for p in sorted(inputs.iterdir())}, "initial_weight_sha256": original_weights,
              "final_weight_sha256": weight_sha(model), "inputs": input_rows, "trials": trials, "rows": report_rows,
              "references_sha256": file_sha(out / "references.pt"),
              "correctness": {"correct_paths_passed": True, "empty_prompt_rejected": empty_rejected, "cached_prefixes_unchanged": True,
                              "trials": len(trials), "observations": len(report_rows)},
              "limits": ["Fixed sinusoidal full Transformer, not RoPE. Original text interface uses right padding; explicit independent adapter supplies logical positions and key masks.",
                         "Compare original per-sequence full forward with batched prefill and four fixed original continuation tokens; no sampling quality, EOS policy or generated PAD behavior claim.",
                         "CPU float64, unchanged weights and original float32 sinusoidal values; no dropout, training, GPU kernel, latency or context-extension result.",
                         "Unused PAD-query logits are not a model target. Cache comparisons include every valid K/V at each layer; masked physical cache slots remain allocated."]}
    save(out / "report.json", report)
    print(json.dumps({"status": "passed", **report["correctness"], "correct_path_max_error": max(max(r[k] for k in ["valid_logits_max_error", "next_logits_max_error", "kv_max_error"]) for r in report_rows if r["method"] == "correct")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--text-directory", required=True, type=Path)
    args = parser.parse_args()
    symbolic.configure_cpu(1)
    run(args.out_dir, args.text_directory)
