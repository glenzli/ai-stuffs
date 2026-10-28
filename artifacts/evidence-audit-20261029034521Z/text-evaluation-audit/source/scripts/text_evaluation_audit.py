"""Re-score fixed language-model weights and retain every document and target loss."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import learning_core as symbolic
import text_training_core as core
from paired_evidence import bootstrap_ratio


def digest(file): return hashlib.sha256(Path(file).read_bytes()).hexdigest()
def save(file, data): Path(file).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def score_model(model, records):
    losses = []; predictions = []
    for start in range(0, len(records), 16):
        batch = core.collate(records[start:start + 16]); logits = model(batch["input_ids"]); labels = batch["labels"]
        values = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none").reshape_as(labels)
        valid = labels != -100; losses.extend(values[valid].double().tolist()); predictions.extend(logits.argmax(-1)[valid].tolist())
    return np.asarray(losses, dtype=np.float64), np.asarray(predictions, dtype=np.int32)


def run(directory, text_directory, baseline_file, expected_baseline_sha):
    if digest(baseline_file) != expected_baseline_sha: raise ValueError("Accepted count-model report changed")
    symbolic.configure_cpu(1); out = Path(directory); out.mkdir(parents=True, exist_ok=False)
    frozen = {"seeds": [11, 29, 47], "checkpoint": "unchanged final update 800; no training or checkpoint selection", "batch_size": 16,
              "scope": "all saved non-padding targets, including document EOS; same frozen tokenizer and windows",
              "aggregation": "micro=sum NLL/sum labels; macro=equal mean of document CE", "repetitions": 2000, "random_seed": 73,
              "resampling": "whole documents; recompute paired NLL difference / sampled label count", "analysis_status": "descriptive follow-up of already inspected fixed data"}
    save(out / "frozen-config.json", frozen); inputs = out / "inputs"; inputs.mkdir(); source = Path(text_directory)
    for name in ("windows.json", "tokenizer.json", "report.json"): shutil.copyfile(source / name, inputs / name)
    shutil.copyfile(baseline_file, inputs / "language-baselines.json")
    baseline = json.loads((inputs / "language-baselines.json").read_text()); original = json.loads((inputs / "report.json").read_text())
    windows = json.loads((inputs / "windows.json").read_text()); bigram = next(r for r in baseline["models"] if r["model"] == "bigram")
    original_runs = {r["seed"]: r for r in original["runs"]}; arrays = {}; results = []; distributions = {}; source_files = {}
    for seed in frozen["seeds"]:
        checkpoint = source / f"seed-{seed}-final.pt"
        if digest(checkpoint) != original_runs[seed]["checkpoint_sha256"]: raise ValueError("Prior checkpoint identity changed")
        shutil.copyfile(checkpoint, inputs / checkpoint.name); source_files[checkpoint.name] = digest(checkpoint)
        saved = torch.load(inputs / checkpoint.name, weights_only=True, map_location="cpu")
        model = core.make_model(seed, saved["config"]).eval(); model.load_state_dict(saved["model"])
        for split, records in windows.items():
            losses, predicted = score_model(model, records); labels = np.asarray([y for r in records for y in r["target"]], dtype=np.int32)
            arrays[f"seed{seed}_{split}_nll"] = losses; arrays[f"seed{seed}_{split}_prediction"] = predicted
            documents = {}; offset = 0
            for record in records:
                count = len(record["target"]); part = documents.setdefault(record["document"], {"tokens": 0, "loss_sum": 0., "correct": 0, "windows": 0})
                part["tokens"] += count; part["loss_sum"] += float(losses[offset:offset+count].sum()); part["correct"] += int((predicted[offset:offset+count] == labels[offset:offset+count]).sum()); part["windows"] += 1; offset += count
            if offset != len(losses) or len(losses) != len(labels): raise ValueError("Target positions are missing or repeated")
            rows = []
            for document, part in documents.items():
                old = bigram["evaluation"][split]["documents"][document]
                if old["tokens"] != part["tokens"]: raise ValueError("Count-model target denominator differs")
                rows.append({"document": document, **part, "ce_nats_per_token": part["loss_sum"] / part["tokens"],
                             "teacher_forced_accuracy": part["correct"] / part["tokens"], "bigram_ce": old["ce_nats_per_token"], "bigram_loss_sum": old["loss_sum"],
                             "token_share": part["tokens"] / len(labels), "gain_nats_per_token": old["ce_nats_per_token"] - part["loss_sum"] / part["tokens"]})
            micro = float(losses.sum()) / len(labels); macro = sum(r["ce_nats_per_token"] for r in rows) / len(rows)
            reference = original_runs[seed]["evaluation"]["learned"][split]
            if abs(micro - reference["ce_nats_per_token"]) > 2e-6 or int((predicted == labels).sum()) / len(labels) != reference["teacher_forced_accuracy"]:
                raise ValueError("Re-scoring differs from the accepted aggregate")
            interval = bootstrap_ratio([(r["bigram_loss_sum"] - r["loss_sum"], r["tokens"]) for r in rows]); distributions[f"seed{seed}_{split}"] = interval.pop("distribution")
            results.append({"seed": seed, "split": split, "documents": rows, "supervised_tokens": len(labels), "micro_ce": micro, "macro_ce": macro,
                            "bigram_micro_ce": bigram["evaluation"][split]["ce_nats_per_token"], "bigram_macro_ce": sum(r["bigram_ce"] for r in rows) / len(rows),
                            "paired_gain": interval, "improved_documents": sum(r["gain_nats_per_token"] > 0 for r in rows),
                            "prior_aggregate_error": abs(micro - reference["ce_nats_per_token"])})
    np.savez_compressed(out / "position-scores.npz", **arrays); save(out / "bootstrap-distributions.json", distributions)
    source_files.update({name: digest(inputs / name) for name in ("windows.json", "tokenizer.json", "report.json", "language-baselines.json")})
    report = {"schema": "text-evaluation-audit-v1", "status": "passed", "frozen": frozen, "inputs": source_files, "runs": results,
              "limits": ["same small domain collection and tokenizer; no new test set or causal attribution", "macro and micro answer different weighting questions; neither automatically replaces the other",
                         "whole-document resampling preserves token weights but requires approximate document independence", "resampling observed documents does not account for tokenizer choices, training seeds, near duplicates, or deployment shift",
                         "all seeds remain separate; ranges are conditional on one saved model"]}
    save(out / "report.json", report); return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out-dir", required=True); p.add_argument("--text-directory", required=True); p.add_argument("--baseline-report", required=True); p.add_argument("--baseline-sha256", required=True)
    a = p.parse_args(); run(a.out_dir, a.text_directory, a.baseline_report, a.baseline_sha256)
