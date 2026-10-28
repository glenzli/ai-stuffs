"""Train-only counts, validation-only smoothing choice, frozen full-split scoring."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def fit_counts(records):
    unigram, bigram = Counter(), defaultdict(Counter)
    for row in records:
        if len(row["input"]) != len(row["target"]): raise ValueError("Shifted input/target length differs")
        for x, y in zip(row["input"], row["target"]): unigram[y] += 1; bigram[x][y] += 1
    return unigram, bigram


def evaluate(records, counts, vocabulary, kind, alpha):
    unigram, bigram = counts; total = sum(unigram.values()); loss = correct = size = 0; documents = {}
    for row in records:
        part = documents.setdefault(row["document"], {"loss_sum": 0., "correct": 0, "tokens": 0})
        for x, y in zip(row["input"], row["target"]):
            table = unigram if kind == "unigram" else bigram.get(x, Counter())
            denominator = total if kind == "unigram" else sum(table.values())
            value = -math.log((table[y] + alpha) / (denominator + alpha * vocabulary))
            predicted = min(table, key=lambda token: (-table[token], token)) if table else 0
            loss += value; correct += predicted == y; size += 1
            part["loss_sum"] += value; part["correct"] += predicted == y; part["tokens"] += 1
    return {"ce_nats_per_token": loss / size, "teacher_forced_accuracy": correct / size, "supervised_tokens": size,
            "documents": {key: {**r, "ce_nats_per_token": r["loss_sum"] / r["tokens"]} for key, r in documents.items()}}


def run(out_dir, text_directory):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=False); source = Path(text_directory)
    frozen = {"models": ["unigram", "bigram"], "alpha_candidates": [.01, .1, 1.], "selection": "validation CE only; tie selects first candidate",
              "target": "same saved 387-token vocabulary and every shifted non-padding label, including document EOS", "test_policy": "evaluate only after alpha selection; no outcome-driven adjustment"}
    (out / "frozen-config.json").write_text(json.dumps(frozen, indent=2) + "\n")
    windows = json.loads((source / "windows.json").read_text()); tokenizer = json.loads((source / "tokenizer.json").read_text())
    counts = fit_counts(windows["train"]); rows = []
    for kind in frozen["models"]:
        candidates = [{"alpha": alpha, "validation": evaluate(windows["validation"], counts, tokenizer["vocab_size"], kind, alpha)} for alpha in frozen["alpha_candidates"]]
        selected = min(candidates, key=lambda row: row["validation"]["ce_nats_per_token"])
        rows.append({"model": kind, "selected_alpha": selected["alpha"], "selection_trace": candidates,
                     "evaluation": {split: evaluate(records, counts, tokenizer["vocab_size"], kind, selected["alpha"]) for split, records in windows.items()}})
    count_file = {"unigram": dict(counts[0]), "bigram": {key: dict(value) for key, value in counts[1].items()}}
    (out / "counts.json").write_text(json.dumps(count_file, sort_keys=True, indent=2) + "\n")
    report = {"schema": "language-baselines-v1", "status": "passed", "frozen": frozen, "vocab_size": tokenizer["vocab_size"], "train_supervised_tokens": sum(counts[0].values()),
              "source": {name: {"path": str(source / name), "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()} for name in ["windows.json", "tokenizer.json", "report.json"]},
              "models": rows, "transformer": json.loads((source / "report.json").read_text())["runs"],
              "limits": ["same-tokenizer next-token prediction; no claim of fluent generation", "count models use only training labels; alpha selection uses validation labels", "document-level slices are descriptive; this small domain collection is not a broad Chinese benchmark"]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n"); return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out-dir", required=True); parser.add_argument("--text-directory", required=True)
    args = parser.parse_args(); run(args.out_dir, args.text_directory)
