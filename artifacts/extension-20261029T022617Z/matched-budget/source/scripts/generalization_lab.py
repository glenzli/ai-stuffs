"""Frozen-checkpoint diagnostics and a declared training-support intervention."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
import learning_core as core

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loaded_model(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = core.make_model(seed=saved["seed"], **saved["model_config"])
    model.load_state_dict(saved["model"]); model.eval()
    return model


@torch.no_grad()
def diagnostic_generation(model, records, suppress_early_eos=False):
    result = []
    for length in sorted({row["length"] for row in records}):
        selected = [row for row in records if row["length"] == length]
        for start in range(0, len(selected), 64):
            rows = selected[start:start + 64]
            ids = torch.tensor([[core.BOS] + row["content"] + [core.SEP] for row in rows])
            predictions = [[] for _ in rows]; active = [True] * len(rows)
            for step in range(core.DEFAULT_MAX_NEW_TOKENS):
                logits = model(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))[:, -1].clone()
                if suppress_early_eos and step < length: logits[:, core.EOS] = -torch.inf
                next_ids = logits.argmax(-1)
                for i, token in enumerate(next_ids.tolist()):
                    if active[i]:
                        predictions[i].append(token)
                        if token == core.EOS: active[i] = False
                if not any(active): break
                ids = torch.cat((ids, next_ids[:, None]), dim=1)
            for row, predicted in zip(rows, predictions):
                result.append({**row, "predicted": predicted, **core.score_generation(predicted, row["target"])})
    return result


@torch.no_grad()
def forced_positions(model, records):
    rows = []
    for start in range(0, len(records), 64):
        selected = records[start:start + 64]; batch = core.collate_records(selected)
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        for i, record in enumerate(selected):
            valid = batch["labels"][i] != -100; z = logits[i][valid]; targets = batch["labels"][i][valid]
            rows.append({"id": record["id"], "length": record["length"],
                         "positions": [{"answer_position": j + 1, "target": int(y), "top1": int(values.argmax()),
                                        "target_probability": float(values.softmax(-1)[y]), "eos_probability": float(values.softmax(-1)[core.EOS])}
                                       for j, (values, y) in enumerate(zip(z, targets))]})
    return rows


def summarize(rows):
    return [{"length": length, "records": len(part), "exact_match": sum(r["exact_match"] for r in part) / len(part),
             "content_accuracy_macro": sum(sum(a == b for a, b in zip(r["predicted"], r["target"])) / r["length"] for r in part) / len(part),
             "early_eos": sum(r["early_eos"] for r in part), "cap": sum(r["termination"] == "cap" for r in part)}
            for length in sorted({r["length"] for r in rows}) for part in [[r for r in rows if r["length"] == length]]]


def run(directory):
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=False)
    core.configure_cpu(1)
    manifest = json.loads((ROOT / "src/experiment-sources.json").read_text())
    original_path = ROOT / manifest["experiments"]["learning"]["report"]["path"]
    original = json.loads(original_path.read_text()); split_path = original_path.parent / "splits.json"
    splits = json.loads(split_path.read_text())
    extra = core.make_task_splits(data_seed=20261029, train_lengths=(6, 7, 8), length_lengths=(9, 10))
    blocked = {r["group_id"] for values in splits.values() for r in values}
    extra_train = [r for r in extra["train"] if r["group_id"] not in blocked]
    mixed = splits["train"] + extra_train
    evaluation = splits["test"] + splits["length"] + extra["length"]
    assert not ({r["group_id"] for r in mixed} & {r["group_id"] for r in evaluation})
    frozen = {"seeds": [11, 29, 47], "updates": 1200, "batch_size": 64, "lr": .003, "weight_decay": .01,
              "clip_norm": 1., "checkpoint_selection": "final fixed budget; no held-out tuning or early stopping",
              "support_intervention": "append disjoint training groups at lengths 6..8; identical optimizer-update budget, NOT identical token/FLOP budget",
              "eos_probe": "suppress EOS before record target length; task-aware diagnostic policy, not the original model's unconstrained generation",
              "position_probe": "shift fixed sinusoidal position signals by +4 at inference; input perturbation, not retraining a different positional architecture"}
    save(directory / "frozen-config.json", frozen)
    save(directory / "splits.json", {"original": splits, "extended_train": mixed, "extra_length": extra["length"]})
    baselines, extended = [], []
    for old in original["runs"]:
        seed = old["seed"]; model = loaded_model(ROOT / old["checkpoints"]["final"])
        standard = diagnostic_generation(model, evaluation)
        for split in ("test", "length"):
            expected = {r["id"]: r["predicted"] for r in old["learned"][split]["predictions"]}
            assert all(row["predicted"] == expected[row["id"]] for row in standard if row["id"] in expected)
        suppressed = diagnostic_generation(model, evaluation, True)
        positions = forced_positions(model, evaluation)
        encoding = model.position_encoding.clone()
        with torch.no_grad(): model.position_encoding[:40].copy_(encoding[4:44])
        shifted = diagnostic_generation(model, evaluation)
        model.position_encoding.copy_(encoding)
        row = {"seed": seed, "checkpoint": old["checkpoints"]["final"], "checkpoint_sha256": file_hash(ROOT / old["checkpoints"]["final"]),
               "standard": standard, "eos_diagnostic": suppressed, "shifted_positions": shifted, "teacher_forced": positions,
               "summaries": {"standard": summarize(standard), "eos_diagnostic": summarize(suppressed), "shifted_positions": summarize(shifted)}}
        baselines.append(row); save(directory / f"seed-{seed}-frozen-probes.json", row)
        core.seed_all(seed)
        model = core.make_model(seed=seed, **original["frozen"]["model"])
        initial = torch.load(ROOT / old["checkpoints"]["initial"], map_location="cpu", weights_only=True)
        model.load_state_dict(initial["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01, foreach=False)
        stream = core.RecordBatchStream(mixed, 64, seed=seed + 1000)
        curve, trace, input_tokens, supervised = [], [], 0, 0
        for step in range(1, 1201):
            model.train(); batch = core.collate_records(stream.next_records())
            optimizer.zero_grad(set_to_none=True); loss = core.forward_loss(model, batch); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            if not torch.isfinite(loss) or not torch.isfinite(norm): raise FloatingPointError("nonfinite support run")
            input_tokens += batch["input_tokens"]; supervised += batch["supervised_tokens"]
            trace.append({"step": step, "loss": float(loss.detach()), "record_ids": batch["record_ids"], "input_tokens": input_tokens, "supervised_tokens": supervised})
            if step % 200 == 0:
                batch_val = core.collate_records(splits["validation"])
                with core.evaluation_mode(model), torch.no_grad(): val = float(core.forward_loss(model, batch_val))
                curve.append({"step": step, "validation_ce": val, "input_tokens": input_tokens, "supervised_tokens": supervised})
                print(f"support seed={seed} step={step} validation CE={val:.4f}", flush=True)
        model.eval(); predictions = diagnostic_generation(model, evaluation)
        checkpoint = directory / f"seed-{seed}-extended.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "model_config": core.model_config(model),
                    "stream": stream.state_dict(), "rng": core.capture_rng_state(), "seed": seed, "steps": 1200}, checkpoint)
        save(directory / f"seed-{seed}-extended-trace.json", trace)
        extended.append({"seed": seed, "predictions": predictions, "summary": summarize(predictions), "curve": curve,
                         "input_tokens": input_tokens, "supervised_tokens": supervised, "baseline_input_tokens": old["progress"]["input_tokens"],
                         "baseline_supervised_tokens": old["progress"]["supervised_tokens"], "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": file_hash(checkpoint)})
    report = {"schema": "generalization-diagnostics-v1", "frozen": frozen, "source_report": str(original_path.relative_to(ROOT)),
              "source_report_sha256": file_hash(original_path), "baseline_predictions_reproduced": True,
              "training_records": {"short": len(splits["train"]), "extended": len(mixed)}, "group_overlap": 0,
              "baselines": baselines, "extended": extended,
              "limits": ["EOS suppression uses task length and changes inference policy; it is diagnostic, not an unconstrained capability score",
                         "position shift tests sensitivity only; it cannot establish which positional architecture would generalize after training",
                         "support comparison holds optimizer updates fixed; token consumption and padded compute change; do not attribute all differences solely to length coverage",
                         "test 9..10 remains outside both training supports; all seeds and outcomes are retained"]}
    save(directory / "report.json", report); return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out-dir", required=True)
    run(parser.parse_args().out_dir)
