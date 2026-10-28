"""Same supervised-label budget, same initial values; retain distinct update/FLOP counts."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import learning_core as core
from generalization_lab import diagnostic_generation, summarize

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {"seeds": [11, 29, 47], "supervised_token_budget": 410000, "batch_size": 64, "lr": .003, "weight_decay": .01, "clip_norm": 1.,
          "support": ["short-2-5", "extended-2-8"], "selection": "final fixed consumed-label budget, no test tuning",
          "final_batch": "keep row-major labels until the exact remaining budget, set all other labels to -100; log discarded labels",
          "matched": "supervised labels and initial weights", "not_matched": "optimizer updates, input/padding positions, FLOPs, length distribution and unique groups"}


def limit_labels(labels, remaining):
    if remaining <= 0: raise ValueError("remaining budget must be positive")
    result = labels.clone(); valid = (result != -100).reshape(-1); indexes = valid.nonzero().reshape(-1)
    if len(indexes) > remaining: result.reshape(-1)[indexes[remaining:]] = -100
    return result, int((result != -100).sum()), max(0, len(indexes) - remaining)


def run(out_dir, practice_directory):
    out = Path(out_dir).resolve(); out.mkdir(parents=True, exist_ok=False); prior = Path(practice_directory).resolve()
    core.configure_cpu(1)
    def save(name, value): (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    save("frozen-config.json", FROZEN)
    splits = json.loads((prior / "generalization/splits.json").read_text())
    original = json.loads((ROOT / json.loads((prior / "generalization/report.json").read_text())["source_report"]).read_text())
    evaluation = splits["original"]["test"] + splits["original"]["length"] + splits["extra_length"]
    populations = {"short-2-5": splits["original"]["train"], "extended-2-8": splits["extended_train"]}
    for rows in populations.values(): assert not ({r["group_id"] for r in rows} & {r["group_id"] for r in evaluation})
    save("splits.json", {"populations": populations, "evaluation": evaluation})
    runs = []
    for old in original["runs"]:
        seed = old["seed"]; initial_file = ROOT / old["checkpoints"]["initial"]
        initial = torch.load(initial_file, map_location="cpu", weights_only=True)
        for name, records in populations.items():
            core.seed_all(seed); model = core.make_model(seed=seed, **original["frozen"]["model"]); model.load_state_dict(initial["model"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01, foreach=False)
            stream = core.RecordBatchStream(records, 64, seed=seed + 1000)
            consumed = inputs = padded = 0; trace = []; curve = []
            while consumed < FROZEN["supervised_token_budget"]:
                model.train(); batch = core.collate_records(stream.next_records())
                batch["labels"], valid, discarded = limit_labels(batch["labels"], FROZEN["supervised_token_budget"] - consumed)
                optimizer.zero_grad(set_to_none=True); loss = core.forward_loss(model, batch); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm): raise FloatingPointError("nonfinite matched-budget update")
                optimizer.step(); consumed += valid; inputs += batch["input_tokens"]; padded += batch["input_ids"].numel()
                update = len(trace) + 1
                trace.append({"update": update, "loss": float(loss.detach()), "supervised_tokens": consumed, "batch_supervised_tokens": valid,
                              "discarded_final_labels": discarded, "input_tokens": inputs, "computed_positions": padded, "record_ids": batch["record_ids"]})
                if update % 100 == 0 or consumed == FROZEN["supervised_token_budget"]:
                    with core.evaluation_mode(model), torch.no_grad(): val = float(core.forward_loss(model, core.collate_records(splits["original"]["validation"])))
                    curve.append({"update": update, "supervised_tokens": consumed, "validation_ce": val})
                    print(f"matched seed={seed} support={name} updates={update} labels={consumed} validation={val:.4f}", flush=True)
            model.eval(); predictions = diagnostic_generation(model, evaluation)
            filename = f"seed-{seed}-{name}"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "model_config": core.model_config(model), "seed": seed,
                        "stream": stream.state_dict(), "rng_torch": torch.get_rng_state(), "updates": len(trace), "supervised_tokens": consumed}, out / (filename + ".pt"))
            save(filename + "-trace.json", trace)
            runs.append({"seed": seed, "support": name, "updates": len(trace), "supervised_tokens": consumed, "input_tokens": inputs, "computed_positions": padded,
                         "discarded_final_labels": trace[-1]["discarded_final_labels"], "initial_sha256": hashlib.sha256(initial_file.read_bytes()).hexdigest(),
                         "checkpoint": filename + ".pt", "checkpoint_sha256": hashlib.sha256((out / (filename + ".pt")).read_bytes()).hexdigest(),
                         "curve": curve, "predictions": predictions, "summary": summarize(predictions)})
    report = {"schema": "matched-budget-v1", "status": "passed", "frozen": FROZEN, "group_overlap": 0,
              "comparison_source": {"path": str(prior.relative_to(ROOT)), "split_sha256": hashlib.sha256((prior / "generalization/splits.json").read_bytes()).hexdigest()},
              "runs": runs, "limits": ["equal supervised-label budget does not imply equal compute or number of updates", "6..8 is in-support for extended training and out-of-support for short training; 9..10 remains external to both", "all final seed results retained; no post-test tuning"]}
    save("report.json", report); return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out-dir", required=True); parser.add_argument("--practice-directory", required=True)
    args = parser.parse_args(); run(args.out_dir, args.practice_directory)
