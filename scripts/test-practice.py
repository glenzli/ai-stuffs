"""Audit saved evidence without retraining or replacing the measured runs."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import unittest
import torch
from torch.nn import functional as F
import learning_core as symbolic
import text_training_core as core
import text_project_lab as lab
import generalization_lab as general
from chunked_ce_reference import chunked_linear_cross_entropy

ROOT = Path(__file__).resolve().parents[1]
symbolic.configure_cpu(1)
manifest = json.loads((ROOT / "src/experiment-sources.json").read_text())
BASE = (ROOT / manifest["experiments"]["practice"]["report"]["path"]).parent
read = lambda p: json.loads(Path(p).read_text())
report = read(BASE / "report.json")
corpus = read(BASE / "corpus.json")
tokenizer = read(BASE / "text/tokenizer.json")
windows = read(BASE / "text/windows.json")


class TextContracts(unittest.TestCase):
    def test_train_only_fitting_reproduces_saved_tokenizer(self):
        train = [d for d in corpus["documents"] if d["split"] == "train"]
        self.assertEqual(core.train_tokenizer(train), tokenizer)
        with self.assertRaises(ValueError): core.train_tokenizer(corpus["documents"])

    def test_every_paragraph_and_unseen_unicode_roundtrip(self):
        for text in [p["text"] for d in corpus["documents"] for p in d["paragraphs"]] + ["罕见字𠮷与🙂\n未知 poem λ", ""]:
            self.assertEqual(core.decode(core.encode(text, tokenizer), tokenizer), text)

    def test_document_and_exact_paragraph_isolation(self):
        ids, hashes = set(), set()
        for doc in corpus["documents"]:
            self.assertNotIn(doc["id"], ids); ids.add(doc["id"])
            for p in doc["paragraphs"]:
                self.assertEqual(hashlib.sha256(p["text"].encode()).hexdigest(), p["id"])
                self.assertNotIn(p["id"], hashes); hashes.add(p["id"])
        self.assertEqual([sum(d["split"] == split for d in corpus["documents"]) for split in windows], [24, 6, 8])

    def test_window_targets_cover_each_document_once_without_cross_document_context(self):
        # Independently reconstruct each target stream, including the sole document EOS.
        for doc in corpus["documents"]:
            encoded = [core.BOS] + core.encode("\n".join(p["text"] for p in doc["paragraphs"]), tokenizer) + [core.EOS]
            rows = [r for r in windows[doc["split"]] if r["document"] == doc["id"]]
            self.assertEqual([x for r in rows for x in r["target"]], encoded[1:])
            for row in rows:
                start, size = row["start"], len(row["target"])
                self.assertEqual(row["input"], encoded[start:start + size])
                self.assertEqual(row["target"], encoded[start + 1:start + size + 1])
                self.assertLessEqual(size, 96)
        self.assertEqual(core.make_windows(corpus, tokenizer), windows)

    def test_padding_and_future_tokens_cannot_change_valid_prefix_logits(self):
        model = core.make_model(5, dict(vocab_size=387, width=16, depth=1, heads=2, context=10)).eval()
        with torch.no_grad():
            a = torch.tensor([[1, 4, 5, 6]]); b = torch.tensor([[1, 4, 88, 89]])
            self.assertTrue(torch.equal(model(a)[:, :2], model(b)[:, :2]))
            padded = core.collate([dict(id="short", input=[1, 4], target=[4, 2]), dict(id="long", input=[1, 7, 8, 9], target=[7, 8, 9, 2])])
            self.assertEqual(padded["labels"][0].tolist(), [4, 2, -100, -100])
            self.assertTrue(torch.allclose(model(torch.tensor([[1, 4]]))[0], model(padded["input_ids"])[0, :2], atol=1e-7, rtol=0))

    def test_saved_checkpoints_reproduce_complete_split_metrics_and_all_generations(self):
        for run in report["reports"]["text"]["runs"]:
            for phase, suffix in [("initial", "initial"), ("learned", "final")]:
                saved = torch.load(BASE / f"text/seed-{run['seed']}-{suffix}.pt", weights_only=True)
                model = core.make_model(run["seed"], saved["config"]).eval(); model.load_state_dict(saved["model"])
                self.assertEqual(sum(p.numel() for p in model.parameters()), 298176)
                for split, records in windows.items():
                    actual = core.evaluate(model, records, batch_size=13)  # different aggregation boundaries
                    expected = run["evaluation"][phase][split]
                    self.assertAlmostEqual(actual["ce_nats_per_token"], expected["ce_nats_per_token"], delta=2e-6)
                    self.assertEqual(actual["teacher_forced_accuracy"], expected["teacher_forced_accuracy"])
                    self.assertEqual(actual["supervised_tokens"], expected["supervised_tokens"])
                for expected in run["generations"][phase]: self.assertEqual(core.generate(model, expected["prompt"], tokenizer), expected)

    def test_midpoint_resumes_all_saved_states_and_sixteen_step_trace(self):
        saved = torch.load(BASE / "text/seed-11-midpoint.pt", weights_only=True)
        expected = torch.load(BASE / "text/resume-expected.pt", weights_only=True)
        model = core.make_model(11, saved["config"]); model.load_state_dict(saved["model"])
        optimizer = lab.new_optimizer(model); optimizer.load_state_dict(saved["optimizer"])
        stream = core.WindowStream(len(windows["train"]), 0); stream.load_state_dict(saved["stream"])
        torch.set_rng_state(saved["torch_rng"])
        trace = [lab.step(model, optimizer, windows["train"], stream) for _ in range(16)]
        actual = lab.checkpoint(model, optimizer, stream, 11, 416)
        self.assertEqual(core.state_error(actual, expected["checkpoint"]), 0)
        self.assertEqual(core.state_error(trace, expected["trace"]), 0)

    def test_delivery_cli_matches_the_saved_sample_and_checks_bundle_identity(self):
        command = [sys.executable, str(ROOT / "scripts/text-inference.py"), "--checkpoint", str(BASE / "text/seed-11-final.pt"), "--tokenizer", str(BASE / "text/tokenizer.json"), "--prompt", "模型训练"]
        value = json.loads(subprocess.check_output(command, cwd=ROOT))
        self.assertEqual(value, report["reports"]["text"]["runs"][0]["generations"]["learned"][0])

    def test_delivery_rejects_a_different_tokenizer_with_the_same_vocabulary_size(self):
        with tempfile.TemporaryDirectory(prefix="ai-stuffs-delivery-") as tmp:
            bundle = Path(tmp) / "practice"; (bundle / "text").mkdir(parents=True)
            shutil.copy(BASE / "SHA256SUMS", bundle / "SHA256SUMS")
            shutil.copy(BASE / "text/seed-11-final.pt", bundle / "text/seed-11-final.pt")
            changed = copy.deepcopy(tokenizer); changed["merges"][0]["pair"] = [4, 5]
            (bundle / "text/tokenizer.json").write_text(json.dumps(changed))
            result = subprocess.run([sys.executable, str(ROOT / "scripts/text-inference.py"), "--checkpoint", str(bundle / "text/seed-11-final.pt"), "--tokenizer", str(bundle / "text/tokenizer.json"), "--prompt", "数据"], capture_output=True, text=True, cwd=ROOT)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match the saved bundle identity", result.stderr)


class GeneralizationEvidence(unittest.TestCase):
    def test_group_isolation_and_training_budget_from_all_trace_rows(self):
        splits = read(BASE / "generalization/splits.json")
        evaluation = splits["original"]["test"] + splits["original"]["length"] + splits["extra_length"]
        self.assertFalse({r["group_id"] for r in splits["extended_train"]} & {r["group_id"] for r in evaluation})
        for run in report["reports"]["generalization"]["extended"]:
            trace = read(BASE / f"generalization/seed-{run['seed']}-extended-trace.json")
            by_id = {r["id"]: r for r in splits["extended_train"]}
            tokens = sum(by_id[i]["length"] + 1 for row in trace for i in row["record_ids"])
            self.assertEqual(tokens, run["supervised_tokens"])
            self.assertEqual(len(trace), 1200)
            self.assertGreater(tokens, run["baseline_supervised_tokens"])

    def test_every_length_summary_recomputes_from_all_raw_predictions(self):
        g = report["reports"]["generalization"]
        for run in g["baselines"]:
            for key in ["standard", "eos_diagnostic", "shifted_positions"]:
                self.assertEqual(general.summarize(run[key]), run["summaries"][key])
                for r in run[key]:
                    score = symbolic.score_generation(r["predicted"], r["target"])
                    for field, value in score.items(): self.assertEqual(r[field], value)
                    if key == "eos_diagnostic": self.assertNotIn(symbolic.EOS, r["predicted"][:r["length"]])
        for run in g["extended"]: self.assertEqual(general.summarize(run["predictions"]), run["summary"])

    def test_saved_short_and_extended_weights_match_cached_generation_at_every_length(self):
        for key in ["baselines", "extended"]:
            for run in report["reports"]["generalization"][key]:
                rows = run["standard"] if key == "baselines" else run["predictions"]
                selected = [part[i] for n in range(2, 11) for part in [[r for r in rows if r["length"] == n]] for i in [0, -1]]
                file = ROOT / run["checkpoint"]
                self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), run["checkpoint_sha256"])
                # Our hash-pinned extended run also stores NumPy RNG; it is a trusted local artifact.
                saved = torch.load(file, weights_only=False, map_location="cpu")
                model = symbolic.make_model(seed=run["seed"], **saved["model_config"])
                model.load_state_dict(saved["model"]); model.eval()
                with torch.no_grad(): actual = symbolic.generate_records(model, selected, use_cache=True)
                self.assertEqual([r["predicted"] for r in actual], [r["predicted"] for r in selected])


class ReproductionAndProfile(unittest.TestCase):
    def test_first_difference_is_verified_from_actual_saved_boundary_tensors(self):
        tensors = torch.load(BASE / "reproduction/comparison-tensors.pt", weights_only=True)["first_update"]
        for case in report["reports"]["reproduction"]["cases"]:
            errors = {key: core.state_error(tensors["baseline"][key], tensors[case["case"]][key]) for key in tensors["baseline"]}
            self.assertEqual(errors, case["rows"][0]["errors"])
            first = next((key for key, value in errors.items() if value != 0), None)
            self.assertEqual(first, case["first_difference"]["stage"] if case["first_difference"] else None)
        self.assertEqual(report["reports"]["reproduction"]["historical_difference"]["status"], "unresolved")

    def test_full_model_dense_and_chunked_loss_hidden_and_parameter_gradients(self):
        saved = torch.load(BASE / "text/seed-11-final.pt", weights_only=True)
        models = [core.make_model(11, saved["config"]).double() for _ in range(2)]
        for m in models: m.load_state_dict(saved["model"])
        batch = core.collate([windows["train"][0], windows["train"][10]])
        hidden = [m.hidden(batch["input_ids"]) for m in models]
        for h in hidden: h.retain_grad()
        loss = [F.cross_entropy(models[0].head(hidden[0]).reshape(-1, 387), batch["labels"].reshape(-1), ignore_index=-100), chunked_linear_cross_entropy(hidden[1].reshape(-1, 96), models[1].head.weight, batch["labels"].reshape(-1), token_block=64, vocab_block=128)]
        for value in loss: value.backward()
        self.assertAlmostEqual(loss[0].item(), loss[1].item(), delta=1e-10)
        self.assertLess(core.state_error(hidden[0].grad, hidden[1].grad), 1e-10)
        self.assertLess(max(core.state_error(a.grad, b.grad) for a, b in zip(models[0].parameters(), models[1].parameters())), 1e-9)


if __name__ == "__main__": unittest.main(verbosity=2)
