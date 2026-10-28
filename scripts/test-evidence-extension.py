"""Independent audits of saved count baselines and equal-label-budget training."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import torch
import learning_core as core
from generalization_lab import loaded_model
from matched_budget_lab import limit_labels

ROOT = Path(__file__).resolve().parents[1]
read = lambda file: json.loads(Path(file).read_text())
manifest = read(ROOT / 'src/experiment-sources.json')
B = (ROOT / manifest['experiments']['language-baselines']['report']['path']).parent
M = (ROOT / manifest['experiments']['matched-budget']['report']['path']).parent
baseline, matched = read(B / 'report.json'), read(M / 'report.json')
core.configure_cpu(1)


class CountEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows = read(B / 'inputs/windows.json'); cls.v = baseline['vocab_size']
        cls.unigram = np.zeros(cls.v, dtype=np.int64); cls.bigram = np.zeros((cls.v, cls.v), dtype=np.int64)
        for row in cls.windows['train']:
            np.add.at(cls.unigram, row['target'], 1); np.add.at(cls.bigram, (row['input'], row['target']), 1)

    def test_count_matrices_are_exactly_train_only(self):
        counts = read(B / 'counts.json')
        for i in range(self.v):
            self.assertEqual(self.unigram[i], counts['unigram'].get(str(i), 0))
            for j in range(self.v): self.assertEqual(self.bigram[i, j], counts['bigram'].get(str(i), {}).get(str(j), 0))
        self.assertEqual(int(self.unigram.sum()), 62977)
        self.assertEqual(int(self.unigram[0] + self.unigram[1]), 0)
        self.assertEqual(int(self.unigram[2]), 24)  # exactly one EOS per training document

    def test_numpy_probability_matrices_reproduce_all_split_losses_and_accuracies(self):
        for model in baseline['models']:
            alpha = model['selected_alpha']; counts = self.unigram if model['model'] == 'unigram' else self.bigram
            probabilities = (counts + alpha) / (counts.sum(axis=-1, keepdims=True) + alpha * self.v)
            self.assertTrue(np.allclose(probabilities.sum(axis=-1), 1.))
            for split, rows in self.windows.items():
                inputs = np.array([x for r in rows for x in r['input']]); targets = np.array([y for r in rows for y in r['target']])
                values = probabilities if model['model'] == 'unigram' else probabilities[inputs]
                p = values[targets] if values.ndim == 1 else values[np.arange(len(targets)), targets]
                prediction = np.full_like(targets, values.argmax()) if values.ndim == 1 else values.argmax(axis=-1)
                saved = model['evaluation'][split]
                self.assertAlmostEqual(float(-np.log(p).mean()), saved['ce_nats_per_token'], delta=1e-11)
                self.assertEqual(float((prediction == targets).mean()), saved['teacher_forced_accuracy'])
                self.assertEqual(len(targets), saved['supervised_tokens'])

    def test_alpha_is_selected_from_validation_without_reading_test_metrics(self):
        for row in baseline['models']:
            selected = min(row['selection_trace'], key=lambda r: r['validation']['ce_nats_per_token'])
            self.assertEqual(selected['alpha'], row['selected_alpha'])
            self.assertEqual([r['alpha'] for r in row['selection_trace']], [.01, .1, 1.])
            self.assertEqual(selected['validation'], row['evaluation']['validation'])

    def test_all_baseline_documents_and_sums_match_the_global_denominator(self):
        for row in baseline['models']:
            for split in self.windows:
                stats = row['evaluation'][split]; parts = stats['documents'].values()
                self.assertEqual(sum(p['tokens'] for p in parts), stats['supervised_tokens'])
                self.assertAlmostEqual(sum(p['loss_sum'] for p in parts) / stats['supervised_tokens'], stats['ce_nats_per_token'], delta=1e-11)
                self.assertEqual(set(stats['documents']), {r['document'] for r in self.windows[split]})

    def test_transformer_comparator_is_the_unchanged_complete_prior_report(self):
        source = baseline['source']['report.json']; data = (ROOT / source['path']).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), source['sha256'])
        self.assertEqual(json.loads(data)['runs'], baseline['transformer'])


class BudgetEvidence(unittest.TestCase):
    def test_last_batch_truncates_labels_once_in_order_without_changing_input(self):
        source = torch.tensor([[-100, 4, 5, 3], [-100, -100, 6, 3]])
        actual, kept, dropped = limit_labels(source, 4)
        self.assertEqual(actual.tolist(), [[-100, 4, 5, 3], [-100, -100, 6, -100]])
        self.assertEqual((kept, dropped), (4, 1)); self.assertEqual(source[1, -1].item(), 3)
        with self.assertRaises(ValueError): limit_labels(source, 0)

    def test_group_isolation_and_initial_identity_are_true_for_every_seed(self):
        splits = read(M / 'splits.json'); evaluation = {r['group_id'] for r in splits['evaluation']}
        for records in splits['populations'].values(): self.assertFalse(evaluation & {r['group_id'] for r in records})
        for seed in [11, 29, 47]:
            runs = [r for r in matched['runs'] if r['seed'] == seed]
            self.assertEqual(len(runs), 2); self.assertEqual(runs[0]['initial_sha256'], runs[1]['initial_sha256'])

    def test_all_step_inputs_reproduce_exact_consumption_and_computed_position_counts(self):
        splits = read(M / 'splits.json')
        for run in matched['runs']:
            records = {r['id']: r for r in splits['populations'][run['support']]}
            trace = read(M / f"seed-{run['seed']}-{run['support']}-trace.json"); consumed = inputs = computed = 0
            self.assertEqual(len(trace), run['updates'])
            for row in trace:
                batch = [records[name] for name in row['record_ids']]; full = sum(r['length'] + 1 for r in batch)
                consumed += full - row['discarded_final_labels']; inputs += 2 * full; computed += len(batch) * (2 * max(r['length'] for r in batch) + 2)
                self.assertEqual(row['supervised_tokens'], consumed); self.assertEqual(row['input_tokens'], inputs); self.assertEqual(row['computed_positions'], computed)
                if row is not trace[-1]: self.assertEqual(row['discarded_final_labels'], 0)
            self.assertEqual(consumed, 410000); self.assertEqual(inputs, run['input_tokens']); self.assertEqual(computed, run['computed_positions'])

    def test_saved_final_weights_reproduce_every_unconstrained_prediction_using_kv_cache(self):
        records = read(M / 'splits.json')['evaluation']
        for run in matched['runs']:
            file = M / run['checkpoint']; self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), run['checkpoint_sha256'])
            model = loaded_model(file)
            with torch.no_grad(): actual = core.generate_records(model, records, use_cache=True)
            expected = {r['id']: r for r in run['predictions']}
            for row in actual: self.assertEqual(row['predicted'], expected[row['id']]['predicted'])

    def test_length_summaries_recompute_without_omitting_failed_seeds(self):
        self.assertEqual([(r['seed'], r['support']) for r in matched['runs']], [(s, name) for s in [11, 29, 47] for name in ['short-2-5', 'extended-2-8']])
        for run in matched['runs']:
            for row in run['summary']:
                selected = [p for p in run['predictions'] if p['length'] == row['length']]
                exact = [p['predicted'] == p['target'] + [core.EOS] for p in selected]
                self.assertEqual(row['records'], len(selected)); self.assertEqual(row['exact_match'], sum(exact) / len(exact))
                self.assertEqual(row['early_eos'], sum(core.EOS in p['predicted'][:p['length']] for p in selected))


if __name__ == '__main__': unittest.main(verbosity=2)
