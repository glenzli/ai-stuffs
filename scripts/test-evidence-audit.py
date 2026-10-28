"""Independent checks of whole-unit resampling, real candidate fusion and saved LM losses."""
import hashlib
import json
import math
from pathlib import Path
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import learning_core as symbolic
import text_training_core as core
from paired_evidence import bootstrap_ratio
from retrieval_comparison import fuse_lists

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'src/experiment-sources.json').read_text())['experiments']
RDIR = (ROOT / MANIFEST['retrieval-comparison']['report']['path']).parent
TDIR = (ROOT / MANIFEST['text-evaluation-audit']['report']['path']).parent
RETRIEVAL = json.loads((RDIR / 'report.json').read_text())
TEXT = json.loads((TDIR / 'report.json').read_text())
symbolic.configure_cpu(1)


def draws(count, repetitions=2000):
    state = 73; values = []
    for _ in range(count * repetitions):
        state = (1664525 * state + 1013904223) & 0xffffffff
        values.append((state * count) // 2**32)
    return np.array(values, dtype=np.int32).reshape(repetitions, count)


class UnitResampling(unittest.TestCase):
    def test_unequal_units_keep_the_sampled_denominator_and_match_independent_numpy(self):
        units = [(2., 2), (-3., 10), (.7, 1)]
        actual = bootstrap_ratio(units); indexes = draws(3)
        totals = np.array([r[0] for r in units]); counts = np.array([r[1] for r in units])
        expected = np.sort(totals[indexes].sum(1) / counts[indexes].sum(1))
        np.testing.assert_allclose(actual['distribution'], expected, atol=1e-15, rtol=0)
        np.testing.assert_allclose(actual['interval'], np.quantile(expected, [.025, .975], method='linear'), atol=1e-15, rtol=0)
        self.assertAlmostEqual(actual['estimate'], -.3 / 13)
        self.assertNotAlmostEqual(actual['estimate'], np.mean(totals / counts))

    def test_empty_nonfinite_and_single_unit_states_are_explicit(self):
        for units in ([], [(1., 0)], [(math.nan, 1)], [(1., 1.2)]):
            with self.assertRaises(ValueError): bootstrap_ratio(units)
        self.assertEqual(bootstrap_ratio([(0., 4)])['interval'], [0., 0.])
        with self.assertRaises(ValueError): bootstrap_ratio([(0., 1)], random_seed=-1)


class RetrievalFusion(unittest.TestCase):
    def test_tiny_hand_computed_rank_fusion_and_missing_contributions(self):
        rows = fuse_lists(['b', 'a'], ['c', 'a'])
        self.assertEqual([r['passage_id'] for r in rows], ['a', 'b', 'c'])
        self.assertAlmostEqual(rows[0]['score'], 2 / 62)
        self.assertAlmostEqual(rows[1]['score'], 1 / 61)
        self.assertIsNone(rows[1]['flat_rank']); self.assertIsNone(rows[2]['bm25_rank'])
        with self.assertRaises(ValueError): fuse_lists(['a', 'a'], ['a'])

    def test_all_candidates_recompute_from_unchanged_real_source_lists(self):
        source = json.loads((RDIR / 'retrieval-source.json').read_text()); by_id = {r['id']: r for r in source['queries']}
        self.assertEqual(hashlib.sha256((RDIR / 'retrieval-source.json').read_bytes()).hexdigest(), MANIFEST['retrieval']['report']['sha256'])
        for query in RETRIEVAL['queries']:
            prior = by_id[query['id']]; lists = [[r['passage_id'] for r in prior['methods'][key]['results']] for key in ('bm25', 'flat')]
            union = sorted(set(lists[0]) | set(lists[1])); ranks = np.zeros((len(union), 2), dtype=np.int32)
            for column, values in enumerate(lists):
                for row, item in enumerate(union): ranks[row, column] = values.index(item) + 1 if item in values else 0
            scores = np.where(ranks > 0, 1 / (60 + ranks), 0).sum(1)
            order = sorted(range(len(union)), key=lambda i: (-scores[i], union[i]))
            self.assertEqual([r['passage_id'] for r in query['candidates']], [union[i] for i in order])
            np.testing.assert_allclose([r['score'] for r in query['candidates']], scores[order], atol=1e-16, rtol=0)
            self.assertEqual(query['gold'], prior['gold']); self.assertEqual(query['group'], prior['group'])

    def test_every_slice_metric_and_coverage_bound_uses_all_its_saved_questions(self):
        per_query = {}
        for query in RETRIEVAL['queries']:
            gold = {r['passage_id'] for r in query['gold']}; ids = [r['passage_id'] for r in query['candidates']]
            pos = next((i + 1 for i, item in enumerate(ids[:10]) if item in gold), None)
            expected = {'hit_at_1': int(pos is not None and pos <= 1), 'hit_at_5': int(pos is not None and pos <= 5), 'hit_at_10': int(pos is not None), 'mrr_at_10': 0 if pos is None else 1 / pos}
            self.assertEqual(expected, query['methods']['hybrid']); per_query[query['id']] = expected
            self.assertEqual(query['candidate_coverage'], int(bool(gold & set(ids))))
        for row in RETRIEVAL['summaries']:
            queries = [q for q in RETRIEVAL['queries'] if q['split'] == row['split'] and (row['kind'] == 'all' or q['kind'] == row['kind'])]
            self.assertEqual(row['questions'], len(queries)); self.assertEqual(row['groups'], len({q['group'] for q in queries}))
            for key in per_query[queries[0]['id']]: self.assertAlmostEqual(row['methods']['hybrid'][key], np.mean([per_query[q['id']][key] for q in queries]))
            self.assertAlmostEqual(row['candidate_coverage'], np.mean([q['candidate_coverage'] for q in queries]))
            self.assertGreaterEqual(row['candidate_coverage'], row['methods']['hybrid']['hit_at_10'])


class DocumentLossEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows = json.loads((TDIR / 'inputs/windows.json').read_text()); cls.scores = np.load(TDIR / 'position-scores.npz')
        cls.distributions = json.loads((TDIR / 'bootstrap-distributions.json').read_text())

    def test_every_position_and_prediction_recomputes_from_saved_weights_with_different_batches(self):
        for seed in (11, 29, 47):
            file = TDIR / f'inputs/seed-{seed}-final.pt'
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), TEXT['inputs'][file.name])
            saved = torch.load(file, weights_only=True, map_location='cpu'); model = core.make_model(seed, saved['config']).eval(); model.load_state_dict(saved['model'])
            for split, records in self.windows.items():
                losses = []; predicted = []
                with torch.no_grad():
                    for start in range(0, len(records), 13):
                        batch = core.collate(records[start:start+13]); logits = model(batch['input_ids']); valid = batch['labels'] != -100
                        logp = F.log_softmax(logits, -1); safe = batch['labels'].clamp_min(0)
                        losses.extend((-logp.gather(-1, safe[..., None]).squeeze(-1))[valid].double().tolist())
                        predicted.extend(logits.argmax(-1)[valid].tolist())
                np.testing.assert_allclose(losses, self.scores[f'seed{seed}_{split}_nll'], atol=3e-6, rtol=0)
                np.testing.assert_array_equal(predicted, self.scores[f'seed{seed}_{split}_prediction'])

    def test_document_sums_weighted_means_and_bigram_baseline_match_independent_probability_matrix(self):
        counts = np.zeros((387, 387), dtype=np.int64)
        for row in self.windows['train']: np.add.at(counts, (row['input'], row['target']), 1)
        baseline = json.loads((TDIR / 'inputs/language-baselines.json').read_text()); alpha = next(r for r in baseline['models'] if r['model'] == 'bigram')['selected_alpha']
        probability = (counts + alpha) / (counts.sum(1)[:, None] + 387 * alpha)
        for run in TEXT['runs']:
            split = run['split']; losses = self.scores[f"seed{run['seed']}_{split}_nll"]; predicted = self.scores[f"seed{run['seed']}_{split}_prediction"]
            expected = {}; offset = 0
            for record in self.windows[split]:
                size = len(record['target']); row = expected.setdefault(record['document'], {'tokens': 0, 'nll': 0., 'old': 0., 'correct': 0})
                row['tokens'] += size; row['nll'] += losses[offset:offset+size].sum(); row['old'] += -np.log(probability[record['input'], record['target']]).sum()
                row['correct'] += (predicted[offset:offset+size] == record['target']).sum(); offset += size
            for row in run['documents']:
                item = expected[row['document']]; self.assertEqual(row['tokens'], item['tokens']); self.assertEqual(row['correct'], item['correct'])
                self.assertAlmostEqual(row['loss_sum'], item['nll'], delta=1e-9); self.assertAlmostEqual(row['bigram_loss_sum'], item['old'], delta=1e-8)
                self.assertAlmostEqual(row['ce_nats_per_token'], item['nll'] / item['tokens'], delta=1e-12)
            self.assertEqual(offset, run['supervised_tokens']); self.assertEqual(len(losses), offset)
            self.assertAlmostEqual(run['micro_ce'], sum(x['nll'] for x in expected.values()) / offset, delta=1e-12)
            self.assertAlmostEqual(run['macro_ce'], np.mean([x['nll'] / x['tokens'] for x in expected.values()]), delta=1e-12)

    def test_whole_document_resampling_recomputes_the_label_denominator_for_every_seed_and_split(self):
        for run in TEXT['runs']:
            rows = run['documents']; sample = draws(len(rows)); totals = np.array([r['bigram_loss_sum'] - r['loss_sum'] for r in rows]); counts = np.array([r['tokens'] for r in rows])
            expected = np.sort(totals[sample].sum(1) / counts[sample].sum(1)); saved = self.distributions[f"seed{run['seed']}_{run['split']}"]
            np.testing.assert_allclose(saved, expected, atol=1e-13, rtol=0)
            np.testing.assert_allclose(run['paired_gain']['interval'], np.quantile(expected, [.025, .975], method='linear'), atol=1e-13, rtol=0)
            self.assertEqual(run['paired_gain']['units'], len(rows)); self.assertEqual(run['paired_gain']['denominator'], sum(counts))


if __name__ == '__main__': unittest.main(verbosity=2)
