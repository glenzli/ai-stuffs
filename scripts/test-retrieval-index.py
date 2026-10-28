"""Independent Faiss/NumPy verification; intentionally never imports PyTorch."""
import hashlib
import json
import math
from pathlib import Path
import unittest
import numpy as np
import faiss
from retrieval_index import bm25_scores, score_ranking

ROOT = Path(__file__).resolve().parents[1]; read = lambda file: json.loads(Path(file).read_text())
R = (ROOT / read(ROOT / 'src/experiment-sources.json')['experiments']['retrieval']['report']['path']).parent
report = read(R / 'report.json'); passages = report['passages']; ids = [p['id'] for p in passages]
arrays = np.load(R / 'embeddings.npz'); x, q = arrays['passages'], arrays['queries']; faiss.omp_set_num_threads(1)


class RetrievalIndexEvidence(unittest.TestCase):
    def test_designated_evidence_and_query_group_isolation(self):
        dataset = read(R / 'queries.json'); by_id = {p['id']: p for p in passages}
        self.assertEqual(hashlib.sha256((R / 'queries.json').read_bytes()).hexdigest(), report['dataset']['queries_sha256'])
        self.assertFalse(set(report['dataset']['validation_groups']) & set(report['dataset']['test_groups']))
        self.assertEqual(len(by_id), 877); self.assertEqual(len(dataset['queries']), 36)
        for query in dataset['queries']:
            for gold in query['gold']:
                self.assertIn(gold['evidence'], by_id[gold['passage_id']]['text']); self.assertEqual(gold['source_document'], by_id[gold['passage_id']]['source_document'])

    def test_bm25_known_tiny_reference_and_saved_complete_score_matrix(self):
        value = bm25_scores([{'text': '甲甲乙'}, {'text': '乙丙'}], [{'query': '甲'}])
        expected = math.log(2) * 2 * 2.2 / (2 + 1.2 * (.25 + .75 * 3 / 2.5))
        self.assertAlmostEqual(value[0, 0], expected, delta=1e-13); self.assertEqual(value[0, 1], 0)
        actual = bm25_scores(passages, report['queries']); saved = np.load(R / 'bm25-scores.npy')
        self.assertTrue(np.array_equal(actual, saved))

    def test_flat_index_matches_normalized_numpy_dot_products_and_every_saved_ranking(self):
        self.assertTrue(np.allclose(np.linalg.norm(x, axis=1), 1., atol=1e-6)); self.assertEqual(x.shape, (877, 512))
        dot = q.astype(np.float64) @ x.astype(np.float64).T
        flat = faiss.read_index(str(R / 'flat.faiss')); scores, indexes = flat.search(q, 10)
        for i, query in enumerate(report['queries']):
            expected = query['methods']['flat']['results']
            self.assertEqual([ids[j] for j in indexes[i]], [r['passage_id'] for r in expected])
            self.assertLess(float(np.max(np.abs(scores[i] - dot[i, indexes[i]]))), 1e-6)
            self.assertGreaterEqual(float(scores[i, -1]), float(np.partition(dot[i], -10)[-10]) - 1e-6)

    def test_each_actual_ivf_probe_and_complete_probe_match_saved_results(self):
        ivf = faiss.read_index(str(R / 'ivf.faiss')); flat = faiss.read_index(str(R / 'flat.faiss')).search(q, 10)[1]
        for nprobe in [1, 4, 16]:
            ivf.nprobe = nprobe; scores, indexes = ivf.search(q, 10)
            for i, query in enumerate(report['queries']):
                saved = query['methods'][f'ivf-{nprobe}']
                self.assertEqual([ids[j] for j in indexes[i]], [r['passage_id'] for r in saved['results']])
                self.assertTrue(np.allclose(scores[i], [r['score'] for r in saved['results']], atol=1e-7, rtol=0))
                self.assertEqual(len(set(indexes[i]) & set(flat[i])) / 10, saved['flat_neighbor_recall_at_10'])
                if nprobe == 16: self.assertEqual(set(indexes[i]), set(flat[i]))

    def test_rrf_recomputes_from_ranks_and_never_adds_a_new_candidate(self):
        lexical = np.load(R / 'bm25-scores.npy')
        for i, query in enumerate(report['queries']):
            order = sorted(range(len(ids)), key=lambda j: (-lexical[i, j], ids[j])); bmrank = {ids[j]: rank + 1 for rank, j in enumerate(order)}
            flat = query['methods']['flat']['results']; scores = {r['passage_id']: 1 / (60 + rank + 1) + 1 / (60 + bmrank[r['passage_id']]) for rank, r in enumerate(flat)}
            expected = sorted(scores, key=lambda name: (-scores[name], name)); actual = query['methods']['rrf-rerank']['results']
            self.assertEqual(expected, [r['passage_id'] for r in actual]); self.assertEqual(set(expected), {r['passage_id'] for r in flat})
            for row in actual: self.assertAlmostEqual(row['score'], scores[row['passage_id']], delta=1e-15)

    def test_all_metrics_and_slice_summaries_recompute_from_complete_candidate_ids(self):
        for query in report['queries']:
            gold = {r['passage_id'] for r in query['gold']}
            for method in query['methods'].values():
                self.assertEqual(method['metrics'], score_ranking([r['passage_id'] for r in method['results']], gold))
        for summary in report['summaries']:
            selected = [r for r in report['queries'] if r['split'] == summary['split'] and (summary['slice'] == 'all' or r['kind'] == summary['slice'])]
            self.assertEqual(len(selected), summary['questions'])
            for key, expected in summary['metrics'].items(): self.assertEqual(sum(r['methods'][summary['method']]['metrics'][key] for r in selected) / len(selected), expected)
        self.assertEqual(score_ranking(['unknown'], {'gold'})['mrr_at_10'], 0)
        self.assertEqual(score_ranking(['gold'], {'gold'})['hit_at_1'], 1)


if __name__ == '__main__': unittest.main(verbosity=2)
