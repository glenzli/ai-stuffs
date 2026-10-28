"""Independent per-record autograd weighting and complete replay of saved update gradients."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import learning_core as symbolic
import text_training_core as core
import accumulation_lab as lab

ROOT = Path(__file__).resolve().parents[1]
ENTRY = json.loads((ROOT / 'src/experiment-sources.json').read_text())['experiments']['accumulation']
BASE = (ROOT / ENTRY['report']['path']).parent
REPORT = json.loads((BASE / 'report.json').read_text())
SAVED = torch.load(BASE / 'inputs/seed-11-final.pt', weights_only=True, map_location='cpu')
symbolic.configure_cpu(1)


class AccumulationEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gradients = {}; cls.per_record = {}; cls.losses = {}
        for case in REPORT['frozen']['cases']:
            rows = lab.masked_records(REPORT['inputs']['records'], case); grads = []; losses = []
            for row in rows:
                model = core.make_model(SAVED['seed'], SAVED['config']).double().eval(); model.load_state_dict(SAVED['model'])
                ids = torch.tensor([row['input']]); target = torch.tensor([row['target']]); valid = target != -100
                logp = F.log_softmax(model(ids), -1); nll = -logp.gather(-1, target.clamp_min(0)[..., None]).squeeze(-1)
                value = (nll * valid).sum(); value.backward(); grads.append(lab.flat_gradient(model)); losses.append(float(value.detach()))
            cls.per_record[case['id']] = grads; cls.losses[case['id']] = losses
        for row in REPORT['rows']:
            file = BASE / row['gradient']['file']; cls.gradients[(row['case'], row['method'])] = np.load(file, allow_pickle=False)

    def test_frozen_inputs_match_original_full_windows_and_checkpoint_bytes(self):
        for name, digest in REPORT['inputs']['source'].items(): self.assertEqual(hashlib.sha256((BASE / 'inputs' / name).read_bytes()).hexdigest(), digest)
        windows = json.loads((BASE / 'inputs/windows.json').read_text())
        self.assertEqual(REPORT['inputs']['records'], [r for r in windows['train'] if len(r['target']) == 96][:4])
        for case in REPORT['frozen']['cases']:
            masked = lab.masked_records(REPORT['inputs']['records'], case)
            self.assertEqual([sum(y != -100 for y in row['target']) for row in masked], case['keep'])
            for actual, source in zip(masked, REPORT['inputs']['records']): self.assertEqual(actual['input'], source['input'])

    def test_all_saved_gradients_match_independently_weighted_per_record_autograd(self):
        for case in REPORT['frozen']['cases']:
            case_id = case['id']; g = self.per_record[case_id]; counts = case['keep']; total = sum(counts)
            global_grad = np.sum(g, axis=0) / total if total else np.zeros_like(g[0])
            for row in [r for r in REPORT['rows'] if r['case'] == case_id]:
                method = row['method']
                if method in ('reference', 'global-count', 'ddp-global'): expected = global_grad
                elif method == 'zero-step': expected = np.zeros_like(g[0])
                elif method in ('micro-means', 'planned-divisor'):
                    divisor = len(case['chunks']) if method == 'micro-means' else case['planned']
                    expected = sum(np.sum([g[i] for i in chunk], axis=0) / (sum(counts[i] for i in chunk) * divisor) for chunk in case['chunks'])
                else:
                    expected = np.zeros_like(g[0])
                    for rank in (0, 1):
                        local = [i for i in range(len(g)) if i % 2 == rank]; count = sum(counts[i] for i in local)
                        if count: expected += np.sum([g[i] for i in local], axis=0) / (2 * count)
                actual = np.load(BASE / row['gradient']['file'], allow_pickle=False)
                np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0, err_msg=case_id + '/' + method)

    def test_loss_scalars_and_counts_match_the_declared_objective_including_the_tail(self):
        for row in REPORT['rows']:
            case = next(c for c in REPORT['frozen']['cases'] if c['id'] == row['case']); counts = case['keep']; sums = self.losses[row['case']]
            self.assertEqual(row['global_labels'], sum(counts))
            if row['execution'] == 'ddp': continue
            if not row['backward_calls']: self.assertIsNone(row['scaled_loss']); continue
            method = row['method']
            if method in ('reference', 'global-count'): expected = sum(sums) / sum(counts)
            elif method == 'zero-step': expected = 0.
            else:
                divisor = len(case['chunks']) if method == 'micro-means' else case['planned']
                expected = sum(sum(sums[i] for i in chunk) / (sum(counts[i] for i in chunk) * divisor) for chunk in case['chunks'])
            self.assertAlmostEqual(row['scaled_loss'], expected, delta=2e-12)

    def test_every_parameter_and_optimizer_state_hash_replays_from_saved_actual_gradients(self):
        outputs = {}
        for row in REPORT['rows']:
            model, optimizer = lab.new_model_optimizer(SAVED, row['optimizer']); initial = lab.flat_model(model)
            gradient = np.load(BASE / row['gradient']['file'], allow_pickle=False); cursor = 0
            self.assertEqual(lab.state_sha(optimizer.state_dict()), row['initial_state_sha256'])
            if row['optimizer_step']:
                for param in model.parameters():
                    size = param.numel(); param.grad = torch.from_numpy(gradient[cursor:cursor+size].copy()).reshape_as(param); cursor += size
                self.assertEqual(cursor, 298176); optimizer.step()
            parameters = lab.flat_model(model)
            self.assertEqual(lab.array_sha(parameters), row['parameters_sha256'])
            self.assertEqual(lab.state_sha(optimizer.state_dict()), row['final_state_sha256'])
            outputs[(row['case'], row['optimizer'], row['method'])] = (parameters, copy.deepcopy(optimizer.state_dict()))
            if row['method'] == 'global-count' and not row['global_labels']: np.testing.assert_array_equal(initial, parameters)
        for row in REPORT['rows']:
            p, state = outputs[(row['case'], row['optimizer'], row['method'])]; reference, expected_state = outputs[(row['case'], row['optimizer'], 'reference')]
            self.assertAlmostEqual(float(np.max(np.abs(p - reference))), row['parameter_max_error'], delta=1e-15)
            self.assertAlmostEqual(core.state_error(state, expected_state), row['optimizer_state_max_error'], delta=1e-15)

    def test_sgd_parameter_changes_follow_the_independent_linear_update_formula(self):
        model, _ = lab.new_model_optimizer(SAVED, 'sgd'); initial = lab.flat_model(model)
        for row in [r for r in REPORT['rows'] if r['optimizer'] == 'sgd' and r['optimizer_step']]:
            g = np.load(BASE / row['gradient']['file'], allow_pickle=False)
            expected = initial - .01 * (g + .01 * initial)
            if row['execution'] == 'ddp':
                actual = torch.load(BASE / row['update_file'], weights_only=True, map_location='cpu')['parameters'].numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
            ref = next(r for r in REPORT['rows'] if r['case'] == row['case'] and r['optimizer'] == 'sgd' and r['method'] == 'reference')
            refg = np.load(BASE / ref['gradient']['file'], allow_pickle=False)
            expected_ref = initial - .01 * (refg + .01 * initial) if ref['optimizer_step'] else initial
            self.assertAlmostEqual(float(np.max(np.abs(expected - expected_ref))), row['parameter_max_error'], delta=1e-15)

    def test_both_actual_gloo_ranks_preserve_the_same_gradients_outputs_and_collective_counts(self):
        ranks = [json.loads((BASE / f'rank-{r}.json').read_text()) for r in (0, 1)]
        self.assertNotEqual(ranks[0]['pid'], ranks[1]['pid'])
        for row in ranks[0]['rows']:
            peer = next(r for r in ranks[1]['rows'] if (r['case'], r['optimizer'], r['method']) == (row['case'], row['optimizer'], row['method']))
            case = next(c for c in REPORT['frozen']['cases'] if c['id'] == row['case'])
            self.assertEqual(row['local_labels'] + peer['local_labels'], row['global_labels'])
            self.assertEqual(row['global_labels'], sum(case['keep']))
            self.assertEqual(row['contexts'], ['no_sync: forward and backward', 'synchronize'])
            self.assertEqual(peer['contexts'], row['contexts'])
            for record in (row, peer):
                self.assertEqual(hashlib.sha256((BASE / record['gradient']['file']).read_bytes()).hexdigest(), record['gradient']['sha256'])
                self.assertEqual(record['replica_digests'][0]['gradient'], record['replica_digests'][1]['gradient'])
                self.assertEqual(record['replica_digests'][0]['parameters'], record['replica_digests'][1]['parameters'])
                actual = torch.load(BASE / record['update_file'], weights_only=True, map_location='cpu')
                self.assertEqual(lab.array_sha(actual['parameters'].numpy()), record['parameters_sha256'])
                self.assertEqual(lab.state_sha(actual['optimizer']), record['final_state_sha256'])
            np.testing.assert_array_equal(np.load(BASE / row['gradient']['file']), np.load(BASE / peer['gradient']['file']))


if __name__ == '__main__': unittest.main(verbosity=2)
