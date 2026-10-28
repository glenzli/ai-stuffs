"""Offline tokenizer and actual model re-encoding; never imports Faiss."""
import hashlib
import json
import os
from pathlib import Path
import unittest
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[1]; read = lambda p: json.loads(Path(p).read_text())
R = (ROOT / read(ROOT / 'src/experiment-sources.json')['experiments']['retrieval']['report']['path']).parent
report = read(R / 'report.json'); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)


class EncoderEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(R / 'model', local_files_only=True, trust_remote_code=False)

    def test_every_model_file_matches_recorded_and_publisher_weight_identity(self):
        identity = report['model']; self.assertEqual(identity['revision'], '7999e1d3359715c523056ef9478215996d62a620')
        for row in identity['files']:
            file = R / 'model' / row['file']; actual = hashlib.sha256(file.read_bytes()).hexdigest()
            self.assertEqual(actual, row['sha256']); self.assertEqual(file.stat().st_size, row['bytes'])
            if row['publisher_lfs_sha256']: self.assertEqual(actual, row['publisher_lfs_sha256'])

    def test_all_saved_token_ids_and_truncation_counts_are_reproducible(self):
        encoding = read(R / 'encoding.json'); prefix = report['frozen']['query_instruction']
        for kind, texts in [('passages', [r['text'] for r in report['passages']]), ('queries', [prefix + r['query'] for r in report['queries']])]:
            raw = self.tokenizer(texts, add_special_tokens=True, truncation=False)['input_ids']
            clipped = self.tokenizer(texts, add_special_tokens=True, truncation=True, max_length=512)['input_ids']
            for a, b, expected in zip(raw, clipped, encoding[kind]):
                self.assertEqual(b, expected['input_ids']); self.assertEqual(len(a), expected['raw_tokens']); self.assertEqual(len(b), expected['encoded_tokens']); self.assertEqual(len(a) > 512, expected['truncated'])
        self.assertEqual(report['dataset']['truncated_passages'], 0); self.assertEqual(report['dataset']['truncated_queries'], 0)

    def test_actual_pinned_encoder_reproduces_saved_vectors_with_declared_float_tolerance(self):
        model = AutoModel.from_pretrained(R / 'model', local_files_only=True, trust_remote_code=False, attn_implementation='eager', use_safetensors=True).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 23953920)
        arrays = np.load(R / 'embeddings.npz'); prefix = report['frozen']['query_instruction']
        for kind, indices, texts in [('passages', [0, 201, 444, 876], [r['text'] for r in report['passages']]), ('queries', [0, 12, 29], [prefix + r['query'] for r in report['queries']])]:
            batch = self.tokenizer([texts[i] for i in indices], padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.inference_mode(): actual = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0], p=2, dim=1).numpy()
            self.assertLess(float(np.max(np.abs(actual - arrays[kind][indices]))), 3e-6)


if __name__ == '__main__': unittest.main(verbosity=2)
