import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { hash } from './experiment-bundles.mjs';
import { retrievalComparison } from '../src/scripts/features/retrieval-comparison-data.js';
import { textEvaluationAudit } from '../src/scripts/features/text-evaluation-audit-data.js';
import { retrievalComparisonView } from '../src/scripts/features/retrieval-comparison.js';
import { textAuditView } from '../src/scripts/features/text-evaluation-audit.js';
import { pairedExample, pairedStatistics, validatePairedData } from '../src/scripts/features/paired-evaluation.js';

const manifest = JSON.parse(await readFile('src/experiment-sources.json', 'utf8'));
const read = async file => JSON.parse(await readFile(file, 'utf8'));
test('both follow-up page projections match complete reports and their executed source copies', async () => {
  for (const [id, snapshot] of [['retrieval-comparison', retrievalComparison], ['text-evaluation-audit', textEvaluationAudit]]) {
    const entry = manifest.experiments[id], full = await read(entry.report.path), { provenance, ...projection } = snapshot;
    assert.deepEqual(projection, full); assert.equal(hash(await readFile(entry.report.path)), provenance.report_sha256);
    assert.equal(hash(await readFile(entry.projection.target)), entry.projection.sha256);
    for (const [file, sha] of Object.entries(provenance.source_sha256)) {
      assert.equal(hash(await readFile(file)), sha, file); assert.equal(hash(await readFile(path.join(entry.bundle_root, 'source', file))), sha);
    }
  }
});
test('every exported real paired dataset reproduces all 2000 saved Python resamples in the actual browser statistics owner', async () => {
  const directory = path.dirname(manifest.experiments['retrieval-comparison'].report.path);
  const saved = await read(directory + '/bootstrap-distributions.json');
  for (const [key, data] of Object.entries(retrievalComparison.paired_exports)) {
    assert.deepEqual(validatePairedData(data).rows.map(({ seed, ...row }) => row), data.rows);
    for (const mode of ['group', 'row']) {
      const actual = pairedStatistics({ data, mode, repetitions: 2000, randomSeed: 73 });
      const expected = saved[key + '-' + mode]; assert.equal(actual.distribution.length, 2000);
      for (let i = 0; i < expected.length; i++) assert.ok(Math.abs(actual.distribution[i] - expected[i]) < 1e-12, key + '/' + mode + '/' + i);
      const [split, baseline, ...metricParts] = key.split('-'), metric = metricParts.join('-');
      const row = retrievalComparison.comparisons.find(r => r.split === split && r.baseline === baseline && r.metric === metric).analyses[mode];
      assert.ok(Math.abs(actual.delta - row.estimate) < 1e-12); assert.equal(actual.units, row.units);
      actual.interval.forEach((value, i) => assert.ok(Math.abs(value - row.interval[i]) < 1e-12));
    }
  }
});
test('the real preset contains unchanged test questions and keeps paired source groups intact', () => {
  const data = pairedExample('retrieval-comparison'); assert.equal(data.rows.length, 24);
  assert.equal(new Set(data.rows.map(r => r.group)).size, 12); assert.deepEqual([...new Set(data.rows.map(r => r.seed))], ['default']);
  for (const row of data.rows) {
    const q = retrievalComparison.queries.find(q => q.id === row.id); assert.equal(q.split, 'test'); assert.equal(row.group, q.group);
    assert.equal(row.old, q.methods.flat.mrr_at_10); assert.equal(row.new, q.methods.hybrid.mrr_at_10);
  }
  const copy = pairedExample('retrieval-comparison'); copy.rows[0].new = -1; assert.ok(pairedExample('retrieval-comparison').rows[0].new >= 0);
});
test('candidate coverage, final ranking and old versus new fusion definitions remain distinct', () => {
  for (const q of retrievalComparison.queries) {
    const union = new Set([...q.bm25_ids, ...q.flat_ids]); assert.equal(union.size, q.candidate_count);
    assert.deepEqual(new Set(q.candidates.map(r => r.passage_id)), union);
    assert.ok(q.candidate_coverage >= q.methods.hybrid.hit_at_10);
  }
  assert.equal(retrievalComparison.frozen.missing_rank_contribution, 0);
  assert.match(retrievalComparison.frozen.old_reranker, /global BM25 ranks/);
  const test = retrievalComparison.summaries.find(r => r.split === 'test' && r.kind === 'all');
  assert.ok(test.methods.hybrid.mrr_at_10 < test.methods.bm25.mrr_at_10); // Preserve the unfavorable observed result.
});
test('text document views conserve label counts and preserve separate weighting targets and seeds', () => {
  for (const seed of [11, 29, 47]) for (const split of ['train', 'validation', 'test']) {
    const view = textAuditView(textEvaluationAudit, seed, split);
    const count = view.documents.reduce((n, row) => n + row.tokens, 0);
    assert.equal(count, view.supervised_tokens);
    const micro = view.documents.reduce((sum, row) => sum + row.loss_sum, 0) / count;
    const macro = view.documents.reduce((sum, row) => sum + row.ce_nats_per_token, 0) / view.documents.length;
    assert.ok(Math.abs(micro - view.micro_ce) < 1e-12); assert.ok(Math.abs(macro - view.macro_ce) < 1e-12);
    assert.equal(view.paired_gain.units, view.documents.length);
    assert.equal(view.paired_gain.denominator, count); assert.ok(view.prior_aggregate_error < 2e-6);
  }
});
test('follow-up view contracts reject unknown selections and missing target denominators', () => {
  const id = retrievalComparison.queries[0].id;
  for (const args of [['test', '__proto__', 'mrr_at_10', id], ['test', 'flat', 'bad', id], ['test', 'flat', 'mrr_at_10', 'absent']]) assert.throws(() => retrievalComparisonView(retrievalComparison, ...args), RangeError);
  assert.throws(() => textAuditView(textEvaluationAudit, 12, 'test'), RangeError);
  assert.throws(() => textAuditView(textEvaluationAudit, 11, 'unknown'), RangeError);
  const broken = structuredClone(textEvaluationAudit); broken.runs[0].supervised_tokens++;
  assert.throws(() => textAuditView(broken, broken.runs[0].seed, broken.runs[0].split), RangeError);
});
test('built follow-up sections, source links and the real preset are reachable in the same production build', async () => {
  const retrieval = await readFile('dist/retrieval-project.html', 'utf8'), text = await readFile('dist/text-training-project.html', 'utf8'), paired = await readFile('dist/paired-evaluation.html', 'utf8');
  assert.match(retrieval, /id="retrieval-hybrid"/); assert.match(retrieval, /data-retrieval-comparison/);
  assert.match(retrieval, /paired-evaluation\.html\?dataset=retrieval-comparison#pe-lab|data-retrieval-comparison/);
  assert.match(text, /id="tp-document-audit"/); assert.match(text, /data-text-evaluation-audit/);
  assert.match(paired, /option value="retrieval-comparison"/); assert.match(paired, /id="pe-real-evidence"/);
  assert.ok(paired.includes(manifest.experiments['retrieval-comparison'].report.sha256));
});
