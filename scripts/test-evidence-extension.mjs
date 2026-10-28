import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir, access } from 'node:fs/promises';
import path from 'node:path';
import { hash } from './experiment-bundles.mjs';
import { parse } from 'parse5';
import { languageBaselines } from '../src/scripts/features/language-baselines-data.js';
import { matchedBudget } from '../src/scripts/features/matched-budget-data.js';
import { retrievalSnapshot } from '../src/scripts/features/retrieval-data.js';
import { baselineComparison } from '../src/scripts/features/language-baselines.js';
import { matchedRuns } from '../src/scripts/features/matched-budget.js';
import { retrievalView } from '../src/scripts/features/retrieval-project.js';
import { experimentQuestions, gradeExperimentAnswer } from '../src/scripts/features/experiment-check.js';
import { resolveViewState, viewUrl } from '../src/scripts/experiment-view.js';
import { learningAttachments } from '../src/learning.mjs';

const manifest = JSON.parse(await readFile('src/experiment-sources.json', 'utf8'));
const read = async file => JSON.parse(await readFile(file, 'utf8'));
const walk = node => [node, ...(node.childNodes || []).flatMap(walk)];
const attrs = node => Object.fromEntries((node.attrs || []).map(a => [a.name, a.value]));

test('accepted display projections preserve complete reports and all executed source identities', async () => {
  for (const [id, snapshot] of [['language-baselines', languageBaselines], ['matched-budget', matchedBudget], ['retrieval', retrievalSnapshot]]) {
    const entry = manifest.experiments[id], full = await read(entry.report.path);
    assert.equal(hash(await readFile(entry.report.path)), snapshot.provenance.report_sha256);
    assert.equal(hash(await readFile(entry.projection.target)), entry.projection.sha256);
    for (const [file, digest] of Object.entries(snapshot.provenance.source_sha256)) {
      assert.equal(hash(await readFile(file)), digest); assert.equal(hash(await readFile(path.join(entry.bundle_root, 'source', file))), digest);
    }
    const { provenance, ...display } = snapshot;
    if (id === 'language-baselines') assert.deepEqual(display, { ...full, transformer: full.transformer.map(row => ({ seed: row.seed, evaluation: row.evaluation })) });
    if (id === 'retrieval') assert.deepEqual(display, full);
    if (id === 'matched-budget') for (const run of display.runs) {
      const original = full.runs.find(r => r.seed === run.seed && r.support === run.support);
      const { predictions, ...stats } = run; const { predictions: all, ...rawStats } = original;
      assert.deepEqual(stats, rawStats); assert.deepEqual(predictions, [...new Set(all.map(r => r.length))].flatMap(n => all.filter(r => r.length === n).slice(0, 6)));
    }
  }
});
test('baseline and matched-budget views expose all seeds and guard invalid choices', () => {
  for (const split of ['train', 'validation', 'test']) {
    const rows = baselineComparison(languageBaselines, split); assert.equal(rows.length, 5);
    assert.equal(new Set(rows.map(r => r.supervised_tokens)).size, 1);
    assert.ok(rows.every(row => Number.isFinite(row.ce_nats_per_token)));
  }
  for (const seed of [11, 29, 47]) {
    const runs = matchedRuns(matchedBudget, seed); assert.equal(runs[0].supervised_tokens, runs[1].supervised_tokens);
    assert.notEqual(runs[0].updates, runs[1].updates);
  }
  assert.throws(() => baselineComparison(languageBaselines, 'train+test'), RangeError);
  assert.throws(() => matchedRuns(matchedBudget, 123), RangeError);
  const invalid = structuredClone(matchedBudget); invalid.runs[0].supervised_tokens++;
  assert.throws(() => matchedRuns(invalid, 11), RangeError);
});
test('all retrieval view methods retain real candidate evidence and invalid names cannot enter the view', () => {
  const methods = ['bm25', 'flat', 'ivf-1', 'ivf-4', 'ivf-16', 'rrf-rerank'];
  for (const query of retrievalSnapshot.queries) for (const method of methods) {
    const view = retrievalView(retrievalSnapshot, query.id, method, 'test', 'all');
    assert.equal(view.result.results.length, 10); assert.equal(view.summary.length, 6);
    for (const row of view.result.results) assert.ok(retrievalSnapshot.passages.some(p => p.id === row.passage_id));
  }
  for (const method of ['__proto__', 'toString', 'not-a-model']) assert.throws(() => retrievalView(retrievalSnapshot, retrievalSnapshot.queries[0].id, method, 'test', 'all'), RangeError);
});
test('URL state survives a reload, keeps anchors and unrelated widgets, and rejects unsupported values', () => {
  const choices = { seed: ['11', '29', '47'], split: ['test', 'validation'] }, defaults = { seed: '11', split: 'test' };
  const url = viewUrl('http://localhost/text-training-project.html?budget_seed=29#tp-results', { seed: '47', split: 'validation' }, defaults);
  assert.deepEqual(resolveViewState(new URL(url).search, choices, defaults).state, { seed: '47', split: 'validation' });
  assert.equal(new URL(url).hash, '#tp-results'); assert.equal(new URL(url).searchParams.get('budget_seed'), '29');
  const invalid = resolveViewState('?seed=<script>&split=train', choices, defaults); assert.deepEqual(invalid.state, defaults); assert.deepEqual(invalid.invalid, ['seed', 'split']);
  const cleared = viewUrl(url, defaults, defaults); assert.equal(new URL(cleared).searchParams.has('seed'), false);
  assert.equal(new URL(cleared).searchParams.get('budget_seed'), '29');
});
test('separate-widget URL prefixes preserve the other panel and defaults remain shareable', () => {
  const defaults = { seed: '11' }, choices = { seed: ['11', '29', '47'] };
  const url = viewUrl('http://localhost/generalization-analysis.html?seed=47', { seed: '29' }, defaults, 'budget_');
  assert.equal(new URL(url).searchParams.get('seed'), '47');
  assert.deepEqual(resolveViewState(new URL(url).search, choices, defaults, 'budget_').state, { seed: '29' });
});
test('each prediction question gives distinct feedback and rejects impossible answers', () => {
  assert.equal(Object.keys(experimentQuestions).length, 6);
  for (const [id, row] of Object.entries(experimentQuestions)) {
    assert.equal(gradeExperimentAnswer(id, row.answer).correct, true);
    assert.equal(gradeExperimentAnswer(id, (row.answer + 1) % row.choices.length).correct, false);
    assert.ok(gradeExperimentAnswer(id, row.answer).explanation.length > 30);
    assert.throws(() => gradeExperimentAnswer(id, -1), RangeError);
  }
  assert.throws(() => gradeExperimentAnswer('__proto__', 0), RangeError);
});
test('all practical attachments have a route back, a prediction and an accepted source card', async () => {
  for (const slug of Object.keys(learningAttachments)) {
    const html = await readFile(`dist/${slug}.html`, 'utf8'), nodes = walk(parse(html));
    assert.ok(nodes.some(n => attrs(n)['aria-label'] === '实战与学习主线的连接'), slug);
    assert.ok(nodes.some(n => 'data-experiment-check' in attrs(n)), slug);
    const expected = Object.values(manifest.experiments).filter(e => e.pages.includes(slug)).length * 3;
    assert.equal(nodes.filter(n => n.tagName === 'a' && 'download' in attrs(n)).length, expected, slug);
  }
  const route = await readFile('dist/deep-dive.html', 'utf8'); assert.ok(route.includes('五个可运行实战')); assert.ok(route.includes('href="retrieval-project.html"'));
});
test('every emitted static JavaScript import resolves, including shared presentation and URL state', async () => {
  async function files(directory) {
    const result = []; for (const item of await readdir(directory, { withFileTypes: true })) {
      const name = path.join(directory, item.name); if (item.isDirectory()) result.push(...await files(name)); else if (item.name.endsWith('.js')) result.push(name);
    } return result;
  }
  for (const file of await files('dist/assets/js')) {
    const source = await readFile(file, 'utf8');
    for (const [, relative] of source.matchAll(/from\s+["'](\.[^"']+)["']/g)) await access(path.resolve(path.dirname(file), relative));
  }
});
