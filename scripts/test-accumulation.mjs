import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { hash } from './experiment-bundles.mjs';
import { accumulationReport as report } from '../src/scripts/features/accumulation-data.js';
import { accumulationView } from '../src/scripts/features/accumulation.js';
import { resolveViewState, viewUrl } from '../src/scripts/experiment-view.js';

const manifest = JSON.parse(await readFile('src/experiment-sources.json', 'utf8')), entry = manifest.experiments.accumulation;
test('the complete Transformer snapshot is the accepted report with actual executed source identities', async () => {
  const { provenance, ...snapshot } = report;
  assert.deepEqual(snapshot, JSON.parse(await readFile(entry.report.path, 'utf8')));
  assert.equal(hash(await readFile(entry.report.path)), provenance.report_sha256);
  assert.equal(hash(await readFile(entry.projection.target)), entry.projection.sha256);
  assert.equal(report.frozen.parameter_count, 298176); assert.equal(report.distributed.world_size, 2);
  for (const [file, digest] of Object.entries(provenance.source_sha256)) {
    assert.equal(hash(await readFile(file)), digest); assert.equal(hash(await readFile(path.join(entry.bundle_root, 'source', file))), digest);
  }
});
test('every condition and optimizer conserves the actual label denominator', () => {
  for (const condition of report.frozen.cases) for (const optimizer of ['adamw', 'sgd']) {
    const view = accumulationView(report, condition.id, optimizer);
    const count = condition.keep.reduce((n, x) => n + x, 0);
    assert.equal(view.reference.global_labels, count);
    for (const row of view.rows) assert.equal(row.global_labels, count);
    assert.ok(view.rows.some(row => row.method === 'reference'));
  }
});
test('correct single-process and real DDP paths preserve the whole update within the frozen tolerance', () => {
  for (const row of report.rows.filter(row => ['global-count', 'ddp-global'].includes(row.method))) {
    for (const key of ['gradient_max_error', 'parameter_max_error', 'optimizer_state_max_error']) assert.ok(row[key] <= report.frozen.tolerance, row.case + '/' + key);
  }
  assert.equal(new Set(report.distributed.ranks.map(row => row.pid)).size, 2);
});
test('balanced, tail and zero-local-rank controls retain their different observed outcomes', () => {
  for (const optimizer of ['adamw', 'sgd']) {
    const find = (id, method) => report.rows.find(row => row.case === id && row.optimizer === optimizer && row.method === method);
    assert.ok(find('balanced', 'micro-means').gradient_max_error < report.frozen.tolerance);
    assert.ok(find('ragged', 'micro-means').gradient_max_error > .001);
    assert.ok(Math.abs(find('tail', 'planned-divisor').gradient_l2_ratio - .75) < 1e-10);
    assert.ok(Math.abs(find('rank-empty', 'ddp-local-mean').gradient_l2_ratio - .5) < 1e-10);
  }
});
test('zero global supervision leaves the mean undefined and the correct optimizer state unchanged', () => {
  for (const optimizer of ['adamw', 'sgd']) {
    const { rows } = accumulationView(report, 'empty', optimizer), good = rows.find(row => row.method === 'global-count'), bad = rows.find(row => row.method === 'zero-step');
    assert.equal(good.scaled_loss, null); assert.equal(good.global_mean_defined, false); assert.equal(good.backward_calls, 0); assert.equal(good.optimizer_step, false);
    assert.equal(good.initial_state_sha256, good.final_state_sha256); assert.equal(good.parameter_move_max, 0);
    assert.equal(bad.global_mean_defined, false); assert.equal(bad.optimizer_step, true); assert.ok(bad.parameter_move_max > 0);
  }
  const adam = report.rows.find(row => row.case === 'empty' && row.optimizer === 'adamw' && row.method === 'zero-step');
  assert.equal(adam.adam_step_after, adam.adam_step_before + 1);
});
test('condition selection rejects invalid names and URL prefixes preserve unrelated controls', () => {
  assert.throws(() => accumulationView(report, '__proto__', 'adamw'), RangeError);
  assert.throws(() => accumulationView(report, 'tail', 'adam'), RangeError);
  const choices = { case: report.frozen.cases.map(row => row.id), optimizer: ['adamw', 'sgd'] }, defaults = { case: 'ragged', optimizer: 'adamw' };
  const url = viewUrl('http://localhost/parallel-correctness.html?other=kept#accumulation-real', { case: 'empty', optimizer: 'sgd' }, defaults, 'accum_');
  assert.deepEqual(resolveViewState(new URL(url).search, choices, defaults, 'accum_').state, { case: 'empty', optimizer: 'sgd' });
  assert.equal(new URL(url).hash, '#accumulation-real'); assert.equal(new URL(url).searchParams.get('other'), 'kept');
  assert.deepEqual(resolveViewState('?accum_case=bad', choices, defaults, 'accum_').invalid, ['accum_case']);
});
test('the production page and recipe reach the real model evidence and all three downloads', async () => {
  const html = await readFile('dist/parallel-correctness.html', 'utf8'), recipe = await readFile('dist/finetuning-recipe.html', 'utf8');
  assert.match(html, /id="accumulation-real"/); assert.match(html, /data-accumulation/);
  assert.ok(html.includes(entry.report.sha256)); assert.match(html, /两个 CPU Gloo/);
  assert.match(recipe, /parallel-correctness\.html#accumulation-real/);
});
