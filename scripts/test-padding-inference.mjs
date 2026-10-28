import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { hash } from './experiment-bundles.mjs';
import { paddingInferenceReport as report } from '../src/scripts/features/padding-inference-data.js';
import { paddingView } from '../src/scripts/features/padding-inference.js';
import { resolveViewState, viewUrl } from '../src/scripts/experiment-view.js';

const manifest = JSON.parse(await readFile('src/experiment-sources.json', 'utf8')), entry = manifest.experiments['padding-inference'];
test('the complete-model padding snapshot is the accepted measured report with executed source identities', async () => {
  const { provenance, ...saved } = report;
  assert.deepEqual(saved, JSON.parse(await readFile(entry.report.path, 'utf8')));
  assert.equal(hash(await readFile(entry.report.path)), provenance.report_sha256);
  assert.equal(hash(await readFile(entry.projection.target)), entry.projection.sha256);
  for (const [file, digest] of Object.entries(provenance.source_sha256)) {
    assert.equal(hash(await readFile(file)), digest); assert.equal(hash(await readFile(path.join(entry.bundle_root, 'source', file))), digest);
  }
  assert.equal(report.parameter_count, 298176); assert.deepEqual(report.initial_weight_sha256, report.final_weight_sha256);
});
test('all 90 condition selections reach exactly six actual method observations with conserved cache counts', () => {
  let count = 0;
  for (const config of report.frozen.cases) for (const side of report.frozen.sides) for (const sample of [0, 1, 2]) for (const event of [0, 1, 2, 3, 4]) {
    const view = paddingView(report, config.id, side, sample, event);
    assert.equal(view.rows.length, 6); assert.equal(view.timeline.length, 30);
    for (const row of view.rows) { assert.equal(row.valid_cache_slots, config.lengths[sample] + event); assert.equal(row.physical_cache_slots, Math.max(...config.lengths) + config.extra_pad + event); }
    count++;
  }
  assert.equal(count, 90); assert.equal(report.rows.length, count * 6);
});
test('correct paths and unpadded balanced controls stay within tolerance without relabeling faulty recipes', () => {
  for (const row of report.rows.filter(row => row.method === 'correct' || row.case === 'balanced')) {
    assert.equal(row.within_tolerance, true); assert.ok(Math.max(row.valid_logits_max_error, row.next_logits_max_error, row.kv_max_error) < report.frozen.tolerance);
  }
  assert.equal(report.correctness.empty_prompt_rejected, true); assert.equal(report.correctness.cached_prefixes_unchanged, true);
});
test('readout errors can leave valid logits and cache correct while missing masks fail only at continuation', () => {
  const get = (side, method, event) => paddingView(report, 'ragged', side, 0, event).rows.find(row => row.method === method);
  for (const [side, method] of [['right', 'last-column'], ['left', 'length-index']]) {
    const row = get(side, method, 0); assert.ok(row.next_logits_max_error > 1); assert.ok(row.valid_logits_max_error < 2e-10); assert.ok(row.kv_max_error < 2e-10); assert.notEqual(row.selected_column, row.last_valid_column);
  }
  for (const method of ['no-key-mask', 'physical', 'cache-physical']) { assert.equal(get('right', method, 0).within_tolerance, true); assert.equal(get('right', method, 1).within_tolerance, false); }
  assert.ok(report.rows.some(row => !row.within_tolerance && row.actual_argmax === row.reference_argmax));
});
test('selection guards and namespaced share links preserve unrelated state and the experiment anchor', () => {
  for (const args of [['bad', 'right', 0, 0], ['ragged', 'bad', 0, 0], ['ragged', 'right', -1, 0], ['ragged', 'right', 3, 0], ['ragged', 'right', 0, NaN], ['ragged', 'right', 0, 5]]) assert.throws(() => paddingView(report, ...args), RangeError);
  const choices = { case: report.frozen.cases.map(row => row.id), side: report.frozen.sides, sample: ['0', '1', '2'], event: ['0', '1', '2', '3', '4'] }, defaults = { case: 'ragged', side: 'right', sample: '0', event: '0' }, state = { case: 'extra', side: 'left', sample: '2', event: '4' };
  const url = viewUrl('http://localhost/position-cache.html?other=kept#padding-real', state, defaults, 'padding_');
  assert.deepEqual(resolveViewState(new URL(url).search, choices, defaults, 'padding_').state, state);
  assert.equal(new URL(url).searchParams.get('other'), 'kept'); assert.equal(new URL(url).hash, '#padding-real');
  assert.deepEqual(resolveViewState('?padding_side=bad', choices, defaults, 'padding_').invalid, ['padding_side']);
});
test('the built position page reaches its full-model evidence and declares fixed positions and fixed suffix inputs', async () => {
  const html = await readFile('dist/position-cache.html', 'utf8'), inference = await readFile('dist/inference-engine.html', 'utf8');
  assert.match(html, /id="padding-real"/); assert.match(html, /data-padding-inference/); assert.ok(html.includes(entry.report.sha256));
  assert.match(html, /保存的固定正弦位置/); assert.match(html, /540 个逐样本阶段观测/); assert.match(inference, /position-cache\.html#padding-real/);
});
