import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { hash } from './experiment-bundles.mjs';
import { retrievalFilteringReport as report } from '../src/scripts/features/retrieval-filtering-data.js';
import { filteringView } from '../src/scripts/features/retrieval-filtering.js';
import { resolveViewState, viewUrl } from '../src/scripts/experiment-view.js';

const manifest = JSON.parse(await readFile('src/experiment-sources.json', 'utf8')), entry = manifest.experiments['retrieval-filtering'];
test('the filtered retrieval display is the exact accepted report with captured executed sources', async () => {
  const { provenance, ...saved } = report;
  assert.deepEqual(saved, JSON.parse(await readFile(entry.report.path, 'utf8')));
  assert.equal(hash(await readFile(entry.report.path)), provenance.report_sha256);
  assert.equal(hash(await readFile(entry.projection.target)), entry.projection.sha256);
  for (const [file, digest] of Object.entries(provenance.source_sha256)) {
    assert.equal(hash(await readFile(file)), digest); assert.equal(hash(await readFile(path.join(entry.bundle_root, 'source', file))), digest);
  }
  assert.equal(report.input_sha256['report.json'], manifest.experiments.retrieval.report.sha256);
});
test('all 720 view selections use the correct scope, query, depth and available result denominator', () => {
  let count = 0;
  for (const scope of report.frozen.filters) for (const query of report.queries) for (const depth of report.frozen.pool_depths) {
    const { request, selected } = filteringView(report, scope.id, query.id, depth);
    assert.equal(request.filter_id, scope.id); assert.equal(report.queries[request.query_index].id, query.id); assert.equal(selected.depth, depth);
    assert.equal(request.reference.length, Math.min(10, request.allowed_count));
    assert.equal(selected.count, selected.results.length); count++;
  }
  assert.equal(count, 720);
});
test('full-depth, fewer-than-k and empty controls retain separate reference and evidence meanings', () => {
  for (const query of report.queries) for (const scope of report.frozen.filters) {
    const { request, selected } = filteringView(report, scope.id, query.id, 877);
    assert.deepEqual(selected.results, request.reference);
    if (scope.id === 'small') { assert.equal(selected.count, 3); assert.equal(selected.fill, 1); assert.equal(selected.evidence_hit, null); }
    if (scope.id === 'empty') { assert.equal(selected.count, 0); assert.equal(selected.fill, null); assert.equal(selected.neighbor_recall, null); assert.equal(selected.evidence_hit, null); }
  }
});
test('the observed resume example hits its label while still returning only three of ten allowed neighbors', () => {
  assert.deepEqual(report.frozen.pool_depths.map(depth => filteringView(report, 'artifacts', 'resume-keyword', depth).selected.count), [3, 7, 10, 10]);
  const { selected, request } = filteringView(report, 'artifacts', 'resume-keyword', 10);
  assert.equal(request.allowed_count, 34); assert.equal(selected.evidence_hit, 1); assert.equal(selected.neighbor_recall, .3);
  const trained = report.summaries.find(row => row.filter_id === 'training' && row.depth === 877);
  assert.equal(trained.evidence_eligible_queries, 4); assert.equal(trained.evidence_hits, 3);
});
test('view guards and namespaced links reject unknown choices and preserve unrelated page state', () => {
  for (const args of [['wrong', 'resume-keyword', 10], ['all', 'absent', 10], ['all', 'resume-keyword', 11], ['all', 'resume-keyword', NaN], ['all', 'resume-keyword', '10']]) assert.throws(() => filteringView(report, ...args), RangeError);
  const choices = { scope: report.frozen.filters.map(row => row.id), query: report.queries.map(row => row.id), depth: report.frozen.pool_depths.map(String) }, defaults = { scope: 'artifacts', query: 'resume-keyword', depth: '10' };
  const state = { scope: 'empty', query: 'dedup-keyword', depth: '877' }, url = viewUrl('http://localhost/ann-index.html?other=kept#filtering-real', state, defaults, 'filter_');
  assert.deepEqual(resolveViewState(new URL(url).search, choices, defaults, 'filter_').state, state);
  assert.equal(new URL(url).searchParams.get('other'), 'kept'); assert.equal(new URL(url).hash, '#filtering-real');
  assert.deepEqual(resolveViewState('?filter_scope=bad', choices, defaults, 'filter_').invalid, ['filter_scope']);
});
test('the built index page owns filtering sources and the original project reaches the new section', async () => {
  const html = await readFile('dist/ann-index.html', 'utf8'), project = await readFile('dist/retrieval-project.html', 'utf8');
  assert.match(html, /id="filtering-real"/); assert.match(html, /data-retrieval-filtering/); assert.ok(html.includes(entry.report.sha256));
  assert.match(project, /ann-index\.html#filtering-real/); assert.match(html, /720 次后过滤/); assert.match(html, /不作为新的独立泛化评估/);
});
