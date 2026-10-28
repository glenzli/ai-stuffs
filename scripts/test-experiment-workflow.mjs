import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, copyFile, rm, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { acceptExperiments, recoverAdoption } from './accept-experiment.mjs';
import { hash, archiveBundle } from './experiment-bundles.mjs';
import { verifyExperimentSources } from './experiment-sources.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const run = promisify(execFile), missing = file => assert.rejects(access(file), /ENOENT/);
async function fixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), 'ai-stuffs-workflow-')); t.after(() => rm(root, { recursive: true, force: true }));
  const bundle = 'artifacts/probe', id = 'language-baselines';
  await mkdir(path.join(root, bundle, id), { recursive: true }); await mkdir(path.join(root, bundle, 'source/scripts'), { recursive: true });
  await mkdir(path.join(root, 'src/scripts/features'), { recursive: true });
  await writeFile(path.join(root, '.gitignore'), '/artifacts/*\n*.log\n');
  await writeFile(path.join(root, 'src/experiment-sources.json'), JSON.stringify({ schema: 'model-engineering-experiments-v1', experiments: {} }) + '\n');
  await writeFile(path.join(root, 'src/scripts/features/language-baselines-data.js'), '// old accepted projection\n');
  await writeFile(path.join(root, bundle, id, 'report.json'), JSON.stringify({ schema: 'language-baselines-v1', status: 'passed', fixture: 'metadata transaction, not a scientific result' }) + '\n');
  await writeFile(path.join(root, bundle, id, 'trace.log'), 'fixture trace\n');
  await writeFile(path.join(root, bundle, 'source/scripts/example.py'), '# executed fixture source\n');
  const projection = Buffer.from('export const languageBaselines = { fixture: true };\n');
  await writeFile(path.join(root, bundle, id, 'page-data.js'), projection);
  const assets = await archiveBundle(path.join(root, bundle), id);
  for (const asset of Object.values(assets)) asset.path = path.relative(root, asset.path);
  const candidate = { schema: 'experiment-candidate-v1', base_manifest_sha256: hash(await readFile(path.join(root, 'src/experiment-sources.json'))), bundle_root: bundle,
    entry: { id, title: 'fixture', pages: ['text-training-project'], schema: 'language-baselines-v1', status: 'passed', bundle_root: bundle, ...assets },
    projection: { path: bundle + '/' + id + '/page-data.js', target: 'src/scripts/features/language-baselines-data.js', bytes: projection.length, sha256: hash(projection) } };
  const file = bundle + '/candidate.json'; await writeFile(path.join(root, file), JSON.stringify(candidate));
  return { root, file, bundle, candidate, targets: ['src/scripts/features/language-baselines-data.js', 'src/experiment-sources.json', '.gitignore'] };
}
async function identities(f) { return Promise.all(f.targets.map(async p => hash(await readFile(path.join(f.root, p))))); }

test('successful adoption updates projection, manifest and inclusion rules as one owned operation', async t => {
  const f = await fixture(t);
  await writeFile(path.join(f.root, '.gitignore'), '/artifacts/*\n*.log\n\n# Accepted experiment bundles (managed by accept-experiment.mjs)\n!/artifacts/old/\n# End accepted experiment bundles\n\n# Reviewed QA evidence\n!/artifacts/qa/\n');
  const result = await acceptExperiments([f.file], { root: f.root });
  assert.equal(result.accepted[0].id, 'language-baselines');
  assert.equal(hash(await readFile(path.join(f.root, f.candidate.projection.target))), f.candidate.projection.sha256);
  assert.ok((await readFile(path.join(f.root, '.gitignore'), 'utf8')).includes('!/artifacts/probe/**'));
  assert.ok((await readFile(path.join(f.root, '.gitignore'), 'utf8')).includes('# Reviewed QA evidence\n!/artifacts/qa/'));
  assert.equal(JSON.parse(await readFile(path.join(f.root, 'src/experiment-sources.json'))).experiments['language-baselines'].report.sha256, f.candidate.entry.report.sha256);
  await missing(path.join(f.root, '.cache/experiment-adoption.lock'));
});
test('an interrupted multi-file write restores every previous file', async t => {
  const f = await fixture(t), before = await identities(f);
  await assert.rejects(acceptExperiments([f.file], { root: f.root, afterWrite: () => { throw new Error('injected failure after first rename'); } }), /injected failure/);
  assert.deepEqual(await identities(f), before); await missing(path.join(f.root, '.cache/experiment-adoption.lock'));
});
test('repeat adoption after a lost acknowledgement is idempotent', async t => {
  const f = await fixture(t); await acceptExperiments([f.file], { root: f.root }); const before = await identities(f);
  const result = await acceptExperiments([f.file], { root: f.root }); assert.equal(result.already_accepted, true); assert.deepEqual(await identities(f), before);
});
test('stale candidates cannot replace newer accepted state', async t => {
  const f = await fixture(t); await writeFile(path.join(f.root, 'src/experiment-sources.json'), '{"experiments":{},"newer":true}'); const before = await identities(f);
  await assert.rejects(acceptExperiments([f.file], { root: f.root }), /Stale candidate/); assert.deepEqual(await identities(f), before);
});
test('changed loose evidence, projection or archive cannot be adopted', async t => {
  for (const which of ['trace', 'projection', 'archive']) {
    const f = await fixture(t), before = await identities(f);
    const file = which === 'trace' ? f.bundle + '/language-baselines/trace.log' : which === 'projection' ? f.candidate.projection.path : f.candidate.entry.archive.path;
    await writeFile(path.join(f.root, file), 'tampered');
    await assert.rejects(acceptExperiments([f.file], { root: f.root }), /changed|identity|mismatch/i); assert.deepEqual(await identities(f), before);
  }
});
test('an archive containing different raw bytes fails even with its own new archive hash', async t => {
  const f = await fixture(t), report = path.join(f.root, f.bundle, 'language-baselines/report.json');
  const bytes = await readFile(report); await writeFile(report, '{"altered":true}');
  await run('tar', ['-czf', path.join(f.root, f.candidate.entry.archive.path), '-C', path.join(f.root, f.bundle), 'language-baselines', 'source', 'SOURCE-SHA256SUMS']);
  await writeFile(report, bytes); const altered = await readFile(path.join(f.root, f.candidate.entry.archive.path));
  f.candidate.entry.archive = { ...f.candidate.entry.archive, bytes: altered.length, sha256: hash(altered) };
  await writeFile(path.join(f.root, f.file), JSON.stringify(f.candidate)); const before = await identities(f);
  await assert.rejects(acceptExperiments([f.file], { root: f.root }), /Archive envelope/); assert.deepEqual(await identities(f), before);
});
test('active adoption blocks build input reads before dist can change', async t => {
  const f = await fixture(t); await mkdir(path.join(f.root, '.cache/experiment-adoption.lock'), { recursive: true });
  await assert.rejects(verifyExperimentSources({ root: f.root }), /adoption is incomplete/);
  await assert.rejects(acceptExperiments([f.file], { root: f.root }), /EEXIST/);
  await access(path.join(f.root, '.cache/experiment-adoption.lock'));
});
test('a crashed writer is recoverable and readers never accept a half update', async t => {
  const f = await fixture(t), before = await identities(f);
  const code = `import {acceptExperiments} from ${JSON.stringify(new URL('./accept-experiment.mjs', import.meta.url).href)}; await acceptExperiments([${JSON.stringify(f.file)}],{root:${JSON.stringify(f.root)},afterWrite:()=>process.exit(86)});`;
  await assert.rejects(run(process.execPath, ['--input-type=module', '-e', code]), error => error.code === 86);
  assert.notDeepEqual(await identities(f), before);
  await assert.rejects(verifyExperimentSources({ root: f.root }), /adoption is incomplete/);
  assert.deepEqual(await recoverAdoption(f.root), { recovered: true, kept: 'previous' }); assert.deepEqual(await identities(f), before);
});
test('packaging failure never mutates any accepted source or invents failure history', async t => {
  const root = await mkdtemp(path.join(tmpdir(), 'ai-stuffs-pack-safe-')); t.after(() => rm(root, { recursive: true, force: true }));
  const bundle = 'artifacts/new-run'; await mkdir(path.join(root, bundle, 'practice'), { recursive: true });
  await mkdir(path.join(root, 'src/scripts/features'), { recursive: true }); await mkdir(path.join(root, bundle, 'source/scripts'), { recursive: true }); await mkdir(path.join(root, 'scripts'));
  await writeFile(path.join(root, 'scripts/example.py'), '# fixture\n'); await copyFile(path.join(root, 'scripts/example.py'), path.join(root, bundle, 'source/scripts/example.py'));
  const old = path.join(repo, 'artifacts/practice-20261028T152417Z/practice');
  for (const id of ['generalization', 'text', 'reproduction', 'profile']) { await mkdir(path.join(root, bundle, 'practice', id)); await copyFile(path.join(old, id, 'report.json'), path.join(root, bundle, 'practice', id, 'report.json')); }
  await writeFile(path.join(root, bundle, 'practice/corpus.json'), '{"documents":[]}');
  await writeFile(path.join(root, bundle, 'practice/run-sources.json'), JSON.stringify({ files: { 'scripts/example.py': hash(await readFile(path.join(root, 'scripts/example.py'))) } }));
  await writeFile(path.join(root, bundle, 'practice/run-environment.json'), JSON.stringify({ runtime: { torch: '2.9.0', numpy: '2.3.4' }, base_manifest_sha256: 'a'.repeat(64), failures: [] }));
  await writeFile(path.join(root, 'src/experiment-sources.json'), '{"old":true}'); await writeFile(path.join(root, 'src/scripts/features/practice-data.js'), '// old baseline\n');
  await writeFile(path.join(root, bundle, 'SOURCE-SHA256SUMS'), 'force an EEXIST after report/projection preparation\n');
  await assert.rejects(run(process.execPath, [path.join(repo, 'scripts/package-practice.mjs'), bundle], { cwd: root }), /EEXIST/);
  assert.equal(await readFile(path.join(root, 'src/experiment-sources.json'), 'utf8'), '{"old":true}');
  assert.equal(await readFile(path.join(root, 'src/scripts/features/practice-data.js'), 'utf8'), '// old baseline\n');
  const prepared = JSON.parse(await readFile(path.join(root, bundle, 'practice/report.json'), 'utf8'));
  assert.equal('failed_path_attempt' in prepared.provenance, false); assert.deepEqual(prepared.provenance.execution.failures, []);
});
