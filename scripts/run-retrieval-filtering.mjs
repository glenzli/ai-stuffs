// Prepare immutable evidence; adopting page data is a separate operation.
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import { hash, checkAsset, checkChecksumFile, archiveBundle, describe, writeCandidate } from './experiment-bundles.mjs';

const args = process.argv.slice(2), root = args[1];
if (args.length !== 2 || args[0] !== '--out-dir' || !/^artifacts\/[\w-]+$/.test(root || '')) throw new Error('Usage: node scripts/run-retrieval-filtering.mjs --out-dir artifacts/<new-folder>');
const manifestBytes = await readFile('src/experiment-sources.json'), manifest = JSON.parse(manifestBytes), prior = manifest.experiments.retrieval;
await checkAsset(prior.report); await checkAsset(prior.checksums); await checkChecksumFile(prior.checksums.path, path.dirname(prior.report.path));
await mkdir(root);
const python = process.env.FRAMEWORK_PYTHON || '.venv-framework/bin/python', sources = {};
for (const file of ['scripts/retrieval_filtering.py', 'scripts/retrieval-filtering-requirements.txt', 'scripts/run-retrieval-filtering.mjs', 'scripts/experiment-bundles.mjs']) {
  const saved = `${root}/source/${file}`; await mkdir(path.dirname(saved), { recursive: true }); await copyFile(file, saved); sources[file] = hash(await readFile(file));
}
const { stdout } = await promisify(execFile)(python, ['-c', "import json,platform,numpy;print(json.dumps(dict(python=platform.python_version(),numpy=numpy.__version__,platform=platform.platform(),device='cpu')))"]);
const environment = { started_at: new Date().toISOString(), runtime: JSON.parse(stdout), source_sha256: sources, base_manifest_sha256: hash(manifestBytes) };
if (environment.runtime.numpy !== '2.3.4') throw new Error('Unexpected reference NumPy version');
await writeFile(`${root}/run-environment.json`, JSON.stringify(environment, null, 2) + '\n', { flag: 'wx' });
await new Promise((resolve, reject) => {
  const child = spawn(python, ['scripts/retrieval_filtering.py', '--out-dir', `${root}/retrieval-filtering`, '--retrieval-directory', path.dirname(prior.report.path), '--source-sha256', prior.report.sha256], { stdio: 'inherit', env: { ...process.env, OPENBLAS_NUM_THREADS: '1', OMP_NUM_THREADS: '1', VECLIB_MAXIMUM_THREADS: '1' } });
  child.once('error', reject); child.once('exit', code => code === 0 ? resolve() : reject(new Error(`Filtering run failed (${code}); partial evidence retained; accepted sources unchanged`)));
});
for (const [file, digest] of Object.entries(sources)) if (hash(await readFile(file)) !== digest || hash(await readFile(`${root}/source/${file}`)) !== digest) throw new Error(`Executed source changed: ${file}`);
const reportPath = `${root}/retrieval-filtering/report.json`, report = JSON.parse(await readFile(reportPath, 'utf8'));
if (report.schema !== 'retrieval-filtering-v1' || report.status !== 'passed' || !report.correctness.full_depth_matches_all_references) throw new Error('Incomplete filtering evidence');
await copyFile(`${root}/run-environment.json`, `${root}/retrieval-filtering/run-environment.json`);
const snapshot = { ...report, provenance: { report_sha256: hash(await readFile(reportPath)), source_sha256: sources, started_at: environment.started_at, runtime: environment.runtime } };
const module = `// Generated from actual saved-vector filtering results.\nexport const retrievalFilteringReport = ${JSON.stringify(snapshot)};\n`;
if (Buffer.byteLength(module) > 1024 * 1024) throw new Error('Display projection exceeds 1 MiB');
await writeFile(`${root}/retrieval-filtering/page-data.js`, module, { flag: 'wx' });
const assets = await archiveBundle(root, 'retrieval-filtering');
await writeCandidate({ root, baseManifestSha256: environment.base_manifest_sha256,
  entry: { id: 'retrieval-filtering', title: '真实向量的来源过滤、候选深度与可用证据', pages: ['ann-index'], schema: report.schema, status: 'passed', runtime: environment.runtime, created_at: environment.started_at,
    command: 'node scripts/run-retrieval-filtering.mjs --out-dir artifacts/filtering-my-run', ...assets },
  projection: { ...await describe(`${root}/retrieval-filtering/page-data.js`), target: 'src/scripts/features/retrieval-filtering-data.js' } });
console.log(`Prepared filtering candidate: ${(assets.archive.bytes / 1048576).toFixed(2)} MiB; accepted sources unchanged.`);
