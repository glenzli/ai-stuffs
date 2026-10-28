// Capture the numerical implementation before running; prepare without page mutation.
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import { hash, checkAsset, checkChecksumFile, archiveBundle, describe, writeCandidate } from './experiment-bundles.mjs';

const args = process.argv.slice(2), root = args[1];
if (args.length !== 2 || args[0] !== '--out-dir' || !/^artifacts\/[\w-]+$/.test(root || '')) throw new Error('Usage: node scripts/run-padding-inference.mjs --out-dir artifacts/<new-folder>');
const manifestBytes = await readFile('src/experiment-sources.json'), manifest = JSON.parse(manifestBytes), prior = manifest.experiments.practice;
await checkAsset(prior.report); await checkAsset(prior.checksums); await checkChecksumFile(prior.checksums.path, path.dirname(prior.report.path));
await mkdir(root);
const python = process.env.FRAMEWORK_PYTHON || '.venv-framework/bin/python', sources = {};
for (const file of ['scripts/padding_inference_lab.py', 'scripts/text_training_core.py', 'scripts/learning_core.py', 'scripts/framework-requirements.txt', 'scripts/run-padding-inference.mjs', 'scripts/experiment-bundles.mjs']) {
  const target = `${root}/source/${file}`; await mkdir(path.dirname(target), { recursive: true }); await copyFile(file, target); sources[file] = hash(await readFile(file));
}
const { stdout } = await promisify(execFile)(python, ['-c', "import sys,json;sys.path.insert(0,'scripts');import learning_core as c;c.configure_cpu(1);print(json.dumps(c.environment_metadata()))"]);
const environment = { started_at: new Date().toISOString(), runtime: JSON.parse(stdout), source_sha256: sources, base_manifest_sha256: hash(manifestBytes) };
if (environment.runtime.torch !== '2.9.0' || environment.runtime.numpy !== '2.3.4') throw new Error('Unexpected reference environment');
await writeFile(`${root}/run-environment.json`, JSON.stringify(environment, null, 2) + '\n', { flag: 'wx' });
await new Promise((resolve, reject) => {
  const child = spawn(python, ['scripts/padding_inference_lab.py', '--out-dir', `${root}/padding-inference`, '--text-directory', path.dirname(prior.report.path) + '/text'], { stdio: 'inherit' });
  child.once('error', reject); child.once('exit', code => code === 0 ? resolve() : reject(new Error(`Padding experiment failed (${code}); partial evidence retained; accepted sources unchanged`)));
});
for (const [file, digest] of Object.entries(sources)) if (hash(await readFile(file)) !== digest || hash(await readFile(`${root}/source/${file}`)) !== digest) throw new Error(`Executed source changed: ${file}`);
const reportPath = `${root}/padding-inference/report.json`, report = JSON.parse(await readFile(reportPath, 'utf8'));
if (report.schema !== 'padding-inference-v1' || report.status !== 'passed' || !report.correctness.correct_paths_passed) throw new Error('Incomplete padding evidence');
await copyFile(`${root}/run-environment.json`, `${root}/padding-inference/run-environment.json`);
const snapshot = { ...report, provenance: { report_sha256: hash(await readFile(reportPath)), source_sha256: sources, started_at: environment.started_at, runtime: environment.runtime } };
const module = `// Generated from actual complete Transformer padding and cache tensors.\nexport const paddingInferenceReport = ${JSON.stringify(snapshot)};\n`;
if (Buffer.byteLength(module) > 1024 * 1024) throw new Error('Display projection exceeds 1 MiB');
await writeFile(`${root}/padding-inference/page-data.js`, module, { flag: 'wx' });
const assets = await archiveBundle(root, 'padding-inference');
await writeCandidate({ root, baseManifestSha256: environment.base_manifest_sha256,
  entry: { id: 'padding-inference', title: '完整 Transformer 的变长填充、下一位置 logits 与缓存续算', pages: ['position-cache'], schema: report.schema, status: 'passed', runtime: environment.runtime, created_at: environment.started_at,
    command: 'node scripts/run-padding-inference.mjs --out-dir artifacts/padding-my-run', ...assets },
  projection: { ...await describe(`${root}/padding-inference/page-data.js`), target: 'src/scripts/features/padding-inference-data.js' } });
console.log(`Prepared padding candidate: ${(assets.archive.bytes / 1048576).toFixed(2)} MiB; accepted sources unchanged.`);
