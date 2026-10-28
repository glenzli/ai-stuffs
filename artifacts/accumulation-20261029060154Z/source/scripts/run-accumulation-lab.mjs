// Prepare one complete experiment candidate; accepting it remains a separate operation.
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import { hash, checkAsset, checkChecksumFile, archiveBundle, describe, writeCandidate } from './experiment-bundles.mjs';

const args = process.argv.slice(2), root = args[1];
if (args.length !== 2 || args[0] !== '--out-dir' || !/^artifacts\/[\w-]+$/.test(root || '')) throw new Error('Usage: node scripts/run-accumulation-lab.mjs --out-dir artifacts/<new-folder>');
const manifestBytes = await readFile('src/experiment-sources.json'), manifest = JSON.parse(manifestBytes), prior = manifest.experiments.practice;
await checkAsset(prior.report); await checkAsset(prior.checksums); await checkChecksumFile(prior.checksums.path, path.dirname(prior.report.path));
const python = process.env.FRAMEWORK_PYTHON || '.venv-framework/bin/python'; await mkdir(root); const sources = {};
for (const file of ['scripts/accumulation_lab.py', 'scripts/text_training_core.py', 'scripts/learning_core.py', 'scripts/framework-requirements.txt', 'scripts/run-accumulation-lab.mjs', 'scripts/experiment-bundles.mjs']) {
  const saved = `${root}/source/${file}`; await mkdir(path.dirname(saved), { recursive: true }); await copyFile(file, saved); sources[file] = hash(await readFile(file));
}
const { stdout } = await promisify(execFile)(python, ['-c', "import sys,json;sys.path.insert(0,'scripts');import learning_core as c;c.configure_cpu(1);print(json.dumps(c.environment_metadata()))"]);
const environment = { started_at: new Date().toISOString(), runtime: JSON.parse(stdout), base_manifest_sha256: hash(manifestBytes), source_sha256: sources };
if (environment.runtime.torch !== '2.9.0' || environment.runtime.numpy !== '2.3.4') throw new Error('Unexpected reference environment');
await writeFile(`${root}/run-environment.json`, JSON.stringify(environment, null, 2) + '\n', { flag: 'wx' });
await new Promise((resolve, reject) => {
  const child = spawn(python, ['scripts/accumulation_lab.py', '--out-dir', `${root}/accumulation`, '--practice-directory', path.dirname(prior.report.path)], { stdio: 'inherit' });
  child.once('error', reject); child.once('exit', code => code === 0 ? resolve() : reject(new Error(`Numerical run failed (${code}); partial evidence preserved under ${root}; accepted pages unchanged`)));
});
for (const [file, digest] of Object.entries(sources)) if (hash(await readFile(file)) !== digest || hash(await readFile(`${root}/source/${file}`)) !== digest) throw new Error(`Executed source changed: ${file}`);
const reportPath = `${root}/accumulation/report.json`, report = JSON.parse(await readFile(reportPath, 'utf8'));
if (report.schema !== 'accumulation-equivalence-v1' || report.status !== 'passed' || !report.correct_paths_passed) throw new Error('Incomplete numerical evidence');
await copyFile(`${root}/run-environment.json`, `${root}/accumulation/run-environment.json`);
const snapshot = { ...report, provenance: { report_sha256: hash(await readFile(reportPath)), source_sha256: sources, started_at: environment.started_at, runtime: environment.runtime } };
const module = `// Generated from complete saved Transformer and Gloo measurements.\nexport const accumulationReport = ${JSON.stringify(snapshot)};\n`;
if (Buffer.byteLength(module) > 1024 * 1024) throw new Error('Display projection exceeds 1 MiB');
await writeFile(`${root}/accumulation/page-data.js`, module, { flag: 'wx' });
const assets = await archiveBundle(root, 'accumulation');
await writeCandidate({ root, baseManifestSha256: environment.base_manifest_sha256,
  entry: { id: 'accumulation', title: '完整 Transformer 的监督分母、累积与双进程 DDP 更新', pages: ['parallel-correctness'], schema: report.schema, status: 'passed', runtime: environment.runtime, created_at: environment.started_at,
    command: 'node scripts/run-accumulation-lab.mjs --out-dir artifacts/accumulation-my-run', ...assets },
  projection: { ...await describe(`${root}/accumulation/page-data.js`), target: 'src/scripts/features/accumulation-data.js' } });
console.log(`Prepared accumulation candidate: ${(assets.archive.bytes / 1048576).toFixed(2)} MiB; accepted sources unchanged.`);
