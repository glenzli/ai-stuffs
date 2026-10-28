import { readFile, writeFile, copyFile, mkdir } from 'node:fs/promises';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import { hash, checkAsset, checkChecksumFile } from './experiment-bundles.mjs';
import { evidenceAuditStudies } from './evidence-audit-studies.mjs';

const args = process.argv.slice(2), options = {};
for (let i = 0; i < args.length; i += 2) {
  if (!['--out-dir', '--only'].includes(args[i]) || !args[i + 1] || options[args[i]]) throw new Error('Usage: node scripts/run-evidence-audit.mjs --out-dir artifacts/<new-folder> [--only retrieval-comparison|text-evaluation-audit]');
  options[args[i]] = args[i + 1];
}
const root = options['--out-dir']; if (!/^artifacts\/[\w-]+$/.test(root || '')) throw new Error('Choose a new artifacts/<folder>');
if (options['--only'] && !evidenceAuditStudies.some(row => row.id === options['--only'])) throw new Error('Unknown study');
const manifestBytes = await readFile('src/experiment-sources.json'), manifest = JSON.parse(manifestBytes);
const python = process.env.FRAMEWORK_PYTHON || '.venv-framework/bin/python';
const run = (command, argv) => new Promise((resolve, reject) => {
  const child = spawn(command, argv, { stdio: 'inherit' }); child.once('error', reject);
  child.once('exit', code => code === 0 ? resolve() : reject(new Error(`Study exited ${code}; partial evidence retained under ${root}; accepted pages unchanged`)));
});
await mkdir(root);
for (const study of evidenceAuditStudies.filter(row => !options['--only'] || row.id === options['--only'])) {
  const directory = `${root}/${study.id}`; await mkdir(directory); const sources = {};
  for (const file of [...study.sources, 'scripts/evidence-audit-studies.mjs', 'scripts/run-evidence-audit.mjs', 'scripts/package-evidence-audit.mjs', 'scripts/experiment-bundles.mjs']) {
    const saved = `${directory}/source/${file}`; await mkdir(path.dirname(saved), { recursive: true }); await copyFile(file, saved); sources[file] = hash(await readFile(file));
  }
  const { stdout } = await promisify(execFile)(python, ['-c', "import sys,json;sys.path.insert(0,'scripts');import learning_core as c;c.configure_cpu(1);print(json.dumps(c.environment_metadata()))"]);
  await writeFile(`${directory}/run-environment.json`, JSON.stringify({ started_at: new Date().toISOString(), runtime: JSON.parse(stdout),
    base_manifest_sha256: hash(manifestBytes), source_sha256: sources, analysis_status: 'exploratory follow-up; already inspected source datasets' }, null, 2) + '\n', { flag: 'wx' });
  let argv;
  if (study.id === 'retrieval-comparison') {
    const entry = manifest.experiments.retrieval; await checkAsset(entry.report);
    argv = ['scripts/retrieval_comparison.py', '--out-dir', `${directory}/${study.id}`, '--source', entry.report.path, '--source-sha256', entry.report.sha256];
  } else {
    const practice = manifest.experiments.practice, baseline = manifest.experiments['language-baselines'];
    await checkAsset(practice.report); await checkAsset(practice.checksums); await checkChecksumFile(practice.checksums.path, path.dirname(practice.report.path)); await checkAsset(baseline.report);
    argv = ['scripts/text_evaluation_audit.py', '--out-dir', `${directory}/${study.id}`, '--text-directory', path.dirname(practice.report.path) + '/text', '--baseline-report', baseline.report.path, '--baseline-sha256', baseline.report.sha256];
  }
  await run(python, argv); await run(process.execPath, ['scripts/package-evidence-audit.mjs', root, study.id]);
}
console.log(`Candidates prepared under ${root}. Accepted page sources unchanged; adopt explicitly with accept:experiment.`);
