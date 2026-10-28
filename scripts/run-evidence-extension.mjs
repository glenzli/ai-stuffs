import { readFile, writeFile, mkdir, copyFile, access } from 'node:fs/promises';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import { hash } from './experiment-bundles.mjs';

const args = process.argv.slice(2), options = {};
for (let i = 0; i < args.length; i += 2) {
  if (!['--out-dir', '--model-directory', '--only'].includes(args[i]) || !args[i + 1] || options[args[i]]) throw new Error('Usage: node scripts/run-evidence-extension.mjs --out-dir artifacts/<new-folder> [--only language-baselines|matched-budget|retrieval] [--model-directory <pinned-local-model>]');
  options[args[i]] = args[i + 1];
}
const root = options['--out-dir']; if (!/^artifacts\/[\w-]+$/.test(root || '')) throw new Error('Choose a new artifacts/<folder>');
const definitions = {
  'language-baselines': ['scripts/language_baselines.py'],
  'matched-budget': ['scripts/matched_budget_lab.py', 'scripts/generalization_lab.py', 'scripts/learning_core.py', 'scripts/framework-requirements.txt'],
  retrieval: ['scripts/retrieval_lab.py', 'scripts/retrieval_index.py', 'scripts/retrieval-requirements.txt', 'scripts/download-retrieval-model.py', 'src/experiments/retrieval-queries.json'],
};
if (options['--only'] && !Object.hasOwn(definitions, options['--only'])) throw new Error('Unknown study');
const manifestBytes = await readFile('src/experiment-sources.json'), manifest = JSON.parse(manifestBytes);
const practice = path.dirname(manifest.experiments.practice.report.path);
const framework = process.env.FRAMEWORK_PYTHON || '.venv-framework/bin/python', retrieval = process.env.RETRIEVAL_PYTHON || '.venv-retrieval/bin/python';
const model = options['--model-directory'] || path.join(path.dirname(manifest.experiments.retrieval.report.path), 'model');
const run = (command, argv) => new Promise((resolve, reject) => {
  const child = spawn(command, argv, { stdio: 'inherit', env: { ...process.env, TOKENIZERS_PARALLELISM: 'false' } });
  child.once('error', reject); child.once('exit', code => code === 0 ? resolve() : reject(new Error(`Study exited ${code}; partial files retained under ${root}; page sources unchanged`)));
});
await mkdir(root);
for (const [id, files] of Object.entries(definitions)) {
  if (options['--only'] && id !== options['--only']) continue;
  const base = root + '/' + id, python = id === 'retrieval' ? retrieval : framework;
  await access(python); if (id === 'retrieval') await access(path.join(model, 'model-manifest.json'));
  await mkdir(base); const hashes = {};
  for (const file of [...files, 'scripts/run-evidence-extension.mjs', 'scripts/package-evidence-extension.mjs', 'scripts/experiment-bundles.mjs']) {
    const to = base + '/source/' + file; await mkdir(path.dirname(to), { recursive: true }); await copyFile(file, to); hashes[file] = hash(await readFile(file));
  }
  const code = id === 'retrieval'
    ? "import json,platform,sys,importlib.metadata as m;print(json.dumps({'python':sys.version.split()[0],'platform':platform.platform(),'machine':platform.machine(),'device':'cpu','threads':1,**{k:m.version(k) for k in ['torch','numpy','faiss-cpu','transformers','tokenizers','safetensors','huggingface-hub']}}))"
    : "import sys,json;sys.path.insert(0,'scripts');import learning_core as c;c.configure_cpu(1);print(json.dumps(c.environment_metadata()))";
  const { stdout } = await promisify(execFile)(python, ['-c', code]);
  await writeFile(base + '/run-environment.json', JSON.stringify({ started_at: new Date().toISOString(), runtime: JSON.parse(stdout),
    base_manifest_sha256: hash(manifestBytes), source_sha256: hashes, tokenizers_parallelism: false }, null, 2) + '\n', { flag: 'wx' });
  const argv = id === 'language-baselines' ? ['scripts/language_baselines.py', '--out-dir', base + '/' + id, '--text-directory', practice + '/text']
    : id === 'matched-budget' ? ['scripts/matched_budget_lab.py', '--out-dir', base + '/' + id, '--practice-directory', practice]
    : ['scripts/retrieval_lab.py', '--out-dir', base + '/' + id, '--corpus', practice + '/corpus.json', '--queries', 'src/experiments/retrieval-queries.json', '--model-directory', model];
  await run(python, argv);
  await run(process.execPath, ['scripts/package-evidence-extension.mjs', root, id]);
}
console.log(`All requested candidates are complete under ${root}. Inspect them before adopting; no page source was changed.`);
