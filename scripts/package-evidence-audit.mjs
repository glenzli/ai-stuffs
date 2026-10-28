import { readFile, writeFile, copyFile } from 'node:fs/promises';
import { hash, readJSON, describe, archiveBundle, writeCandidate } from './experiment-bundles.mjs';
import { evidenceAuditStudies } from './evidence-audit-studies.mjs';

const base = process.argv[2], selected = process.argv[3];
if (!/^artifacts\/[\w-]+$/.test(base || '') || process.argv.length !== 4 || !evidenceAuditStudies.some(row => row.id === selected)) throw new Error('Usage: node scripts/package-evidence-audit.mjs artifacts/<complete-audit> <study-id>');
const definition = evidenceAuditStudies.find(row => row.id === selected), directory = `${base}/${selected}`, environment = await readJSON(`${directory}/run-environment.json`);
if (environment.runtime.torch !== '2.9.0' || environment.runtime.numpy !== '2.3.4') throw new Error('Unexpected reference environment');
for (const [file, digest] of Object.entries(environment.source_sha256)) if (hash(await readFile(file)) !== digest || hash(await readFile(`${directory}/source/${file}`)) !== digest) throw new Error(`Executed source changed: ${file}`);
const report = await readJSON(`${directory}/${selected}/report.json`);
if (report.schema !== definition.schema || report.status !== 'passed') throw new Error('Incomplete audit');
await copyFile(`${directory}/run-environment.json`, `${directory}/${selected}/run-environment.json`);
const snapshot = { ...report, provenance: { report_sha256: hash(await readFile(`${directory}/${selected}/report.json`)), source_sha256: environment.source_sha256, started_at: environment.started_at, runtime: environment.runtime } };
const module = `// Generated from complete saved follow-up evidence; no measured value is hand-edited.\nexport const ${definition.variable} = ${JSON.stringify(snapshot)};\n`;
if (Buffer.byteLength(module) > 1024 * 1024) throw new Error('Display projection exceeds 1 MiB');
await writeFile(`${directory}/${selected}/page-data.js`, module, { flag: 'wx' });
const assets = await archiveBundle(directory, selected);
await writeCandidate({ root: directory, baseManifestSha256: environment.base_manifest_sha256,
  entry: { id: selected, title: definition.title, pages: definition.pages, schema: definition.schema, status: 'passed', runtime: environment.runtime, created_at: environment.started_at,
    command: `node scripts/run-evidence-audit.mjs --out-dir artifacts/audit-my-run --only ${selected}`, ...assets },
  projection: { ...await describe(`${directory}/${selected}/page-data.js`), target: `src/scripts/features/${selected}-data.js` } });
console.log(`${selected}: ${(assets.archive.bytes / 1048576).toFixed(2)} MiB candidate; accepted page sources unchanged.`);
