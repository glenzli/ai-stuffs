import { readFile, writeFile, copyFile, mkdir } from "node:fs/promises";
import { hash, readJSON, describe, archiveBundle, writeCandidate } from "./experiment-bundles.mjs";

const base = process.argv[2];
if (!/^artifacts\/[\w-]+$/.test(base || "") || process.argv.length > 4) throw new Error("Usage: node scripts/package-evidence-extension.mjs artifacts/<complete-extension> [study-id]");
const definitions = [
  { id: "language-baselines", title: "自然语言模型的词频与相邻统计基线", pages: ["text-training-project"], schema: "language-baselines-v1", variable: "languageBaselines" },
  { id: "matched-budget", title: "相同监督 token 预算的长度支持对照", pages: ["generalization-analysis"], schema: "matched-budget-v1", variable: "matchedBudget" },
  { id: "retrieval", title: "真实中文文档、Embedding 与索引检索实战", pages: ["retrieval-project"], schema: "wiki-retrieval-v1", variable: "retrievalSnapshot" },
];
if (process.argv[3] && !definitions.some(row => row.id === process.argv[3])) throw new Error("Unknown study ID");
for (const definition of definitions.filter(row => !process.argv[3] || row.id === process.argv[3])) {
  const { id, variable } = definition, root = `${base}/${id}`, environment = await readJSON(`${root}/run-environment.json`);
  for (const [file, digest] of Object.entries(environment.source_sha256)) {
    if (hash(await readFile(file)) !== digest || hash(await readFile(`${root}/source/${file}`)) !== digest) throw new Error(`Executed source changed: ${file}`);
  }
  const report = await readJSON(`${root}/${id}/report.json`);
  if (report.schema !== definition.schema || report.status !== "passed") throw new Error(`Incomplete experiment: ${id}`);
  await copyFile(`${root}/run-environment.json`, `${root}/${id}/run-environment.json`);
  const snapshot = structuredClone(report);
  if (id === "language-baselines") {
    snapshot.transformer = report.transformer.map(run => ({ seed: run.seed, evaluation: run.evaluation }));
    await mkdir(`${root}/${id}/inputs`);
    for (const [name, file] of Object.entries(report.source)) {
      if (hash(await readFile(file.path)) !== file.sha256) throw new Error("Baseline input changed");
      await copyFile(file.path, `${root}/${id}/inputs/${name}`);
    }
  }
  if (id === "matched-budget") {
    for (const run of snapshot.runs) run.predictions = [...new Set(run.predictions.map(r => r.length))].flatMap(length => run.predictions.filter(r => r.length === length).slice(0, 6));
    snapshot.prediction_subset = "first six saved records per length; all metric summaries use the full evaluation set";
  }
  snapshot.provenance = { report_sha256: hash(await readFile(`${root}/${id}/report.json`)), source_sha256: environment.source_sha256,
    started_at: environment.started_at, runtime: environment.runtime };
  const module = `// Generated from this study's complete saved report.\nexport const ${variable} = ${JSON.stringify(snapshot)};\n`;
  if (Buffer.byteLength(module) > 1024 * 1024) throw new Error(`Display projection exceeds 1 MiB: ${id}`);
  await writeFile(`${root}/${id}/page-data.js`, module, { flag: "wx" });
  const assets = await archiveBundle(root, id);
  const entry = { id, title: definition.title, pages: definition.pages, schema: definition.schema, status: "passed", runtime: environment.runtime,
    created_at: environment.started_at, command: "node scripts/run-evidence-extension.mjs --out-dir artifacts/extension-my-run", ...assets };
  await writeCandidate({ root, entry, baseManifestSha256: environment.base_manifest_sha256,
    projection: { ...await describe(`${root}/${id}/page-data.js`), target: `src/scripts/features/${id}-data.js` } });
  console.log(`${id}: complete candidate, ${(assets.archive.bytes / 1048576).toFixed(1)} MiB; page sources unchanged`);
}
