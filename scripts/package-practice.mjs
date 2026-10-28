import { readFile, writeFile, readdir, mkdir, copyFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import path from "node:path";

const root = process.argv[2];
if (process.argv.length !== 3 || !/^artifacts\/[\w-]+$/.test(root)) throw new Error("Usage: node scripts/package-practice.mjs artifacts/<completed-practice-set>");
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const read = async file => JSON.parse(await readFile(file, "utf8"));
const describe = async file => { const bytes = await readFile(file); return { path: file, bytes: bytes.length, sha256: hash(bytes) }; };
async function filesIn(directory) {
  const files = [];
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, item.name);
    if (item.isDirectory()) files.push(...await filesIn(file));
    else if (item.isFile()) files.push(file);
    else throw new Error(`Unsupported file entry: ${file}`);
  }
  return files.sort();
}
async function checksums(directory, output) {
  const rows = [];
  for (const file of await filesIn(directory)) rows.push(`${hash(await readFile(file))}  ${path.relative(directory, file)}`);
  await writeFile(output, rows.join("\n") + "\n", { flag: "wx" });
}
const reports = {};
for (const id of ["generalization", "text", "reproduction", "profile"]) reports[id] = await read(`${root}/practice/${id}/report.json`);
if (!reports.generalization.baseline_predictions_reproduced || !reports.text.resume.passed || !reports.reproduction.matched_repeat_exact || !reports.profile.numerical_comparison.passed) throw new Error("Practice correctness did not pass");
const previous = await read("src/experiment-sources.json");
// Runtime identity comes from the saved experiment environment, not a guessed host.
const { execFile } = await import("node:child_process");
const { promisify } = await import("node:util");
const { stdout } = await promisify(execFile)(process.env.FRAMEWORK_PYTHON || ".venv-framework/bin/python", ["-c", "import sys,json;sys.path.insert(0,'scripts');import learning_core as c;c.configure_cpu(1);print(json.dumps(c.environment_metadata()))"]);
const actualRuntime = JSON.parse(stdout);
if (actualRuntime.torch !== "2.9.0" || actualRuntime.numpy !== "2.3.4") throw new Error("Unexpected reference environment");
const sourceBytes = await readFile(`${root}/practice/run-sources.json`), sources = JSON.parse(sourceBytes);
for (const [file, sha] of Object.entries(sources.files)) if (hash(await readFile(`${root}/source/${file}`)) !== sha || hash(await readFile(file)) !== sha) throw new Error(`Run-time source changed: ${file}`);
const corpus = await read(`${root}/practice/corpus.json`);
for (const doc of corpus.documents) if (hash(await readFile(`${root}/practice/corpus-sources/${doc.id}.html`)) !== doc.source_sha256) throw new Error(`Corpus source changed: ${doc.id}`);
const report = { schema: "model-practice-v1", status: "passed", created_at: new Date().toISOString(), runtime: actualRuntime,
  provenance: { run_id: path.basename(root), report_path: `${root}/practice/report.json`, source_sha256: sources.files,
    comparison_source_report: reports.generalization.source_report, comparison_source_report_sha256: reports.generalization.source_report_sha256,
    no_downloaded_models_or_corpus: true, failed_path_attempt: "First attempt failed while serializing relative paths; its artifacts remain separate. Budgets and data were unchanged." },
  scope: "Four bounded CPU practice studies; correctness status does not mean fluent text, length extrapolation or GPU speed", reports };
try {
  const history = `${root}/practice/measurement-history/initial-concurrent-profile/report.json`;
  report.provenance.profile_remeasurement = { previous_report_sha256: hash(await readFile(history)),
    previous_report: "measurement-history/initial-concurrent-profile/report.json", workload_changed: false,
    reason: "Initial profile overlapped another experiment's tail; repeat the same frozen workload after other experiment processes completed. Training, data and reproduction outcomes are unchanged." };
} catch (error) { if (error.code !== "ENOENT") throw error; }
await writeFile(`${root}/practice/report.json`, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
const cap = rows => [...new Set(rows.map(r => r.length))].flatMap(length => rows.filter(r => r.length === length).slice(0, 6));
const snapshot = structuredClone(report);
for (const run of snapshot.reports.generalization.baselines) {
  run.position_summary = [...new Set(run.teacher_forced.map(row => row.length))].map(length => {
    const rows = run.teacher_forced.filter(row => row.length === length), positions = rows.flatMap(row => row.positions.slice(0, -1));
    return { length, records: rows.length, teacher_content_accuracy: positions.filter(p => p.target === p.top1).length / positions.length,
      mean_premature_eos_probability: positions.reduce((sum, p) => sum + p.eos_probability, 0) / positions.length,
      mean_gold_eos_probability: rows.reduce((sum, row) => sum + row.positions.at(-1).eos_probability, 0) / rows.length };
  });
  delete run.teacher_forced;
  for (const key of ["standard", "eos_diagnostic", "shifted_positions"]) run[key] = cap(run[key]);
}
for (const run of snapshot.reports.generalization.extended) run.predictions = cap(run.predictions);
snapshot.provenance.report_sha256 = hash(await readFile(`${root}/practice/report.json`));
snapshot.provenance.prediction_subset = "first six saved records at each length; summaries and metrics retain the entire split; position statistics are means over all saved teacher-forced positions";
const module = `// Generated from a complete saved practice report; measured outcomes are never hand-edited.\nexport const practiceSnapshot = ${JSON.stringify(snapshot)};\n`;
if (Buffer.byteLength(module) > 1024 * 1024) throw new Error("Practice snapshot exceeds 1 MiB");
await writeFile(`${root}/practice/page-data.js`, module, { flag: "wx" });
await copyFile(`${root}/practice/page-data.js`, "src/scripts/features/practice-data.js");
await checksums(`${root}/practice`, `${root}/practice/SHA256SUMS`);
const sourceRows = [];
for (const file of await filesIn(`${root}/source`)) sourceRows.push(`${hash(await readFile(file))}  ${path.relative(root, file)}`);
await writeFile(`${root}/SOURCE-SHA256SUMS`, sourceRows.join("\n") + "\n", { flag: "wx" });
await mkdir(`${root}/downloads`);
const archive = `${root}/downloads/practice-sources.tar.gz`;
await new Promise((resolve, reject) => {
  const child = spawn("tar", ["-czf", path.resolve(archive), "-C", path.resolve(root), "practice", "source", "SOURCE-SHA256SUMS"], { stdio: "inherit" });
  child.once("error", reject); child.once("exit", code => code === 0 ? resolve() : reject(new Error("Practice archive failed")));
});
previous.experiments.practice = { id: "practice", title: "从诊断到自然语言项目的四项实战", pages: ["generalization-analysis", "text-training-project", "reproducibility-lab", "training-step-profile"],
  schema: report.schema, status: "passed", runtime: actualRuntime, created_at: report.created_at,
  command: "node scripts/run-practice-labs.mjs --out-dir artifacts/practice-my-run",
  report: await describe(`${root}/practice/report.json`), archive: await describe(archive), checksums: await describe(`${root}/practice/SHA256SUMS`),
  source: await describe(`${root}/SOURCE-SHA256SUMS`) };
await writeFile("src/experiment-sources.json", JSON.stringify(previous, null, 2) + "\n");
await writeFile(`${root}/manifest.json`, JSON.stringify(previous.experiments.practice, null, 2) + "\n", { flag: "wx" });
console.log(`Packaged four studies; snapshot ${Buffer.byteLength(module)} bytes; archive ${(previous.experiments.practice.archive.bytes / 1048576).toFixed(1)} MiB`);
