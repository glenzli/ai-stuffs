import { access, mkdir, readFile, writeFile, copyFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import path from "node:path";

const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== "--out-dir" || !/^artifacts\/[\w-]+$/.test(args[1])) throw new Error("Usage: node scripts/run-practice-labs.mjs --out-dir artifacts/<new-directory>");
const root = args[1], python = process.env.FRAMEWORK_PYTHON || ".venv-framework/bin/python";
await access(python); await mkdir(root); await mkdir(`${root}/practice`);
const sourceFiles = ["scripts/learning_core.py", "scripts/chunked_ce_reference.py", "scripts/generalization_lab.py", "scripts/text_training_core.py", "scripts/text_project_lab.py", "scripts/export-text-corpus.mjs", "scripts/framework-requirements.txt", "scripts/run-practice-labs.mjs", "scripts/package-practice.mjs", "scripts/experiment-bundles.mjs"];
const hashes = {};
for (const file of sourceFiles) { const target = `${root}/source/${file}`; await mkdir(path.dirname(target), { recursive: true }); await copyFile(file, target); hashes[file] = createHash("sha256").update(await readFile(file)).digest("hex"); }
await writeFile(`${root}/practice/run-sources.json`, JSON.stringify({ files: hashes, source_snapshot: "source/", timing_exclusivity: "runs sequentially; desktop processes not controlled" }, null, 2) + "\n");
const run = (command, arguments_) => new Promise((resolve, reject) => { const child = spawn(command, arguments_, { stdio: "inherit" }); child.once("error", reject); child.once("exit", code => code === 0 ? resolve() : reject(new Error(`Practice command exited ${code}; partial artifacts retained at ${root}`))); });
// Reproduce the saved corpus even if the living Wiki's prose has since changed.
const manifestBytes = await readFile("src/experiment-sources.json"), manifest = JSON.parse(manifestBytes);
const { execFile } = await import("node:child_process"); const { promisify } = await import("node:util");
const { stdout } = await promisify(execFile)(python, ["-c", "import sys,json;sys.path.insert(0,'scripts');import learning_core as c;c.configure_cpu(1);print(json.dumps(c.environment_metadata()))"]);
await writeFile(`${root}/practice/run-environment.json`, JSON.stringify({ started_at: new Date().toISOString(),
  runtime: JSON.parse(stdout), base_manifest_sha256: createHash("sha256").update(manifestBytes).digest("hex"),
  command: [process.execPath, ...process.argv.slice(1)], failures: [] }, null, 2) + "\n", { flag: "wx" });
const savedDirectory = path.dirname(manifest.experiments.practice.report.path);
const savedReportBytes = await readFile(manifest.experiments.practice.report.path);
if (createHash("sha256").update(savedReportBytes).digest("hex") !== manifest.experiments.practice.report.sha256) throw new Error("Saved practice report identity mismatch");
const savedReport = JSON.parse(savedReportBytes);
if (createHash("sha256").update(await readFile(`${savedDirectory}/corpus.json`)).digest("hex") !== savedReport.reports.text.corpus.sha256) throw new Error("Frozen corpus identity mismatch");
await copyFile(`${savedDirectory}/corpus.json`, `${root}/practice/corpus.json`);
const corpus = JSON.parse(await readFile(`${root}/practice/corpus.json`, "utf8"));
await mkdir(`${root}/practice/corpus-sources`);
for (const doc of corpus.documents) {
  const file = `${savedDirectory}/corpus-sources/${doc.id}.html`;
  if (createHash("sha256").update(await readFile(file)).digest("hex") !== doc.source_sha256) throw new Error(`Frozen corpus hash mismatch: ${doc.id}`);
  await copyFile(file, `${root}/practice/corpus-sources/${doc.id}.html`);
}
await run(python, ["scripts/generalization_lab.py", "--out-dir", `${root}/practice/generalization`]);
await run(python, ["scripts/text_project_lab.py", "--out-dir", `${root}/practice`, "--corpus", `${root}/practice/corpus.json`]);
await run(process.execPath, ["scripts/package-practice.mjs", root]);
