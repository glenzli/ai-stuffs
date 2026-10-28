import { readFile, writeFile, readdir, mkdir, access } from "node:fs/promises";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import path from "node:path";

// Package completed runs; measured values stay owned by the experiment runners.
const root = process.cwd();
const directory = process.argv[2];
if (process.argv.length !== 3 || !directory?.startsWith("artifacts/") || directory.split("/").includes("..")) {
  throw new Error("Usage: node scripts/package-experiments.mjs artifacts/<completed-run-set>");
}
const specs = [
  { id: "learning", title: "小模型学习与泛化", pages: ["learning-run"], schema: "learning-run-v1", status: "passed",
    command: "node scripts/run-learning-lab.mjs --out-dir artifacts/learning-my-run --steps 1200 --seeds 11,29,47 --batch-size 64 --lr 0.003 --threads 1" },
  { id: "failures", title: "故障与确定性回放", pages: ["training-failures"], schema: "failure-replay-v1", status: "passed",
    command: "node scripts/run-failure-lab.mjs --out-dir artifacts/failure-my-run" },
  { id: "dynamics", title: "初始化、宽度与 Batch", pages: ["initialization-width", "batch-noise"], schema: "training-dynamics-v1", status: "complete",
    command: "node scripts/run-dynamics-lab.mjs --out-dir artifacts/dynamics-my-run --write-page-data" },
  { id: "output", title: "分块交叉熵与 CPU 测量", pages: ["chunked-cross-entropy"], schema: "output-lab-v1", status: "passed",
    command: "node scripts/run-output-lab.mjs --out-dir artifacts/output-my-run" },
];
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
async function filesIn(dir) {
  const files = [];
  for (const item of await readdir(dir, { withFileTypes: true })) {
    const target = path.join(dir, item.name);
    if (item.isDirectory()) files.push(...await filesIn(target));
    else if (item.isFile()) files.push(target);
    else throw new Error(`Unsupported artifact entry: ${target}`);
  }
  return files.sort();
}
async function describe(file) {
  const bytes = await readFile(file);
  return { path: file, bytes: bytes.length, sha256: sha(bytes) };
}
async function checksumList(files, relativeTo) {
  return (await Promise.all(files.map(async file => `${sha(await readFile(file))}  ${path.relative(relativeTo, file)}`))).join("\n") + "\n";
}
const sourceFiles = (await filesIn(path.join(directory, "source")))
  .filter(file => !file.includes("/__pycache__/") && !file.endsWith("/.DS_Store"));
await writeFile(path.join(directory, "SOURCE-SHA256SUMS"), await checksumList(sourceFiles, directory), { flag: "wx" });
await mkdir(path.join(directory, "downloads"));
const experiments = {};
for (const spec of specs) {
  const runDir = path.join(directory, spec.id), reportPath = path.join(runDir, "report.json");
  const report = JSON.parse(await readFile(reportPath, "utf8"));
  if (report.schema !== spec.schema || report.status !== spec.status) throw new Error(`Incomplete experiment: ${reportPath}`);
  const files = await filesIn(runDir);
  const checksumsPath = path.join(runDir, "SHA256SUMS");
  await writeFile(checksumsPath, await checksumList(files, runDir), { flag: "wx" });
  const archive = path.join(directory, "downloads", `${spec.id}-sources.tar.gz`);
  try { await access(archive); throw new Error(`Refusing to overwrite: ${archive}`); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  await new Promise((resolve, reject) => {
    const child = spawn("tar", ["-czf", path.resolve(archive), "-C", path.resolve(directory), spec.id, "source", "SOURCE-SHA256SUMS"], { stdio: "inherit" });
    child.once("error", reject);
    child.once("exit", code => code === 0 ? resolve() : reject(new Error(`Archive failed: ${spec.id}`)));
  });
  experiments[spec.id] = { ...spec, runtime: report.runtime,
    created_at: report.created_at || report.provenance?.created_at || report.timestamp,
    report: await describe(reportPath), archive: await describe(archive), checksums: await describe(checksumsPath) };
  console.log(`${spec.id}: report + ${files.length} raw files + exact run-time source packaged`);
}
const manifest = { schema: "model-engineering-experiments-v1", generated_at: new Date().toISOString(),
  source: await describe(path.join(directory, "SOURCE-SHA256SUMS")), experiments };
await writeFile(path.join(directory, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
await writeFile(path.join(root, "src/experiment-sources.json"), JSON.stringify(manifest, null, 2) + "\n");
