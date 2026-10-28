import { access, mkdir, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { annInputs, annSearch } from "../src/scripts/features/ann-index.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
if (args.length && (args.length !== 2 || args[0] !== "--out-dir" || !args[1])) throw new Error("Usage: npm run lab:ann -- [--out-dir <new-directory>]");
const directory = path.resolve(args[1] || path.join(root, "artifacts", `ann-lab-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}`));
const python = process.env.FRAMEWORK_PYTHON || path.join(root, ".venv-framework/bin/python");
await access(python); await mkdir(path.dirname(directory), { recursive: true }); await mkdir(directory);
const cases = [];
for (const query of ["center", "left", "upper"]) for (const nprobe of [1, 2, 3]) {
  const r = annSearch({ query, nprobe }); cases.push({ query, nprobe, exact: r.exact.map(x => x.id), approximate: r.approximate.map(x => x.id), distances: r.approximate.map(x => x.distance) });
}
await writeFile(path.join(directory, "fixture.json"), JSON.stringify({ inputs: annInputs(), cases }) + "\n", { flag: "wx" });
await new Promise((resolve, reject) => {
  const child = spawn(python, ["scripts/ann-reference.py", "--out-dir", directory], { cwd: root, stdio: "inherit" });
  child.once("error", reject); child.once("exit", code => code === 0 ? resolve() : reject(new Error(`Faiss experiment exited with ${code}`)));
});
console.log(`Real CPU indexes verified. Report: ${directory}/report.json`);
