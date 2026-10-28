import { mkdir, access } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2), outIndex = args.indexOf("--out-dir");
const directory = path.resolve(outIndex < 0 ? path.join(root, "artifacts", `framework-lab-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}`) : args[outIndex + 1]);
const python = process.env.FRAMEWORK_PYTHON || path.join(root, ".venv-framework/bin/python");
try { await access(python); } catch { throw new Error("Create .venv-framework and install scripts/framework-requirements.txt, or set FRAMEWORK_PYTHON to that environment's Python."); }
// New directories prevent an experiment from overwriting previous evidence.
await mkdir(path.dirname(directory), { recursive: true });
await mkdir(directory);
function run(command, commandArgs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, commandArgs, { cwd: root, stdio: "inherit" });
    child.once("error", reject);
    child.once("exit", code => code === 0 ? resolve() : reject(new Error(`${path.basename(command)} exited with ${code}`)));
  });
}
const fixture = path.join(directory, "reference.json");
await run(process.execPath, ["scripts/export-framework-fixtures.mjs", fixture]);
await run(python, ["scripts/framework-reference.py", "--fixtures", fixture, "--out-dir", directory]);
console.log(`Framework experiment passed. Records: ${directory}`);
