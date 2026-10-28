import { access, mkdir } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
if (args.length && (args.length !== 2 || args[0] !== "--out-dir" || !args[1])) throw new Error("Usage: npm run lab:memory -- [--out-dir <new-directory>]");
const directory = path.resolve(args[1] || path.join(root, "artifacts", `memory-lab-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}`));
const python = process.env.FRAMEWORK_PYTHON || path.join(root, ".venv-framework/bin/python");
await access(python); await mkdir(path.dirname(directory), { recursive: true }); await mkdir(directory);
const run = (command, commandArgs) => new Promise((resolve, reject) => {
  const child = spawn(command, commandArgs, { cwd: root, stdio: "inherit" });
  child.once("error", reject); child.once("exit", code => code === 0 ? resolve() : reject(new Error(`${command} exited with ${code}`)));
});
await run(process.execPath, ["scripts/export-memory-fixtures.mjs", path.join(directory, "reference.json")]);
await run(python, ["scripts/memory-reference.py", "--out-dir", directory]);
console.log(`Memory and gradient experiment passed. Report: ${directory}/report.json`);
