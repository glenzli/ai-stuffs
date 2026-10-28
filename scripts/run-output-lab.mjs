import { access, mkdir, copyFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2), options = {};
for (let i = 0; i < args.length; i++) {
  if (["--out-dir", "--correctness-report", "--reference", "--contention-note"].includes(args[i]) && args[i + 1]) options[args[i].slice(2)] = args[++i];
  else if (["--correctness-only", "--timing-only"].includes(args[i])) options[args[i].slice(2)] = true;
  else throw new Error("Usage: node scripts/run-output-lab.mjs [--out-dir <new-directory>] [--correctness-only | --timing-only --correctness-report <passed.json> --reference <reference.json>] [--contention-note <text>]");
}
if (options["correctness-only"] && options["timing-only"]) throw new Error("Select one mode.");
if (options["timing-only"] && (!options["correctness-report"] || !options.reference)) throw new Error("Timing-only must reuse an exact passing correctness report and reference fixture.");
const directory = path.resolve(options["out-dir"] || path.join(root, "artifacts", `output-lab-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}`));
const python = process.env.FRAMEWORK_PYTHON || path.join(root, ".venv-framework/bin/python");
await access(python);
await mkdir(path.dirname(directory), { recursive: true });
await mkdir(directory); // EEXIST protects prior reports; never overwrite runs.
const run = (command, commandArgs) => new Promise((resolve, reject) => {
  const child = spawn(command, commandArgs, { cwd: root, stdio: "inherit", env: { ...process.env, OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1", OPENBLAS_NUM_THREADS: "1" } });
  child.once("error", reject); child.once("exit", code => code === 0 ? resolve() : reject(new Error(`${command} exited with ${code}`)));
});
const reference = path.join(directory, "reference.json");
if (options["timing-only"]) await copyFile(path.resolve(options.reference), reference, 1);
else await run(process.execPath, [path.join(root, "scripts/export-chunked-ce-fixtures.mjs"), reference]);
const pythonArgs = [path.join(root, "scripts/chunked_ce_reference.py"), "--out-dir", directory];
if (options["correctness-only"]) pythonArgs.push("--correctness-only");
if (options["timing-only"]) pythonArgs.push("--timing-only", "--correctness-report", path.resolve(options["correctness-report"]));
if (options["contention-note"]) pythonArgs.push("--contention-note", options["contention-note"]);
await run(python, pythonArgs);
console.log(`Saved ${options["correctness-only"] ? "correctness" : "output lab"} artifacts: ${directory}`);
