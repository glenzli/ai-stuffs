import { access } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
if (args.length && (args.length !== 2 || args[0] !== "--out-dir" || !args[1])) {
  throw new Error("Usage: node scripts/run-failure-lab.mjs [--out-dir <new-directory>]");
}
const directory = path.resolve(args[1] || path.join(root, "artifacts", `failure-lab-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}`));
const python = process.env.FRAMEWORK_PYTHON || path.join(root, ".venv-framework/bin/python");
await access(python);
await new Promise((resolve, reject) => {
  const child = spawn(python, ["scripts/failure_replay.py", "--out-dir", directory], { cwd: root, stdio: "inherit" });
  child.once("error", reject);
  child.once("exit", code => code === 0 ? resolve() : reject(new Error(`Actual failure/replay experiment exited with ${code}; partial evidence preserved at ${directory}`)));
});
console.log(`Actual faults and deterministic replays verified. Report: ${directory}/report.json`);
