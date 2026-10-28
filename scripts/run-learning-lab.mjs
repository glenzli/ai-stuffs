import { access } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const python = process.env.FRAMEWORK_PYTHON || path.join(root, ".venv-framework/bin/python");
const args = process.argv.slice(2);
const allowed = new Set(["--out-dir", "--steps", "--seeds", "--batch-size", "--lr", "--threads", "--snapshot-out", "--revalidate-report"]);
for (let i = 0; i < args.length; i += 2) {
  if (!allowed.has(args[i]) || !args[i + 1] || args[i + 1].startsWith("--")) throw new Error("Usage: node scripts/run-learning-lab.mjs [--out-dir <new directory>] [--steps N] [--seeds 11,29,47] [--batch-size N] [--lr X] [--threads N] [--snapshot-out <new JS file>]");
}
await access(python);
await new Promise((resolve, reject) => {
  const child = spawn(python, ["-u", "scripts/learning-run.py", ...args], { cwd: root, stdio: "inherit" });
  child.once("error", reject);
  child.once("exit", code => code === 0 ? resolve() : reject(new Error(`Learning lab exited with ${code}`)));
});
