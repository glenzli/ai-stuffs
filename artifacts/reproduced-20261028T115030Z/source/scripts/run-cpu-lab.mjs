import { mkdir, readFile, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import { initialTrainingState, runTraining, restoreTrainingState, checkpointExperiment } from "../src/scripts/features/checkpoint-replay.js";
import { cacheConsistency } from "../src/scripts/features/position-cache.js";

const args = process.argv.slice(2);
const argument = name => { const index = args.indexOf(name); return index < 0 ? null : args[index + 1]; };
const writeJson = (file, value) => writeFile(file, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });

if (args.includes("--resume")) {
  const checkpoint = JSON.parse(await readFile(argument("--resume"), "utf8"));
  const state = restoreTrainingState(checkpoint, argument("--omit") || "none");
  const rows = runTraining(state, 8 - checkpoint.step);
  await writeJson(argument("--output"), { rows, state });
} else {
  const directory = path.resolve(argument("--out-dir") || `artifacts/cpu-lab-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}`);
  await mkdir(directory, { recursive: true });
  const baselineState = initialTrainingState(), baseline = runTraining(baselineState, 8);
  const state = initialTrainingState(); runTraining(state, 3);
  const checkpointPath = path.join(directory, "checkpoint.json");
  await writeJson(checkpointPath, state);
  await writeJson(path.join(directory, "uninterrupted.json"), { rows: baseline, state: baselineState });
  const comparisons = [];
  for (const omission of ["none", "optimizer", "scheduler", "rng", "data", "weights-only"]) {
    const output = path.join(directory, `resume-${omission}.json`);
    const child = spawnSync(process.execPath, [fileURLToPath(import.meta.url), "--resume", checkpointPath, "--omit", omission, "--output", output], { encoding: "utf8" });
    if (child.status !== 0) throw new Error(child.stderr || `恢复进程退出：${child.status}`);
    const restored = JSON.parse(await readFile(output, "utf8"));
    const reference = checkpointExperiment({ omission });
    assert.deepEqual(restored.rows, reference.rows.map(row => row.resumed));
    if (omission === "none") assert.deepEqual(restored.state, baselineState);
    else assert.ok(reference.firstDivergence !== null, `${omission} should expose a trajectory difference`);
    comparisons.push({ omission, firstDivergence: reference.firstDivergence, maxParameterError: reference.maxError });
  }
  const cache = cacheConsistency({ padding: 2, chunk: 2 });
  assert.ok(cache.maxError < 1e-12 && cache.invalidRows === 0);
  const report = { runtime: process.version, platform: process.platform, architecture: process.arch, dtype: "JavaScript binary64",
    model: "single-head pre-RMSNorm block; hidden=2; FFN=3; vocabulary=3; 2 input positions; last-position CE",
    training: { optimizer: "AdamW", beta1: .9, beta2: .99, epsilon: 1e-8, decay: .01, steps: 8, checkpointAfter: 3,
      data: "four fixed examples; additive input noise in [-0.05,0.05)", evaluation: "trajectory equivalence; no generalization claim" },
    cache: { model: "separate fixed 4D attention example with RoPE", maxLogitError: cache.maxError }, comparisons };
  await writeJson(path.join(directory, "report.json"), report);
  console.table(comparisons);
  console.log(`CPU 实验通过；完整恢复与未中断运行逐值一致。\n记录：${directory}`);
}
