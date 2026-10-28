import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const allowed = new Set(["--out-dir", "--only", "--write-page-data", "--reanalyze"]);
let directory = path.join(root, "artifacts", `dynamics-lab-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}`);
let only = "all", pageData = false, reanalysis = null;
for (let i = 0; i < args.length; i++) {
  if (!allowed.has(args[i])) throw new RangeError(`Unknown argument: ${args[i]}`);
  if (args[i] === "--write-page-data") { pageData = true; continue; }
  if (!args[i + 1] || args[i + 1].startsWith("--")) throw new RangeError(`Missing value: ${args[i]}`);
  if (args[i] === "--out-dir") directory = path.resolve(args[++i]);
  else if (args[i] === "--reanalyze") reanalysis = path.resolve(args[++i]);
  else only = args[++i];
}
if (!["all", "width"].includes(only)) throw new RangeError("--only must be all or width");
if (pageData && only !== "all") throw new RangeError("Page data requires both slices from a complete run");
const python = process.env.FRAMEWORK_PYTHON || path.join(root, ".venv-framework/bin/python");
await access(python);
await mkdir(path.dirname(directory), { recursive: true });
// Python atomically creates the new directory and refuses existing artifacts.
await new Promise((resolve, reject) => {
  const child = spawn(python, ["scripts/training_dynamics.py", "--out-dir", directory, "--only", only,
    ...(reanalysis ? ["--reanalyze", reanalysis] : [])], { cwd: root, stdio: "inherit" });
  child.once("error", reject);
  child.once("exit", code => code === 0 ? resolve() : reject(new Error(`Dynamics experiment exited with ${code}`)));
});
if (pageData) {
  const bytes = await readFile(path.join(directory, "report.json"));
  const report = JSON.parse(bytes);
  if (report.schema !== "training-dynamics-v1" || report.status !== "complete") throw new RangeError("Incomplete dynamics report");
  const provenance = { ...report.provenance, report_sha256: createHash("sha256").update(bytes).digest("hex"),
    source_report: path.relative(root, path.join(directory, "report.json")) };
  const compactWidthRun = run => ({ run_id: run.run_id, width: run.width, seed: run.seed,
    parameterization: run.parameterization, base_lr: run.base_lr, role: run.role,
    updates: run.updates, consumed_examples: run.consumed_examples, parameter_count: run.parameter_count,
    selected_for_transfer: run.selected_for_transfer, ledger: run.ledger, coordinates: run.coordinates,
    curve: run.curve.filter(point => point.validation !== null), final_validation: run.final_validation, test: run.test });
  const width = { ...report.width, runs: report.width.runs.map(run => ({
    run_id: run.run_id, width: run.width, seed: run.seed, parameterization: run.parameterization,
    base_lr: run.base_lr, role: run.role, final_validation: run.final_validation,
    selected_for_transfer: !!run.selected_for_transfer })),
    selected_runs: report.width.selected_runs.map(compactWidthRun) };
  const briefStats = row => ({ rms: row.rms, mean_abs: row.mean_abs, max_abs: row.max_abs });
  const depth = { ...report.depth, runs: report.depth.runs.map(run => ({
    run_id: run.run_id, depth: run.depth, norm: run.norm, seed: run.seed, residual_rule: run.residual_rule,
    residual_scale: run.residual_scale, base_lr: run.base_lr, parameter_count: run.parameter_count,
    config: run.config, minibatch_losses: run.minibatch_losses,
    probes: run.probes.map(probe => ({ ...probe, parameters: probe.parameters
      .filter(row => row.name.endsWith("mlp.2.weight")).map(row => ({ name: row.name, gradient: briefStats(row.gradient) })),
      rows: probe.rows.filter(row => !/\.attention$|\.mlp$/.test(row.layer))
        .map(row => ({ layer: row.layer, activation: briefStats(row.activation), gradient: briefStats(row.gradient) })) })) })) };
  const noise = { ...report.noise, enumerations: report.noise.enumerations.map(row => ({ ...row, rows: undefined })) };
  const compactEvaluation = (evaluation, predictions) => Object.fromEntries(Object.entries(evaluation)
    .map(([split, value]) => [split, { metrics: value.metrics,
      predictions: predictions ? value.predictions.slice(0, 8).map(row => ({
        id: row.id, group_id: row.group_id, content: row.content, target: row.target,
        length: row.length, predicted: row.predicted, exact_match: row.exact_match,
        token_accuracy: row.token_accuracy, teacher_forced_loss: row.teacher_forced_loss,
        terminated: row.terminated })) : undefined }]));
  const batch = { ...report.batch, page_prediction_subset: "first 8 records of each saved evaluation split; metrics use the complete split",
    runs: report.batch.runs.map(row => ({ run_id: row.run_id, seed: row.seed, batch_size: row.batch_size,
      base_lr: row.base_lr, config: row.config, parameter_count: row.parameter_count,
      updates: row.updates, consumed_records: row.consumed_records,
      consumed_input_tokens: row.consumed_input_tokens, consumed_supervised_tokens: row.consumed_supervised_tokens,
      scheduler: row.scheduler, data_stream_sha256: row.data_stream_sha256,
      evaluation_split_sha256: row.evaluation_split_sha256, initial_model_sha256: row.initial_model_sha256,
      curve: row.curve, initial_noise: row.initial_noise, final_noise: row.final_noise,
      cached_logits_max_error: row.cached_logits_max_error, resume: row.resume,
      initial_evaluation: compactEvaluation(row.initial_evaluation, false),
      final_evaluation: compactEvaluation(row.final_evaluation, true) })) };
  const exports = [
    ["initialization-width", "initializationWidthData", { schema: "initialization-width-v1", status: "complete", runtime: report.runtime, provenance, width, depth }],
    ["batch-noise", "batchNoiseData", { schema: "batch-noise-v1", status: "complete", runtime: report.runtime, provenance, noise, batch }],
  ];
  for (const [slug, name, data] of exports) {
    const source = `// Generated from ${provenance.source_report}; do not hand-edit measured values.\nexport const ${name} = ${JSON.stringify(data)};\n`;
    const size = Buffer.byteLength(source);
    if (size > 500 * 1024) throw new RangeError(`${slug} default snapshot exceeds 500 KiB (${size} bytes)`);
    await writeFile(path.join(root, "src/scripts/features", `${slug}-data.js`), source);
    console.log(`${slug} default snapshot: ${size.toLocaleString("en-US")} bytes`);
  }
}
console.log(`Dynamics experiment complete: ${directory}`);
