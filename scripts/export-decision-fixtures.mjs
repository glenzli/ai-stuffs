import { writeFile } from "node:fs/promises";
import { contextParallel } from "../src/scripts/features/context-parallel.js";
import { attentionBackwardInputs } from "../src/scripts/features/attention-backward.js";
import { policyInputs, policyTrajectory, enumeratedPolicyGradient } from "../src/scripts/features/rl-update.js";
import { quantizedLinear, quantizedLinearInputs } from "../src/scripts/features/quantized-linear.js";
import { pairedExample, pairedStatistics, checkpointSelection } from "../src/scripts/features/paired-evaluation.js";
import { annInputs, annSearch } from "../src/scripts/features/ann-index.js";

const context = [], policy = [], quantization = [], statistics = [], selection = [], ann = [];
for (const split of [1, 2, 3, 4, 5]) for (const block of [1, 2, 3, 5]) for (const mask of ["causal", "full", "empty-first"]) for (const fault of ["none", "local-softmax", "local-position", "drop-remote", "local-mean"]) {
  const options = { split, block, mask, fault }, r = contextParallel(options);
  context.push({ options, O: r.O, dQ: r.dQ, dK: r.dK, dV: r.dV, loss: r.loss });
}
const extractStep = r => ({ weights: r.weights, oldWeights: r.oldWeights, updated: r.updated, gradient: r.gradient, loss: r.loss, kl: r.kl, count: r.count, ratios: r.rows.map(row => row.ratio), probabilities: r.rows.map(row => row.p) });
for (const normalization of ["token", "sequence"]) for (const fault of ["none", "old-graph", "include-controls", "refresh-old", "wrong-denominator"]) for (const limit of [0, 1, 2, 3]) for (const beta of [0, .08, .5]) for (const equal of [false, true]) {
  const options = { normalization, fault, limit, beta, rewards: equal ? [1, 1, 1, 1] : policyInputs().rewards, steps: 4 }, r = policyTrajectory(options);
  policy.push({ options, history: r.history.map(extractStep), weights: r.weights });
}
for (const scenario of ["standard", "outlier", "saturation", "zero"]) for (const mode of ["w8a8", "w4a16"]) for (const fault of mode === "w8a8" ? ["none", "zero-point", "scale-axis", "bias-unit"] : ["none"]) {
  const options = { scenario, mode, fault }, r = quantizedLinear(options);
  quantization.push({ options, inputs: quantizedLinearInputs(scenario), floating: r.floating, restored: r.restored, actual: r.integerRestored, output: r.output, qW: r.qW, qX: r.qX, qBias: r.qBias });
}
for (const kind of ["correlated", "identical", "multi-seed"]) for (const mode of ["row", "group"]) for (const randomSeed of [0, 73, 4294967295]) {
  const data = pairedExample(kind), seeds = [...new Set(data.rows.map(row => row.seed))];
  for (const seed of seeds) {
    const options = { mode, seed, randomSeed, repetitions: 2000 }, r = pairedStatistics({ data, ...options });
    statistics.push({ data, options, delta: r.delta, interval: r.interval, distribution: r.distribution, permutation: r.permutation });
  }
}
for (const candidates of [1, 5, 10, 30, 50]) selection.push(checkpointSelection({ candidates }));
for (const nprobe of [1, 2, 3]) for (const k of [1, 3, 5, 12]) for (const query of ["center", "left", "upper"]) for (const filter of ["all", "A", "B", "empty"]) for (const version of ["aligned", "stale", "rebuilt"]) {
  const options = { nprobe, k, query, filter, version }, r = annSearch(options);
  ann.push({ options, exact: r.exact.map(x => x.id), approximate: r.approximate.map(x => x.id), distances: r.approximate.map(x => x.distance), annRecall: r.annRecall, oracleRecall: r.oracleRecall, taskRecall: r.taskRecall, compatible: r.compatible });
}
const output = { schema: 1, contextInputs: attentionBackwardInputs(), context, policyInputs: policyInputs(), policy, expectation: enumeratedPolicyGradient(), quantization, statistics, selection, annInputs: annInputs(), ann };
await writeFile(process.argv[2], JSON.stringify(output) + "\n", { flag: "wx" });
console.log(`Exported ${context.length} context, ${policy.length} four-step policy, ${quantization.length} quantization, ${statistics.length} statistics, ${selection.length} selection and ${ann.length} ANN configurations.`);
