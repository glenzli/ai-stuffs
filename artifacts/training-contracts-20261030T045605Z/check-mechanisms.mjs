// Run from any directory. Default verifies only; --write creates a new result once.
// All examples have fixed author-supplied inputs. No model or vendor benchmark runs.
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseFragment } from "parse5";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const inputs = JSON.parse(await readFile(resolve(here, "inputs.json"), "utf8"));
const sum = a => a.reduce((v, x) => v + x, 0);
const near = (a, b, eps = 1e-8) => assert.ok(Number.isFinite(a) && Math.abs(a - b) <= eps, `${a} != ${b}`);
const sigmoid = t => 1 / (1 + Math.exp(-t));
const derivative = (fn, t) => (fn(t + 1e-5) - fn(t - 1e-5)) / 2e-5;
const walk = (n, p) => [...(p(n) ? [n] : []), ...(n.childNodes || []).flatMap(c => walk(c, p))];
const attr = (n, key) => n.attrs?.find(a => a.name === key)?.value;
const text = n => n.nodeName === "#text" ? n.value : (n.childNodes || []).map(text).join("");
async function table(file, marker) {
  const tree = parseFragment(await readFile(resolve(root, "src/content", file), "utf8"));
  const matches = walk(tree, n => attr(n, marker) !== undefined);
  assert.equal(matches.length, 1, marker);
  const body = walk(matches[0], n => n.tagName === "tbody")[0];
  return walk(body, n => n.tagName === "tr").map(row => row.childNodes.filter(c => c.tagName === "td").map(c => text(c).trim().replaceAll("−", "-")));
}

const { sampling: q, proximal: p, current, advantages: advantage, threshold } = inputs.rollout;
function distribution(values) {
  assert.ok(values.length === 2 && values.every(x => Number.isFinite(x) && x >= 0));
  near(sum(values), 1);
}
function fullWeights(sampling, proximal) {
  distribution(sampling); distribution(proximal);
  assert.ok(proximal.every(x => x > 0), "positive proximal probabilities required");
  assert.ok(sampling.every((x, i) => x > 0 || proximal[i] === 0), "sampling support missing");
  return proximal.map((x, i) => x / sampling[i]);
}
distribution(current);
const weights = fullWeights(q, p);
const logit = Math.log(current[0] / current[1]);
const modes = [
  { label: "忽略采样差异", weights: [1, 1] },
  { label: "完整比率", weights },
  { label: "TIS：权重上限 1.5", weights: weights.map(x => Math.min(x, threshold)) },
  { label: "RS：丢弃 w>1.5 的项", weights: weights.map(x => x > threshold ? 0 : x) },
];
const objective = (t, fixedWeights) => {
  const a = sigmoid(t), pi = [a, 1 - a];
  return sum(q.map((mass, i) => mass * fixedWeights[i] * (pi[i] / p[i]) * advantage[i]));
};
const gradient = (t, fixedWeights) => {
  const a = sigmoid(t);
  return a * (1 - a) * (q[0] * fixedWeights[0] * advantage[0] / p[0] - q[1] * fixedWeights[1] * advantage[1] / p[1]);
};
let rolloutDifferenceChecks = 0;
for (const t of [-0.7, logit, 0.9]) {
  const a = sigmoid(t);
  const target = a * advantage[0] + (1 - a) * advantage[1];
  near(objective(t, weights), target);
  near(gradient(t, weights), a * (1 - a) * (advantage[0] - advantage[1]));
  for (const mode of modes) {
    near(gradient(t, mode.weights), derivative(x => objective(x, mode.weights), t));
    rolloutDifferenceChecks++;
  }
}
assert.throws(() => fullWeights([1, 0], p), /sampling support/);
near(objective(logit, modes[1].weights), 0.1);
assert.ok(Math.abs(gradient(logit, modes[0].weights) - gradient(logit, weights)) > 0.01);
const rolloutRows = modes.map(mode => ({
  label: mode.label, fixedWeights: mode.weights,
  objective: objective(logit, mode.weights), derivative: gradient(logit, mode.weights),
  retainedSamplingMass: sum(q.map((mass, i) => mode.weights[i] === 0 ? 0 : mass)),
}));
assert.deepEqual(await table("rl-posttraining.html", "data-rollout-correction-example"), rolloutRows.map(row => [
  row.label, row.fixedWeights.map(x => Number(x.toPrecision(12))).join(" / "), row.objective.toFixed(6), row.derivative.toFixed(6),
]));

const levels = inputs.fp4.nonnegativeLevels;
function nearestE2M1(value) {
  assert.ok(Number.isFinite(value), "finite input required");
  const magnitude = Math.abs(value);
  let best = levels[0];
  for (const level of levels) if (Math.abs(level - magnitude) < Math.abs(best - magnitude)) best = level;
  return best === 0 ? 0 : Math.sign(value) * best;
}
function blockQuantize(values, blockSize) {
  assert.ok(Number.isInteger(blockSize) && blockSize > 0);
  assert.ok(values.every(Number.isFinite), "finite input required");
  const quantized = [], scales = [];
  for (let start = 0; start < values.length; start += blockSize) {
    const block = values.slice(start, start + blockSize);
    const amax = Math.max(...block.map(Math.abs)), scale = amax === 0 ? 1 : amax / 6;
    scales.push(scale);
    quantized.push(...block.map(x => scale * nearestE2M1(x / scale)));
  }
  return { quantized, scales };
}
const original = inputs.fp4.activations;
assert.equal(original.length, 32);
const quantizations = [
  { label: "高精度输入参考", quantized: original, scales: [], scaleText: "无量化" },
  ...inputs.fp4.blockSizes.map(blockSize => {
    const value = blockQuantize(original, blockSize);
    return { label: blockSize === 32 ? "32 个数共用 scale" : "每 16 个数共用 scale", ...value,
      scaleText: value.scales.map(x => x === 1 ? "1" : x.toFixed(6)).join(" / ") };
  }),
];
let fp4DifferenceChecks = 0;
const fp4Rows = quantizations.map(({ label, quantized, scales, scaleText }) => {
  const y = sum(quantized.map(x => inputs.fp4.weight * x));
  const dW = quantized.map(x => inputs.fp4.upstreamGradient * x);
  for (let i = 0; i < quantized.length; i++) {
    // Quantized activations and scales stay fixed; perturb only one linear weight.
    const fn = wi => sum(quantized.map((x, j) => x * (j === i ? wi : inputs.fp4.weight))) * inputs.fp4.upstreamGradient;
    near(dW[i], derivative(fn, inputs.fp4.weight)); fp4DifferenceChecks++;
  }
  return { label, scales, scaleText, quantized, output: y, dW,
    gradientMaxAbsoluteError: Math.max(...dW.map((g, i) => Math.abs(g - original[i] * inputs.fp4.upstreamGradient))) };
});
near(fp4Rows[0].output, 22.6); near(fp4Rows[1].output, 21); near(fp4Rows[2].output, 22.6);
assert.deepEqual(blockQuantize([0, 0], 16), { quantized: [0, 0], scales: [1] });
assert.deepEqual(blockQuantize([], 16), { quantized: [], scales: [] });
assert.throws(() => blockQuantize([Infinity], 16), /finite/);
for (const value of [0.1, 0.75, 1.6, 3.5, 6]) near(nearestE2M1(-value), -nearestE2M1(value));
assert.equal(nearestE2M1(0.25), 0); assert.equal(nearestE2M1(0.75), 0.5);
assert.deepEqual(await table("numerical-computing.html", "data-fp4-scaling-example"), fp4Rows.map(row => [
  row.label, row.scaleText, row.output.toFixed(6), row.gradientMaxAbsoluteError.toFixed(6),
]));

function rubricScore(checks, rubricWeights) {
  assert.equal(checks.length, rubricWeights.length);
  assert.ok(checks.every(x => x === 0 || x === 1));
  assert.ok(rubricWeights.every(x => Number.isFinite(x) && x >= 0) && sum(rubricWeights) > 0, "positive total of nonnegative weights required");
  return sum(checks.map((x, i) => x * rubricWeights[i])) / sum(rubricWeights);
}
const schemes = inputs.rubric.schemes;
const rubricRows = inputs.rubric.candidates.map(candidate => ({ ...candidate,
  scores: Object.fromEntries(Object.entries(schemes).map(([name, ws]) => [name, rubricScore(candidate.checks, ws)])),
}));
near(rubricRows[0].scores.A, 4 / 7); near(rubricRows[1].scores.A, 3 / 7);
near(rubricRows[0].scores.B, 1 / 5); near(rubricRows[1].scores.B, 4 / 5);
const centeredAdvantages = Object.fromEntries(Object.keys(schemes).map(name => {
  const scores = rubricRows.slice(0, 2).map(row => row.scores[name]);
  const mean = sum(scores) / scores.length;
  return [name, scores.map(score => score - mean)];
}));
assert.ok(centeredAdvantages.A[0] > 0 && centeredAdvantages.B[0] < 0);
const changedChecks = [...inputs.rubric.candidates[inputs.rubric.judgeError.candidateIndex].checks];
changedChecks[inputs.rubric.judgeError.criterionIndex] = inputs.rubric.judgeError.replacement;
const afterJudgeError = rubricScore(changedChecks, schemes.A);
assert.equal(afterJudgeError, 1);
assert.throws(() => rubricScore([1], [0]), /positive total/);
assert.deepEqual(await table("model-evaluation.html", "data-rubric-rewards-example"), rubricRows.map(row => [
  row.label, row.checks.join(" / "), row.scores.A.toFixed(6), row.scores.B.toFixed(6),
]));

const result = {
  schema: "training-contracts-results-v1", passed: true, boundary: inputs.boundary,
  rollout: { inputs: inputs.rollout, modes: rolloutRows, finiteDifferenceChecks: rolloutDifferenceChecks, supportGuardChecked: true },
  fp4: { inputs: inputs.fp4, modes: fp4Rows, finiteDifferenceChecks: fp4DifferenceChecks, zeroEmptyNonfiniteAndTieCasesChecked: true },
  rubric: { inputs: inputs.rubric, candidates: rubricRows, centeredAdvantages, afterJudgeError, zeroDenominatorGuardChecked: true },
  authoredTablesMatched: 3,
};
if (process.argv.includes("--write")) await writeFile(resolve(here, "results.json"), JSON.stringify(result, null, 2) + "\n", { flag: "wx" });
console.log(JSON.stringify(result, null, 2));
