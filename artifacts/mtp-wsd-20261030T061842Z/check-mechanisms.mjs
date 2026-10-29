// Fixed teaching inputs only. Default verifies; --write creates results once.
// No vendor model, API, GPU kernel, throughput or benchmark is executed.
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseFragment } from "parse5";
import { speculativeMass } from "../../src/scripts/features/speculative-probability.js";

const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, "../..");
const inputs = JSON.parse(await readFile(resolve(here, "inputs.json"), "utf8"));
const sum = a => a.reduce((s, x) => s + x, 0);
const dot = (a, b) => sum(a.map((x, i) => x * b[i]));
const near = (a, b, eps = 1e-8) => assert.ok(Number.isFinite(a) && Math.abs(a - b) <= eps, `${a} != ${b}`);
const fd = (fn, x) => (fn(x + 1e-5) - fn(x - 1e-5)) / 2e-5;
const softmax = z => { const m = Math.max(...z), a = z.map(x => Math.exp(x - m)); return a.map(x => x / sum(a)); };
const ce = (z, y) => { const m = Math.max(...z); return m + Math.log(sum(z.map(x => Math.exp(x - m)))) - z[y]; };
const walk = (n, pred) => [...(pred(n) ? [n] : []), ...(n.childNodes || []).flatMap(c => walk(c, pred))];
const attr = (n, name) => n.attrs?.find(a => a.name === name)?.value;
const text = n => n.nodeName === "#text" ? n.value : (n.childNodes || []).map(text).join("");
const clean = s => s.trim().replaceAll("−", "-");
async function marked(file, marker) {
  const tree = parseFragment(await readFile(resolve(root, "src/content", file), "utf8"));
  const found = walk(tree, n => attr(n, marker) !== undefined);
  assert.equal(found.length, 1, marker); return found[0];
}
async function table(file, marker) {
  const body = walk(await marked(file, marker), n => n.tagName === "tbody")[0];
  return walk(body, n => n.tagName === "tr").map(row => row.childNodes.filter(c => c.tagName === "td").map(c => clean(text(c))));
}

const mt = inputs.mtp;
const initialParameters = [mt.theta, ...mt.heads.flat()];
function mtp(parameters, records = [mt.tokens], coefficients = mt.lossWeights) {
  assert.equal(parameters.length, 5);
  assert.ok(parameters.every(Number.isFinite));
  assert.equal(coefficients.length, 2);
  assert.ok(coefficients.every(x => Number.isFinite(x) && x >= 0));
  assert.ok(records.every(record => record.every(x => x === 0 || x === 1)));
  const gradient = Array(5).fill(0), heads = [];
  let loss = 0;
  for (let j = 0; j < 2; j++) {
    const k = mt.offsets[j], positions = [];
    records.forEach((record, r) => {
      for (let t = 0; t + k < record.length; t++) positions.push({ record: r, t, input: record[t], target: record[t + k] });
    });
    let meanLoss = 0, trunkGradient = 0;
    const headGradient = [0, 0], w = parameters.slice(1 + 2 * j, 3 + 2 * j);
    for (const item of positions) {
      const e = mt.embedding[item.input], h = parameters[0] * e, z = w.map(x => x * h), p = softmax(z);
      const dz = p.map((x, i) => x - Number(i === item.target));
      meanLoss += ce(z, item.target) / positions.length;
      trunkGradient += dot(dz, w) * e / positions.length;
      for (let i = 0; i < 2; i++) headGradient[i] += dz[i] * h / positions.length;
    }
    const weight = coefficients[j], weightedLoss = weight * meanLoss, weightedTrunkGradient = weight * trunkGradient;
    loss += weightedLoss; gradient[0] += weightedTrunkGradient;
    for (let i = 0; i < 2; i++) gradient[1 + 2 * j + i] = weight * headGradient[i];
    heads.push({ offset: k, positions, count: positions.length, meanLoss, weight, weightedLoss, weightedTrunkGradient, headGradient });
  }
  return { loss, gradient, heads };
}
const before = mtp(initialParameters);
let mtpFiniteDifferences = 0;
for (const coefficients of [[1, 0], mt.lossWeights, [1, 1]]) {
  const exact = mtp(initialParameters, [mt.tokens], coefficients);
  for (let i = 0; i < 5; i++) {
    const fn = x => { const changed = [...initialParameters]; changed[i] = x; return mtp(changed, [mt.tokens], coefficients).loss; };
    near(exact.gradient[i], fd(fn, initialParameters[i])); mtpFiniteDifferences++;
  }
}
const head1 = mtp(initialParameters, [mt.tokens], [1, 0]), head2 = mtp(initialParameters, [mt.tokens], [0, mt.lossWeights[1]]);
before.gradient.forEach((x, i) => near(x, head1.gradient[i] + head2.gradient[i]));
assert.ok(before.heads[0].weightedTrunkGradient < 0 && before.heads[1].weightedTrunkGradient > 0);
for (const record of mt.emptyAndBoundaryRecords.slice(0, 2)) {
  const empty = mtp(initialParameters, [record]);
  assert.equal(empty.loss, 0); assert.deepEqual(empty.gradient, Array(5).fill(0));
}
const separated = mtp(initialParameters, [[0, 1], [1, 0]]);
assert.deepEqual(separated.heads.map(h => h.count), [2, 0]);
assert.deepEqual(mtp(initialParameters).heads.map(h => h.count), [3, 2]);
const afterParameters = initialParameters.map((x, i) => x - mt.learningRate * before.gradient[i]);
const after = mtp(afterParameters);
assert.ok(after.loss < before.loss);
assert.deepEqual(await table("generation-mechanisms.html", "data-mtp-targets"), mt.tokens.map((_, t) => [
  String(t), mt.tokens.slice(0, t + 1).join(" / "), ...mt.offsets.map(k => t + k < mt.tokens.length ? String(mt.tokens[t + k]) : "不计"),
]));
assert.deepEqual(await table("generation-mechanisms.html", "data-mtp-gradients"), [
  ...before.heads.map(h => [`偏移 ${h.offset}`, String(h.count), h.weightedLoss.toFixed(6), h.weightedTrunkGradient.toFixed(6)]),
  ["合计", String(sum(before.heads.map(h => h.count))), before.loss.toFixed(6), before.gradient[0].toFixed(6)],
]);
assert.equal(clean(text(await marked("generation-mechanisms.html", "data-mtp-after-loss"))), after.loss.toFixed(6));

function validateProbability(p) {
  assert.ok(p.length > 0 && p.every(x => Number.isFinite(x) && x >= 0)); near(sum(p), 1);
}
function kl(p, q) {
  validateProbability(p); validateProbability(q); assert.equal(p.length, q.length);
  assert.ok(q.every((x, i) => p[i] === 0 || x > 0));
  return sum(p.map((x, i) => x === 0 ? 0 : x * Math.log(x / q[i])));
}
const boost = inputs.boosting, teacherCopy = [...boost.teacher], qBefore = softmax(boost.studentLogits);
const boostGradient = qBefore.map((x, i) => x - boost.teacher[i]);
let boostFiniteDifferences = 0;
for (let i = 0; i < 2; i++) {
  const fn = x => { const z = [...boost.studentLogits]; z[i] = x; return kl(boost.teacher, softmax(z)); };
  near(boostGradient[i], fd(fn, boost.studentLogits[i])); boostFiniteDifferences++;
}
const logitsAfter = boost.studentLogits.map((x, i) => x - boost.learningRate * boostGradient[i]);
const boostingRows = [boost.studentLogits, logitsAfter].map((z, i) => {
  const q = softmax(z), mass = speculativeMass(boost.teacher, q);
  mass.output.forEach((x, k) => near(x, boost.teacher[k]));
  return { label: i === 0 ? "更新前" : "一次 logit 更新后", logits: [...z], q, kl: kl(boost.teacher, q), acceptance: sum(mass.accepted), finalOutput: mass.output };
});
assert.deepEqual(boost.teacher, teacherCopy);
assert.ok(boostingRows[1].kl < boostingRows[0].kl && boostingRows[1].acceptance > boostingRows[0].acceptance);
assert.deepEqual(await table("efficient-inference.html", "data-mtp-boosting-example"), boostingRows.map(r => [r.label, r.q[0].toFixed(6), r.kl.toFixed(6), r.acceptance.toFixed(6)]));
const counter = inputs.counterexamples.klVersusAcceptance;
const counterRows = [counter.draftA, counter.draftB].map((q, i) => ({ label: i === 0 ? "A" : "B", q, kl: kl(counter.teacher, q), acceptance: sum(speculativeMass(counter.teacher, q).accepted) }));
assert.ok(counterRows[1].kl < counterRows[0].kl && counterRows[1].acceptance < counterRows[0].acceptance);
assert.deepEqual(await table("efficient-inference.html", "data-mtp-kl-counterexample"), counterRows.map(r => [r.label, r.q.join(" / "), r.kl.toFixed(6), r.acceptance.toFixed(6)]));
const pairConfig = inputs.counterexamples, marginal0 = [0, 0], marginal1 = [0, 0];
pairConfig.jointPairs.forEach((pair, i) => { marginal0[pair[0]] += pairConfig.jointPairProbabilities[i]; marginal1[pair[1]] += pairConfig.jointPairProbabilities[i]; });
const inconsistentMass = marginal0[0] * marginal1[1] + marginal0[1] * marginal1[0];
assert.equal(inconsistentMass, 0.5);

const wsd = inputs.wsd, horizon = wsd.warmupSteps + wsd.stableSteps + wsd.decaySteps;
const evaluate = theta => 0.5 * dot(wsd.curvature, theta.map(x => x * x));
function rate(t) {
  assert.ok(Number.isInteger(t) && t >= 1 && t <= horizon, "schedule step out of range");
  if (t <= wsd.warmupSteps) return wsd.peakLearningRate * t / wsd.warmupSteps;
  if (t <= wsd.warmupSteps + wsd.stableSteps) return wsd.peakLearningRate;
  return wsd.peakLearningRate + (wsd.minimumLearningRate - wsd.peakLearningRate) * (t - wsd.warmupSteps - wsd.stableSteps) / wsd.decaySteps;
}
const perturbation = t => wsd.alternatingPerturbations[(t - 1) % wsd.alternatingPerturbations.length];
const step = (theta, t) => theta.map((x, i) => x - rate(t) * (wsd.curvature[i] * x + perturbation(t)[i]));
const trajectory = [[...wsd.initial]], updates = [];
let wsdFiniteDifferences = 0;
for (let t = 1; t <= horizon; t++) {
  const previous = trajectory.at(-1), noise = perturbation(t), gradient = previous.map((x, i) => wsd.curvature[i] * x + noise[i]);
  for (let i = 0; i < 2; i++) {
    const fn = x => { const theta = [...previous]; theta[i] = x; return evaluate(theta) + dot(noise, theta); };
    near(gradient[i], fd(fn, previous[i])); wsdFiniteDifferences++;
  }
  const theta = step(previous, t);
  trajectory.push(theta); updates.push({ step: t, learningRate: rate(t), before: previous, perturbation: noise, gradient, after: theta, diagnostic: evaluate(theta) });
}
near(rate(1), 0.1); near(rate(2), 0.2); near(rate(8), 0.2); near(rate(9), 0.155); near(rate(12), 0.02);
assert.throws(() => rate(0), /out of range/); assert.throws(() => rate(13), /out of range/);
function average(points, weights) {
  assert.ok(points.length > 0 && points.length === weights.length && points.every(p => p.length === 2 && p.every(Number.isFinite)), "matching two-dimensional points required");
  assert.ok(weights.every(x => Number.isFinite(x) && x >= 0) && sum(weights) > 0, "positive coefficient sum required");
  return [0, 1].map(i => sum(points.map((p, j) => p[i] * weights[j])) / sum(weights));
}
assert.throws(() => average([[1, 2]], [0]), /positive coefficient/);
assert.throws(() => average([[1]], [1]), /two-dimensional/);
const readouts = [];
for (const t of wsd.readoutSteps) {
  const points = trajectory.slice(t - wsd.mergeWindow + 1, t + 1), stage = t === 8 ? "稳定末" : "衰减末";
  const choices = [
    { label: "末次参数", selectedSteps: [t], weights: [1], theta: [...trajectory[t]] },
    { label: "窗口等权平均", selectedSteps: points.map((_, i) => t - wsd.mergeWindow + 1 + i), weights: points.map(() => 1), theta: average(points, points.map(() => 1)) },
    { label: "窗口 1:2:3:4", selectedSteps: points.map((_, i) => t - wsd.mergeWindow + 1 + i), weights: [...wsd.mergeWeights], theta: average(points, wsd.mergeWeights) },
  ];
  choices.forEach(item => readouts.push({ step: t, stage, ...item, diagnostic: evaluate(item.theta) }));
}
assert.ok(readouts[1].diagnostic < readouts[0].diagnostic && readouts[2].diagnostic < readouts[0].diagnostic);
assert.ok(readouts[4].diagnostic > readouts[3].diagnostic && readouts[5].diagnostic > readouts[3].diagnostic);
assert.deepEqual(await table("optimization-dynamics.html", "data-wsd-average-example"), readouts.map(r => [`${r.stage} · step ${r.step}`, r.label, r.theta.map(x => x.toFixed(6)).join(" / "), r.diagnostic.toFixed(6)]));
const meanTheta = readouts[1].theta, resumedStep9 = step(meanTheta, 9), restartedStep1 = step(meanTheta, 1);
assert.ok(resumedStep9.some((x, i) => Math.abs(x - restartedStep1[i]) > 1e-3));
assert.ok(meanTheta.some((x, i) => Math.abs(x - trajectory[8][i]) > 1e-3));
const averageParameter = sum(pairConfig.parameterAverage) / pairConfig.parameterAverage.length;
const outputOfMeanParameter = averageParameter ** 2;
const meanOutput = sum(pairConfig.parameterAverage.map(x => x ** 2)) / pairConfig.parameterAverage.length;
assert.equal(outputOfMeanParameter, 0); assert.equal(meanOutput, 1);

const interval = inputs.publishedIntervals, positionsPerStep = interval.globalBatchSequences * interval.sequenceLength;
const reportPositions = positionsPerStep * interval.reportIterations, recipePositions = positionsPerStep * interval.recipeIterations;
assert.equal(reportPositions, 25165824000); assert.equal(recipePositions, 50331648000);
assert.deepEqual(await table("optimization-dynamics.html", "data-wsd-source-intervals"), [
  ["Super 报告 · 页首 2026-04-03", "1,000 步", reportPositions.toLocaleString("en-US")],
  ["官方配方 · commit 99a7e45", "2,000 步", recipePositions.toLocaleString("en-US")],
]);
const result = {
  schema: "mtp-wsd-results-v1", passed: true, boundary: inputs.boundary,
  mtp: { inputs: mt, initialParameters, before, afterParameters, after, finiteDifferenceChecks: mtpFiniteDifferences, emptyAndRecordBoundaryGuardsChecked: true },
  boosting: { inputs: boost, gradient: boostGradient, rows: boostingRows, finiteDifferenceChecks: boostFiniteDifferences, teacherUnchanged: true, lossIsTeachingForwardKLNotNvidiaLK: true },
  counterexamples: { klVersusAcceptance: counterRows, independentMarginals: { marginal0, marginal1, inconsistentMass }, parameterAverageVersusEnsemble: { averageParameter, outputOfMeanParameter, meanOutput } },
  wsd: { inputs: wsd, updates, trajectory, readouts, finiteDifferenceChecks: wsdFiniteDifferences, continuation: { initialMean: meanTheta, resumedStep9, restartedStep1, originalStep8: trajectory[8], boundary: "New initial weights; neither continuation is a proof of exact original-trajectory recovery." } },
  publishedIntervals: { ...interval, positionsPerStep, reportPositions, recipePositions },
  authoredTablesMatched: 6, authoredScalarValuesMatched: 1,
};
if (process.argv.includes("--write")) await writeFile(resolve(here, "results.json"), JSON.stringify(result, null, 2) + "\n", { flag: "wx" });
else assert.deepEqual(JSON.parse(await readFile(resolve(here, "results.json"), "utf8")), result, "saved results must match the fixed-input recomputation");
console.log(JSON.stringify(result, null, 2));
