import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { contextParallel } from "../src/scripts/features/context-parallel.js";
import { attentionBackwardInputs } from "../src/scripts/features/attention-backward.js";
import { policyInputs, policyStep, policyTrajectory, enumeratedPolicyGradient } from "../src/scripts/features/rl-update.js";
import { quantizedLinear, affineCalibration, roundAway } from "../src/scripts/features/quantized-linear.js";
import { pairedExample, pairedStatistics, validatePairedData, exactPairedPermutation, checkpointSelection } from "../src/scripts/features/paired-evaluation.js";
import { annSearch, annInputs } from "../src/scripts/features/ann-index.js";
import { maxDifference } from "../src/scripts/features/objective-math.js";

const near = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
const matrixNear = (a, b, tolerance = 1e-9) => assert.ok(maxDifference(a, b) < tolerance, `matrix error ${maxDifference(a, b)}`);
test("context partitions, block tails and all masks preserve full outputs and gradients", () => {
  for (const split of [1, 2, 3, 4, 5]) for (const block of [1, 2, 3, 5]) for (const mask of ["causal", "full", "empty-first"]) {
    const r = contextParallel({ split, block, mask }); near(r.outputError, 0); near(r.gradientError, 0); near(r.loss, r.reference.loss);
  }
});
test("context independently differentiated losses agree with finite differences", () => {
  const epsilon = 1e-5;
  for (const fault of ["none", "local-softmax", "local-position", "local-mean"]) {
    const r = contextParallel({ fault });
    for (const name of ["Q", "K", "V"]) for (let i = 0; i < 5; i++) for (let j = 0; j < 2; j++) {
      const plus = attentionBackwardInputs(), minus = attentionBackwardInputs(); plus[name][i][j] += epsilon; minus[name][i][j] -= epsilon;
      near(r[`d${name}`][i][j], (contextParallel({ fault, inputs: plus }).loss - contextParallel({ fault, inputs: minus }).loss) / (2 * epsilon), 1e-8);
    }
  }
});
test("remote gradient loss keeps forward and dQ but removes a KV contribution", () => {
  const good = contextParallel(), bad = contextParallel({ fault: "drop-remote" });
  matrixNear(good.O, bad.O); matrixNear(good.dQ, bad.dQ); assert.ok(bad.gradientError > .01);
  for (let i = 0; i < 5; i++) {
    const rank = i < 2 ? 0 : 1; matrixNear([bad.dK[i]], [good.contributions[rank].dK[i]]);
  }
});
test("context faults have declared masking and partition boundaries", () => {
  near(contextParallel({ split: 5, fault: "drop-remote" }).gradientError, 0);
  near(contextParallel({ split: 5, fault: "local-position" }).gradientError, 0);
  near(contextParallel({ fault: "local-position", mask: "full" }).gradientError, 0);
  near(contextParallel({ split: 5, block: 5, fault: "local-softmax" }).gradientError, 0);
  assert.ok(contextParallel({ fault: "local-mean" }).lossError > .01);
});
test("context communication accounts sent bytes once and handles no-peer case", () => {
  const r = contextParallel(); assert.equal(r.ranks.reduce((s, r) => s + r.sendBytes, 0), r.ranks.reduce((s, r) => s + r.receiveBytes, 0));
  assert.equal(r.ranks.reduce((s, r) => s + r.sendBytes, 0), 480);
  assert.equal(contextParallel({ split: 5 }).ranks[0].sendBytes, 0);
  assert.throws(() => contextParallel({ split: 0 })); assert.throws(() => contextParallel({ block: 0 }));
});
test("policy full parameter gradients match finite differences away from clipping boundaries", () => {
  const epsilon = 1e-5;
  for (const normalization of ["token", "sequence"]) for (const fault of ["none", "old-graph", "include-controls", "wrong-denominator"]) for (const beta of [0, .08]) {
    const weights = policyInputs().weights.map(row => row.map(w => w * 1.4));
    const r = policyStep({ normalization, fault, beta, weights });
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
      const plus = structuredClone(weights), minus = structuredClone(weights); plus[i][j] += epsilon; minus[i][j] -= epsilon;
      const f = w => policyStep({ normalization, fault, beta, weights: w }).loss;
      near(r.gradient[i][j], (f(plus) - f(minus)) / (2 * epsilon), 1e-8);
    }
  }
});
test("policy gradient includes exact KL and zero advantage still permits KL updates", () => {
  const noReward = policyStep({ rewards: [1, 1, 1, 1], beta: 0 }); matrixNear(noReward.gradient, [[0, 0, 0], [0, 0, 0]]);
  assert.ok(maxDifference(policyStep({ rewards: [1, 1, 1, 1], beta: .5 }).gradient, noReward.gradient) > 1e-4);
  const cancelled = policyStep({ fault: "old-graph", beta: 0 }); matrixNear(cancelled.gradient, noReward.gradient);
});
test("policy displayed logits, reference distributions and upstream gradients use one calculation", () => {
  const r = policyStep();
  for (const row of r.rows) {
    near(row.p.reduce((a, b) => a + b, 0), 1); near(row.referenceP.reduce((a, b) => a + b, 0), 1);
    near(row.dLogits.reduce((a, b) => a + b, 0), 0);
    for (let k = 0; k < 3; k++) near(row.logits[k], row.x[0] * r.weights[0][k] + row.x[1] * r.weights[1][k]);
    near(row.referenceLogp, Math.log(row.referenceP[row.action]));
  }
});
test("policy prompt and padding are excluded, zero targets skip updates", () => {
  const r = policyStep(); assert.equal(r.count, 8); assert.ok(r.rows.filter(row => row.kind !== "response").every(row => row.factor === 0));
  const empty = policyStep({ limit: 0 }); assert.equal(empty.loss, null); assert.equal(empty.skipped, true); matrixNear(empty.updated, empty.weights);
  assert.equal(policyStep({ fault: "include-controls", limit: 0 }).count, 8);
  const seq = policyStep({ normalization: "sequence" }); for (let sample = 0; sample < 4; sample++) near(seq.rows.filter(t => t.sample === sample).reduce((s, t) => s + t.factor, 0), .25);
});
test("behavior policy remains fixed over reused rollout updates", () => {
  const good = policyTrajectory({ steps: 4 }), bad = policyTrajectory({ steps: 4, fault: "refresh-old" });
  matrixNear(good.history[0].updated, bad.history[0].updated);
  assert.ok(maxDifference(good.weights, bad.weights) > 1e-4);
  for (const step of good.history) matrixNear(step.oldWeights, good.behavior);
  assert.throws(() => policyTrajectory({ steps: -1 }));
});
test("policy clipping handles positive and negative advantages with finite gradients", () => {
  const r = policyStep({ weights: [[3, -2, .5], [-1, 2, 0]], beta: 0 });
  assert.ok(r.rows.some(t => t.enabled && t.advantage > 0 && t.clipped)); assert.ok(r.rows.some(t => t.enabled && t.advantage < 0 && t.clipped));
  assert.ok(r.gradient.flat().every(Number.isFinite));
});
test("enumerated nine-trajectory policy gradient differentiates the complete expectation", () => {
  const r = enumeratedPolicyGradient(); near(r.rows.reduce((s, row) => s + row.probability, 0), 1);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
    const plus = policyInputs().weights, minus = policyInputs().weights; plus[i][j] += 1e-5; minus[i][j] -= 1e-5;
    near(r.gradient[i][j], (enumeratedPolicyGradient(plus).expectation - enumeratedPolicyGradient(minus).expectation) / 2e-5, 1e-8);
  }
});
test("integer quantization agrees with restored matmul for every declared scenario", () => {
  for (const scenario of ["standard", "outlier", "saturation", "zero"]) {
    const r = quantizedLinear({ scenario }); near(r.arithmeticError, 0);
    for (const row of r.rows) { assert.equal(row.accumulator, row.direct); assert.ok(Number.isSafeInteger(row.accumulator + row.qBias)); }
  }
});
test("integer quantization separates approximation, zero-point, axis and bias faults", () => {
  assert.ok(quantizedLinear().approximationError > 0);
  for (const fault of ["zero-point", "scale-axis", "bias-unit"]) { assert.ok(quantizedLinear({ fault }).arithmeticError > .01); near(quantizedLinear({ scenario: "zero", fault }).arithmeticError, 0); }
  const base = quantizedLinear(), saturated = quantizedLinear({ scenario: "saturation" }); assert.equal(base.saturation, 0); assert.ok(saturated.saturation > 0);
  assert.deepEqual(base.sx, saturated.sx); assert.deepEqual(base.sy, saturated.sy);
});
test("rounding, zero tensors and weight-only recovery have explicit ranges", () => {
  assert.equal(roundAway(-1.5), -2); assert.equal(roundAway(1.5), 2); assert.equal(affineCalibration([0, 0]).scale, 1);
  assert.throws(() => affineCalibration([])); assert.throws(() => affineCalibration([NaN]));
  const r = quantizedLinear({ mode: "w4a16" }); assert.ok(r.qW.flat().every(x => x >= -7 && x <= 7)); assert.equal(r.bytes.weightPayload, 3); assert.ok(r.approximationError > 0);
});
test("paired bootstrap is deterministic and keeps new and old scores coupled", () => {
  const a = pairedStatistics(), b = pairedStatistics(); assert.deepEqual(a.interval, b.interval); near(a.delta, .084375);
  const reversed = pairedExample(); reversed.rows.forEach(row => { [row.old, row.new] = [row.new, row.old]; });
  const negative = pairedStatistics({ data: reversed }); near(negative.delta, -a.delta); near(negative.interval[0], -a.interval[1]);
});
test("cluster bootstrap retains unequal cluster sizes and exposes correlated uncertainty", () => {
  const rows = pairedStatistics({ mode: "row" }), groups = pairedStatistics({ mode: "group" });
  near(rows.delta, groups.delta); assert.equal(rows.units, 16); assert.equal(groups.units, 4);
  assert.ok(groups.interval[1] - groups.interval[0] > rows.interval[1] - rows.interval[0]);
  const groupMean = groups.slices.reduce((s, x) => s + x.delta, 0) / groups.slices.length; assert.ok(Math.abs(groupMean - groups.delta) > .005);
});
test("exact paired sign swapping reproduces tiny enumerable distributions", () => {
  const rows = [1, 2].map(i => ({ id: `${i}`, group: `${i}`, old: 0, new: 1 }));
  const r = exactPairedPermutation(rows); assert.equal(r.permutations, 4); assert.equal(r.p, .5);
  const grouped = exactPairedPermutation(rows.map(row => ({ ...row, group: "same" })), "group"); assert.equal(grouped.p, 1);
  const identical = pairedStatistics({ data: pairedExample("identical") }); assert.deepEqual(identical.interval, [0, 0]); assert.equal(identical.permutation.p, 1);
});
test("paired input rejects invalid records and separates training seeds", () => {
  for (const data of [{ schema: 1, rows: [] }, { schema: 2, rows: pairedExample().rows }, { schema: 1, rows: [{ id: "x", group: "a", old: .2 }] }]) assert.throws(() => validatePairedData(data));
  const duplicate = pairedExample(); duplicate.rows.push(duplicate.rows[0]); assert.throws(() => validatePairedData(duplicate));
  const data = pairedExample("multi-seed"), r = pairedStatistics({ data, seed: "29" }); assert.equal(r.count, 16); assert.equal(r.perSeed.length, 3); assert.equal(r.rows.length, 16);
  const one = pairedStatistics({ data: { schema: 1, rows: [{ id: "q", group: "doc", old: .2, new: .8 }] } }); assert.equal(one.exceedsThreshold, false); assert.equal(one.degenerate, true);
});
test("checkpoint selection keeps independent test noise separate", () => {
  const one = checkpointSelection({ candidates: 1 }), many = checkpointSelection({ candidates: 50 });
  assert.ok(many.selected > one.selected + .1); assert.ok(Math.abs(many.independent - .6) < .025);
  assert.deepEqual(checkpointSelection(), checkpointSelection());
});
test("all IVF lists recover Flat for all filters, queries and encoding versions", () => {
  for (const query of ["center", "left", "upper"]) for (const filter of ["all", "A", "B", "empty"]) for (const version of ["aligned", "stale", "rebuilt"]) for (const k of [1, 3, 5, 12]) {
    const r = annSearch({ nprobe: 3, query, filter, version, k }); assert.deepEqual(r.approximate, r.exact); assert.equal(r.annRecall, r.exact.length ? 1 : null);
  }
});
test("IVF candidate budgets and relevance denominators are independent", () => {
  const small = annSearch(), full = annSearch({ nprobe: 3 }); assert.ok(small.annRecall < 1); assert.ok(small.distanceCount < full.distanceCount);
  assert.equal(full.annRecall, 1); near(full.taskRecall, 2 / 3); assert.equal(annSearch({ query: "left" }).taskRecall, null);
});
test("same-dimensional stale query encoding can pass ANN recall while failing its contract", () => {
  const old = annSearch({ nprobe: 3 }), stale = annSearch({ nprobe: 3, version: "stale" }), rebuilt = annSearch({ nprobe: 3, version: "rebuilt" });
  assert.equal(stale.annRecall, 1); assert.ok(stale.oracleRecall < 1); assert.equal(stale.compatible, false);
  assert.equal(rebuilt.compatible, true); assert.deepEqual(old.approximate.map(x => x.id), rebuilt.approximate.map(x => x.id));
});
test("IVF ties use deterministic IDs and empty ranges have undefined recall", () => {
  const inputs = annInputs(); inputs.documents[1].vector = [...inputs.documents[0].vector];
  const r = annSearch({ inputs, nprobe: 3, k: 12, query: "left" }); assert.ok(r.exact.findIndex(x => x.id === "d01") < r.exact.findIndex(x => x.id === "d02"));
  assert.equal(annSearch({ filter: "empty" }).annRecall, null); assert.throws(() => annSearch({ nprobe: 0 }));
});
test("five new pages expose initialized controls and declared local lab entry", async () => {
  for (const [slug, control] of [["context-parallel", "data-cp-split"], ["rl-update", "data-ru-fault"], ["quantized-linear", "data-ql-mode"], ["paired-evaluation", "data-pe-file"], ["ann-index", "data-ai-nprobe"]]) {
    const source = await readFile(new URL(`../src/content/${slug}.html`, import.meta.url), "utf8"); assert.ok(source.includes(control)); assert.ok(source.includes("<h1>"));
  }
});
