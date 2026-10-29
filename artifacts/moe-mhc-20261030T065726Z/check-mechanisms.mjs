// Fixed-input teaching checks, not a vendor training/quality/speed reproduction.
// Default verifies saved results and authored tables; --write creates results once.
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseFragment } from "parse5";

const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, "../..");
const inputs = JSON.parse(await readFile(resolve(here, "inputs.json"), "utf8"));
const sum = a => a.reduce((s, x) => s + x, 0);
const dot = (a, b) => sum(a.map((x, i) => x * b[i]));
const near = (a, b, eps = 1e-8) => assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps, String(a) + " != " + b);
const fd = (fn, x) => (fn(x + 1e-5) - fn(x - 1e-5)) / 2e-5;
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

function route(affinities, bias, k, expertOutputs = bias.map((_, i) => i)) {
  const e = bias.length;
  assert.ok(e > 0 && bias.every(Number.isFinite));
  assert.ok(Number.isInteger(k) && k > 0 && k <= e, "invalid Top-k");
  assert.ok(expertOutputs.length === e && expertOutputs.every(Number.isFinite));
  assert.ok(affinities.every(row => row.length === e && row.every(x => Number.isFinite(x) && x > 0 && x < 1)), "positive affinity inputs required");
  const counts = bias.map(() => 0);
  const tokens = affinities.map((s, t) => {
    const selected = s.map((_, i) => i).sort((i, j) => (s[j] + bias[j]) - (s[i] + bias[i]) || i - j).slice(0, k).sort((i, j) => i - j);
    const denominator = sum(selected.map(i => s[i])), weights = selected.map(i => s[i] / denominator);
    selected.forEach(i => counts[i]++);
    return { token: t, affinity: [...s], biasedScores: s.map((x, i) => x + bias[i]), selected, denominator, weights, output: dot(weights, selected.map(i => expertOutputs[i])) };
  });
  const mean = affinities.length * k / e;
  return { tokens, counts, mean, maxViolation: mean === 0 ? null : Math.max(...counts) / mean - 1 };
}
function adjust(bias, counts, rate) {
  assert.equal(bias.length, counts.length);
  assert.ok(Number.isFinite(rate) && rate >= 0);
  assert.ok(counts.every(x => Number.isInteger(x) && x >= 0));
  const mean = sum(counts) / counts.length;
  return bias.map((b, i) => b + rate * Math.sign(mean - counts[i]));
}
const r = inputs.routing, replays = [];
let bias = [...r.initialBias];
for (let replay = 0; replay < r.replays; replay++) {
  const current = route(r.affinities, bias, r.topK, r.expertOutputs);
  const nextBias = adjust(bias, current.counts, r.biasUpdateRate);
  replays.push({ replay, bias: [...bias], ...current, nextBias });
  bias = nextBias;
}
assert.deepEqual(replays.map(x => x.counts), [[4, 4, 0], [3, 3, 2], [2, 2, 4]]);
assert.ok(replays[1].maxViolation < replays[0].maxViolation && replays[2].maxViolation > replays[1].maxViolation);
assert.deepEqual(route([[0.5, 0.5, 0.5]], [0, 0, 0], 2).tokens[0].selected, [0, 1]);
assert.throws(() => route(r.affinities, r.initialBias, 0), /invalid Top-k/);
assert.throws(() => route(r.affinities, r.initialBias, 4), /invalid Top-k/);
assert.throws(() => route([[0, 0.5, 0.5]], r.initialBias, 2), /positive affinity/);
assert.equal(route([], [0.1, -0.1], 1).maxViolation, null);
assert.deepEqual(adjust([0.1, -0.1], [0, 0], 0.2), [0.1, -0.1]);
assert.deepEqual(adjust([0.1, -0.1], [2, 2], 0.2), [0.1, -0.1]);
assert.deepEqual(adjust(r.initialBias, replays[0].counts, 0), r.initialBias);
const outputs = replays[1].tokens;
const gateGradient = outputs.map((item, t) => item.affinity.map((_, e) => item.selected.includes(e) ? r.upstream[t] * (r.expertOutputs[e] - item.output) / item.denominator : 0));
const routingLoss = scores => dot(route(scores, replays[1].bias, r.topK, r.expertOutputs).tokens.map(x => x.output), r.upstream);
let routerFiniteDifferences = 0;
for (let t = 0; t < r.affinities.length; t++) for (let e = 0; e < r.initialBias.length; e++) {
  const fn = x => {
    const changed = r.affinities.map(row => [...row]); changed[t][e] = x;
    assert.deepEqual(route(changed, replays[1].bias, r.topK).tokens[t].selected, outputs[t].selected, "finite difference must stay inside a fixed selection region");
    return routingLoss(changed);
  };
  near(gateGradient[t][e], fd(fn, r.affinities[t][e])); routerFiniteDifferences++;
}
const t2 = outputs[2], wrongScores = t2.selected.map(e => t2.biasedScores[e]);
const biasedGateOutput = dot(wrongScores.map(x => x / sum(wrongScores)), t2.selected.map(e => r.expertOutputs[e]));
near(biasedGateOutput, 1.4545454545454546);
assert.ok(Math.abs(biasedGateOutput - t2.output) > 0.1);

const m = inputs.modalities, zeroModalBias = [0, 0];
const textRoute = route(m.text, zeroModalBias, m.topK), imageRoute = route(m.image, zeroModalBias, m.topK), totalRoute = route([...m.text, ...m.image], zeroModalBias, m.topK);
const modalityRows = [
  { label: "总体", count: m.text.length + m.image.length, ...totalRoute },
  { label: "文本", count: m.text.length, ...textRoute },
  { label: "图像", count: m.image.length, ...imageRoute },
];
assert.equal(totalRoute.maxViolation, 0); assert.equal(textRoute.maxViolation, 1); assert.equal(imageRoute.maxViolation, 1);
const globalBias = adjust(zeroModalBias, totalRoute.counts, m.biasUpdateRate);
const separateBiases = [textRoute, imageRoute].map(x => adjust(zeroModalBias, x.counts, m.biasUpdateRate));
assert.deepEqual(globalBias, [0, 0]); assert.deepEqual(separateBiases, [[-0.2, 0.2], [0.2, -0.2]]);
const afterSeparateBiases = [route(m.text, separateBiases[0], m.topK), route(m.image, separateBiases[1], m.topK)];
assert.ok(afterSeparateBiases.every(x => x.maxViolation === 1), "separate statistics do not guarantee one-step balance");
const c = inputs.causality;
const causalityRows = [c.tailA, c.tailB].map(tail => {
  const current = route([c.prefix, ...tail], c.historicalBias, 1);
  const futureDependentBias = adjust(c.historicalBias, current.counts, c.biasUpdateRate);
  return { counts: current.counts, correctPrefixExpert: current.tokens[0].selected[0], futureDependentBias, wrongReroutedPrefixExpert: route([c.prefix], futureDependentBias, 1).tokens[0].selected[0] };
});
assert.equal(causalityRows[0].correctPrefixExpert, causalityRows[1].correctPrefixExpert);
assert.notEqual(causalityRows[0].wrongReroutedPrefixExpert, causalityRows[1].wrongReroutedPrefixExpert);

const pack = flat => [flat.slice(0, 2), flat.slice(2, 4)];
function sinkhorn(raw, iterations, order = "row-column") {
  assert.ok(raw.length === 4 && raw.every(Number.isFinite));
  assert.ok(Number.isInteger(iterations) && iterations >= 0 && iterations <= 200, "invalid iteration count");
  assert.ok(order === "row-column" || order === "column-row");
  const positive = raw.map(Math.exp);
  assert.ok(positive.every(x => Number.isFinite(x) && x > 0), "positive finite exponential required");
  let values = [...positive];
  const tape = [], rowGroups = [[0, 1], [2, 3]], columnGroups = [[0, 2], [1, 3]];
  const groupsInOrder = order === "row-column" ? [rowGroups, columnGroups] : [columnGroups, rowGroups];
  for (let t = 0; t < iterations; t++) for (const groups of groupsInOrder) {
    const previous = values, denominators = groups.map(indices => sum(indices.map(i => previous[i]))), next = Array(4).fill(0);
    groups.forEach((indices, j) => indices.forEach(i => { next[i] = previous[i] / denominators[j]; }));
    values = next; tape.push({ groups, denominators, normalized: [...values] });
  }
  return { values, positive, tape };
}
function sinkhornGradient(forward, readout) {
  let grad = [...readout];
  for (const op of [...forward.tape].reverse()) {
    const before = Array(4).fill(0);
    op.groups.forEach((indices, j) => {
      const weighted = sum(indices.map(i => grad[i] * op.normalized[i]));
      indices.forEach(i => { before[i] = (grad[i] - weighted) / op.denominators[j]; });
    });
    grad = before;
  }
  return grad.map((x, i) => x * forward.positive[i]);
}
function norm2(a) {
  const trace = sum(a.map(x => x * x)), determinant = a[0] * a[3] - a[1] * a[2];
  return Math.sqrt((trace + Math.sqrt(Math.max(0, trace * trace - 4 * determinant * determinant))) / 2);
}
function matrixMetrics(a) {
  return { rowError: Math.max(Math.abs(a[0] + a[1] - 1), Math.abs(a[2] + a[3] - 1)), columnError: Math.max(Math.abs(a[0] + a[2] - 1), Math.abs(a[1] + a[3] - 1)), norm2: norm2(a) };
}
const sk = inputs.sinkhorn, raw = sk.initialPositiveMatrix.flat().map(Math.log), readout = sk.linearReadout.flat();
const normalizationRows = sk.readoutIterations.map(iterations => {
  const { values } = sinkhorn(raw, iterations);
  return { iterations, matrix: pack(values), ...matrixMetrics(values) };
});
assert.ok(normalizationRows[1].rowError > 0.09 && normalizationRows[1].norm2 > 1);
assert.ok(normalizationRows.at(-1).rowError < 1e-14 && normalizationRows.at(-1).columnError < 1e-14);
const reversedOnce = sinkhorn(raw, 1, "column-row").values, reversedMetrics = matrixMetrics(reversedOnce);
assert.ok(reversedMetrics.columnError > 0.09 && reversedMetrics.rowError < 1e-14);
assert.ok(reversedOnce.some((x, i) => Math.abs(x - sinkhorn(raw, 1).values[i]) > 0.09));
assert.throws(() => sinkhorn(raw, -1), /invalid iteration/);
assert.throws(() => sinkhorn([1000, 0, 0, 0], 1), /positive finite/);
assert.throws(() => sinkhorn([-1000, 0, 0, 0], 1), /positive finite/);
const forward = sinkhorn(raw, sk.trainingIterations), gradient = sinkhornGradient(forward, readout);
const sinkhornLoss = z => dot(sinkhorn(z, sk.trainingIterations).values, readout);
let sinkhornFiniteDifferences = 0;
for (let i = 0; i < raw.length; i++) {
  const fn = x => { const changed = [...raw]; changed[i] = x; return sinkhornLoss(changed); };
  near(gradient[i], fd(fn, raw[i])); sinkhornFiniteDifferences++;
}
const afterRaw = raw.map((x, i) => x - sk.learningRate * gradient[i]), beforeLoss = sinkhornLoss(raw), afterLoss = sinkhornLoss(afterRaw);
assert.ok(afterLoss < beforeLoss);
const matvec = (a, v) => [a[0] * v[0] + a[1] * v[1], a[2] * v[0] + a[3] * v[1]];
const multiply = (a, b) => [a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3], a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3]];
const comp = inputs.composition, bExact = comp.exactMatrix.flat(), compositionRows = [];
const exactMetrics = matrixMetrics(bExact);
near(exactMetrics.rowError, 0); near(exactMetrics.columnError, 0); assert.ok(bExact.every(x => x >= 0));
for (const depth of comp.readoutDepths) {
  let product = [1, 0, 0, 1];
  for (let t = 0; t < depth; t++) product = multiply(bExact, product);
  const contrast = matvec(product, comp.contrast), contrastRatio = Math.hypot(...contrast) / Math.hypot(...comp.contrast);
  const metrics = matrixMetrics(product);
  near(contrastRatio, (1 / 3) ** depth); near(metrics.rowError, 0); near(metrics.columnError, 0); near(metrics.norm2, 1);
  compositionRows.push({ depth, product: pack(product), contrast, contrastRatio, norm2: metrics.norm2 });
}
const jac = inputs.fullJacobianCounterexample;
assert.equal(jac.residual, 1);
const fullMap = x => jac.residual * x + jac.post * jac.branchSlope * jac.pre * x;
const fullJacobian = jac.residual + jac.post * jac.branchSlope * jac.pre;
near(fullJacobian, fd(fullMap, jac.input)); assert.equal(fullJacobian, 3);
const sp = inputs.singlePass, residual = matvec(sp.residualMatrix.flat(), sp.state);
const transition = a => { const mixed = dot(a, sp.state); return { mixed, output: residual.map((x, i) => x + sp.post[i] * sp.branchSlope * mixed) }; };
const singlePassRows = [
  { label: "本层 Aℓ", ...transition(sp.currentMixing) },
  { label: "前层 Aℓ-1", ...transition(sp.previousMixing) },
];
assert.ok(singlePassRows[0].output.some((x, i) => Math.abs(x - singlePassRows[1].output[i]) > 0.05));
assert.deepEqual(transition(sp.currentMixing), transition([...sp.currentMixing]), "same coefficients must preserve the transition");

assert.deepEqual(await table("transformer-anatomy.html", "data-moe-control"), replays.map(x => [String(x.replay), x.bias.map(v => v.toFixed(1)).join(" / "), x.counts.join(" / "), x.maxViolation.toFixed(6)]));
assert.deepEqual(await table("transformer-anatomy.html", "data-moe-gates"), outputs.map(x => ["T" + x.token, x.selected.map(e => "E" + e).join(" / "), x.weights.map(v => v.toFixed(6)).join(" / "), x.output.toFixed(6)]));
assert.deepEqual(await table("transformer-anatomy.html", "data-moe-modalities"), modalityRows.map(x => [x.label, String(x.count), x.counts.join(" / "), x.maxViolation.toFixed(6)]));
const errorText = x => x < 1e-14 ? "<1e-14" : x.toExponential(3);
assert.deepEqual(await table("architecture-variants.html", "data-mhc-sinkhorn"), normalizationRows.map(x => [String(x.iterations), errorText(x.rowError), errorText(x.columnError), x.norm2.toFixed(6)]));
assert.deepEqual(await table("architecture-variants.html", "data-mhc-composition"), compositionRows.map(x => [String(x.depth), x.contrastRatio.toExponential(6), x.norm2.toFixed(6)]));
assert.deepEqual(await table("architecture-variants.html", "data-mhc-single-pass"), singlePassRows.map(x => [x.label, x.mixed.toFixed(6), x.output.map(v => v.toFixed(6)).join(" / ")]));
for (const [marker, value] of [["data-mhc-before-loss", beforeLoss], ["data-mhc-after-loss", afterLoss], ["data-mhc-full-jacobian", fullJacobian]]) {
  assert.equal(clean(text(await marked("architecture-variants.html", marker))), value.toFixed(6));
}
const result = {
  schema: "moe-mhc-results-v1", passed: true, boundary: inputs.boundary,
  routing: { inputs: r, replays, gateGradient, linearReadout: routingLoss(r.affinities), finiteDifferenceChecks: routerFiniteDifferences, biasedGateCounterexample: { wrong: biasedGateOutput, correct: t2.output }, guards: ["invalid Top-k", "positive affinities", "empty batch", "equal loads", "zero update rate", "deterministic ties", "fixed-selection finite differences"] },
  modalities: { inputs: m, rows: modalityRows, globalBias, separateBiases, afterSeparateBiases },
  causality: { inputs: c, rows: causalityRows, wrongProtocol: "Count the full current sequence, update bias, then reroute its unchanged prefix." },
  sinkhorn: { inputs: sk, rows: normalizationRows, reversedOrderOnce: { matrix: pack(reversedOnce), ...reversedMetrics }, initialRaw: raw, beforeLoss, gradient, afterRaw, afterLoss, afterMatrix: pack(sinkhorn(afterRaw, sk.trainingIterations).values), finiteDifferenceChecks: sinkhornFiniteDifferences, guards: ["positive finite exponential", "nonnegative iteration count", "finite iteration order difference", "nonzero residual error"] },
  composition: { inputs: comp, rows: compositionRows },
  fullJacobianCounterexample: { inputs: jac, fullJacobian, finiteDifferenceChecks: 1 },
  singlePass: { inputs: sp, rows: singlePassRows },
  authoredTablesMatched: 6, authoredScalarValuesMatched: 3, finiteDifferenceChecks: routerFiniteDifferences + sinkhornFiniteDifferences + 1,
};
if (process.argv.includes("--write")) await writeFile(resolve(here, "results.json"), JSON.stringify(result, null, 2) + "\n", { flag: "wx" });
else assert.deepEqual(JSON.parse(await readFile(resolve(here, "results.json"), "utf8")), result, "saved results must match fixed-input recomputation");
console.log(JSON.stringify(result, null, 2));
