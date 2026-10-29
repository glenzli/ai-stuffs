import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { parse } from "parse5";
import { speculativeMass, speculativeToken } from "../../src/scripts/features/speculative-probability.js";

const directory = fileURLToPath(new URL(".", import.meta.url));
const root = resolve(directory, "../..");
const inputs = JSON.parse(await readFile(resolve(directory, "inputs.json"), "utf8"));
const close = (a, b, label = "number") => assert.ok(Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b)), `${label}: ${a} != ${b}`);
const sum = xs => xs.reduce((a, b) => a + b, 0);
const f = x => x.toFixed(6);

function prefixProcess(conditional) {
  assert.ok(conditional.every(c => Number.isFinite(c) && c >= 0 && c <= 1));
  const survival = [1], terminal = [];
  for (const c of conditional) {
    terminal.push(survival.at(-1) * (1 - c));
    survival.push(survival.at(-1) * c);
  }
  terminal.push(survival.at(-1));
  close(sum(terminal), 1, "terminal probability");
  const accepted = sum(terminal.map((mass, n) => mass * n));
  close(accepted, sum(survival.slice(1)), "tail-sum identity");
  return { survival, terminal, expectedAccepted: accepted, expectedEmitted: accepted + 1 };
}

const conditional = inputs.conditionalAcceptance;
const prefix = prefixProcess(conditional);
// Exact finite quadrature for these decimal inputs: each c lies on the grid.
// Independent grid uniforms realize the specified conditional prefix process;
// independence is not needed for the general conditional chain-rule identity.
const grid = inputs.uniformGridSize;
assert.ok(Number.isSafeInteger(grid) && grid > 0);
conditional.forEach(c => close(c * grid, Math.round(c * grid), "grid representability"));
const gridOutcomes = grid ** conditional.length;
const counts = Array(conditional.length + 1).fill(0);
for (let code = 0; code < gridOutcomes; code++) {
  let digits = code, accepted = 0;
  for (const c of conditional) {
    const accept = ((digits % grid) + 0.5) / grid;
    digits = Math.floor(digits / grid);
    const trial = speculativeToken([c, 1 - c], [1, 0], { draft: 0.5, accept, repair: 0.5 });
    if (!trial.accepted) {
      assert.equal(trial.token, 1, "first rejection emits a residual token");
      break;
    }
    assert.equal(trial.token, 0);
    accepted++;
  }
  counts[accepted]++;
}
const enumeratedTerminal = counts.map(n => n / gridOutcomes);
enumeratedTerminal.forEach((p, n) => close(p, prefix.terminal[n], "exact grid terminal mass"));
const capped = Array.from({ length: conditional.length + 3 }, (_, cap) => {
  const direct = sum(enumeratedTerminal.map((p, n) => p * Math.min(n + 1, cap)));
  const tails = cap === 0 ? 0 : 1 + sum(prefix.survival.slice(1, cap));
  close(direct, tails, "output cap tail sum");
  return { cap, expectedEmitted: direct };
});
const boundaryCases = [[], [0, 0, 0, 0], [1, 1, 1, 1], [0.7, 0, 0.9]].map(c => ({ conditional: c, ...prefixProcess(c) }));
close(boundaryCases[0].expectedEmitted, 1);
close(boundaryCases[1].expectedEmitted, 1);
close(boundaryCases[2].expectedEmitted, 5);
close(boundaryCases[3].expectedEmitted, 1.7);

const budgets = Array.from({ length: conditional.length + 1 }, (_, K) => {
  const expectedEmitted = prefixProcess(conditional.slice(0, K)).expectedEmitted;
  const profiles = Object.fromEntries(Object.entries(inputs.verificationTime).map(([name, times]) => {
    assert.equal(times.length, conditional.length + 1);
    assert.ok(times.every(t => Number.isFinite(t) && t > 0));
    const time = times[K] + inputs.overheadTime + (K === 0 ? 0 : inputs.draftTimeWhenEnabled);
    return [name, { time, tokensPerTime: expectedEmitted / time }];
  }));
  return { K, survival: prefix.survival[K], expectedEmitted, profiles };
});
const argmax = values => values.reduce((best, value, i) => value > values[best] ? i : best, 0);
const bestK = Object.fromEntries(Object.keys(inputs.verificationTime).map(name => [name, argmax(budgets.map(row => row.profiles[name].tokensPerTime))]));
assert.deepEqual(bestK, { A: 4, B: 2 });

const biasInput = inputs.selectionBias;
const firstMass = speculativeMass(biasInput.targetFirst, biasInput.draftFirst);
const c1 = sum(firstMass.accepted);
close(c1, 1);
const biasBranches = biasInput.vocabulary.map((token, i) => {
  const second = speculativeMass(biasInput.targetSecondByFirst[i], biasInput.draftSecondByFirst[i]);
  const c2 = sum(second.accepted);
  const scores = [1, 1 + c1, 1 + c1 + c1 * c2].map((n, K) => n * biasInput.stepsPerTimeByBatchSize[K]);
  const retrospectiveK = argmax(scores);
  // Evaluate no continuation-dependent quantity for the safe decision: K=1
  // already loses to K=0. c2 is computed above only to record the bad branch.
  const safeK = scores[1] <= scores[0] ? 0 : null;
  assert.equal(safeK, 0);
  return { token, c2, scores, retrospectiveK, safeK };
});
assert.deepEqual(biasBranches.map(row => row.retrospectiveK), [2, 0]);

function firstOutputMass(selectK) {
  const output = biasInput.targetFirst.map(() => 0);
  for (let candidate = 0; candidate < biasInput.draftFirst.length; candidate++) {
    const branch = biasInput.draftFirst[candidate];
    if (selectK(candidate) === 0) {
      output.forEach((_, v) => { output[v] += branch * biasInput.targetFirst[v]; });
    } else {
      const acceptance = firstMass.acceptance[candidate];
      output[candidate] += branch * acceptance;
      output.forEach((_, v) => { output[v] += branch * (1 - acceptance) * (firstMass.correction?.[v] ?? 0); });
    }
  }
  close(sum(output), 1);
  return output;
}
const bias = {
  branches: biasBranches,
  target: biasInput.targetFirst,
  retrospective: firstOutputMass(i => biasBranches[i].retrospectiveK),
  safe: firstOutputMass(i => biasBranches[i].safeK),
  fixedK2: firstOutputMass(() => 2),
};
bias.retrospective.forEach((p, i) => close(p, [0.75, 0.25][i], "selection-bias counterexample"));
bias.safe.forEach((p, i) => close(p, bias.target[i], "pre-token decision"));
bias.fixedK2.forEach((p, i) => close(p, bias.target[i], "fixed horizon"));
const results = { schema: 1, prefix, exactGrid: { gridOutcomes, counts, terminal: enumeratedTerminal }, capped, boundaryCases, budgets, bestK, bias };

const document = parse(await readFile(resolve(root, "src/content/efficient-inference.html"), "utf8"));
function find(node, predicate) {
  if (predicate(node)) return node;
  for (const child of node.childNodes ?? []) { const found = find(child, predicate); if (found) return found; }
}
const attr = (node, name) => node.attrs?.some(a => a.name === name);
const text = node => node.nodeName === "#text" ? node.value : (node.childNodes ?? []).map(text).join("");
let readoutTables = 0;
function bindTable(attribute, expected) {
  const table = find(document, node => node.tagName === "table" && attr(node, attribute));
  assert.ok(table, `missing ${attribute}`);
  const body = find(table, node => node.tagName === "tbody");
  const actual = body.childNodes.filter(node => node.tagName === "tr").map(row => row.childNodes.filter(node => ["td", "th"].includes(node.tagName)).map(cell => text(cell).trim()));
  assert.deepEqual(actual, expected, attribute);
  readoutTables++;
}
bindTable("data-dspark-prefix", conditional.map((c, i) => [String(i + 1), f(c), f(prefix.survival[i + 1])]));
bindTable("data-dspark-budget", budgets.map(row => [String(row.K), f(row.expectedEmitted), f(row.profiles.A.time), f(row.profiles.A.tokensPerTime), f(row.profiles.B.time), f(row.profiles.B.tokensPerTime)]));
bindTable("data-dspark-selection", biasBranches.map(row => [row.token, f(row.c2), ...row.scores.map(f), String(row.retrospectiveK)]));
bindTable("data-dspark-bias", [["直接从目标采样", ...bias.target.map(f)], ["看完后续再选 K", ...bias.retrospective.map(f)], ["在首个下降处停止", ...bias.safe.map(f)]]);
assert.equal(text(find(document, node => attr(node, "data-dspark-emitted"))).trim(), prefix.expectedEmitted.toFixed(4));
assert.equal(text(find(document, node => attr(node, "data-dspark-best"))).trim(), `A：K=${bestK.A}；B：K=${bestK.B}`);

function compare(actual, expected, label = "results") {
  if (typeof expected === "number") return close(actual, expected, label);
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, label);
    expected.forEach((value, i) => compare(actual[i], value, `${label}[${i}]`));
  } else if (expected && typeof expected === "object") {
    assert.deepEqual(Object.keys(actual), Object.keys(expected), label);
    Object.entries(expected).forEach(([key, value]) => compare(actual[key], value, `${label}.${key}`));
  } else assert.equal(actual, expected, label);
}
const flags = process.argv.slice(2);
assert.ok(flags.length === 0 || flags.length === 1 && flags[0] === "--write", "use no arguments or --write on first creation");
const resultPath = resolve(directory, "results.json");
if (flags[0] === "--write") await writeFile(resultPath, JSON.stringify(results, null, 2) + "\n", { flag: "wx" });
else compare(results, JSON.parse(await readFile(resultPath, "utf8")));
console.log(JSON.stringify({ ok: true, gridOutcomes, terminalCounts: counts, expectedEmitted: prefix.expectedEmitted, bestK, retrospectiveFirstOutput: bias.retrospective, readoutTables, scalarReadouts: 2, outputCaps: capped.length, boundaryCases: boundaryCases.length, mode: flags[0] === "--write" ? "created" : "verified" }, null, 2));
