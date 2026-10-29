// Fixed-input checks, not model training or a reproduction of vendor benchmarks.
// Run from any directory; --write creates evidence once, default only verifies.
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { parseFragment } from "parse5";
import { clippedSurrogate } from "../../src/scripts/features/rl-dynamics.js";

const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, "../..");
const near = (a, b, eps = 2e-7) => assert.ok(Math.abs(a - b) <= eps, String(a) + " != " + String(b));
const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
const soft = (r, tau) => 4 / tau * sigmoid(tau * (r - 1));
const weight = (r, tau) => { const p = sigmoid(tau * (r - 1)); return 4 * p * (1 - p); };
const derivative = (fn, x) => (fn(x + 1e-6) - fn(x - 1e-6)) / 2e-6;
const sum = values => values.reduce((a, b) => a + b, 0);
const walk = (n, p) => [...(p(n) ? [n] : []), ...(n.childNodes || []).flatMap(c => walk(c, p))];
const attr = (n, key) => n.attrs?.find(a => a.name === key)?.value;
const text = n => n.nodeName === "#text" ? n.value : (n.childNodes || []).map(text).join("");
async function table(file, marker) {
  const tree = parseFragment(await readFile(resolve(root, file), "utf8"));
  const node = walk(tree, n => attr(n, marker) !== undefined)[0];
  assert.ok(node, marker);
  const body = walk(node, n => n.tagName === "tbody")[0];
  return walk(body, n => n.tagName === "tr").map(row => row.childNodes.filter(c => c.tagName === "td").map(c => text(c).trim().replaceAll("−", "-")));
}

const policy = [];
for (const r of [0.6, 1, 1.4]) {
  const row = { ratio: r, hard_positive: null, hard_negative: null, soft_derivative: weight(r, 2) };
  for (const advantage of [1, -1]) {
    const fn = ratio => clippedSurrogate({ old: 0.5, current: 0.5 * ratio, advantage }).objective;
    const expected = advantage > 0 ? (r > 1.2 ? 0 : 1) : (r < 0.8 ? 0 : -1);
    near(derivative(fn, r), expected);
    row[advantage > 0 ? "hard_positive" : "hard_negative"] = expected;
  }
  policy.push(row);
}
for (const tau of [0.5, 2, 10]) for (const r of [0.3, 0.6, 1, 1.4, 2]) {
  near(derivative(x => soft(x, tau), r), weight(r, tau));
  assert.ok(weight(r, tau) >= 0 && weight(r, tau) <= 1);
}
assert.deepEqual(await table("src/content/rl-posttraining.html", "data-training-policy-example"),
  policy.map(row => [row.ratio.toFixed(1), String(row.hard_positive), String(row.hard_negative), row.soft_derivative.toFixed(3)]));

const gspo = [[2, 0.5], [2, 2]].map(ratios => ({
  token_ratios: ratios, sequence_ratio: ratios.reduce((a, b) => a * b, 1),
  length_normalized_ratio: Math.exp(sum(ratios.map(Math.log)) / ratios.length),
}));
near(gspo[0].length_normalized_ratio, 1); near(gspo[1].length_normalized_ratio, 2);
assert.equal(gspo[1].sequence_ratio, 4);

const stages = [{ main: 2, sub: [3, 5] }, { main: 1, sub: [4, 2] }];
const critical = sum(stages.map(stage => stage.main + Math.max(...stage.sub)));
const work = sum(stages.map(stage => stage.main + sum(stage.sub)));
assert.equal(critical, 12); assert.equal(work, 17);
assert.deepEqual(await table("src/content/training-environments.html", "data-training-parl-example"),
  stages.map((stage, i) => [String(i + 1), String(stage.main), stage.sub.join(" / "), String(stage.main + Math.max(...stage.sub))]));

// Same response-token losses; changing storage splits must not redefine rollout weight.
const grouped = [[2, 2], [4, 4]], splitA = [[2], [2], [4, 4]];
const mean = values => sum(values) / values.length;
const originalSampleMean = mean(grouped.map(mean)), splitSampleMean = mean(splitA.map(mean));
const equalRolloutMean = mean(grouped.map(mean));
assert.equal(originalSampleMean, 3); near(splitSampleMean, 8 / 3); assert.equal(equalRolloutMean, 3);
near(mean([mean(splitA.slice(0, 2).flat()), mean(splitA[2])]), equalRolloutMean);

const evidence = {
  schema: "training-techniques-fixed-input-v1", passed: true,
  boundary: "No model, GPU, rollout training or vendor benchmark was executed.",
  policy, soft_derivative_cases: 15, gspo,
  parl: { assumptions: "Subtasks concurrent within each stage; barrier between stages; fixed supplied step counts.", stages, critical_steps: critical, serial_work_steps: work },
  normalization: { grouped_token_losses: grouped, split_token_losses: splitA, original_sample_mean: originalSampleMean, split_sample_mean: splitSampleMean, equal_rollout_mean: equalRolloutMean },
};
if (process.argv.includes("--write")) await writeFile(resolve(here, "mechanism-results.json"), JSON.stringify(evidence, null, 2) + "\n", { flag: "wx" });
console.log(JSON.stringify(evidence, null, 2));
