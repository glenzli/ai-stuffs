import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { parseFragment } from "parse5";
import { packingTokens, packingParameters, packingExperiment, packingComparison, initPackingCorrectness } from "../src/scripts/features/packing-correctness.js";
import { loraParameters, loraCalculation, loraExperiment, loraQuantize, initLoraMechanics } from "../src/scripts/features/lora-mechanics.js";
import { contrastiveCalculation, initContrastiveDistributed } from "../src/scripts/features/contrastive-distributed.js";
import { maxDifference } from "../src/scripts/features/objective-math.js";

const near = (actual, expected, tolerance = 1e-10) => assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance, `${actual} versus ${expected}`);
const matricesNear = (a, b, tolerance = 1e-10) => { assert.deepEqual(a.map(row => row.length), b.map(row => row.length)); near(maxDifference(a, b), 0, tolerance); };

test("multi-turn records preserve template, EOS targets and original last-turn identity", () => {
  const tokens = packingTokens();
  assert.deepEqual(tokens.map(record => record.length), [11, 13]);
  assert.deepEqual(tokens.map(record => record.filter(row => row.supervised).length), [4, 5]);
  assert.deepEqual(tokens[0].map(row => row.id), [1, 2, 5, 3, 6, 4, 2, 7, 3, 8, 4]);
  assert.deepEqual(packingTokens({ supervise: "last" }).map(record => record.filter(row => row.supervised).length), [2, 3]);
  assert.deepEqual(packingTokens({ supervise: "last", limit: 6 }).map(record => record.filter(row => row.supervised).length), [0, 0]);
  assert.equal(packingExperiment({ supervise: "last", limit: 6 }).loss, null);
});

test("independent, padded and packed objectives agree across lengths, supervision and perturbations", () => {
  for (const limit of [1, 3, 6, 10, 32]) for (const supervise of ["all", "last"]) for (const changeA of [false, true]) {
    const { variants } = packingComparison({ limit, supervise, changeA });
    const reference = variants[0];
    for (const result of variants.slice(0, 3)) {
      assert.equal(result.denominator, reference.denominator);
      near(result.logitError, 0);
      if (!reference.denominator) {
        assert.equal(result.loss, null); assert.equal(result.gradients, null); assert.equal(result.gradientError, null);
      } else { near(result.loss, reference.loss); near(result.gradientError, 0); }
    }
  }
});

test("packed full parameter gradients pass independent finite differences", () => {
  for (const options of [{ mode: "packed" }, { mode: "leaky", supervise: "last" }, { mode: "double-shift", limit: 10 }]) {
    const parameters = packingParameters(), result = packingExperiment({ ...options, parameters }), epsilon = 1e-5;
    for (const [key, matrix] of Object.entries(parameters)) for (let i = 0; i < matrix.length; i++) for (let j = 0; j < matrix[i].length; j++) {
      const value = matrix[i][j];
      matrix[i][j] = value + epsilon; const plus = packingExperiment({ ...options, parameters }).loss;
      matrix[i][j] = value - epsilon; const minus = packingExperiment({ ...options, parameters }).loss;
      matrix[i][j] = value;
      near(result.gradients[key][i][j], (plus - minus) / (2 * epsilon), 1e-8);
    }
  }
});

test("fixed-parameter perturbation stays inside an isolated sample and leaks with causal-only masking", () => {
  const b = result => result.outputs.filter(row => row.record === "B").map(row => row.logits);
  for (const mode of ["separate", "padded", "packed"]) matricesNear(b(packingExperiment({ mode })), b(packingExperiment({ mode, changeA: true })));
  assert.ok(maxDifference(b(packingExperiment({ mode: "leaky" })), b(packingExperiment({ mode: "leaky", changeA: true }))) > .01);
});

test("padding contributes no labels or lookup gradient and never becomes a valid key", () => {
  const result = packingExperiment({ mode: "padded" }), run = result.runs[0];
  const padding = run.rows.flatMap((row, i) => row.valid ? [] : [i]);
  assert.equal(padding.length, 2);
  for (const i of padding) {
    assert.equal(run.labels[i], null); assert.ok(run.mask[i].every(value => !value));
    assert.ok(run.mask.every(row => !row[i]));
  }
  assert.deepEqual(result.gradients.embedding[0], [0, 0]);
  near(run.probabilities[0].reduce((a, b) => a + b, 0), 1);
});

test("duplicate shift preserves logits and label counts while changing targets and gradients", () => {
  const proper = packingExperiment(), wrong = packingExperiment({ mode: "double-shift" });
  assert.equal(proper.denominator, wrong.denominator);
  matricesNear(proper.outputs.map(row => row.logits), wrong.outputs.map(row => row.logits));
  assert.notDeepEqual(proper.outputs.map(row => row.label), wrong.outputs.map(row => row.label));
  assert.ok(maxDifference(proper.gradients.out, wrong.gradients.out) > .1);
});

test("LoRA initialization and first simultaneous SGD update have explicit reference values", () => {
  const initial = loraExperiment();
  matricesNear(initial.output, [[1.8], [-1.3]]);
  matricesNear(initial.gradients.A, [[0, 0]]); matricesNear(initial.gradients.B, [[2.25], [-2.875]]);
  matricesNear(initial.gradients.X, [[2.49], [-1.25]]);
  const next = loraExperiment({ steps: 1 });
  matricesNear(next.parameters.A, [[.5, -.25]]); matricesNear(next.parameters.B, [[-.225], [.2875]]);
  matricesNear(next.output, [[1.51875], [-.940625]]);
  for (const steps of [1, 3, 5]) {
    const zero = loraExperiment({ steps, initialization: "both-zero" });
    matricesNear(zero.parameters.A, [[0, 0]]); matricesNear(zero.parameters.B, [[0], [0]]);
  }
});

test("LoRA A/B/input gradients pass finite differences for full and quantized fixed bases", () => {
  const p = loraParameters(); p.B = [[.2], [-.1]];
  for (const precision of ["full", "int4"]) {
    const result = loraCalculation(p, { precision }), epsilon = 1e-5;
    for (const key of ["A", "B", "X"]) for (let i = 0; i < p[key].length; i++) for (let j = 0; j < p[key][i].length; j++) {
      const value = p[key][i][j];
      p[key][i][j] = value + epsilon; const plus = loraCalculation(p, { precision }).loss;
      p[key][i][j] = value - epsilon; const minus = loraCalculation(p, { precision }).loss;
      p[key][i][j] = value;
      near(result.gradients[key][i][j], (plus - minus) / (2 * epsilon), 1e-9);
    }
  }
});

test("frozen parameters and detached branches produce different input-gradient semantics", () => {
  const proper = loraExperiment(), detached = loraExperiment({ graph: "detach-base" }), all = loraExperiment({ graph: "detach-all", steps: 5 });
  near(proper.loss, detached.loss); matricesNear(proper.gradients.B, detached.gradients.B);
  matricesNear(detached.gradients.X, [[0], [0]]);
  assert.equal(all.gradients, null); assert.equal(all.updates, 0);
  matricesNear(all.parameters.A, proper.parameters.A); matricesNear(all.parameters.B, proper.parameters.B);
});

test("same-base merge is equivalent, while base quantization and requantization remain separate", () => {
  for (const precision of ["full", "int4"]) for (let steps = 0; steps <= 5; steps++) {
    const result = loraExperiment({ steps, precision });
    near(result.mergeError, 0); matricesNear(result.parameters.W, loraParameters().W);
    if (precision === "full") near(result.baseError, 0); else assert.ok(result.baseError > .05);
  }
  assert.ok(loraExperiment({ steps: 1, precision: "int4" }).requantizeError > .01);
  matricesNear(loraQuantize([[0, 0]]).restored, [[0, 0]]);
  const rounded = loraQuantize([[7, .5, -.5]]); assert.deepEqual(rounded.integers, [[7, 1, -1]]);
});

test("global contrastive gradients match finite differences at multiple temperatures", () => {
  for (const temperature of [.5, 1, 2]) {
    const query = [1, 2], document = [1, -1], r = contrastiveCalculation({ temperature }), epsilon = 1e-5;
    for (const [values, expected] of [[query, r.queryGradients], [document, r.documentGradients]]) for (let i = 0; i < 2; i++) {
      const original = values[i]; values[i] = original + epsilon;
      const plus = contrastiveCalculation({ temperature, query, document }).loss;
      values[i] = original - epsilon; const minus = contrastiveCalculation({ temperature, query, document }).loss; values[i] = original;
      near(expected[i], (plus - minus) / (2 * epsilon), 1e-9);
    }
  }
});

test("detached remote documents preserve loss while losing cross-query contributions", () => {
  const reference = contrastiveCalculation(), detached = contrastiveCalculation({ mode: "detached" });
  near(reference.loss, 2.0725389694803913); near(detached.loss, reference.loss);
  near(reference.documentGradients[0], .9224123290268496); near(detached.documentGradients[0], -.05960146101105884);
  near(reference.parameterGradients.query, reference.parameterGradients.document);
  near(detached.parameterGradients.document, reference.parameterGradients.document / 2);
  const micro = contrastiveCalculation({ mode: "micro" }); near(micro.loss, 0);
  assert.deepEqual(micro.queryGradients, [0, 0]); assert.deepEqual(micro.documentGradients, [0, 0]);
});

test("invalid experiment configurations fail explicitly", () => {
  for (const fn of [() => packingExperiment({ limit: 0 }), () => packingExperiment({ mode: "unknown" }), () => packingExperiment({ supervise: "unknown" }), () => loraExperiment({ steps: -1 }), () => loraExperiment({ initialization: "unknown" }), () => loraCalculation(undefined, { graph: "unknown" }), () => contrastiveCalculation({ temperature: 0 }), () => contrastiveCalculation({ document: [1] })]) assert.throws(fn, RangeError);
});

async function mount(t, slug) {
  const nodes = new Map(), source = await readFile(new URL(`../src/content/${slug}.html`, import.meta.url), "utf8");
  const text = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(text).join("");
  function visit(node) {
    const attributes = Object.fromEntries((node.attrs || []).map(a => [a.name, a.value]));
    const keys = Object.keys(attributes).filter(key => key.startsWith("data-"));
    if (keys.length) {
      const options = (node.childNodes || []).filter(child => child.tagName === "option"), selected = options.find(child => child.attrs?.some(a => a.name === "selected")) || options[0];
      const listeners = new Map(), optionAttributes = Object.fromEntries((selected?.attrs || []).map(a => [a.name, a.value]));
      const element = { value: selected ? optionAttributes.value ?? text(selected) : attributes.value ?? "", checked: "checked" in attributes, textContent: "", innerHTML: "", max: attributes.max,
        addEventListener(name, fn) { listeners.set(name, [...(listeners.get(name) || []), fn]); }, emit(name) { for (const fn of listeners.get(name) || []) fn(); } };
      for (const key of keys) nodes.set(`[${key}]`, element);
    }
    (node.childNodes || []).forEach(visit);
  }
  visit(parseFragment(source));
  const original = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: selector => nodes.get(selector) || null } });
  t.after(() => { if (original) Object.defineProperty(globalThis, "document", original); else delete globalThis.document; });
  return key => { const node = nodes.get(`[data-${key}]`); assert.ok(node, key); return node; };
}

test("packing controls expose leakage, duplicate shifts and no-label state", async t => {
  const node = await mount(t, "packing-correctness"); initPackingCorrectness();
  assert.equal(Number(node("pk-count").textContent), 9); assert.match(node("pk-isolation").textContent, /0\.00e\+0/);
  node("pk-mode").value = "leaky"; node("pk-mode").emit("change"); assert.ok(Number(node("pk-gradient").textContent) > .001);
  node("pk-mode").value = "double-shift"; node("pk-mode").emit("change"); assert.match(node("pk-note").textContent, /t\+2/);
  node("pk-limit").value = "3"; node("pk-limit").emit("change"); assert.equal(node("pk-loss").textContent, "未定义");
  assert.equal(node("pk-gradient").textContent, "无有效目标");
});

test("LoRA controls update gradient, merge and detached-output states", async t => {
  const node = await mount(t, "lora-mechanics"); initLoraMechanics();
  assert.equal(node("lm-output").textContent, "[1.8, -1.3]");
  node("lm-steps").value = "1"; node("lm-steps").emit("input"); assert.equal(node("lm-output").textContent, "[1.51875, -0.940625]");
  node("lm-precision").value = "int4"; node("lm-precision").emit("change"); assert.ok(Number(node("lm-base-error").textContent) > 0);
  node("lm-graph").value = "detach-all"; node("lm-graph").emit("change"); assert.equal(Number(node("lm-updates").textContent), 0);
  assert.match(node("lm-gradients").innerHTML, /无计算图/);
});

test("contrastive controls preserve forward loss and expose the missing derivative", async t => {
  const node = await mount(t, "parallel-correctness"); initContrastiveDistributed(); const original = node("cd-loss").textContent;
  node("cd-mode").value = "detached"; node("cd-mode").emit("change"); assert.equal(node("cd-loss").textContent, original); assert.match(node("cd-gradients").innerHTML, /-0.059601/);
  node("cd-mode").value = "micro"; node("cd-mode").emit("change"); assert.equal(node("cd-loss").textContent, "0"); assert.match(node("cd-scores").innerHTML, /未进入候选/);
});
