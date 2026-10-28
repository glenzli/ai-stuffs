import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { parseFragment } from "parse5";
import { attentionBackwardInputs, denseAttentionReference, tiledAttentionBackward, initAttentionBackward } from "../src/scripts/features/attention-backward.js";
import { gradientCacheInputs, gradientCacheExperiment, initGradientCache } from "../src/scripts/features/gradient-cache.js";
import { shardedClipping, initShardedClipping } from "../src/scripts/features/sharded-clipping.js";
import { validateMemoryReport, memoryReportRows, initMemoryReport } from "../src/scripts/features/memory-report.js";
import { maxDifference } from "../src/scripts/features/objective-math.js";

const near = (a, b, tolerance = 1e-10) => assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tolerance, `${a} versus ${b}`);

test("tiled forward/backward equals dense across block boundaries and mask rules", () => {
  for (const queryBlock of [1, 2, 3, 5]) for (const keyBlock of [1, 2, 3, 5]) for (const mask of ["full", "causal", "empty-first"]) {
    const r = tiledAttentionBackward({ queryBlock, keyBlock, mask });
    near(r.error, 0); near(r.outputError, 0);
    assert.equal(r.events.length, Math.ceil(5 / queryBlock) * Math.ceil(5 / keyBlock));
    assert.equal(r.forwardPairs, mask === "full" ? 25 : mask === "causal" ? 15 : 14);
    assert.equal(r.forwardPairs, r.reconstructedPairs);
    for (const name of ["dQ", "dK", "dV"]) near(maxDifference(r.events.at(-1)[name], r[name]), 0);
  }
});

test("tiled Q/K/V gradients independently match finite differences", () => {
  for (const mask of ["full", "causal", "empty-first"]) {
    const inputs = attentionBackwardInputs(), actual = tiledAttentionBackward({ inputs, mask, queryBlock: 3, keyBlock: 2 }), epsilon = 1e-5;
    for (const name of ["Q", "K", "V"]) for (let i = 0; i < 5; i++) for (let j = 0; j < 2; j++) {
      const original = inputs[name][i][j];
      inputs[name][i][j] = original + epsilon; const plus = denseAttentionReference(inputs, mask).loss;
      inputs[name][i][j] = original - epsilon; const minus = denseAttentionReference(inputs, mask).loss;
      inputs[name][i][j] = original;
      near(actual[`d${name}`][i][j], (plus - minus) / (2 * epsilon), 1e-9);
    }
  }
});

test("empty query row has explicit zero output/gradient and never reconstructs NaN", () => {
  const r = tiledAttentionBackward({ mask: "empty-first" });
  assert.equal(r.lse[0], -Infinity); assert.deepEqual(r.O[0], [0, 0]); assert.deepEqual(r.dQ[0], [0, 0]);
  assert.ok([r.O, r.dQ, r.dK, r.dV].flat(3).every(Number.isFinite));
  for (const event of r.events.filter(e => e.qStart === 0)) assert.ok(event.probability[0].every(value => value === 0));
});

test("local probability normalization and overwritten KV accumulators have distinct failures", () => {
  assert.ok(tiledAttentionBackward({ fault: "local-softmax" }).error > .1);
  assert.ok(tiledAttentionBackward({ fault: "overwrite-kv" }).error > .1);
  near(tiledAttentionBackward({ fault: "local-softmax", keyBlock: 5 }).error, 0);
  near(tiledAttentionBackward({ fault: "overwrite-kv", queryBlock: 5 }).error, 0);
  for (const fault of ["local-softmax", "overwrite-kv"]) near(tiledAttentionBackward({ fault }).outputError, 0);
});

test("logical attention storage separately counts shared tensors, saved statistics and tile capacity", () => {
  const r = tiledAttentionBackward();
  assert.deepEqual(r.storage, { sharedInputsOutput: 40, denseProbabilities: 25, savedStatistics: 5, tileCapacity: 4, upstreamAndGradients: 40 });
  assert.equal(tiledAttentionBackward({ queryBlock: 3, keyBlock: 5 }).storage.tileCapacity, 15);
});

test("gradient cache replays the same global objective and leaves next-step RNG unchanged", () => {
  for (const chunk of [1, 2, 3, 4]) for (const dropout of [0, .25, .5]) {
    const r = gradientCacheExperiment({ chunk, dropout });
    near(r.gradientError, 0); near(r.replayError, 0); assert.equal(r.rngAfterReplay, r.rngAfterFirstPass);
    assert.equal(r.events.at(-1).forwardRows, 16);
    assert.equal(r.chunks.length, 2 * Math.ceil(4 / chunk));
    assert.equal(r.storage.baselineHidden, 96); assert.equal(r.storage.cachedHidden, chunk * 12);
    assert.equal(r.storage.scoreMatrix, 16); assert.equal(r.storage.representations, 16); assert.equal(r.storage.representationGradients, 16);
  }
});

test("cached encoder parameter gradients match finite differences of the global loss", () => {
  for (const dropout of [0, .25]) {
    const inputs = gradientCacheInputs(), actual = gradientCacheExperiment({ inputs, dropout, chunk: 3 }), epsilon = 1e-5;
    for (const key of ["W1", "W2"]) for (let i = 0; i < inputs[key].length; i++) for (let j = 0; j < inputs[key][i].length; j++) {
      const value = inputs[key][i][j];
      inputs[key][i][j] = value + epsilon; const plus = gradientCacheExperiment({ inputs, dropout, chunk: 3 }).global.loss;
      inputs[key][i][j] = value - epsilon; const minus = gradientCacheExperiment({ inputs, dropout, chunk: 3 }).global.loss;
      inputs[key][i][j] = value; near(actual.actual[key][i][j], (plus - minus) / (2 * epsilon), 1e-9);
    }
  }
});

test("fresh dropout during replay changes gradients despite identical cached loss", () => {
  const good = gradientCacheExperiment(), bad = gradientCacheExperiment({ fault: "fresh-rng" });
  near(good.global.loss, bad.global.loss); assert.ok(bad.gradientError > .01); assert.ok(bad.replayError > .1);
  assert.notEqual(bad.rngAfterReplay, bad.rngAfterFirstPass);
  near(gradientCacheExperiment({ fault: "fresh-rng", dropout: 0 }).gradientError, 0);
});

test("deterministic global objective stays equivalent under non-divisible chunk sizes", () => {
  const reference = gradientCacheExperiment({ chunk: 4, dropout: 0 });
  for (const chunk of [1, 2, 3]) {
    const r = gradientCacheExperiment({ chunk, dropout: 0 }); near(r.global.loss, reference.global.loss);
    for (const name of ["W1", "W2"]) near(maxDifference(r.actual[name], reference.actual[name]), 0);
  }
});

test("global clip respects independent element ownership and thresholds", () => {
  for (const scenario of ["shards", "mixed", "replicated", "zero"]) for (const threshold of [0, 1, 20]) {
    const r = shardedClipping({ scenario, threshold });
    near(r.error, 0); assert.ok(r.outputNorm <= threshold + 1e-12); assert.ok(r.consistentDecision);
    near(r.norm, { shards: 5, mixed: 13, replicated: 5, zero: 0 }[scenario]);
    assert.ok(r.ranks.every(rank => rank.coefficient === r.correctCoefficient));
  }
  const r = shardedClipping(); near(r.ranks[0].output[0].gradient, .6); near(r.ranks[1].output[0].gradient, .8);
});

test("local clipping and double-counted replicas fail only when ownership exposes them", () => {
  const local = shardedClipping({ mode: "local" }); near(local.outputNorm, Math.sqrt(2));
  near(shardedClipping({ mode: "duplicate" }).error, 0);
  const duplicate = shardedClipping({ mode: "duplicate", scenario: "mixed" }); near(duplicate.countedNorm, Math.sqrt(313)); assert.ok(duplicate.error > .1);
  near(shardedClipping({ mode: "local", scenario: "replicated" }).error, 0);
});

test("one-rank overflow triggers coordinated skip and local-only decisions diverge", () => {
  for (const mode of ["global", "local", "duplicate"]) {
    const r = shardedClipping({ mode, scenario: "overflow" }); assert.ok(r.ranks.every(rank => rank.skip));
    assert.equal(r.norm, null); assert.equal(r.outputNorm, null); assert.equal(r.error, null);
    assert.ok(r.ranks.every(rank => rank.output.every(row => row.parameterAfter === 1)));
  }
  const wrong = shardedClipping({ mode: "local-skip", scenario: "overflow" }); assert.equal(wrong.consistentDecision, false);
  assert.ok(wrong.ranks[0].output[0].parameterAfter < 1); assert.equal(wrong.ranks[1].output[0].parameterAfter, 1);
});

test("invalid configurations reject unsupported dimensions and modes", () => {
  for (const fn of [() => tiledAttentionBackward({ queryBlock: 0 }), () => tiledAttentionBackward({ keyBlock: 6 }), () => tiledAttentionBackward({ mask: "invalid" }), () => gradientCacheExperiment({ chunk: 0 }), () => gradientCacheExperiment({ dropout: 1 }), () => shardedClipping({ threshold: -1 }), () => shardedClipping({ scale: 0 }), () => shardedClipping({ scenario: "unknown" })]) assert.throws(fn, RangeError);
});

function reportFixture() {
  return { schema: "memory-lab-v1", status: "passed", runtime: { python: "3.12", torch: "2.9.0", device: "cpu", dtype: "float64", platform: "test" }, max_error: 1e-15, comparisons: 12,
    measurements: [{ name: "<img src=x onerror=alert(1)>", wall_ms: { median: 1, min: .5, max: 2 }, saved_tensor_storage: { peak_referenced_bytes: 100 }, profiler_top_self_cpu: [{ operation: "aten::mm", self_cpu_us: 2, calls: 1, self_cpu_memory_bytes: -16 }] }] };
}
test("report reader validates measurements and escapes external strings", () => {
  const r = reportFixture(); assert.equal(validateMemoryReport(r), r); assert.match(memoryReportRows(r), /&lt;img/); assert.ok(!memoryReportRows(r).includes("<img"));
  for (const bad of [{}, { ...r, status: "failed" }, { ...r, max_error: Infinity }, { ...r, measurements: [{ ...r.measurements[0], wall_ms: { median: -1, min: 0, max: 2 } }] }]) assert.throws(() => validateMemoryReport(bad), RangeError);
});

async function mount(t, slug) {
  const nodes = new Map(), source = await readFile(new URL(`../src/content/${slug}.html`, import.meta.url), "utf8");
  const content = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(content).join("");
  function visit(node) {
    const attrs = Object.fromEntries((node.attrs || []).map(a => [a.name, a.value]));
    const keys = Object.keys(attrs).filter(key => key.startsWith("data-"));
    if (keys.length) {
      const options = (node.childNodes || []).filter(child => child.tagName === "option"), selected = options.find(child => child.attrs?.some(a => a.name === "selected")) || options[0];
      const selectedAttrs = Object.fromEntries((selected?.attrs || []).map(a => [a.name, a.value])), listeners = new Map();
      const element = { value: selected ? selectedAttrs.value ?? content(selected) : attrs.value ?? "", textContent: "", innerHTML: "", disabled: false, hidden: "hidden" in attrs, files: [],
        addEventListener(name, fn) { listeners.set(name, [...(listeners.get(name) || []), fn]); }, async emit(name) { for (const fn of listeners.get(name) || []) await fn(); } };
      for (const key of keys) nodes.set(`[${key}]`, element);
    }
    (node.childNodes || []).forEach(visit);
  }
  visit(parseFragment(source));
  const original = Object.getOwnPropertyDescriptor(globalThis, "document"); Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: selector => nodes.get(selector) || null } });
  t.after(() => { if (original) Object.defineProperty(globalThis, "document", original); else delete globalThis.document; });
  return key => { const node = nodes.get(`[data-${key}]`); assert.ok(node, key); return node; };
}
const finish = async (node, prefix) => { let limit = 100; while (!node(`${prefix}-next`).disabled && limit--) await node(`${prefix}-next`).emit("click"); assert.ok(limit > 0); };

test("attention stepper withholds final verification and exposes faulty accumulation", async t => {
  const node = await mount(t, "attention-kernels"); initAttentionBackward(); assert.equal(node("ab-error").textContent, "反向尚未完成");
  await finish(node, "ab"); assert.ok(Number(node("ab-error").textContent) < 1e-12);
  node("ab-fault").value = "overwrite-kv"; await node("ab-fault").emit("change"); assert.equal(node("ab-step").textContent, "0/9");
  await finish(node, "ab"); assert.ok(Number(node("ab-error").textContent) > .1);
  await node("ab-prev").emit("click"); assert.equal(node("ab-error").textContent, "反向尚未完成");
});

test("cache stepper separates encoding, global loss and replay completion", async t => {
  const node = await mount(t, "activation-checkpointing"); initGradientCache(); assert.equal(node("gc-loss").textContent, "尚未计算全局 loss");
  for (let i = 0; i < 5; i++) await node("gc-next").emit("click"); assert.ok(Number(node("gc-loss").textContent) > 1);
  assert.equal(node("gc-error").textContent, "反向尚未完成"); await finish(node, "gc"); assert.equal(Number(node("gc-error").textContent), 0);
  node("gc-fault").value = "fresh-rng"; await node("gc-fault").emit("change"); await finish(node, "gc"); assert.ok(Number(node("gc-error").textContent) > .01);
});

test("clipping controls expose local norm and divergent skip decisions", async t => {
  const node = await mount(t, "parallel-correctness"); initShardedClipping(); assert.equal(node("sc-norm").textContent, "5");
  node("sc-mode").value = "local"; await node("sc-mode").emit("change"); assert.equal(node("sc-output").textContent, "1.414214");
  node("sc-scenario").value = "overflow"; node("sc-mode").value = "local-skip"; await node("sc-scenario").emit("change"); assert.match(node("sc-decision").textContent, /分叉/);
});

test("report upload handles success, failure and newer selection without stale overwrite", async t => {
  const node = await mount(t, "performance-analysis"); initMemoryReport();
  node("mr-file").files = [{ size: 100, text: async () => JSON.stringify(reportFixture()) }]; await node("mr-file").emit("change");
  assert.equal(node("mr-output").hidden, false); assert.match(node("mr-rows").innerHTML, /&lt;img/);
  node("mr-file").files = [{ size: 100, text: async () => "{}" }]; await node("mr-file").emit("change"); assert.equal(node("mr-output").hidden, true); assert.match(node("mr-status").textContent, /无法读取/);
  let release; node("mr-file").files = [{ size: 100, text: () => new Promise(resolve => { release = resolve; }) }]; const pending = node("mr-file").emit("change");
  node("mr-file").files = []; await node("mr-file").emit("change"); release(JSON.stringify(reportFixture())); await pending; assert.equal(node("mr-output").hidden, true);
});
