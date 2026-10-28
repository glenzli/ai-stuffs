import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { parseFragment } from "parse5";
import { renderMath } from "./render-math.mjs";
import { validateTrainingFailuresReport, failureReplayView, failureEventRows, failureLossChart, initTrainingFailures } from "../src/scripts/features/training-failures.js";
import { trainingFailuresSnapshot as snapshot } from "../src/scripts/features/training-failures-data.js";

const manifest = JSON.parse(await readFile(new URL("../src/experiment-sources.json", import.meta.url), "utf8"));
const fullPath = process.env.FAILURE_REPORT || new URL(`../${manifest.experiments.failures.report.path}`, import.meta.url);
const fullBytes = await readFile(fullPath), full = JSON.parse(fullBytes);
const clone = value => structuredClone(value);

test("default is a compact projection of the saved 115-check actual report", async () => {
  assert.equal(validateTrainingFailuresReport(snapshot), snapshot);
  assert.equal(validateTrainingFailuresReport(full), full);
  assert.equal(snapshot.checks.length, 115);
  assert.equal(snapshot.provenance.report_sha256, createHash("sha256").update(fullBytes).digest("hex"));
  assert.equal(snapshot.provenance.saved_after_updates, 8); assert.equal(snapshot.provenance.pending_step, 9);
  assert.ok((await readFile(new URL("../src/scripts/features/training-failures-data.js", import.meta.url))).length < 1024 * 1024);
  for (const item of snapshot.cases) {
    const original = full.cases.find(row => row.id === item.id);
    for (const [key, run] of Object.entries(item.runs)) {
      const source = original.runs[key];
      for (const name of ["loss", "gradient", "update", "next_loss", "actual_labels", "loaded_pre_step_digest", "final_state_digest"]) assert.deepEqual(run[name], source[name]);
      for (const event of run.events) assert.deepEqual(event, source.events.find(row => row.key === event.key));
    }
  }
});

test("actual overflow, finite LR fault and symbol label rotation retain accurate evidence", () => {
  const overflow = failureReplayView(snapshot);
  assert.equal(overflow.first.name, "loss.exp"); assert.equal(overflow.first.stats.positive_infinity_count, 768);
  assert.equal(overflow.run.loss.status, "nan"); assert.equal(overflow.run.loss.value, null);
  assert.equal(overflow.run.update.applied, false);
  const high = failureReplayView(snapshot, { caseId: "high-lr" });
  assert.equal(high.run.all_finite, true); assert.equal(high.comparison.gradients.equal, true);
  assert.ok(high.run.update.relative_l2 > high.run.update.threshold);
  assert.equal(high.comparison.first_divergence.phase, "post-update");
  assert.ok(high.comparison.next_logits.max_abs > 20);
  assert.equal(high.item.anomaly_probe.optimizer_update_executed, false);
  const labels = failureReplayView(snapshot, { caseId: "labels" });
  assert.equal(labels.run.all_finite, true); assert.equal(labels.run.supervised_count, 44);
  assert.equal(labels.run.labels_changed, 36); assert.ok(labels.comparison.gradients.l2 > 0.9);
  assert.equal(labels.comparison.first_divergence.name, "batch.labels");
  labels.run.actual_labels.forEach((row, i) => row.forEach((value, j) => {
    const gold = labels.run.batch.labels[i][j];
    assert.equal(value, gold >= 4 ? 4 + (gold - 4 + 1) % 4 : gold);
  }));
});

test("all repaired and normal-control tensor/state proofs match actual report", () => {
  for (const item of snapshot.cases) for (const name of ["repaired_vs_baseline", "control_faulty_vs_baseline", "control_repaired_vs_baseline"]) {
    const proof = item.comparisons[name]; assert.equal(proof.final_state_equal, true); assert.equal(proof.all_compared_tensors_equal, true);
    assert.equal(proof.gradients.l2, 0); assert.equal(proof.parameters_after.max_abs, 0); assert.equal(proof.next_logits.max_abs, 0);
  }
});

test("report reader rejects bad NaN encoding, count/shape, first finding and repair proof", () => {
  const mutations = [
    value => { value.status = "failed"; },
    value => { value.cases[0].runs["target-faulty"].loss.value = NaN; },
    value => { value.cases[0].runs["target-faulty"].events[0].stats.finite_count++; },
    value => { value.cases[0].runs["target-faulty"].first_forward_nonfinite.name = "invented"; value.cases[0].runs["target-faulty"].first_forward_nonfinite.index--; },
    value => { value.cases[0].runs["target-repaired"].final_state_digest = "0".repeat(64); },
    value => { value.cases[2].runs["target-faulty"].actual_labels[0][0] = 4; },
    value => { value.cases[1].anomaly_probe.optimizer_update_executed = true; },
    value => { value.provenance.pending_step = 8; },
  ];
  for (const mutate of mutations) { const bad = clone(snapshot); mutate(bad); assert.throws(() => validateTrainingFailuresReport(bad), RangeError); }
  for (const bad of [{}, null, { ...snapshot, cases: [] }]) assert.throws(() => validateTrainingFailuresReport(bad), RangeError);
});

test("external names are escaped and chart uses saved finite next-forward losses", () => {
  const value = clone(snapshot), view = failureReplayView(value);
  view.events[0].name = '<img src=x onerror="window.unsafe=1">';
  const html = failureEventRows(view); assert.match(html, /&lt;img/); assert.ok(!html.includes("<img"));
  const chart = failureLossChart(value.cases[1], "target");
  assert.ok(chart.includes(value.cases[1].runs["target-faulty"].next_loss.value.toFixed(6)));
  assert.ok(!chart.includes("NaN") && !chart.includes("undefined"));
});

async function mount(t) {
  const html = await readFile(new URL("../src/content/training-failures.html", import.meta.url), "utf8");
  const nodes = new Map();
  const text = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(text).join("");
  function visit(source) {
    const attrs = Object.fromEntries((source.attrs || []).map(a => [a.name, a.value]));
    const keys = Object.keys(attrs).filter(key => key.startsWith("data-tf-"));
    if (keys.length) {
      const option = (source.childNodes || []).find(node => node.tagName === "option"), optionAttrs = Object.fromEntries((option?.attrs || []).map(a => [a.name, a.value]));
      const listeners = new Map();
      const node = { value: option ? optionAttrs.value ?? text(option) : attrs.value ?? "", textContent: "", innerHTML: "", hidden: "hidden" in attrs, disabled: false, files: [],
        addEventListener(name, fn) { listeners.set(name, [...(listeners.get(name) || []), fn]); },
        async emit(name) { for (const fn of listeners.get(name) || []) await fn(); },
        setAttribute(name, value) { this[name] = value; } };
      keys.forEach(key => nodes.set(`[${key}]`, node));
    }
    (source.childNodes || []).forEach(visit);
  }
  visit(parseFragment(html));
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: selector => nodes.get(selector) || null } });
  t.after(() => { if (previous) Object.defineProperty(globalThis, "document", previous); else delete globalThis.document; });
  return name => { const node = nodes.get(`[data-tf-${name}]`); assert.ok(node, name); return node; };
}

test("actual page controls cover 18 branches, first evidence, phase and default reset", async t => {
  const node = await mount(t); initTrainingFailures(); assert.equal(node("output").hidden, false); assert.match(node("loss").textContent, /NaN/);
  await node("first").emit("click"); assert.match(node("event-name").textContent, /loss\.exp/);
  for (const caseId of ["overflow", "high-lr", "labels"]) for (const selection of ["target", "control"]) for (const variant of ["baseline", "faulty", "repaired"]) {
    node("case").value = caseId; node("selection").value = selection; node("variant").value = variant; await node("case").emit("change");
    assert.equal(node("output").hidden, false); assert.ok(node("event-rows").innerHTML.length > 0); assert.match(node("verification").textContent, /最终状态相同/);
  }
  node("case").value = "high-lr"; node("variant").value = "faulty"; node("selection").value = "target"; await node("case").emit("change");
  assert.match(node("case-description").textContent, /未执行高 LR/); assert.match(node("update").textContent, /超阈值/);
  node("phase").value = "post-update"; await node("phase").emit("change"); assert.match(node("event-name").textContent, /更新后.*parameter/);
  node("phase").value = "all"; await node("default").emit("click"); assert.match(node("status").textContent, /默认快照/);
});

test("local import handles full report, escaped text, errors, empty state and stale reads", async t => {
  const node = await mount(t); initTrainingFailures();
  const upload = async value => { node("file").files = [{ name: "saved-report.json", size: 100, text: async () => JSON.stringify(value) }]; await node("file").emit("change"); };
  await upload(full); assert.equal(node("output").hidden, false); assert.match(node("status").textContent, /115/);
  const unsafe = clone(snapshot); unsafe.cases[0].runs["target-faulty"].events[0].name = '<img src=x onerror="window.unsafe=1">';
  await upload(unsafe); assert.match(node("event-rows").innerHTML, /&lt;img/); assert.ok(!node("event-rows").innerHTML.includes("<img"));
  await upload({}); assert.equal(node("output").hidden, true); assert.equal(node("event-rows").innerHTML, ""); assert.equal(node("provenance").textContent, ""); assert.match(node("status").textContent, /无法读取/);
  node("file").files = []; await node("file").emit("change"); assert.equal(node("output").hidden, true); assert.match(node("status").textContent, /未选择/);
  let release; node("file").files = [{ size: 100, text: () => new Promise(resolve => { release = resolve; }) }]; const pending = node("file").emit("change");
  await upload({}); release(JSON.stringify(snapshot)); await pending; assert.equal(node("output").hidden, true); assert.match(node("status").textContent, /无法读取/);
  await node("default").emit("click"); assert.equal(node("output").hidden, false);
  node("file").files = [{ size: 13 * 1024 * 1024, text: async () => { throw new Error("must not read oversized file"); } }]; await node("file").emit("change"); assert.match(node("status").textContent, /12 MiB/);
});

test("owner prose renders all math and names scroll regions without shared build", async () => {
  const source = await readFile(new URL("../src/content/training-failures.html", import.meta.url), "utf8");
  const rendered = renderMath(source, "training-failures.html"); assert.equal(rendered.count, 4); assert.match(rendered.html, /<math/);
  const fragment = parseFragment(source), ids = new Set(); let regions = 0;
  function visit(node) { const attrs = Object.fromEntries((node.attrs || []).map(a => [a.name, a.value])); if (attrs.id) { assert.ok(!ids.has(attrs.id)); ids.add(attrs.id); } if (attrs.class?.includes("compare-wrap")) { regions++; assert.equal(attrs.tabindex, "0"); assert.equal(attrs.role, "region"); assert.ok(attrs["aria-label"]); } (node.childNodes || []).forEach(visit); }
  visit(fragment); assert.equal(regions, 4); assert.ok(!source.includes("标签错位"));
});
