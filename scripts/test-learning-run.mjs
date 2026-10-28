import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { parseFragment } from "parse5";
import { renderMath } from "./render-math.mjs";
import { validatePairedData } from "../src/scripts/features/paired-evaluation.js";
import { learningRunSnapshot } from "../src/scripts/features/learning-run-data.js";
import { LEARNING_REPORT_MAX_BYTES, validateLearningReport, scoreLearningPrediction, learningEvaluation,
  learningCurveSeries, learningCurveMarkup, learningSummaryRows, learningPredictionRows,
  learningPairedData, initLearningRun } from "../src/scripts/features/learning-run.js";

const root = new URL("../", import.meta.url);
const reportPath = learningRunSnapshot.provenance.report_path;
const [reportBytes, content, feature, runnerSource] = await Promise.all([
  readFile(new URL(reportPath, root)),
  readFile(new URL("src/content/learning-run.html", root), "utf8"),
  readFile(new URL("src/scripts/features/learning-run.js", root), "utf8"),
  readFile(new URL("scripts/learning-run.py", root), "utf8"),
]);
const report = JSON.parse(reportBytes);
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const clone = value => structuredClone(value);
// Deliberately synthetic failed-check fixture. It is never saved as a lab run.
const failedFixture = clone(report);
failedFixture.status = "failed";
failedFixture.checks.cache[0].dtype_checks[0].max_scaled_error = 2;
failedFixture.checks.cache[0].dtype_checks[0].passed = false;
failedFixture.checks.cache[0].passed = false;
const failedBytes = Buffer.from(JSON.stringify(failedFixture));
const pct = value => `${(value * 100).toFixed(2)}%`;

test("default snapshot comes from the current complete saved report with matching source and checkpoint hashes", async () => {
  assert.equal(validateLearningReport(learningRunSnapshot), learningRunSnapshot);
  assert.equal(learningRunSnapshot.provenance.report_path, reportPath);
  assert.equal(learningRunSnapshot.provenance.report_sha256, sha(reportBytes));
  assert.equal(learningRunSnapshot.status, "passed");
  assert.equal(learningRunSnapshot.checks.all_seeds_fit, report.runs.every(run => run.fit_passed));
  assert.equal(report.provenance.sources.find(s => s.path === "scripts/learning-run.py").sha256, sha(runnerSource));
  for (const file of report.provenance.files) assert.equal(sha(await readFile(new URL(file.path, root))), file.sha256, file.path);
  for (const run of learningRunSnapshot.runs) {
    const saved = report.runs.find(s => s.seed === run.seed);
    assert.deepEqual(run.curve, saved.curve);
    for (const phase of ["initial", "learned"]) for (const split of ["train", "validation", "test", "length"]) {
      assert.deepEqual(run[phase][split].metrics, saved[phase][split].metrics);
      assert.ok(run[phase][split].predictions.every(row => saved[phase][split].predictions.some(actual => JSON.stringify(row) === JSON.stringify(actual))));
    }
  }
});

test("reader keeps failed computation separate from measured learning capability", () => {
  assert.equal(validateLearningReport(failedFixture), failedFixture);
  assert.equal(validateLearningReport(report), report);
  assert.equal(failedFixture.status, "failed");
  assert.deepEqual(report.runs, failedFixture.runs);
  assert.deepEqual(report.baselines, failedFixture.baselines);
  assert.deepEqual(report.checks.resume, failedFixture.checks.resume);
  assert.equal(report.frozen.correctness.cache.float32.atol, 1e-4);
  assert.equal(report.frozen.correctness.cache.float64.atol, 1e-9);
  for (const run of report.runs) assert.equal(run.learned.length.metrics.exact_match, 0);
  for (const run of report.runs) assert.equal(run.fit_passed, run.learned.train.metrics.exact_match >= .95);
});

test("generation scorer includes EOS, missing/extra slots, PAD errors and incomplete versus cap", () => {
  assert.equal(scoreLearningPrediction([4, 5, 3], [4, 5]).exact_match, true);
  assert.equal(scoreLearningPrediction([4, 3], [4, 5]).early_eos, true);
  assert.equal(scoreLearningPrediction([4, 5, 6, 3], [4, 5]).token_accuracy, .5);
  assert.equal(scoreLearningPrediction([4, 5], [4, 5]).termination, "incomplete");
  assert.equal(scoreLearningPrediction(Array(20).fill(0), [4, 5]).termination, "cap");
  assert.equal(scoreLearningPrediction([0, 4, 3], [4, 5]).correct_tokens, 1);
  assert.throws(() => scoreLearningPrediction([4, 3, 5], [4, 5]), RangeError);
  for (const run of report.runs) for (const phase of ["initial", "learned"]) for (const split of ["train", "validation", "test", "length"]) {
    for (const row of run[phase][split].predictions) {
      const scored = scoreLearningPrediction(row.predicted, row.target, report.frozen.generation.max_new_tokens);
      for (const [key, value] of Object.entries(scored)) assert.equal(row[key], value);
    }
  }
});

test("invalid ranges, inconsistent metrics, partial reports and mislabeled checks are rejected", () => {
  for (const mutate of [
    r => { r.status = "failed"; },
    r => { r.frozen.generation.max_new_tokens = 0; },
    r => { r.runs[0].learned.test.metrics.free_token_accuracy = Infinity; },
    r => { r.runs[0].learned.test.predictions.pop(); },
    r => { r.runs[0].learned.test.predictions[0].correct_tokens++; },
    r => { r.runs[0].curve[1].step = 0; },
    r => { r.checks.all_seeds_fit = !r.checks.all_seeds_fit; },
    r => { r.checks.cache[0].dtype_checks[0].max_scaled_error = 2; },
  ]) { const bad = clone(report); mutate(bad); assert.throws(() => validateLearningReport(bad), RangeError); }
});

test("curves use only saved train/validation points and explicit units", () => {
  for (const metric of ["teacher_forced_loss", "free_token_accuracy", "exact_match"]) {
    const series = learningCurveSeries(report, { seed: 47, metric, axis: "supervised_tokens" });
    assert.deepEqual(series.map(s => s.name), ["train", "validation"]);
    const run = report.runs.find(r => r.seed === 47);
    for (const s of series) assert.deepEqual(s.points, run.curve.map(p => ({ x: p.supervised_tokens, y: p[s.name][metric] * (metric === "teacher_forced_loss" ? 1 : 100) })));
    assert.match(learningCurveMarkup(report, { seed: 47, metric, axis: "supervised_tokens" }), /累计消费监督 token/);
  }
  assert.match(learningSummaryRows(report), /95% 训练阈值/);
  assert.match(learningCurveMarkup(report), /nats/);
});

test("complete paired export preserves every question, seed and reversal group", () => {
  for (const old of ["initial", "identity"]) for (const metric of ["exact_match", "token_accuracy"]) {
    const paired = learningPairedData(report, { split: "test", old, metric });
    assert.equal(paired.rows.length, 139 * 3);
    assert.equal(validatePairedData(paired).rows.length, 417);
    for (const row of paired.rows) {
      const actual = learningEvaluation(report, row.seed, "test", "learned").predictions.find(r => r.id === row.id);
      assert.equal(row.group, actual.group_id); assert.equal(row.new, Number(actual[metric]));
    }
  }
  assert.throws(() => learningPairedData(learningRunSnapshot), /完整 report/);
});

test("prediction pagination and HTML text escaping retain actual results", () => {
  const first = learningPredictionRows(report, { seed: 47, split: "test" });
  assert.equal(first.total, 139); assert.equal(first.pages, 6); assert.equal(first.shown, 25);
  assert.equal(learningPredictionRows(report, { seed: 47, split: "test", page: 5 }).shown, 14);
  assert.equal(learningPredictionRows(report, { seed: 47, split: "test", errorsOnly: true }).total,
    report.runs.find(run => run.seed === 47).learned.test.predictions.filter(row => !row.exact_match).length);
  const unsafe = clone(report);
  unsafe.runs[0].learned.test.predictions[0].id = '<img src=x onerror="window.unsafe=1">';
  unsafe.runs[0].learned.test.predictions[0].group_id = '<svg onload="window.unsafe=1">';
  const html = learningPredictionRows(unsafe).html;
  assert.match(html, /&lt;img/); assert.match(html, /&lt;svg/); assert.ok(!html.includes("<img")); assert.ok(!html.includes("<svg"));
});

function mount(t) {
  const nodes = new Map();
  const text = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(text).join("");
  const create = (tag = "div") => {
    const listeners = new Map();
    return { tagName: tag, value: "", textContent: "", innerHTML: "", hidden: false, disabled: false, checked: false, files: [], children: [], attributes: {},
      addEventListener(name, fn) { const entries = listeners.get(name) || []; entries.push(fn); listeners.set(name, entries); },
      async emit(name) { for (const fn of listeners.get(name) || []) await fn(); },
      replaceChildren(...children) { this.children = children; this.innerHTML = ""; this.textContent = ""; },
      setAttribute(name, value) { this.attributes[name] = value; }, click() {} };
  };
  const visit = node => {
    const attrs = Object.fromEntries((node.attrs || []).map(a => [a.name, a.value]));
    const keys = Object.keys(attrs).filter(k => k.startsWith("data-lr-"));
    if (keys.length) {
      const el = create(node.tagName), options = (node.childNodes || []).filter(n => n.tagName === "option");
      const option = options.find(n => n.attrs?.some(a => a.name === "selected")) || options[0];
      const selected = Object.fromEntries((option?.attrs || []).map(a => [a.name, a.value]));
      el.value = option ? selected.value ?? text(option) : attrs.value ?? "";
      el.hidden = "hidden" in attrs; el.disabled = "disabled" in attrs;
      for (const key of keys) nodes.set(`[${key}]`, el);
    }
    (node.childNodes || []).forEach(visit);
  };
  visit(parseFragment(content));
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: selector => nodes.get(selector) || null, createElement: create } });
  t.after(() => originalDocument ? Object.defineProperty(globalThis, "document", originalDocument) : delete globalThis.document);
  return key => { const result = nodes.get(`[data-lr-${key}]`); assert.ok(result, `missing data-lr-${key}`); return result; };
}

test("page initializes, selects seed47 and OOD, and distinguishes compute status from capability", async t => {
  const node = mount(t); initLearningRun();
  assert.equal(node("output").hidden, false); assert.equal(node("paired").disabled, true);
  assert.ok(node("provenance").textContent.includes(reportPath));
  assert.ok(node("fit").textContent.includes(`${report.runs.filter(run => run.fit_passed).length}/${report.runs.length}`));
  node("seed").value = "47"; await node("seed").emit("change");
  const selected = report.runs.find(run => run.seed === 47);
  assert.ok(node("outcome").textContent.includes(pct(selected.learned.train.metrics.exact_match)));
  assert.ok(node("outcome").textContent.includes(pct(selected.learned.test.metrics.exact_match)));
  node("split").value = "length"; await node("split").emit("change");
  assert.equal(node("em").textContent, "0.00%"); assert.match(node("termination").textContent, /143 提前 EOS/);
  node("file").files = [{ name: "synthetic-failed-check.json", size: failedBytes.length, text: async () => failedBytes.toString() }];
  await node("file").emit("change"); assert.equal(node("output").hidden, false);
  assert.match(node("computation").textContent, /未通过/); assert.equal(node("paired").disabled, false);
});

test("local import clears failed/stale results and newest file wins the asynchronous race", async t => {
  const node = mount(t); initLearningRun();
  node("file").files = [{ name: "validated.json", size: reportBytes.length, text: async () => reportBytes.toString() }];
  await node("file").emit("change"); assert.equal(node("output").hidden, false); assert.equal(node("paired").disabled, false);
  node("file").files = [{ name: "invalid.json", size: 2, text: async () => "{}" }];
  await node("file").emit("change"); assert.equal(node("output").hidden, true);
  for (const key of ["rows", "chart", "summary-rows", "provenance", "model"]) assert.equal(node(key).textContent + node(key).innerHTML, "");
  assert.match(node("status").textContent, /无法读取/);
  let release;
  node("file").files = [{ name: "slow.json", size: reportBytes.length, text: () => new Promise(resolve => { release = resolve; }) }];
  const pending = node("file").emit("change");
  node("file").files = [{ name: "newest.json", size: failedBytes.length, text: async () => failedBytes.toString() }];
  await node("file").emit("change"); release(reportBytes.toString()); await pending;
  assert.match(node("status").textContent, /newest.json/); assert.match(node("computation").textContent, /未通过/);
  node("file").files = [{ name: "oversize.json", size: LEARNING_REPORT_MAX_BYTES + 1, text: async () => { throw new Error("should not read"); } }];
  await node("file").emit("change"); assert.match(node("status").textContent, /16 MiB/); assert.equal(node("output").hidden, true);
  await node("default").emit("click"); assert.equal(node("output").hidden, false); assert.equal(node("paired").disabled, true);
});

test("scoped HTML selectors, math, table accessibility and future runner cache standard are present", () => {
  const identifiers = new Set(), markers = new Set(), bad = [];
  function visit(node) {
    const attrs = Object.fromEntries((node.attrs || []).map(a => [a.name, a.value]));
    if (attrs.id) { assert.ok(!identifiers.has(attrs.id), attrs.id); identifiers.add(attrs.id); }
    for (const key of Object.keys(attrs)) if (key.startsWith("data-lr-")) markers.add(key);
    if (attrs.class?.split(" ").includes("compare-wrap")) assert.ok(attrs.tabindex === "0" && attrs.role === "region" && attrs["aria-label"]);
    if (node.tagName === "script") bad.push(node.tagName);
    (node.childNodes || []).forEach(visit);
  }
  visit(parseFragment(content));
  for (const marker of new Set(feature.match(/data-lr-[a-z-]+/gu))) assert.ok(markers.has(marker), marker);
  assert.equal(bad.length, 0);
  const rendered = renderMath(content, "learning-run.html"); assert.equal(rendered.count, 2);
  assert.match(rendered.html, /<math/); assert.ok(!rendered.html.includes("katex-error"));
  assert.match(runnerSource, /initial_cache = validated_cache_checks/);
  assert.match(runnerSource, /final_cache = validated_cache_checks/);
  assert.match(runnerSource, /def enforce_reference_versions/);
});
