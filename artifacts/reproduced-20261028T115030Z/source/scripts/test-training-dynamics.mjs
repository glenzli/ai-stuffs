import test from "node:test";
import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import { createHash } from "node:crypto";
import { parseFragment } from "parse5";
import { renderMath } from "./render-math.mjs";
import { initializationWidthData as widthData } from "../src/scripts/features/initialization-width-data.js";
import { batchNoiseData as batchData } from "../src/scripts/features/batch-noise-data.js";
import { validateInitializationWidthReport, widthCoordinateRows, depthProbeRows, initInitializationWidth } from "../src/scripts/features/initialization-width.js";
import { validateGradientGram, gradientGramStatistics, batchNoiseVariance, enumerateGradientBatches,
  validateBatchNoiseReport, batchTrainingRows, initBatchNoise } from "../src/scripts/features/batch-noise.js";
import { dynamicsProvenance } from "../src/scripts/features/initialization-width-utils.js";

const root = new URL("../", import.meta.url);
const source = path => readFile(new URL(path, root), "utf8");
const clone = value => structuredClone(value);
const near = (actual, expected, tolerance = 1e-11) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const fullBytes = await readFile(new URL(batchData.provenance.source_report, root));
const full = JSON.parse(fullBytes);
const pages = await Promise.all(["initialization-width", "batch-noise"].map(sourceName => source(`src/content/${sourceName}.html`)));

test("compact snapshots and full saved report validate with both provenance hashes", async () => {
  assert.equal(validateInitializationWidthReport(widthData), widthData);
  assert.equal(validateBatchNoiseReport(batchData), batchData);
  validateInitializationWidthReport(full); validateBatchNoiseReport(full);
  for (const [slug, data] of [["initialization-width", widthData], ["batch-noise", batchData]]) {
    assert.ok((await stat(new URL(`src/scripts/features/${slug}-data.js`, root))).size <= 500 * 1024);
    assert.equal(data.provenance.kind, "saved-measurement-reanalysis");
    assert.equal(data.provenance.no_full_retraining, true);
    assert.equal(data.provenance.report_sha256, sha(fullBytes));
    assert.equal(data.provenance.source_sha256["training_dynamics.py"], "35d3f729d25c28f5e0d2f3a385133d2ee1f179be7775f56485504b2155fccf58");
    assert.equal(data.provenance.analysis_source_sha256["training_dynamics.py"], sha(await readFile(new URL("scripts/training_dynamics.py", root))));
    const text = dynamicsProvenance(data);
    assert.match(text, /未重新完整训练/); assert.match(text, /原训练代码/); assert.match(text, /当前分析代码/);
    assert.match(text, /587611efd52b00d9b0209af948bc72a54eb92075eb3aebea6e2d14bf9bfff645/);
  }
  assert.equal(sha(await readFile(new URL("artifacts/dynamics-lab-20261028-01/report.json", root))),
    "587611efd52b00d9b0209af948bc72a54eb92075eb3aebea6e2d14bf9bfff645");
});

test("population and sampling laws match an independent vector enumeration", () => {
  const vectors = [[1, 0], [0, 2], [3, -1]], mean = [4 / 3, 1 / 3];
  const gram = vectors.map(x => vectors.map(y => x.reduce((sum, value, i) => sum + value * y[i], 0)));
  const stats = gradientGramStatistics(gram); near(stats.mean_norm_sq, 17 / 9); near(stats.covariance_trace, 28 / 9);
  for (const batchSize of [1, 2, 3]) for (const scheme of ["with-replacement", "without-replacement", "repeated", "correlated"]) {
    const draws = [];
    function visit(prefix) {
      if (prefix.length === batchSize) { draws.push(prefix); return; }
      for (let i = 0; i < 3; i++) if (scheme !== "without-replacement" || !prefix.length || i > prefix.at(-1)) visit([...prefix, i]);
    }
    if (scheme === "repeated") for (let i = 0; i < 3; i++) draws.push(Array(batchSize).fill(i)); else visit([]);
    let weights = draws.map(() => 1 / draws.length);
    if (scheme === "correlated") {
      weights = weights.map(x => x / 2);
      for (let i = 0; i < 3; i++) { draws.push(Array(batchSize).fill(i)); weights.push(1 / 6); }
    }
    const exact = draws.reduce((sum, indices, i) => {
      const average = mean.map((_, coordinate) => indices.reduce((total, index) => total + vectors[index][coordinate], 0) / batchSize);
      return sum + weights[i] * average.reduce((distance, value, j) => distance + (value - mean[j]) ** 2, 0);
    }, 0);
    const actual = enumerateGradientBatches(gram, { batchSize, scheme });
    near(actual.enumerated_trace, exact); near(actual.formula_trace, exact); near(actual.probability_sum, 1);
  }
  assert.equal(batchNoiseVariance(0, 1, 1, false), 0);
  assert.equal(batchNoiseVariance(7, 3, 3, false), 0);
  near(batchNoiseVariance(7, 3, 8), 7 / 8);
  assert.throws(() => batchNoiseVariance(1, 3, 4, false));
});

test("PSD rejection and near-zero means never produce Infinity or a false zero ratio", () => {
  for (const invalid of [[[1, 2], [2, 1]], [[-1]], [[1, 1], [0, 1]], [[NaN]]]) assert.throws(() => validateGradientGram(invalid));
  const zero = gradientGramStatistics([[5, -5], [-5, 5]]);
  assert.equal(zero.noise_ratio_state, "undefined-zero-mean"); assert.equal(zero.noise_ratio, null);
  assert.equal(gradientGramStatistics([[0]]).noise_ratio, null);
  const nearZero = gradientGramStatistics([[1, -1 + 2e-12], [-1 + 2e-12, (-1 + 2e-12) ** 2]]);
  assert.notEqual(nearZero.noise_ratio_state, "defined"); assert.equal(nearZero.noise_ratio, null);
  assert.ok(enumerateGradientBatches([[1, -1], [-1, 1]], { batchSize: 2 }).rows.every(row => row.cosine_to_mean === null));
});

test("width selection uses base16 only, held128 is reserved, and parameter group violations are rejected", () => {
  assert.ok(widthData.width.selection.every(row => row.selection_width === 16 && !row.held_width_used && row.grid_boundary && row.selected_lr === .03));
  assert.ok(widthData.width.selected_runs.filter(run => run.width === 128).every(run => run.role === "held-out-transfer"));
  assert.equal(widthCoordinateRows(widthData).length, 8);
  assert.equal(depthProbeRows(widthData, { depth: 8, norm: "post", residualRule: "inverse-sqrt-depth", afterUpdates: 4 }).length, 11);
  const badLR = clone(widthData), parameter = badLR.width.selected_runs.find(run => run.width === 128 && run.parameterization === "mup").ledger.find(row => row.name === "input.weight");
  parameter.actual_lr /= 8;
  assert.throws(() => validateInitializationWidthReport(badLR), /实际 LR/);
  const badChoice = clone(widthData); badChoice.width.selection[0].selection_width = 64;
  assert.throws(() => validateInitializationWidthReport(badChoice), /宽度 16/);
  const badSource = clone(widthData); delete badSource.provenance.original_measurement;
  assert.throws(() => validateInitializationWidthReport(badSource), /原测量来源/);
});

test("batch requires common token grids across seeds and actual derived RNG evidence", () => {
  const grids = batchData.batch.runs.map(run => run.curve.map(row => row.consumed_input_tokens));
  for (const grid of grids) assert.deepEqual(grid, [0, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]);
  const badGrid = clone(batchData); badGrid.batch.runs[1].curve.splice(1, 1);
  assert.throws(() => validateBatchNoiseReport(badGrid), /共同的 token/);
  const badRNG = clone(batchData); delete badRNG.batch.runs.find(run => run.resume).resume.max_errors.rng;
  assert.throws(() => validateBatchNoiseReport(badRNG), /实际 RNG/);
  const badStream = clone(batchData); badStream.batch.runs[1].data_stream_sha256 = "0".repeat(64);
  assert.throws(() => validateBatchNoiseReport(badStream), /不同流/);
  assert.match(batchData.batch.runs.find(run => run.resume).resume.rng_reference_kind, /^derived/);
  assert.ok(batchTrainingRows(batchData, { split: "length" }).every(row => row.exact_match === 0));
  assert.equal(batchTrainingRows(batchData, { seed: 11 })[0].updates, 1024);
});

function mount(t, html, prefix) {
  const nodes = new Map();
  const text = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(text).join("");
  function create(tagName = "div") {
    const listeners = new Map(); let markup = "", words = "";
    return { tagName, value: "", hidden: false, disabled: false, files: [], options: [],
      get textContent() { return words; }, set textContent(value) { words = String(value); markup = ""; },
      get innerHTML() { return markup; }, set innerHTML(value) { markup = String(value); words = ""; },
      addEventListener(event, callback) { listeners.set(event, [...(listeners.get(event) || []), callback]); },
      async emit(event) { for (const callback of listeners.get(event) || []) await callback(); },
      querySelector(selector) { const match = selector.match(/^option\[value="([^"]+)"\]$/); return match ? this.options.find(option => option.value === match[1]) || null : null; } };
  }
  function visit(node) {
    const attrs = Object.fromEntries((node.attrs || []).map(a => [a.name, a.value]));
    const markers = Object.keys(attrs).filter(key => key.startsWith(`data-${prefix}-`));
    if (markers.length) {
      const el = create(node.tagName);
      el.options = (node.childNodes || []).filter(child => child.tagName === "option").map(option => {
        const a = Object.fromEntries((option.attrs || []).map(attr => [attr.name, attr.value]));
        return { value: a.value ?? text(option), selected: "selected" in a, disabled: "disabled" in a };
      });
      el.value = (el.options.find(option => option.selected) || el.options[0])?.value || attrs.value || "";
      el.hidden = "hidden" in attrs;
      for (const marker of markers) nodes.set(`[${marker}]`, el);
    }
    (node.childNodes || []).forEach(visit);
  }
  visit(parseFragment(html));
  const prior = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: selector => nodes.get(selector) || null, createElement: create } });
  t.after(() => prior ? Object.defineProperty(globalThis, "document", prior) : delete globalThis.document);
  return name => { const node = nodes.get(`[data-${prefix}-${name}]`); assert.ok(node, `missing ${prefix}/${name}`); return node; };
}

test("both pages initialize from defaults and render scope, ledger, Norm, sampling and evaluation changes", async t => {
  const width = mount(t, pages[0], "iw"); initInitializationWidth();
  assert.equal(width("output").hidden, false); assert.match(width("selection").textContent, /只由宽度 16/);
  assert.match(width("selection").textContent, /边界/); assert.match(width("provenance").textContent, /原训练代码/);
  width("depth").value = "8"; width("norm").value = "post"; width("depth-step").value = "4";
  await width("norm").emit("change"); assert.match(width("depth-note").textContent, /POST LayerNorm/);
  assert.match(width("depth-rows").innerHTML, /block7.output/);
  width("ledger-width").value = "128"; await width("ledger-width").emit("change");
  assert.match(width("ledger").innerHTML, /hidden.weight/);
  const batch = mount(t, pages[1], "bn"); initBatchNoise();
  assert.equal(batch("output").hidden, false); assert.match(batch("checks").textContent, /derived/);
  assert.match(batch("checks").textContent, /原 run 未保存/); assert.match(batch("budget").textContent, /32,768/);
  batch("scheme").value = "without-replacement"; batch("batch").value = "6";
  await batch("scheme").emit("change"); assert.equal(batch("variance").textContent, "0");
  batch("scheme").value = "correlated"; await batch("scheme").emit("change");
  assert.equal(batch("batch").value, "4"); assert.match(batch("noise-note").textContent, /ρ=.5/);
  batch("population").value = "centered"; await batch("population").emit("change");
  assert.equal(batch("ratio").textContent, "未定义／病态");
  batch("split").value = "length"; await batch("split").emit("change");
  assert.match(batch("training-rows").innerHTML, /0%/);
});

test("local imports on both pages clear errors, escape text, cap files and resolve races by last selection", async t => {
  for (const [prefix, html, initialize, output] of [["iw", pages[0], initInitializationWidth, "ledger"], ["bn", pages[1], initBatchNoise, "records"]]) {
    const node = mount(t, html, prefix); initialize();
    const submit = async (name, bytes) => { node("file").files = [{ name, size: bytes.length, text: async () => bytes.toString() }]; await node("file").emit("change"); };
    await submit("full.json", fullBytes); assert.equal(node("output").hidden, false);
    await submit("invalid.json", "{}"); assert.equal(node("output").hidden, true);
    assert.equal(node(output).innerHTML, ""); assert.equal(node("provenance").textContent, "");
    assert.match(node("status").textContent, /无法读取/);
    const unsafe = clone(prefix === "iw" ? widthData : batchData);
    unsafe.provenance.run_id = '<img src=x onerror="unsafe=1">';
    if (prefix === "bn") unsafe.noise.records[0].id = '<svg onload="unsafe=1">';
    await submit("unsafe.json", JSON.stringify(unsafe)); assert.equal(node("output").hidden, false);
    assert.match(node("provenance").textContent, /<img/);
    if (prefix === "bn") { assert.match(node("records").innerHTML, /&lt;svg/); assert.ok(!node("records").innerHTML.includes("<svg")); }
    let release;
    node("file").files = [{ name: "slow.json", size: fullBytes.length, text: () => new Promise(resolve => { release = resolve; }) }];
    const pending = node("file").emit("change");
    await submit("latest-invalid.json", "{}"); release(fullBytes.toString()); await pending;
    assert.equal(node("output").hidden, true); assert.match(node("status").textContent, /无法读取/);
    node("file").files = [{ name: "oversize.json", size: 32 * 1024 * 1024 + 1, text: async () => { throw new Error("must not read"); } }];
    await node("file").emit("change"); assert.match(node("status").textContent, /32 MiB/);
    await node("default").emit("click"); assert.equal(node("output").hidden, false);
  }
});

test("scoped HTML math renders and tables, SVGs and controls remain accessible without a site build", () => {
  for (const [i, html] of pages.entries()) {
    const ids = new Set();
    function visit(node, parent) {
      const attrs = Object.fromEntries((node.attrs || []).map(a => [a.name, a.value]));
      if (attrs.id) { assert.ok(!ids.has(attrs.id), attrs.id); ids.add(attrs.id); }
      if (attrs.class?.split(" ").includes("compare-wrap")) assert.ok(attrs.tabindex === "0" && attrs.role === "region" && attrs["aria-label"]);
      if (node.tagName === "svg") assert.ok(attrs.role === "img" && attrs["aria-label"]);
      if (["input", "select"].includes(node.tagName)) assert.equal(parent.tagName, "label");
      assert.notEqual(node.tagName, "script");
      (node.childNodes || []).forEach(child => visit(child, node));
    }
    visit(parseFragment(html));
    const rendered = renderMath(html, i ? "batch-noise.html" : "initialization-width.html");
    assert.equal(rendered.count, i ? 6 : 3); assert.match(rendered.html, /<math/); assert.ok(!rendered.html.includes("katex-error"));
  }
});
