import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile, writeFile, mkdir, realpath } from "node:fs/promises";
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseFragment } from "parse5";
import { renderMath } from "./render-math.mjs";
import { makeChunkedCEFixtures } from "./export-chunked-ce-fixtures.mjs";
import { chunkedCEInputs, denseCEReference, chunkedCrossEntropy, validateChunkedCEReport, chunkedCEReportRows, createChunkedCEReportLoader, initChunkedCrossEntropy } from "../src/scripts/features/chunked-cross-entropy.js";
import { chunkedCEReportSnapshot } from "../src/scripts/features/chunked-cross-entropy-data.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const near = (a, b, tolerance = 2e-10) => assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tolerance, `${a} vs ${b}; tolerance ${tolerance}`);
const error = (a, b) => {
  const aa = Array.isArray(a) ? a.flat(Infinity) : [a], bb = Array.isArray(b) ? b.flat(Infinity) : [b];
  assert.equal(aa.length, bb.length); return Math.max(0, ...aa.map((x, i) => Math.abs(x - bb[i])));
};
const fixture = makeChunkedCEFixtures();

test("all 360 exact fixtures independently match dense loss/dH/dW/db", () => {
  assert.equal(fixture.cases.length, 360);
  for (const c of fixture.cases) {
    const inputs = fixture.inputs[c.input], r = chunkedCrossEntropy({ inputs, snapshots: false, ...c.options });
    const dense = denseCEReference(inputs, c.options);
    for (const name of ["loss", "dH", "dW", "db"]) near(error(r[name], dense[name]), 0);
    near(r.errors.loss, 0); near(r.errors.dH, 0); near(r.errors.dW, 0); near(r.errors.db, 0);
  }
});

test("all H/W/b coordinates match central differences with negative nonunit upstream", () => {
  for (const bias of [true, false]) for (const mask of ["all", "mixed"]) {
    const inputs = chunkedCEInputs({ bias, mask }), upstream = -2.25, epsilon = 1e-5;
    const actual = chunkedCrossEntropy({ inputs, tokenBlock: 3, vocabBlock: 2, upstream, snapshots: false });
    for (const key of ["H", "W", ...(bias ? ["bias"] : [])]) {
      for (let i = 0; i < inputs[key].length; i++) for (let c = 0; c < (key === "bias" ? 1 : inputs[key][i].length); c++) {
        const value = key === "bias" ? inputs[key][i] : inputs[key][i][c];
        const put = next => { if (key === "bias") inputs[key][i] = next; else inputs[key][i][c] = next; };
        put(value + epsilon); const plus = denseCEReference(inputs).loss;
        put(value - epsilon); const minus = denseCEReference(inputs).loss;
        put(value);
        near(key === "bias" ? actual.db[i] : actual["d" + key][i][c], upstream * (plus - minus) / (2 * epsilon), 2e-9);
      }
    }
  }
});

test("mask and ignored rows have zero dH and one global valid-position denominator", () => {
  const inputs = chunkedCEInputs({ mask: "mixed" }), r = chunkedCrossEntropy({ inputs });
  assert.equal(r.validCount, 3);
  assert.deepEqual(r.dH[1], [0, 0, 0]); assert.deepEqual(r.dH[2], [0, 0, 0]);
  assert.equal(r.lse[1], null); assert.equal(r.perPositionLoss[2], null);
  near(r.loss, r.perPositionLoss.filter(x => x !== null).reduce((a, b) => a + b, 0) / 3);
});

test("empty labels/mask/N return explicit zero sentinel without pretending a defined mean", () => {
  for (const key of ["empty-bias", "ignore-all-no-bias", "zero-positions"]) {
    const r = chunkedCrossEntropy({ inputs: fixture.inputs[key], upstream: -2.25 });
    assert.equal(r.skipped, true); assert.equal(r.validCount, 0); assert.equal(r.loss, 0); assert.equal(r.events.length, 0);
    near(error(r.dH, r.dH.map(row => row.map(() => 0))), 0); near(error(r.dW, r.dW.map(row => row.map(() => 0))), 0);
    assert.equal(r.storage.savedRowStatistics, 0); assert.equal(r.work.forwardScores, 0); assert.equal(r.work.recomputedScores, 0);
  }
});

test("upstream scales gradients alone, with unit/nonunit/negative/zero cases", () => {
  const one = chunkedCrossEntropy({ snapshots: false });
  for (const upstream of [0, .375, -2.25, 3.5]) {
    const r = chunkedCrossEntropy({ upstream, snapshots: false }); near(r.loss, one.loss);
    for (const key of ["dH", "dW", "db"]) near(error(r[key], one[key].map(row => Array.isArray(row) ? row.map(x => x * upstream) : row * upstream)), 0);
  }
});

test("large score shifts remain stable and probabilities keep one global mass", () => {
  const baseline = chunkedCrossEntropy({ snapshots: false });
  for (const scoreShift of [1e4, -1e4, 1e6, -1e6]) {
    const r = chunkedCrossEntropy({ scoreShift, vocabBlock: 2, tokenBlock: 3 });
    near(r.loss, baseline.loss); near(error(r.dH, baseline.dH), 0);
    r.probabilityMass.forEach((mass, i) => near(mass, r.inputs.labels[i] === -100 ? 0 : 1));
    assert.ok([r.dH, r.dW, r.db].flat(Infinity).every(Number.isFinite));
  }
});

test("target captured correctly in every vocab block including final tail", () => {
  for (let target = 0; target < 7; target++) for (const vocabBlock of [1, 2, 3, 4]) {
    const inputs = chunkedCEInputs({ target }), r = chunkedCrossEntropy({ inputs, vocabBlock, tokenBlock: 3 });
    const dense = denseCEReference(inputs); near(r.targetLogit[3], dense.logits[3][target]);
    const forward = r.events.filter(e => e.stage === "forward" && e.tStart <= 3 && e.tEnd > 3);
    const firstHit = forward.find(e => e.rows.find(row => row.position === 3).targetLogit !== null);
    assert.ok(firstHit.vStart <= target && firstHit.vEnd > target);
  }
});

test("forward running statistics and tail shapes agree with dense prefixes", () => {
  const r = chunkedCrossEntropy({ tokenBlock: 3, vocabBlock: 2 }), inputs = r.inputs, full = r.reference.logits;
  assert.equal(r.events.length, 2 * Math.ceil(5 / 3) * Math.ceil(7 / 2));
  const lastForward = r.events.filter(e => e.stage === "forward").at(-1);
  assert.deepEqual([lastForward.tEnd - lastForward.tStart, lastForward.vEnd - lastForward.vStart], [2, 1]);
  for (const e of r.events.filter(e => e.stage === "forward")) for (const row of e.rows) if (row.active) {
    const logits = full[row.position].slice(0, e.vEnd), maximum = Math.max(...logits), sum = logits.reduce((s, z) => s + Math.exp(z - maximum), 0);
    near(row.maximum, maximum); near(row.denominator, sum); near(row.lse, maximum + Math.log(sum));
    if (inputs.labels[row.position] < e.vEnd) near(row.targetLogit, full[row.position][inputs.labels[row.position]]);
    else assert.equal(row.targetLogit, null);
  }
});

test("UI backward snapshots match final results; snapshots are explicitly excluded", () => {
  const r = chunkedCrossEntropy(), final = r.events.at(-1);
  assert.equal(final.stage, "backward");
  for (const name of ["dH", "dW", "db"]) near(error(final[name], r[name]), 0);
  assert.equal(r.storage.snapshotsExcluded, true);
  const noSnapshots = chunkedCrossEntropy({ snapshots: false }); assert.equal(noSnapshots.events.length, 0);
  near(error(noSnapshots.dH, r.dH), 0);
});

test("local normalization and overwritten dH produce distinct gradient failures", () => {
  const good = chunkedCrossEntropy(), local = chunkedCrossEntropy({ fault: "local-softmax" }), overwrite = chunkedCrossEntropy({ fault: "overwrite-dh" });
  for (const r of [local, overwrite]) { near(r.loss, good.loss); assert.ok(r.errors.dH > .01); }
  assert.ok(local.errors.dW > .01); assert.ok(local.errors.db > .01);
  near(overwrite.errors.dW, 0); near(overwrite.errors.db, 0);
  local.probabilityMass.forEach((mass, i) => near(mass, good.inputs.labels[i] === -100 ? 0 : 3));
  for (const fault of ["local-softmax", "overwrite-dh"]) {
    near(chunkedCrossEntropy({ fault, vocabBlock: 7 }).errors.dH, 0);
    near(chunkedCrossEntropy({ fault, upstream: 0 }).errors.dH, 0);
  }
});

test("logical categories distinguish sub-full blocks from full-sized edge choice", () => {
  const r = chunkedCrossEntropy();
  assert.equal(r.storage.denseLogits, 35); assert.equal(r.storage.sharedH, 15); assert.equal(r.storage.sharedW, 21);
  assert.equal(r.storage.sharedBias, 7); assert.equal(r.storage.savedRowStatistics, 10); assert.equal(r.storage.gradientBuffers, 43);
  assert.equal(r.storage.maximumScoreTile, 6); assert.equal(r.work.forwardScores, 28); assert.equal(r.work.recomputedScores, 28);
  assert.equal(chunkedCrossEntropy({ tokenBlock: 8, vocabBlock: 9 }).storage.maximumScoreTile, 35);
});

test("reject invalid shapes, finite values, labels, masks and options", () => {
  const bad = mutator => { const inputs = chunkedCEInputs(); mutator(inputs); assert.throws(() => chunkedCrossEntropy({ inputs }), RangeError); };
  bad(i => { i.H[0] = [0]; }); bad(i => { i.W[0][0] = NaN; }); bad(i => { i.bias[0] = Infinity; });
  bad(i => { i.labels[1] = 9; i.mask[1] = false; }); bad(i => { i.mask[0] = 1; });
  bad(i => { i.labels[0] = 1.5; }); bad(i => { i.bias = [0]; });
  for (const options of [{ tokenBlock: 0 }, { vocabBlock: -1 }, { tokenBlock: 1.5 }, { fault: "other" }, { scoreShift: Infinity }, { upstream: NaN }, { snapshots: 1 }]) assert.throws(() => chunkedCrossEntropy(options), RangeError);
  const overflow = chunkedCEInputs(); overflow.H[0][0] = 1e308; overflow.W[0][0] = 1e308;
  assert.throws(() => chunkedCrossEntropy({ inputs: overflow }), /溢出/);
});

const readSnapshot = async () => {
  if (chunkedCEReportSnapshot) return validateChunkedCEReport(chunkedCEReportSnapshot);
  const index = process.argv.indexOf("--report");
  if (index >= 0 && process.argv[index + 1]) return validateChunkedCEReport(JSON.parse(await readFile(process.argv[index + 1], "utf8")));
  return null;
};
const report = await readSnapshot();

test("default snapshot is provenance extracted from an actual saved run", { skip: !report }, async () => {
  validateChunkedCEReport(report);
  assert.equal(report.runtime.device, "cpu"); assert.equal(report.runtime.dtype, "float64");
  const bytes = await readFile(path.resolve(root, report.provenance.report_path));
  const full = JSON.parse(bytes); assert.equal(full.schema, "output-lab-v1");
  if (report.provenance.report_sha256) assert.equal(createHash("sha256").update(bytes).digest("hex"), report.provenance.report_sha256);
  for (let i = 0; i < report.measurements.length; i++) {
    assert.deepEqual(report.measurements[i].wall_ms, full.measurements[i].wall_ms);
    assert.equal(report.measurements[i].saved_tensor_storage.peak_referenced_bytes, full.measurements[i].saved_tensor_storage.peak_referenced_bytes);
  }
});

test("report validator rejects inconsistent samples, bounds, shapes and storage", { skip: !report }, () => {
  for (const mutate of [
    r => { r.status = "failed"; }, r => { r.runtime.device = "cuda"; }, r => { r.runtime.dtype = "float32"; },
    r => { r.provenance.python_sha256 = "invalid"; }, r => { r.correctness.no_full_score_allocation = false; },
    r => { r.measurements[0].wall_ms.samples[0] = NaN; }, r => { r.measurements[0].wall_ms.median += 1; },
    r => { r.measurements[0].shape[0] = 0; }, r => { r.measurements[0].logical.dense_logits_elements++; },
    r => { r.measurements[0].saved_tensor_storage.live_bytes_after_backward = 1; },
    r => { r.measurements[1].saved_tensor_storage.full_score_shape_save_requests = 1; },
    r => { r.measurements[1].logical.maximum_score_tile_elements++; },
    r => { r.measurements[0].reference_errors.dH = -1; }, r => { r.measurements.push(structuredClone(r.measurements[0])); },
  ]) {
    const invalid = structuredClone(report); mutate(invalid); assert.throws(() => validateChunkedCEReport(invalid), RangeError);
  }
  assert.throws(() => validateChunkedCEReport({ schema: "output-lab-v1" }), RangeError);
});

test("report table safely escapes text and filters incomparable shapes", { skip: !report }, () => {
  const copy = structuredClone(report); copy.measurements[0].name = '<img src=x onerror="window.__unsafe=1">';
  const rows = chunkedCEReportRows(copy, copy.measurements[0].shape.join("×"));
  assert.ok(rows.includes("&lt;img")); assert.ok(!rows.includes("<img"));
  assert.equal((rows.match(/<tr>/g) || []).length, 4);
});

test("async import race keeps newest file and cancellation discards old pending file", { skip: !report }, async () => {
  let release, displayed = null, status = "", clears = 0;
  const loader = createChunkedCEReportLoader({ onClear: () => { displayed = null; clears++; }, onReport: r => { displayed = r; }, onStatus: s => { status = s; } });
  const slow = loader.read({ name: "slow.json", size: 2, text: () => new Promise(resolve => { release = resolve; }) });
  await loader.read({ name: "fast.json", size: 2, text: async () => JSON.stringify(report) });
  assert.ok(displayed); assert.ok(status.includes("fast.json"));
  release("{}"); await slow; assert.ok(displayed); assert.ok(status.includes("fast.json"));
  const pending = loader.read({ name: "old.json", size: 2, text: () => new Promise(resolve => { release = resolve; }) });
  loader.cancel(); release(JSON.stringify(report)); await pending; assert.equal(displayed, null); assert.equal(clears, 4);
});

test("bad/oversize/cancelled imports clear results and can recover", { skip: !report }, async () => {
  let displayed = report, status = "";
  const loader = createChunkedCEReportLoader({ onClear: () => { displayed = null; }, onReport: r => { displayed = r; }, onStatus: s => { status = s; } });
  await loader.read({ size: 6 * 1024 * 1024, text: async () => { throw new Error("not called"); } }); assert.equal(displayed, null); assert.ok(status.includes("5 MiB"));
  await loader.read({ size: 2, text: async () => "not json" }); assert.equal(displayed, null); assert.ok(status.startsWith("无法读取"));
  await loader.read(undefined); assert.equal(displayed, null); assert.ok(status.includes("请选择"));
  await loader.read({ name: "report.json", size: 2, text: async () => JSON.stringify(report) }); assert.ok(displayed); assert.ok(status.includes("report.json"));
});

test("page TeX/structure/labels/anchors and CSS follow existing conventions", async () => {
  const content = await readFile(path.join(root, "src/content/chunked-cross-entropy.html"), "utf8"), rendered = renderMath(content, "chunked-cross-entropy.html");
  assert.equal(rendered.count, 4); assert.ok(rendered.html.includes('<math xmlns="http://www.w3.org/1998/Math/MathML"')); assert.ok(!rendered.html.includes("katex-error"));
  const tree = parseFragment(content), nodes = [];
  const walk = node => { nodes.push(node); for (const child of node.childNodes || []) walk(child); }; walk(tree);
  const attrs = node => Object.fromEntries((node.attrs || []).map(a => [a.name, a.value]));
  const ids = nodes.map(n => attrs(n).id).filter(Boolean); assert.equal(new Set(ids).size, ids.length);
  for (const heading of nodes.filter(n => n.tagName === "h2")) assert.ok(attrs(heading).id);
  assert.equal(nodes.filter(n => n.tagName === "h2").length, 7);
  for (const wrap of nodes.filter(n => attrs(n).class?.split(" ").includes("compare-wrap"))) assert.equal(attrs(wrap).tabindex, "0");
  for (const svg of nodes.filter(n => n.tagName === "svg")) assert.ok(attrs(svg)["aria-label"]);
  for (const button of nodes.filter(n => n.tagName === "button")) assert.ok(attrs(button).class.includes("button"));
  for (const node of nodes.filter(n => attrs(n).href)) {
    const [filename, anchor] = attrs(node).href.split("#");
    const linked = await readFile(path.join(root, "src/content", filename), "utf8");
    if (anchor) assert.ok(linked.includes(`id="${anchor}"`), attrs(node).href);
  }
  assert.ok(content.includes("未定义")); assert.ok(content.includes("快照仅供教学"));
  assert.ok(content.includes("allocator")); assert.ok(content.includes("RSS"));
});

test("feature initialization is inert when the page is absent", () => {
  const old = globalThis.document;
  try { globalThis.document = { querySelector: () => null }; assert.doesNotThrow(initChunkedCrossEntropy); }
  finally { if (old) globalThis.document = old; else delete globalThis.document; }
});

async function runBrowserQA() {
  if (!report) throw new Error("Browser QA requires an actual successful report snapshot or --report <report.json>.");
  const args = process.argv.slice(2), q = args.indexOf("--qa-dir");
  const directory = path.resolve(q >= 0 ? args[q + 1] : path.join(root, "artifacts", `output-ui-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}`));
  await mkdir(directory); // QA never overwrites earlier screenshots/reports.
  const content = renderMath(await readFile(path.join(root, "src/content/chunked-cross-entropy.html"), "utf8"), "chunked-cross-entropy.html").html;
  const styles = ["tokens", "layout", "shared", "modern", "exact-mechanics", "decision-chains", "chunked-cross-entropy", "overrides", "wiki", "math"];
  const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>分块 CE 独立验收</title><link rel="icon" href="/src/assets/favicon.svg">${styles.map(s => `<link rel="stylesheet" href="/src/styles/${s}.css">`).join("")}<link rel="stylesheet" href="/node_modules/katex/dist/katex.min.css"></head><body data-page="chunked-cross-entropy"><div class="wiki-workspace"><aside class="wiki-library"><nav aria-label="验收目录"><p>分块交叉熵独立验收</p></nav></aside><main id="main">${content}</main></div><script type="module">import{initChunkedCrossEntropy}from'/src/scripts/features/chunked-cross-entropy.js';initChunkedCrossEntropy();document.documentElement.dataset.ceReady='true';</script></body></html>`;
  await writeFile(path.join(directory, "isolated-page.html"), html, { flag: "wx" });
  const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".json": "application/json" };
  const server = createServer(async (req, res) => {
    try {
      const requested = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
      if (requested === "/" || requested === "/chunked-cross-entropy.html") { res.writeHead(200, { "Content-Type": "text/html" }); res.end(html); return; }
      const file = path.resolve(root, "." + requested);
      if (!file.startsWith(root + path.sep) || (!requested.startsWith("/src/") && !requested.startsWith("/node_modules/katex/"))) throw new Error("Invalid path");
      res.writeHead(200, { "Content-Type": types[path.extname(file)] || "application/octet-stream" }); res.end(await readFile(file));
    } catch { if (!res.headersSent) res.writeHead(404); res.end("Not found"); }
  });
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  const port = server.address().port, profile = path.join(directory, "chrome-profile");
  await mkdir(profile);
  const chrome = spawn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check", "--disable-background-networking", "--disable-component-update", "--disable-sync", "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
  const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
  let socket = null, target = null, devtoolsPort, sequence = 0, active = "initialization", stderr = "";
  const errors = [], results = [], pending = new Map();
  chrome.stderr.on("data", data => { stderr += data; });
  chrome.on("error", e => { stderr += e.message; });
  const check = (name, ok, detail) => { results.push({ name, ok: Boolean(ok), ...(detail === undefined ? {} : { detail }) }); if (!ok) console.error(name, detail); };
  try {
    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      try { devtoolsPort = Number((await readFile(path.join(profile, "DevToolsActivePort"), "utf8")).split("\n")[0]); if (devtoolsPort) break; } catch { /* Chrome starting */ }
      if (chrome.exitCode !== null) throw new Error(`Chrome exited: ${stderr}`);
      await pause(100);
    }
    if (!devtoolsPort) throw new Error(`Chrome failed to open DevTools: ${stderr}`);
    target = await (await fetch(`http://127.0.0.1:${devtoolsPort}/json/new`, { method: "PUT" })).json();
    socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { socket.addEventListener("open", resolve, { once: true }); socket.addEventListener("error", reject, { once: true }); });
    socket.addEventListener("message", event => {
      const message = JSON.parse(event.data);
      if (message.method === "Runtime.exceptionThrown") errors.push(`${active}: ${message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text}`);
      if (message.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(message.params.type)) errors.push(`${active}: ${message.params.args.map(a => a.value || a.description).join(" ")}`);
      if (message.method === "Network.responseReceived" && message.params.response.status >= 400) errors.push(`${active}: HTTP ${message.params.response.status} ${message.params.response.url}`);
      if (message.method === "Network.loadingFailed" && !message.params.canceled) errors.push(`${active}: ${message.params.errorText}`);
      const p = pending.get(message.id); if (!p) return; clearTimeout(p.timer); pending.delete(message.id); message.error ? p.reject(new Error(message.error.message)) : p.resolve(message.result);
    });
    const send = (method, params = {}) => new Promise((resolve, reject) => { const id = ++sequence, timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timeout ${method}`)); }, 10000); pending.set(id, { resolve, reject, timer }); socket.send(JSON.stringify({ id, method, params })); });
    const evaluate = async expression => { const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text); return r.result.value; };
    const value = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).textContent`);
    const change = (selector, v) => evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});e.value=${JSON.stringify(String(v))};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
    const click = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
    const importReport = async (data, name) => { await evaluate(`(()=>{const input=document.querySelector('[data-ce-report-file]'),transfer=new DataTransfer();transfer.items.add(new File([${JSON.stringify(JSON.stringify(data))}],${JSON.stringify(name)},{type:'application/json'}));input.files=transfer.files;input.dispatchEvent(new Event('change',{bubbles:true}));})()`); await pause(60); };
    const layout = async label => {
      await evaluate("document.fonts.ready");
      const r = await evaluate(`(()=>{const main=document.querySelector('main');return{width:innerWidth,doc:document.documentElement.scrollWidth,main:main.scrollWidth,client:main.clientWidth,badLabels:[...document.querySelectorAll('main input,main select')].filter(e=>!e.labels?.length&&!e.getAttribute('aria-label')).length,badMath:document.querySelectorAll('.katex-error').length,scrollable:[...document.querySelectorAll('main .compare-wrap')].filter(e=>e.scrollWidth>e.clientWidth+2).length,unfocusable:[...document.querySelectorAll('main .compare-wrap')].filter(e=>e.scrollWidth>e.clientWidth+2&&e.tabIndex<0).length};})()`);
      check(label, r.doc <= r.width + 1 && r.main <= r.client + 2 && !r.badLabels && !r.badMath && !r.unfocusable, r);
    };
    await send("Runtime.enable"); await send("Page.enable"); await send("Network.enable"); await send("Network.setCacheDisabled", { cacheDisabled: true });
    for (const [width, theme] of [[390, "light"], [768, "light"], [1440, "dark"]]) {
      active = `${width}/${theme}`;
      await send("Emulation.setDeviceMetricsOverride", { width, height: 960, deviceScaleFactor: 1, mobile: false });
      await send("Page.navigate", { url: `http://127.0.0.1:${port}/chunked-cross-entropy.html` });
      const loadDeadline = Date.now() + 10000;
      while (Date.now() < loadDeadline) { if (await evaluate("document.documentElement.dataset.ceReady==='true' && document.readyState==='complete'")) break; await pause(60); }
      check(`Initialization ${active}`, await evaluate("document.documentElement.dataset.ceReady==='true'"));
      await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)}`);
      check(`Baseline dH ${active}`, Number(await value("[data-ce-dh-error]")) < 1e-10);
      check(`Default actual report ${active}`, await evaluate("!document.querySelector('[data-ce-report-output]').hidden && document.querySelector('[data-ce-report-provenance]').textContent.includes('PyTorch 2.9.0')"));
      for (const fault of ["local-softmax", "overwrite-dh"]) { await change("[data-ce-fault]", fault); check(`Fault ${fault} ${active}`, Number(await value("[data-ce-dh-error]")) > .01); }
      check(`Overwrite isolates dW ${active}`, Number(await value("[data-ce-dw-error]")) < 1e-10);
      await change("[data-ce-upstream]", "0"); check(`Zero upstream hides fault ${active}`, Number(await value("[data-ce-dh-error]")) === 0);
      await change("[data-ce-upstream]", "-2.25"); await change("[data-ce-fault]", "none");
      for (const key of ["dh", "dw", "db"]) check(`Negative upstream ${key} ${active}`, Number(await value(`[data-ce-${key}-error]`)) < 1e-10);
      await change("[data-ce-mask]", "empty"); check(`Zero valid sentinel ${active}`, (await value("[data-ce-loss]")).includes("未定义") && await evaluate("document.querySelector('[data-ce-next]').disabled"));
      await change("[data-ce-mask]", "mixed"); await change("[data-ce-token]", "3"); await change("[data-ce-vocab]", "2"); await change("[data-ce-target]", "5"); await change("[data-ce-shift]", "1000000");
      check(`Tails mask and shift ${active}`, Number(await value("[data-ce-dh-error]")) < 2e-10 && (await value("[data-ce-count]")).startsWith("3 /"));
      const initialEvent = await value("[data-ce-event]"); await click("[data-ce-next]"); check(`Step action ${active}`, initialEvent !== await value("[data-ce-event]"));
      await evaluate("for(let i=0;i<80&&!document.querySelector('[data-ce-next]').disabled;i++)document.querySelector('[data-ce-next]').click()");
      check(`Backward final ${active}`, (await value("[data-ce-event]")).includes("反向") && await evaluate("document.querySelector('[data-ce-next]').disabled"));
      await change("[data-ce-bias]", "off"); check(`Bias optional ${active}`, await value("[data-ce-db-error]") === "无 bias");
      await change("[data-ce-report-metric]", "storage"); check(`Measured storage graph ${active}`, (await value("[data-ce-measure-chart]")).includes("KiB"));
      const malicious = structuredClone(report); malicious.measurements[0].name = '<img src=x onerror="window.__unsafe=1">';
      await importReport(malicious, "safe-text.json");
      check(`Import escapes text ${active}`, await evaluate("!document.querySelector('[data-ce-report-rows] img') && !window.__unsafe && document.querySelector('[data-ce-report-rows]').textContent.includes('<img')"));
      await importReport({ schema: "invalid" }, "bad.json"); check(`Invalid clears old report ${active}`, await evaluate("document.querySelector('[data-ce-report-output]').hidden && document.querySelector('[data-ce-report-rows]').textContent===''"));
      await importReport(report, "real-report.json"); check(`Valid import recovers ${active}`, await evaluate("!document.querySelector('[data-ce-report-output]').hidden"));
      await evaluate(`(async()=>{const input=document.querySelector('[data-ce-report-file]'), slow=new File(['ignored'],'slow.json');slow.text=()=>new Promise(resolve=>setTimeout(()=>resolve('{}'),120));const one=new DataTransfer();one.items.add(slow);input.files=one.files;input.dispatchEvent(new Event('change',{bubbles:true}));const fast=new File([${JSON.stringify(JSON.stringify(report))}],'fast.json');const two=new DataTransfer();two.items.add(fast);input.files=two.files;input.dispatchEvent(new Event('change',{bubbles:true}));await new Promise(resolve=>setTimeout(resolve,170));})()`);
      check(`Newest file race ${active}`, (await value("[data-ce-report-status]")).includes("fast.json") && await evaluate("!document.querySelector('[data-ce-report-output]').hidden"));
      await click("[data-ce-report-default]");
      await evaluate("document.querySelectorAll('main details').forEach(e=>e.open=true);document.querySelectorAll('main .math-display').forEach(e=>e.tabIndex=0)");
      await layout(`All expanded layout ${active}`);
      for (const [name, selector] of [["tiles", "#ce-lab"], ["measurements", "#ce-cpu-lab"]]) {
        await evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({behavior:'instant',block:'start'})`); await pause(80);
        const png = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
        await writeFile(path.join(directory, `${name}-${width}-${theme}.png`), Buffer.from(png.data, "base64"), { flag: "wx" });
      }
    }
    check("No page errors or resource failures", !errors.length, errors);
    const result = { schema: "chunked-ce-ui-qa-v1", passed: results.every(r => r.ok), created_at: new Date().toISOString(), checks: results.length, results, scope: "Isolated exact page with actual site styles/math/feature. No shared metadata registration or site build; final integration/browser audit remains integrator-owned." };
    await writeFile(path.join(directory, "interaction-results.json"), JSON.stringify(result, null, 2) + "\n", { flag: "wx" });
    console.log(JSON.stringify({ checks: results.length, failed: results.filter(r => !r.ok), artifacts: directory }, null, 2));
    if (!result.passed) process.exitCode = 1;
  } finally {
    if (socket) socket.close();
    if (devtoolsPort && target) await fetch(`http://127.0.0.1:${devtoolsPort}/json/close/${target.id}`).catch(() => {});
    chrome.kill("SIGTERM"); server.close();
  }
}

if (process.argv.includes("--ui")) await runBrowserQA();
