import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { pairedExample } from "../src/scripts/features/paired-evaluation.js";

const port = process.env.CDP_PORT || "9222", baseUrl = process.env.BASE_URL || "http://127.0.0.1:4173";
const directory = path.resolve(process.env.QA_DIR || "artifacts/decision-chains-qa");
await mkdir(directory, { recursive: true });
const target = await (await fetch(`http://127.0.0.1:${port}/json/new`, { method: "PUT" })).json();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.addEventListener("open", resolve, { once: true }); socket.addEventListener("error", reject, { once: true }); });
let sequence = 0, active = "initialization"; const pending = new Map(), errors = [], results = [];
socket.addEventListener("message", event => {
  const message = JSON.parse(event.data);
  if (message.method === "Runtime.exceptionThrown") errors.push(`${active}: ${message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text}`);
  if (message.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(message.params.type)) errors.push(`${active}: ${message.params.args.map(arg => arg.value || arg.description).join(" ")}`);
  if (message.method === "Network.responseReceived" && message.params.response.status >= 400) errors.push(`${active}: HTTP ${message.params.response.status} ${message.params.response.url}`);
  if (message.method === "Network.loadingFailed" && !message.params.canceled) errors.push(`${active}: ${message.params.errorText}`);
  const request = pending.get(message.id); if (!request) return; clearTimeout(request.timer); pending.delete(message.id);
  message.error ? request.reject(new Error(message.error.message)) : request.resolve(message.result);
});
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++sequence, timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timeout: ${method}, ${active}`)); }, 10000);
  pending.set(id, { resolve, reject, timer }); socket.send(JSON.stringify({ id, method, params }));
});
const evaluate = async expression => {
  const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  return result.result.value;
};
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const check = (name, ok, detail) => { results.push({ name, ok: Boolean(ok), ...(detail === undefined ? {} : { detail }) }); };
const value = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).textContent`);
async function change(selector, choice) {
  await evaluate(`(() => { const el=document.querySelector(${JSON.stringify(selector)});el.value=${JSON.stringify(String(choice))};el.dispatchEvent(new Event('change',{bubbles:true})); })()`);
}
async function click(selector) { await evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`); }
async function navigate(slug) {
  active = slug; const pathname = `/${slug}.html`; await send("Page.navigate", { url: `${baseUrl}${pathname}` });
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    if (await evaluate(`location.pathname===${JSON.stringify(pathname)} && document.readyState==='complete' && document.documentElement.dataset.wikiReady==='true'`)) { await evaluate("document.fonts.ready.then(()=>true)"); return; }
    await pause(70);
  }
  throw new Error(`Page failed to initialize: ${slug}`);
}
async function layout(name) {
  await pause(80);
  const state = await evaluate(`(() => {
    const main=document.querySelector('main'); const rect=main.getBoundingClientRect();
    const badControls=[...document.querySelectorAll('main input,main select,main textarea')].filter(el=>!el.labels?.length&&!el.getAttribute('aria-label')).length;
    const scroll=[...document.querySelectorAll('main .compare-wrap')].filter(el=>el.getBoundingClientRect().width>0&&el.scrollWidth>el.clientWidth+2);
    return {width:innerWidth,doc:document.documentElement.scrollWidth,main:main.scrollWidth,client:main.clientWidth,right:rect.right,badControls,toc:document.querySelectorAll('[data-toc-links] a').length,headings:document.querySelectorAll('main h2[id]').length,badMath:document.querySelectorAll('.katex-error').length,unfocusable:scroll.filter(el=>el.tabIndex<0).length};
  })()`);
  check(name, state.doc <= state.width + 1 && state.main <= state.client + 2 && state.right <= state.width + 1 && !state.badControls && !state.badMath && !state.unfocusable && state.toc === state.headings, state);
}
async function screenshot(slug, width, theme, selector) {
  await evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block:'start',behavior:'instant'});`); await pause(90);
  const png = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  await writeFile(path.join(directory, `${slug}-${width}-${theme}.png`), Buffer.from(png.data, "base64"));
}
async function importData(data, name = "paired.json") {
  await evaluate(`(() => {const input=document.querySelector('[data-pe-file]');const transfer=new DataTransfer();transfer.items.add(new File([${JSON.stringify(JSON.stringify(data))}],${JSON.stringify(name)},{type:'application/json'}));input.files=transfer.files;input.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await pause(80);
}

try {
  await send("Page.enable"); await send("Runtime.enable"); await send("Network.enable"); await send("Network.setCacheDisabled", { cacheDisabled: true });
  for (const [width, theme] of [[390, "light"], [768, "light"], [1440, "dark"]]) {
    await send("Emulation.setDeviceMetricsOverride", { width, height: 960, deviceScaleFactor: 1, mobile: false });
    await navigate("context-parallel"); await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)}`);
    check(`CP baseline ${width}`, Number(await value("[data-cp-gradient]")) < 1e-10);
    for (const fault of ["local-softmax", "local-position", "drop-remote", "local-mean"]) {
      await change("[data-cp-fault]", fault); check(`CP ${fault} ${width}`, Number(await value("[data-cp-gradient]")) > 1e-3);
    }
    await change("[data-cp-split]", 5); check(`CP one rank local-mean ${width}`, Number(await value("[data-cp-gradient]")) < 1e-10);
    await change("[data-cp-fault]", "none"); await change("[data-cp-mask]", "empty-first"); await change("[data-cp-split]", 2);
    check(`CP empty row ${width}`, Number(await value("[data-cp-gradient]")) < 1e-10);
    await evaluate(`while(!document.querySelector('[data-cp-next]').disabled)document.querySelector('[data-cp-next]').click();`);
    check(`CP last event ${width}`, (await value("[data-cp-event]")).includes("反向")); await click("[data-cp-prev]"); await click("[data-cp-reset]");
    await evaluate("document.querySelectorAll('main details').forEach(el=>el.open=true)"); await layout(`CP expanded ${width}`); await screenshot("context-parallel", width, theme, "#cp-lab");

    await navigate("rl-update"); await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)}`);
    check(`RL complete distributions ${width}`, await evaluate("document.querySelectorAll('[data-ru-distributions] tr').length===8 && document.querySelector('[data-ru-inputs]').textContent.includes('referenceWeights')"));
    const before = await value("[data-ru-parameters]"); await click("[data-ru-update]"); await click("[data-ru-update]");
    check(`RL updates ${width}`, (await value("[data-ru-steps]")) === "2" && (await value("[data-ru-parameters]")) !== before);
    const proper = await value("[data-ru-parameters]"); await change("[data-ru-fault]", "refresh-old"); await click("[data-ru-update]"); await click("[data-ru-update]");
    check(`RL old version divergence ${width}`, (await value("[data-ru-parameters]")) !== proper);
    for (const fault of ["old-graph", "include-controls", "wrong-denominator"]) { await change("[data-ru-fault]", fault); await click("[data-ru-update]"); check(`RL ${fault} ${width}`, (await value("[data-ru-steps]")) === "1"); }
    await change("[data-ru-fault]", "none"); await change("[data-ru-limit]", 0);
    check(`RL zero token skip ${width}`, await evaluate("document.querySelector('[data-ru-update]').disabled && document.querySelector('[data-ru-loss]').textContent==='未定义'"));
    await change("[data-ru-limit]", 3); await change("[data-ru-rewards]", "equal"); await change("[data-ru-beta]", 0);
    const unchanged = await value("[data-ru-parameters]"); await click("[data-ru-update]"); check(`RL zero advantage ${width}`, (await value("[data-ru-parameters]")) === unchanged);
    await change("[data-ru-rewards]", "mixed"); await change("[data-ru-beta]", .08); await change("[data-ru-normalization]", "sequence");
    await evaluate("document.querySelectorAll('main details').forEach(el=>el.open=true)"); await layout(`RL expanded ${width}`); await screenshot("rl-update", width, theme, "#ru-lab");

    await navigate("quantized-linear"); await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)}`);
    check(`Quant baseline ${width}`, Number(await value("[data-ql-arithmetic]")) < 1e-10);
    for (const fault of ["zero-point", "scale-axis", "bias-unit"]) { await change("[data-ql-fault]", fault); check(`Quant ${fault} ${width}`, Number(await value("[data-ql-arithmetic]")) > .01); }
    await change("[data-ql-scenario]", "zero"); check(`Quant zero ${width}`, Number(await value("[data-ql-arithmetic]")) === 0);
    await change("[data-ql-fault]", "none"); await change("[data-ql-scenario]", "saturation"); check(`Quant saturation ${width}`, !(await value("[data-ql-saturation]")).startsWith("0 个输入"));
    await change("[data-ql-mode]", "w4a16"); check(`W4A16 controls ${width}`, await evaluate("document.querySelector('[data-ql-fault]').disabled"));
    await change("[data-ql-mode]", "w8a8"); await change("[data-ql-scenario]", "outlier");
    await evaluate("document.querySelectorAll('main details').forEach(el=>el.open=true)"); await layout(`Quant expanded ${width}`); await screenshot("quantized-linear", width, theme, "#ql-lab");

    await navigate("paired-evaluation"); await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)}`);
    const grouped = await value("[data-pe-interval]"); await change("[data-pe-mode]", "row"); check(`Paired grouping ${width}`, grouped !== await value("[data-pe-interval]"));
    await change("[data-pe-preset]", "multi-seed"); const first = await value("[data-pe-delta]"); await change("[data-pe-seed]", "29"); check(`Paired seed separation ${width}`, first !== await value("[data-pe-delta]"));
    await importData({ schema: 1, rows: [{ id: '<img src=x onerror="window.__unsafe=1">', group: "document", old: .4, new: .6 }] }, "local-example.json");
    check(`Paired import escapes text ${width}`, await evaluate("!document.querySelector('[data-pe-results]').hidden && !document.querySelector('[data-pe-rows] img') && !window.__unsafe && document.querySelector('[data-pe-rows]').textContent.includes('<img')"));
    check(`Single unit warning ${width}`, (await value("[data-pe-decision]")).includes("只有一个"));
    await importData({ schema: 1, rows: [] }); check(`Invalid import clears state ${width}`, await evaluate("document.querySelector('[data-pe-results]').hidden && document.querySelector('[data-pe-status]').textContent.includes('读取失败')"));
    await importData(pairedExample("multi-seed")); check(`Import recovers ${width}`, await evaluate("!document.querySelector('[data-pe-results]').hidden && document.querySelectorAll('[data-pe-seed] option').length===3"));
    await evaluate(`(async()=>{
      const input=document.querySelector('[data-pe-file]');
      const a=new File(['ignored'],'slow.json',{type:'application/json'});a.text=()=>new Promise(resolve=>setTimeout(()=>resolve(JSON.stringify({schema:1,rows:[{id:'slow',group:'s',old:0,new:1}]})),120));
      const one=new DataTransfer();one.items.add(a);input.files=one.files;input.dispatchEvent(new Event('change',{bubbles:true}));
      const b=new File([JSON.stringify({schema:1,rows:[{id:'fast',group:'f',old:.5,new:.6}]})],'fast.json',{type:'application/json'});const two=new DataTransfer();two.items.add(b);input.files=two.files;input.dispatchEvent(new Event('change',{bubbles:true}));
      await new Promise(resolve=>setTimeout(resolve,170));return true;
    })()`);
    check(`Import race keeps newest file ${width}`, (await value("[data-pe-status]")).includes("fast.json") && (await value("[data-pe-rows]")).includes("fast"));
    await change("[data-pe-preset]", "correlated"); await change("[data-pe-mode]", "group"); await change("[data-pe-candidates]", 50);
    check(`Selection simulation ${width}`, (await value("[data-pe-selection]")).includes("300"));
    await evaluate("document.querySelectorAll('main details').forEach(el=>el.open=true)"); await layout(`Paired expanded ${width}`); await screenshot("paired-evaluation", width, theme, "#pe-lab");

    await navigate("ann-index"); await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)}`);
    check(`ANN approximate baseline ${width}`, (await value("[data-ai-recall]")) === "33.3%"); await change("[data-ai-nprobe]", 3);
    check(`ANN all lists ${width}`, (await value("[data-ai-recall]")) === "100.0%");
    await change("[data-ai-version]", "stale"); check(`ANN stale contract ${width}`, (await value("[data-ai-contract]")).includes("失败") && (await value("[data-ai-oracle]")) !== "100.0%" && (await value("[data-ai-recall]")) === "100.0%");
    await change("[data-ai-version]", "rebuilt"); check(`ANN rebuilt ${width}`, (await value("[data-ai-oracle]")) === "100.0%");
    await change("[data-ai-filter]", "empty"); check(`ANN empty ${width}`, (await value("[data-ai-recall]")) === "未定义");
    await change("[data-ai-filter]", "A"); await change("[data-ai-query]", "left"); check(`ANN unlabeled query ${width}`, (await value("[data-ai-task]")) === "未定义");
    await change("[data-ai-filter]", "all"); await change("[data-ai-query]", "center"); await change("[data-ai-version]", "aligned"); await change("[data-ai-nprobe]", 1);
    await evaluate("document.querySelectorAll('main details').forEach(el=>el.open=true)"); await layout(`ANN expanded ${width}`); await screenshot("ann-index", width, theme, "#ai-lab");
  }
  check("No page exceptions, warnings or resource failures", !errors.length, errors);
  await writeFile(path.join(directory, "interaction-results.json"), JSON.stringify({ passed: results.every(row => row.ok), checks: results.length, results }, null, 2) + "\n");
  const failures = results.filter(row => !row.ok); console.log(JSON.stringify({ checks: results.length, failures, artifacts: directory }, null, 2));
  if (failures.length) process.exitCode = 1;
} finally {
  socket.close(); await fetch(`http://127.0.0.1:${port}/json/close/${target.id}`);
}
