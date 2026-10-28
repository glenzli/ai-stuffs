import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { pages } from "../src/pages.mjs";

const slugs = ["learning-run", "training-failures", "initialization-width", "batch-noise", "chunked-cross-entropy"];
const port = process.env.CDP_PORT || "9222";
const baseUrl = process.env.BASE_URL || "http://127.0.0.1:4173";
const runId = `ui-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}`;
const directory = path.resolve(process.env.QA_DIR || path.join("artifacts/learning-dynamics-qa", runId));
const targeted = process.argv.includes("--targeted");
const timeoutMs = 12000;
const results = [], errors = [], screenshots = [], reports = [];
let target, socket, active = "initialization", sequence = 0;
const pending = new Map();
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const check = (name, ok, detail) => results.push({ name, ok: Boolean(ok), ...(detail === undefined ? {} : { detail }) });
const clone = value => structuredClone(value);
const sha256 = value => createHash("sha256").update(value).digest("hex");

async function request(url, options = {}) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(timeoutMs) });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
  return response;
}

function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout ${method}: ${active}`)); }, timeoutMs);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression) {
  const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  return result.result.value;
}

async function waitFor(expression, label = expression, timeout = timeoutMs) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await evaluate(expression)) return;
    await pause(60);
  }
  throw new Error(`Timed out waiting for ${label}: ${active}`);
}

async function navigate(slug, theme) {
  active = slug;
  const pathname = `/${slug}.html`;
  const result = await send("Page.navigate", { url: `${baseUrl}${pathname}` });
  if (result.errorText) throw new Error(result.errorText);
  await waitFor(`location.pathname === ${JSON.stringify(pathname)} && document.readyState === 'complete' && document.documentElement.dataset.wikiReady === 'true'`, "page initialization");
  await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)};document.fonts.ready.then(()=>true)`);
}

const text = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).textContent`);
async function click(selector) { await evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`); }
async function toggle(selector, checked) {
  await evaluate(`(() => {const el=document.querySelector(${JSON.stringify(selector)});el.checked=${JSON.stringify(checked)};el.dispatchEvent(new Event('change',{bubbles:true}));})()`);
}
async function change(selector, choice) {
  await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) throw new Error('Missing control: '+${JSON.stringify(selector)});
    el.value=${JSON.stringify(String(choice))};
    if(el.value!==${JSON.stringify(String(choice))})throw new Error('Unsupported control value: '+${JSON.stringify(String(choice))});
    el.dispatchEvent(new Event('input',{bubbles:true}));
    el.dispatchEvent(new Event('change',{bubbles:true}));
  })()`);
}

async function key(key, code, virtualKey) {
  await send("Page.bringToFront");
  await send("Emulation.setFocusEmulationEnabled", { enabled:true });
  const event = { key, code, windowsVirtualKeyCode: virtualKey, nativeVirtualKeyCode: virtualKey };
  await send("Input.dispatchKeyEvent", { type: "keyDown", ...event });
  await send("Input.dispatchKeyEvent", { type: "keyUp", ...event });
}

async function layout(name) {
  await evaluate("document.fonts.ready.then(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve(true)))))");
  const state = await evaluate(`(() => {
    const main=document.querySelector('main'), rect=main.getBoundingClientRect();
    const controls=[...main.querySelectorAll('input,select,textarea')];
    const visible=el=>el.getClientRects().length>0;
    const regions=[...main.querySelectorAll('.compare-wrap')].filter(el=>visible(el)&&el.scrollWidth>el.clientWidth+2);
    return { width:innerWidth, documentWidth:document.documentElement.scrollWidth, mainWidth:main.scrollWidth, mainClient:main.clientWidth, right:rect.right,
      unlabeled:controls.filter(el=>!el.labels?.length&&!el.closest('label')&&!el.getAttribute('aria-label')).length,
      unnamedSvg:[...main.querySelectorAll('svg[role=img]')].filter(el=>!el.getAttribute('aria-label')&&!el.querySelector('title')).length,
      badMath:main.querySelectorAll('.katex-error').length, math:main.querySelectorAll('[data-math]').length, mathML:main.querySelectorAll('[data-math] math').length,
      toc:document.querySelectorAll('[data-toc-links] a').length, headings:main.querySelectorAll('h2[id]').length,
      scrollRegions:regions.length, inaccessible:regions.filter(el=>el.tabIndex<0||el.getAttribute('role')!=='region'||!el.getAttribute('aria-label')).map(el=>el.outerHTML.slice(0,160)) };
  })()`);
  check(name, state.documentWidth<=state.width+1 && state.mainWidth<=state.mainClient+2 && state.right<=state.width+1 && !state.unlabeled && !state.unnamedSvg && !state.badMath && state.math===state.mathML && state.toc===state.headings && !state.inaccessible.length, state);
}

async function tableKeyboard(name, scope) {
  const state = await evaluate(`(() => {
    const root=document.querySelector(${JSON.stringify(scope)});
    const el=[...root.querySelectorAll('.compare-wrap')].find(el=>el.getClientRects().length&&el.scrollWidth>el.clientWidth+4);
    if(!el)return null;
    el.dataset.learningQaScroll='true';el.scrollLeft=0;el.scrollIntoView({block:'center',behavior:'instant'});el.focus({preventScroll:true});
    return {label:el.getAttribute('aria-label'),tabIndex:el.tabIndex,role:el.getAttribute('role'),focused:document.activeElement===el,client:el.clientWidth,scroll:el.scrollWidth};
  })()`);
  if (!state) { check(`${name}: table fits current viewport`, true); return; }
  check(`${name}: overflowing table is named and focused`, state.focused && state.tabIndex===0 && state.role==="region" && Boolean(state.label), state);
  await key("ArrowRight", "ArrowRight", 39);
  await pause(180);
  const moved = await evaluate("document.querySelector('[data-learning-qa-scroll]').scrollLeft");
  check(`${name}: keyboard scroll moves table`, moved>0, { scrollLeft:moved });
  await evaluate("(() => {const el=document.querySelector('[data-learning-qa-scroll]');el.scrollLeft=0;delete el.dataset.learningQaScroll;})()");
}

async function screenshot(slug, width, theme, selector, suffix = "experiment") {
  await evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block:'start',behavior:'instant'})`);
  await pause(100);
  const region = await evaluate(`(() => {const el=document.querySelector(${JSON.stringify(selector)}),r=el.getBoundingClientRect();return {selector:${JSON.stringify(selector)},x:r.x,y:r.y,width:r.width,height:r.height,scrollY}})()`);
  const png = await send("Page.captureScreenshot", { format:"png", captureBeyondViewport:false });
  const filename=`${slug}-${width}-${theme}-${suffix}.png`, bytes=Buffer.from(png.data,"base64");
  await writeFile(path.join(directory,filename),bytes,{flag:"wx"});
  screenshots.push({file:filename,slug,width,theme,region,sha256:sha256(bytes),bytes:bytes.length});
}

async function importText(selector, source, name = "report.json") {
  await evaluate(`(() => {
    const input=document.querySelector(${JSON.stringify(selector)}),transfer=new DataTransfer();
    transfer.items.add(new File([${JSON.stringify(source)}],${JSON.stringify(name)},{type:'application/json'}));
    input.files=transfer.files;input.dispatchEvent(new Event('change',{bubbles:true}));
  })()`);
}

async function importReport(selector, report, name) { await importText(selector,JSON.stringify(report),name); }

async function importRace(selector, slowReport, fastReport) {
  await evaluate(`(async () => {
    const input=document.querySelector(${JSON.stringify(selector)});
    const slow=new File(['delayed test read'],'slow-report.json',{type:'application/json'});
    slow.text=()=>new Promise(resolve=>setTimeout(()=>resolve(${JSON.stringify(JSON.stringify(slowReport))}),220));
    const first=new DataTransfer();first.items.add(slow);input.files=first.files;input.dispatchEvent(new Event('change',{bubbles:true}));
    const fast=new File([${JSON.stringify(JSON.stringify(fastReport))}],'newest-report.json',{type:'application/json'});
    const last=new DataTransfer();last.items.add(fast);input.files=last.files;input.dispatchEvent(new Event('change',{bubbles:true}));
    await new Promise(resolve=>setTimeout(resolve,320));return true;
  })()`);
}

async function readActualReport(file, label) {
  const bytes=await readFile(file),report=JSON.parse(bytes);
  reports.push({label,file:path.resolve(file),sha256:sha256(bytes),bytes:bytes.length,schema:report.schema,status:report.status});
  return report;
}

async function checkImports({prefix,width,report,unsafe,invalid,clearFields,inspectUnsafe}) {
  const input=`[data-${prefix}-file]`,output=`[data-${prefix}-output]`,status=`[data-${prefix}-status]`;
  const name=`verified-${prefix}-report.json`;
  await importReport(input,report,name);
  await waitFor(`!document.querySelector(${JSON.stringify(output)}).hidden&&document.querySelector(${JSON.stringify(status)}).textContent.includes(${JSON.stringify(name)})`,`${prefix} actual full report import`);
  check(`${prefix} imports actual saved report ${width}`,true,{schema:report.schema,bytes:Buffer.byteLength(JSON.stringify(report))});
  await tableKeyboard(`${prefix} imported dynamic wide table ${width}`,output);
  await importReport(input,unsafe,`escaped-${prefix}.json`);
  await waitFor(`document.querySelector(${JSON.stringify(status)}).textContent.includes('escaped-${prefix}.json')`,`${prefix} escaped report`);
  await inspectUnsafe();
  const clearExpression=`document.querySelector(${JSON.stringify(output)}).hidden&&${JSON.stringify(clearFields)}.every(selector=>!document.querySelector(selector).textContent)&&!window.__learningUnsafe`;
  for (const [filename,source] of [["empty.json",""],["empty-object.json","{}"],["invalid-json.json","{broken"],["invalid-data.json",JSON.stringify(invalid)]]) {
    await importText(input,source,filename);
    await waitFor(`document.querySelector(${JSON.stringify(status)}).textContent.includes('无法读取')`,`${prefix} rejects ${filename}`);
    check(`${prefix} rejects ${filename} and clears prior report ${width}`,await evaluate(clearExpression));
  }
  await importRace(input,report,unsafe);
  check(`${prefix} newest valid import wins race ${width}`,(await text(status)).includes("newest-report.json")&&await evaluate(`!document.querySelector(${JSON.stringify(output)}).hidden`));
  await inspectUnsafe();
  await importRace(input,report,{});
  check(`${prefix} newest invalid import cancels older valid read ${width}`,(await text(status)).includes("无法读取")&&await evaluate(clearExpression));
  await click(`[data-${prefix}-default]`);
  check(`${prefix} default snapshot recovers ${width}`,await evaluate(`!document.querySelector(${JSON.stringify(output)}).hidden&&document.querySelector(${JSON.stringify(status)}).textContent.includes('默认')`));
}

async function testLearningRun(width,theme,report,original) {
  await navigate("learning-run",theme);
  check(`Learning default shows accepted correctness separately ${width}`,(await text("[data-lr-computation]")).includes("计算核验通过")&&(await text("[data-lr-fit]")).includes("2/3")&&(await text("[data-lr-cache]")).includes("诊断后"));
  await change("[data-lr-seed]",47);
  const selected=report.runs.find(run=>run.seed===47);
  check(`Learning seed47 actual capability limits ${width}`,(await text("[data-lr-outcome]")).includes("93.33%")&&(await text("[data-lr-outcome]")).includes("91.37%")&&(await text("[data-lr-outcome]")).includes("0.00%"));
  for (const split of ["train","validation","test","length"]) {
    await change("[data-lr-split]",split);await change("[data-lr-phase]","learned");
    check(`Learning ${split} EM follows actual seed47 ${width}`,(await text("[data-lr-em]"))===`${(selected.learned[split].metrics.exact_match*100).toFixed(2)}%`);
  }
  check(`Learning length failures show early EOS ${width}`,(await text("[data-lr-termination]")).includes("143 提前 EOS"));
  await change("[data-lr-phase]","oracle");
  check(`Learning rule baseline has no teacher-forced loss ${width}`,(await text("[data-lr-ce]"))==="不适用"&&(await text("[data-lr-em]"))==="100.00%");
  await toggle("[data-lr-errors]",true);
  check(`Learning empty errors filter is explicit ${width}`,await evaluate("!document.querySelector('[data-lr-rows]').textContent&&document.querySelector('[data-lr-page]').textContent.includes('当前筛选无题目')&&document.querySelector('[data-lr-table-next]').disabled"));
  await toggle("[data-lr-errors]",false);await change("[data-lr-phase]","initial");
  check(`Learning initial model differs from oracle ${width}`,(await text("[data-lr-em]"))!=="100.00%");
  for (const metric of ["free_token_accuracy","exact_match","teacher_forced_loss"]) {
    await change("[data-lr-curve-metric]",metric);await change("[data-lr-axis]","supervised_tokens");
    check(`Learning curve ${metric} declares actual units ${width}`,await evaluate(`document.querySelector('[data-lr-chart]').getAttribute('aria-label').includes(${JSON.stringify(metric==="teacher_forced_loss"?"nats":"百分比")})&&document.querySelector('[data-lr-chart]').getAttribute('aria-label').includes('监督 token')`));
  }
  const unsafe=clone(report),markup='<img src=x onerror="window.__learningUnsafe=1">';
  unsafe.provenance.report_path='<svg onload="window.__learningUnsafe=1">';
  unsafe.runs[0].learned.test.predictions[0].id=markup;unsafe.runs[0].learned.test.predictions[0].group_id=markup;
  const invalid=clone(report);invalid.runs[0].learned.test.predictions=[];
  await checkImports({prefix:"lr",width,report,unsafe,invalid,clearFields:["[data-lr-provenance]","[data-lr-rows]","[data-lr-summary-rows]","[data-lr-chart]"],inspectUnsafe:async()=>{
    await change("[data-lr-seed]",report.runs[0].seed);await change("[data-lr-split]","test");await change("[data-lr-phase]","learned");
    check(`Learning imported IDs and provenance are escaped ${width}`,await evaluate("!window.__learningUnsafe&&!document.querySelector('[data-lr-rows] img')&&!document.querySelector('[data-lr-provenance] svg')&&document.querySelector('[data-lr-rows]').textContent.includes('<img')&&document.querySelector('[data-lr-provenance]').textContent.includes('<svg')"));
  }});
  await importReport("[data-lr-file]",original,"original-failed-report.json");
  await waitFor("document.querySelector('[data-lr-status]').textContent.includes('original-failed-report.json')","original failed learning report");
  check(`Learning original numerical failure stays visible ${width}`,(await text("[data-lr-computation]")).includes("未通过")&&(await text("[data-lr-fit]")).includes("2/3"));
  await importReport("[data-lr-file]",report,"complete-learning-report.json");
  await waitFor("document.querySelector('[data-lr-status]').textContent.includes('complete-learning-report.json')","complete learning report");
  await change("[data-lr-split]","test");await change("[data-lr-phase]","learned");await toggle("[data-lr-errors]",false);
  const firstPage=await text("[data-lr-rows]");await click("[data-lr-table-next]");
  check(`Learning full report pagination uses next saved rows ${width}`,firstPage!==await text("[data-lr-rows]")&&(await text("[data-lr-page]")).includes("第 2/"));
  await click("[data-lr-table-prev]");
  check(`Learning paired export enabled only for full data ${width}`,await evaluate("!document.querySelector('[data-lr-paired]').disabled"));
  await change("[data-lr-seed]",47);await change("[data-lr-split]","length");await toggle("[data-lr-errors]",true);
  await evaluate("document.querySelectorAll('main details').forEach(el=>el.open=true)");
  await layout(`Learning expanded layout ${width}`);await tableKeyboard(`Learning final predictions ${width}`,"[data-lr-output]");
  await screenshot("learning-run",width,theme,"[data-lr-chart]","curve");
  await screenshot("learning-run",width,theme,"[data-lr-rows]","predictions");
}

async function testInitializationWidth(width,theme,report) {
  await navigate("initialization-width",theme);
  check(`Width default snapshot has actual provenance ${width}`,await evaluate("!document.querySelector('[data-iw-output]').hidden&&document.querySelector('[data-iw-provenance]').textContent.includes('PyTorch')&&document.querySelector('[data-iw-provenance]').textContent.includes('SHA-256')"));
  check(`Width held128 omitted from selection grid ${width}`,await evaluate("[...document.querySelectorAll('[data-iw-sweep-rows] tr')].every(row=>!row.children[0].textContent.startsWith('128'))&&document.querySelector('[data-iw-selection]').textContent.includes('128')"));
  await change("[data-iw-metric]","displacement_rms");await change("[data-iw-step]",0);
  check(`Width zero-update displacement is zero ${width}`,await evaluate("[...document.querySelectorAll('[data-iw-coordinates] tr')].every(row=>Number(row.children[1].textContent)===0)"));
  await change("[data-iw-step]",1);
  check(`Width actual updates change coordinates ${width}`,await evaluate("[...document.querySelectorAll('[data-iw-coordinates] tr')].some(row=>Number(row.children[1].textContent)>0)"));
  for (const layer of ["input_pre","hidden_relu","readout_logits"]) {
    await change("[data-iw-layer]",layer);await change("[data-iw-metric]","rms");await change("[data-iw-seed]",47);
    check(`Width ${layer} single seed rows ${width}`,await evaluate("document.querySelectorAll('[data-iw-coordinates] tr').length===8&&[...document.querySelectorAll('[data-iw-coordinates] tr')].every(row=>row.children[3].textContent==='1')"));
  }
  for (const parameterization of ["sp","mup"]) {
    await change("[data-iw-sweep-param]",parameterization);await change("[data-iw-ledger-param]",parameterization);await change("[data-iw-ledger-width]",report.width.held_width);
    const expected=report.width.selection.find(row=>row.parameterization===parameterization).selected_lr/(parameterization==="mup"?report.width.held_width/report.width.base_width:1);
    const group=await evaluate("[...document.querySelectorAll('[data-iw-ledger] tr')].find(row=>row.children[0].textContent==='hidden.weight').children[4].textContent");
    check(`Width ${parameterization} ledger actual matrix LR ${width}`,Math.abs(Number(group.split('/').at(-1).trim())-expected)<5e-5,{observed:group,expected});
  }
  for (const norm of ["post","none","pre"]) {
    await change("[data-iw-depth]",report.depth.depths.at(-1));await change("[data-iw-norm]",norm);await change("[data-iw-residual]","inverse-sqrt-depth");await change("[data-iw-depth-step]",4);await change("[data-iw-depth-seed]",29);
    check(`Width depth ${norm} shows saved layers and scope ${width}`,await evaluate(`document.querySelectorAll('[data-iw-depth-rows] tr').length===${report.depth.depths.at(-1)+3}&&document.querySelector('[data-iw-depth-note]').textContent.includes('宽度 32')&&document.querySelector('[data-iw-depth-note]').textContent.includes('更新 4')`));
  }
  const unsafe=clone(report);unsafe.provenance.run_id='<svg onload="window.__learningUnsafe=1">';
  const invalid=clone(report);invalid.width.runs=[];
  await checkImports({prefix:"iw",width,report,unsafe,invalid,clearFields:["[data-iw-provenance]","[data-iw-coordinates]","[data-iw-ledger]","[data-iw-chart]","[data-iw-depth-chart]"],inspectUnsafe:async()=>{
    check(`Width imported provenance text escapes markup ${width}`,await evaluate("!window.__learningUnsafe&&!document.querySelector('[data-iw-provenance] svg')&&document.querySelector('[data-iw-provenance]').textContent.includes('<svg')"));
  }});
  await change("[data-iw-metric]","rms");await change("[data-iw-step]",report.width.steps);await change("[data-iw-seed]","all");
  await evaluate("document.querySelectorAll('main details').forEach(el=>el.open=true)");
  await layout(`Width expanded layout ${width}`);await tableKeyboard(`Width final ledger ${width}`,"[aria-labelledby=iw-ledger]");
  await screenshot("initialization-width",width,theme,"[aria-labelledby=iw-lab]");
  await screenshot("initialization-width",width,theme,"[aria-labelledby=iw-ledger]","ledger");
}

async function testBatchNoise(width,theme,report) {
  await navigate("batch-noise",theme);
  check(`Batch default uses saved CPU provenance ${width}`,await evaluate("!document.querySelector('[data-bn-output]').hidden&&document.querySelector('[data-bn-provenance]').textContent.includes('PyTorch')"));
  await change("[data-bn-population]","measured");await change("[data-bn-batch]",2);
  for (const scheme of ["with-replacement","without-replacement","repeated","correlated"]) {
    await change("[data-bn-scheme]",scheme);
    check(`Batch ${scheme} exact variance agrees with formula ${width}`,Number(await text("[data-bn-noise-error]"))<1e-9);
    if(scheme==="repeated")check(`Batch repeated draws keep population variance ${width}`,await text("[data-bn-variance]")===await text("[data-bn-trace]"));
  }
  await change("[data-bn-scheme]","without-replacement");await change("[data-bn-batch]",6);
  check(`Batch full population without replacement has zero variance ${width}`,Number(await text("[data-bn-variance]"))<1e-9&&await evaluate("document.querySelectorAll('[data-bn-draws] tr').length===1"));
  await change("[data-bn-population]","centered");
  check(`Batch zero mean leaves ratio and direction undefined ${width}`,(await text("[data-bn-ratio]")).includes("未定义")&&await evaluate("[...document.querySelectorAll('[data-bn-draws] tr')].every(row=>row.children[3].textContent==='未定义')"));
  await change("[data-bn-batch]",2);await change("[data-bn-population]","duplicate");await change("[data-bn-scheme]","with-replacement");
  check(`Batch duplicate population is separately measured ${width}`,(await text("[data-bn-noise-note]")).includes("总体项重复")&&Number(await text("[data-bn-noise-error]"))<1e-9);
  for (const lr of report.batch.lr_grid) {
    await change("[data-bn-lr]",lr);await change("[data-bn-seed]",47);await change("[data-bn-split]","test");
    check(`Batch LR ${lr} preserves equal actual token budgets ${width}`,(await text("[data-bn-budget]")).includes("32,768")&&(await text("[data-bn-budget]")).includes("16,384")&&await evaluate(`document.querySelectorAll('[data-bn-training-rows] tr').length===${report.batch.batch_sizes.length}`));
  }
  await change("[data-bn-curve-split]","train");await change("[data-bn-split]","length");await change("[data-bn-pred-batch]",report.batch.batch_sizes.at(-1));await change("[data-bn-pred-seed]",29);
  check(`Batch prediction controls select saved OOD run ${width}`,(await text("[data-bn-prediction-note]")).includes("seed=29")&&(await text("[data-bn-prediction-note]")).includes("length")&&await evaluate("document.querySelectorAll('[data-bn-predictions] tr').length>0"));
  const unsafe=clone(report);unsafe.provenance.run_id='<svg onload="window.__learningUnsafe=1">';unsafe.noise.records[0].id='<img src=x onerror="window.__learningUnsafe=1">';
  const invalid=clone(report);invalid.batch.runs=[];
  await checkImports({prefix:"bn",width,report,unsafe,invalid,clearFields:["[data-bn-provenance]","[data-bn-draws]","[data-bn-training-rows]","[data-bn-predictions]","[data-bn-noise-chart]","[data-bn-training-chart]"],inspectUnsafe:async()=>{
    check(`Batch imported IDs and provenance escape markup ${width}`,await evaluate("!window.__learningUnsafe&&!document.querySelector('[data-bn-provenance] svg')&&!document.querySelector('[data-bn-records] img')&&document.querySelector('[data-bn-provenance]').textContent.includes('<svg')&&document.querySelector('[data-bn-records]').textContent.includes('<img')"));
  }});
  await change("[data-bn-population]","measured");await change("[data-bn-scheme]","without-replacement");await change("[data-bn-batch]",2);await change("[data-bn-seed]","all");
  await evaluate("document.querySelectorAll('main details').forEach(el=>el.open=true)");
  await layout(`Batch expanded layout ${width}`);await tableKeyboard(`Batch final training table ${width}`,"[aria-labelledby=bn-training]");
  await screenshot("batch-noise",width,theme,"[aria-labelledby=bn-lab]");
  await screenshot("batch-noise",width,theme,"[aria-labelledby=bn-training]","training");
}

async function testTrainingFailures(width,theme,report) {
  await navigate("training-failures",theme);
  check(`Failures default contains actual saved provenance ${width}`,await evaluate(`!document.querySelector('[data-tf-output]').hidden&&document.querySelector('[data-tf-provenance]').textContent.includes('SHA-256')&&document.querySelector('[data-tf-verification]').textContent.includes('${report.checks.length} 项')`));
  await change("[data-tf-case]","overflow");await change("[data-tf-variant]","faulty");await change("[data-tf-selection]","target");
  await click("[data-tf-first]");
  check(`Failures jumps to actual first nonfinite exp ${width}`,(await text("[data-tf-event-name]")).includes("loss.exp")&&(await text("[data-tf-loss]"))==="NaN"&&(await text("[data-tf-case-description]")).includes("跳过"));
  await click("[data-tf-next]");await click("[data-tf-prev]");await click("[data-tf-reset]");
  check(`Failures stepping resets to first recorded input ${width}`,await evaluate("document.querySelector('[data-tf-prev]').disabled&&document.querySelector('[data-tf-event-name]').textContent.includes('input')"));
  for (const caseId of ["overflow","high-lr","labels"]) {
    await change("[data-tf-case]",caseId);await change("[data-tf-variant]","faulty");await change("[data-tf-selection]","target");
    const run=report.cases.find(row=>row.id===caseId).runs["target-faulty"];
    if(caseId==="high-lr")check(`Failures high LR remains finite with abnormal update ${width}`,(await text("[data-tf-update]")).includes("超阈值")&&(await text("[data-tf-case-description]")).includes("全部有限")&&Number(await text("[data-tf-next-loss]"))>0);
    if(caseId==="labels")check(`Failures legal wrong labels remain finite ${width}`,(await text("[data-tf-batch-summary]")).includes(`${run.labels_changed} 处标签差异`)&&(await text("[data-tf-gradient]"))!=="非有限／未定义"&&(await text("[data-tf-diagnosis]")).includes("batch.labels"));
    await change("[data-tf-variant]","repaired");
    check(`Failures ${caseId} repair matches saved baseline ${width}`,(await text("[data-tf-verification]")).includes("全部张量相同")&&(await text("[data-tf-verification]")).includes("最终状态相同"));
    await change("[data-tf-variant]","baseline");await change("[data-tf-variant]","faulty");await change("[data-tf-selection]","control");
    check(`Failures ${caseId} normal control is finite and equal ${width}`,(await text("[data-tf-diagnosis]")).includes("未观察到非有限")&&(await text("[data-tf-verification]")).includes("正常 batch 对照通过"));
  }
  await testFailurePhases(width,report);
  await change("[data-tf-phase]","all");await change("[data-tf-selection]","target");
  const unsafe=clone(report);unsafe.provenance.report_path='<svg onload="window.__learningUnsafe=1">';
  unsafe.cases[0].runs["target-faulty"].batch.record_ids[0]='<img src=x onerror="window.__learningUnsafe=1">';
  const invalid=clone(report);invalid.cases=[];
  await checkImports({prefix:"tf",width,report,unsafe,invalid,clearFields:["[data-tf-provenance]","[data-tf-event-rows]","[data-tf-label-rows]","[data-tf-chart]","[data-tf-proof]"],inspectUnsafe:async()=>{
    await change("[data-tf-case]","overflow");await change("[data-tf-variant]","faulty");await change("[data-tf-selection]","target");
    check(`Failures imported IDs and provenance escape markup ${width}`,await evaluate("!window.__learningUnsafe&&!document.querySelector('[data-tf-provenance] svg')&&!document.querySelector('[data-tf-label-rows] img')&&document.querySelector('[data-tf-provenance]').textContent.includes('<svg')&&document.querySelector('[data-tf-label-rows]').textContent.includes('<img')"));
  }});
  await change("[data-tf-case]","high-lr");await change("[data-tf-variant]","faulty");await change("[data-tf-selection]","target");await change("[data-tf-phase]","next-forward");
  await evaluate("document.querySelectorAll('main details').forEach(el=>el.open=true)");
  await layout(`Failures expanded layout ${width}`);await tableKeyboard(`Failures final dynamic event table ${width}`,"[data-tf-output]");
  await screenshot("training-failures",width,theme,"[data-tf-output]");
}

async function testFailurePhases(width,report) {
  await importReport("[data-tf-file]",report,"complete-failure-events.json");
  await waitFor("document.querySelector('[data-tf-status]').textContent.includes('complete-failure-events.json')","full failure event report");
  await change("[data-tf-case]","high-lr");
  await change("[data-tf-selection]","target");
  await change("[data-tf-variant]","faulty");
  const events=report.cases.find(row=>row.id==="high-lr").runs["target-faulty"].events;
  for(const phase of ["step","post-update","next-forward"]){
    await change("[data-tf-phase]",phase);
    const expected=events.filter(event=>event.phase===phase);
    const actual=await evaluate("[...document.querySelectorAll('[data-tf-event-rows] tr')].map(row=>({index:Number(row.children[0].textContent),label:row.children[1].textContent}))");
    const label={step:"本步", "post-update":"更新后", "next-forward":"下一前向"}[phase];
    check(`Failures ${phase} filter uses recorded event phases ${width}`,expected.length>0&&actual.length===expected.length&&actual.every((row,index)=>row.index===expected[index].index+1&&row.label===`${label} / ${expected[index].name}`),{case:"high-lr",selection:"target",variant:"faulty",expected:expected.length,actual:actual.length});
  }
}

async function retestFailures(width,theme,report) {
  await navigate("training-failures",theme);
  await testFailurePhases(width,report);
}

async function retestBatch390(theme) {
  await navigate("batch-noise",theme);
  await change("[data-bn-population]","measured");
  await change("[data-bn-scheme]","without-replacement");
  await change("[data-bn-batch]",2);
  await change("[data-bn-seed]","all");
  await evaluate("document.querySelectorAll('main details').forEach(el=>el.open=true)");
  await layout("Batch expanded layout 390");
  const styling=await evaluate(`(() => {
    const grids=[...document.querySelectorAll('main .dynamics-controls')].map(el=>({display:getComputedStyle(el).display,columns:getComputedStyle(el).gridTemplateColumns}));
    const provenance=getComputedStyle(document.querySelector('[data-bn-provenance]')).overflowWrap;
    const charts=[...document.querySelectorAll('main .dynamics-chart')].map(el=>({width:el.getBoundingClientRect().width,client:el.parentElement.clientWidth}));
    const colors=[...document.querySelectorAll('main .dynamics-teal,main .dynamics-blue,main .dynamics-coral')].map(el=>({color:getComputedStyle(el).color,stroke:getComputedStyle(el).stroke}));
    return {grids,provenance,charts,colors};
  })()`);
  check("Batch shared chart/control/provenance styles load 390",styling.grids.length>0&&styling.grids.every(grid=>grid.display==="grid"&&!grid.columns.includes(" "))&&styling.provenance==="anywhere"&&styling.charts.every(chart=>chart.width<=chart.client+1)&&styling.colors.length>0&&new Set(styling.colors.map(row=>row.color)).size>1,styling);
  await tableKeyboard("Batch final training table 390","[aria-labelledby=bn-training]");
  await screenshot("batch-noise",390,theme,"[aria-labelledby=bn-lab]");
  await screenshot("batch-noise",390,theme,"[aria-labelledby=bn-training]","training");
}

async function searchAndGuide(width, theme) {
  await navigate("training-engineer-guide",theme);
  const route=await evaluate("[...document.querySelectorAll('[aria-labelledby=learning-dynamics-route] a')].map(el=>el.getAttribute('href'))");
  check(`Guide links all five topics ${width}`, JSON.stringify(route)===JSON.stringify(slugs.map(slug=>`${slug}.html`)),route);
  await click("[data-open-search]");
  check(`Global search starts with 61 topics ${width}`,await evaluate("document.querySelector('[data-search-dialog]').open&&document.querySelectorAll('[data-global-search-results] a').length===61"));
  for (const [query,slug] of [["自由生成","learning-run"],["确定性回放","training-failures"],["μP 参数组","initialization-width"],["总体方差","batch-noise"],["分块 交叉熵","chunked-cross-entropy"]]) {
    await change("[data-global-search]",query);
    check(`Search ${query} routes to ${slug} ${width}`,await evaluate(`Boolean(document.querySelector('[data-global-search-results] a[href="${slug}.html"]'))`));
  }
  await change("[data-global-search]","<img src=x onerror=window.__learningUnsafe=1>");
  check(`Unmatched search is safe ${width}`,await evaluate("!window.__learningUnsafe&&!document.querySelector('[data-global-search-results] img')&&document.querySelectorAll('[data-global-search-results] a').length===0"));
  await key("Escape","Escape",27);
  await waitFor("!document.querySelector('[data-search-dialog]').open&&document.activeElement===document.querySelector('[data-open-search]')","search closes and returns focus");
  check(`Search Escape restores opener focus ${width}`,await evaluate("!document.querySelector('[data-search-dialog]').open&&document.activeElement===document.querySelector('[data-open-search]')"));
}

async function testChunkedCE(width,theme,report) {
  await navigate("chunked-cross-entropy",theme);
  for(const name of ["loss","dh","dw","db"])check(`CE baseline ${name} ${width}`,Number(await text(`[data-ce-${name}-error]`))<1e-10);
  const baselineLoss=await text("[data-ce-loss]");
  await change("[data-ce-token]",1);await change("[data-ce-vocab]",1);
  check(`CE one-element tiles keep loss ${width}`,(await text("[data-ce-loss]"))===baselineLoss);
  await change("[data-ce-token]",2);await change("[data-ce-vocab]",3);
  for(const target of [0,3,6]) {await change("[data-ce-target]",target);check(`CE target block ${target} ${width}`,Number(await text("[data-ce-loss-error]"))<1e-10&&Number(await text("[data-ce-dh-error]"))<1e-10);}
  for (const shift of [10000,-10000,1000000,-1000000]) {
    await change("[data-ce-shift]",shift);
    check(`CE score shift ${shift} stays stable ${width}`,Number(await text("[data-ce-loss-error]"))<1e-9&&Number(await text("[data-ce-dh-error]"))<1e-9);
  }
  for(const fault of ["local-softmax","overwrite-dh"]) {await change("[data-ce-fault]",fault);check(`CE ${fault} changes gradient but keeps forward ${width}`,Number(await text("[data-ce-dh-error]"))>1e-3&&Number(await text("[data-ce-loss-error]"))<1e-9);}
  await change("[data-ce-upstream]",0);
  check(`CE zero upstream suppresses all gradient contributions ${width}`,Number(await text("[data-ce-dh-error]"))===0&&Number(await text("[data-ce-dw-error]"))===0&&await evaluate("[...document.querySelectorAll('[data-ce-gradient-rows] tr')].every(row=>[row.children[2],row.children[3]].every(cell=>{const numbers=cell.textContent.match(/[-+]?(?:\\d*\\.)?\\d+(?:[eE][-+]?\\d+)?/g);return numbers?.length>0&&numbers.every(value=>Number(value)===0)}))"));
  await change("[data-ce-upstream]",1);await change("[data-ce-fault]","none");await change("[data-ce-shift]",0);
  await change("[data-ce-bias]","off");check(`CE optional bias ${width}`,(await text("[data-ce-db-error]"))==="无 bias");await change("[data-ce-bias]","on");
  for(const mask of ["empty","ignore-all"]) {
    await change("[data-ce-mask]",mask);
    check(`CE ${mask} skips undefined mean ${width}`,await evaluate("document.querySelector('[data-ce-loss]').textContent.includes('未定义')&&document.querySelector('[data-ce-next]').disabled&&document.querySelector('[data-ce-step]').textContent==='没有块事件'&&document.querySelector('[data-ce-count]').textContent.startsWith('0 /')"));
  }
  await change("[data-ce-mask]","mixed");await change("[data-ce-mask]","all");
  await evaluate("for(let count=0;count<150&&!document.querySelector('[data-ce-next]').disabled;count++)document.querySelector('[data-ce-next]').click()");
  check(`CE completes backward tail event ${width}`,(await text("[data-ce-event]")).includes("反向")&&await evaluate("document.querySelector('[data-ce-next]').disabled"));
  await click("[data-ce-prev]");await click("[data-ce-reset]");
  check(`CE reset returns first forward block ${width}`,(await text("[data-ce-event]")).includes("前向")&&await evaluate("document.querySelector('[data-ce-prev]').disabled"));

  const input="[data-ce-report-file]", output="[data-ce-report-output]", status="[data-ce-report-status]";
  check(`CE default snapshot has saved provenance ${width}`,await evaluate("!document.querySelector('[data-ce-report-output]').hidden&&document.querySelector('[data-ce-report-provenance]').textContent.includes('PyTorch')&&document.querySelector('[data-ce-report-provenance]').textContent.includes('artifacts/')"));
  await importReport(input,report,"verified-output-report.json");
  await waitFor(`!document.querySelector(${JSON.stringify(output)}).hidden&&document.querySelector(${JSON.stringify(status)}).textContent.includes('verified-output-report.json')`,"actual CE report import");
  check(`CE imported report shows actual case count ${width}`,(await text("[data-ce-report-checks]")).includes(`${report.correctness.cases} 个配置`));
  await tableKeyboard(`CE imported dynamic table ${width}`,output);
  const shapes=[...new Set(report.measurements.map(row=>row.shape.join("×")))];
  if(shapes.length>1) {await change("[data-ce-report-shape]",shapes.at(-1));check(`CE report shape filters recorded measurements ${width}`,await evaluate(`document.querySelectorAll('[data-ce-report-rows] tr').length===${report.measurements.filter(row=>row.shape.join("×")===shapes.at(-1)).length}`));}
  await change("[data-ce-report-metric]","time");
  check(`CE time chart declares units ${width}`,await evaluate("document.querySelector('[data-ce-measure-chart]').getAttribute('aria-label').includes('毫秒')"));
  await change("[data-ce-report-metric]","storage");
  check(`CE storage chart declares reference semantics ${width}`,await evaluate("document.querySelector('[data-ce-measure-chart]').getAttribute('aria-label').includes('KiB')"));
  const unsafe=clone(report);unsafe.measurements[0].name='<img src=x onerror="window.__learningUnsafe=1">';unsafe.provenance.run_id='<svg onload="window.__learningUnsafe=1">';
  await importReport(input,unsafe,"escaping-report.json");await waitFor(`document.querySelector(${JSON.stringify(status)}).textContent.includes('escaping-report.json')`,"escaped CE report");
  check(`CE report escapes row and provenance text ${width}`,await evaluate("!window.__learningUnsafe&&!document.querySelector('[data-ce-report-rows] img')&&!document.querySelector('[data-ce-report-provenance] svg')&&document.querySelector('[data-ce-report-provenance]').textContent.includes('<svg')"));
  for(const [name,source] of [["empty-file.json",""],["empty-object.json","{}"],["broken-json.json","{broken"],["empty-measurements.json",JSON.stringify({...report,measurements:[]})]]) {
    await importText(input,source,name);await waitFor(`document.querySelector(${JSON.stringify(status)}).textContent.includes('无法读取')`,name);
    check(`CE rejects ${name} and clears prior state ${width}`,await evaluate("document.querySelector('[data-ce-report-output]').hidden&&!document.querySelector('[data-ce-report-rows]').textContent&&!document.querySelector('[data-ce-report-provenance]').textContent&&!document.querySelector('[data-ce-measure-chart]').textContent"));
  }
  const inconsistent=clone(report);inconsistent.measurements[0].wall_ms.median+=1;
  await importReport(input,inconsistent,"inconsistent-timing.json");await waitFor(`document.querySelector(${JSON.stringify(status)}).textContent.includes('无法读取')`,"inconsistent timing summary");
  check(`CE rejects timing summary inconsistent with samples ${width}`,await evaluate("document.querySelector('[data-ce-report-output]').hidden"));
  await importRace(input,report,unsafe);
  check(`CE import race keeps newest file ${width}`,(await text(status)).includes("newest-report.json")&&(await text("[data-ce-report-provenance]")).includes("<svg"));
  await importRace(input,report,{});
  check(`CE invalid newest file cancels older valid import ${width}`,(await text(status)).includes("无法读取")&&await evaluate("document.querySelector('[data-ce-report-output]').hidden"));
  await click("[data-ce-report-default]");
  check(`CE reset recovers verified default ${width}`,await evaluate("!document.querySelector('[data-ce-report-output]').hidden&&document.querySelector('[data-ce-report-status]').textContent.includes('默认快照')"));
  await evaluate("document.querySelectorAll('main details').forEach(el=>el.open=true)");
  await layout(`CE expanded layout ${width}`);await tableKeyboard(`CE experiment table ${width}`,"#ce-lab");
  await screenshot("chunked-cross-entropy",width,theme,"#ce-lab");
  await screenshot("chunked-cross-entropy",width,theme,"#ce-cpu-lab","report");
}

async function runTopics(width,theme,fixtures) {
  await testLearningRun(width,theme,fixtures.learning,fixtures.learningOriginal);
  await testTrainingFailures(width,theme,fixtures.failures);
  await testInitializationWidth(width,theme,fixtures.width);
  await testBatchNoise(width,theme,fixtures.batch);
  await testChunkedCE(width,theme,fixtures.output);
}

try {
  await mkdir(directory,{recursive:true});
  await access(path.join(directory,"interaction-results.json")).then(()=>{throw new Error(`QA directory already contains results: ${directory}`)},error=>{if(error.code!=="ENOENT")throw error});
  for(const slug of slugs) {
    if(!pages.some(page=>page.slug===slug))throw new Error(`Missing page mapping ${slug}`);
    await access(`src/content/${slug}.html`);await access(`src/scripts/features/${slug}.js`);await access(`dist/${slug}.html`);
  }
  const { chunkedCEReportSnapshot }=await import("../src/scripts/features/chunked-cross-entropy-data.js");
  const { learningRunSnapshot }=await import("../src/scripts/features/learning-run-data.js");
  const { trainingFailuresSnapshot }=await import("../src/scripts/features/training-failures-data.js");
  const { initializationWidthData }=await import("../src/scripts/features/initialization-width-data.js");
  const { batchNoiseData }=await import("../src/scripts/features/batch-noise-data.js");
  const outputPath=process.env.OUTPUT_REPORT || chunkedCEReportSnapshot?.provenance?.report_path;
  if(!outputPath)throw new Error("CE report snapshot and final saved report are required for browser QA");
  const learningPath=process.env.LEARNING_REPORT||learningRunSnapshot.provenance.report_path;
  const failuresPath=process.env.FAILURES_REPORT||trainingFailuresSnapshot.provenance.report_path;
  const widthPath=process.env.DYNAMICS_REPORT||initializationWidthData.provenance.source_report||initializationWidthData.provenance.report_path;
  const batchPath=process.env.DYNAMICS_REPORT||batchNoiseData.provenance.source_report||batchNoiseData.provenance.report_path;
  const fixtures={output:await readActualReport(outputPath,"chunked-cross-entropy"),learning:await readActualReport(learningPath,"learning-run"),failures:await readActualReport(failuresPath,"training-failures"),width:await readActualReport(widthPath,"initialization-width")};
  fixtures.batch=batchPath===widthPath?fixtures.width:await readActualReport(batchPath,"batch-noise");
  fixtures.learningOriginal=await readActualReport(fixtures.learning.provenance.original_report.path,"learning-original-failed");
  const {validateLearningReport}=await import("../src/scripts/features/learning-run.js");
  const {validateTrainingFailuresReport}=await import("../src/scripts/features/training-failures.js");
  const {validateInitializationWidthReport}=await import("../src/scripts/features/initialization-width.js");
  const {validateBatchNoiseReport}=await import("../src/scripts/features/batch-noise.js");
  const {validateChunkedCEReport}=await import("../src/scripts/features/chunked-cross-entropy.js");
  validateLearningReport(fixtures.learning);validateLearningReport(fixtures.learningOriginal);
  validateTrainingFailuresReport(fixtures.failures);validateInitializationWidthReport(fixtures.width);validateBatchNoiseReport(fixtures.batch);validateChunkedCEReport(fixtures.output);
  target=await(await request(`http://127.0.0.1:${port}/json/new`,{method:"PUT"})).json();
  socket=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error("CDP connection timeout")),timeoutMs);socket.addEventListener("open",()=>{clearTimeout(timer);resolve()},{once:true});socket.addEventListener("error",()=>{clearTimeout(timer);reject(new Error("CDP connection error"))},{once:true});});
  socket.addEventListener("message",event=>{
    const message=JSON.parse(event.data);
    if(message.method==="Runtime.exceptionThrown")errors.push(`${active}: ${message.params.exceptionDetails.exception?.description||message.params.exceptionDetails.text}`);
    if(message.method==="Runtime.consoleAPICalled"&&["error","warning"].includes(message.params.type))errors.push(`${active}: console ${message.params.type} ${message.params.args.map(arg=>arg.value??arg.description).join(" ")}`);
    if(message.method==="Network.responseReceived"&&message.params.response.status>=400)errors.push(`${active}: HTTP ${message.params.response.status} ${message.params.response.url}`);
    if(message.method==="Network.loadingFailed"&&!message.params.canceled)errors.push(`${active}: ${message.params.errorText}`);
    const item=pending.get(message.id);if(!item)return;clearTimeout(item.timer);pending.delete(message.id);message.error?item.reject(new Error(message.error.message)):item.resolve(message.result);
  });
  await send("Page.enable");await send("Runtime.enable");await send("Network.enable");await send("Network.setCacheDisabled",{cacheDisabled:true});
  for(const [width,theme] of [[390,"light"],[768,"light"],[1440,"dark"]]) {
    await send("Emulation.setDeviceMetricsOverride",{width,height:1000,deviceScaleFactor:1,mobile:false});
    if(targeted) {
      await retestFailures(width,theme,fixtures.failures);
      if(width===390)await retestBatch390(theme);
    } else {
      await searchAndGuide(width,theme);await runTopics(width,theme,fixtures);
    }
  }
  check("No runtime warnings, exceptions or failed resources",!errors.length,errors);
} catch(error) {
  check(`${active}: browser QA completes`,false,{error:error.stack||String(error)});
} finally {
  socket?.close();
  for(const item of pending.values()){clearTimeout(item.timer);item.reject(new Error("CDP session closed"));}pending.clear();
  if(target?.id) {try {await request(`http://127.0.0.1:${port}/json/close/${target.id}`)} catch(error) {check("Closes owned QA tab",false,error.message)}}
  await mkdir(directory,{recursive:true});
  const summary={createdAt:new Date().toISOString(),mode:targeted?"targeted":"full",passed:results.every(row=>row.ok),checks:results.length,expectedPages:61,topics:targeted?["training-failures","batch-noise"]:slugs,viewports:[[390,"light"],[768,"light"],[1440,"dark"]],reports,screenshots,results};
  await writeFile(path.join(directory,"interaction-results.json"),JSON.stringify(summary,null,2)+"\n",{flag:"wx"});
  const failures=results.filter(row=>!row.ok);console.log(JSON.stringify({checks:results.length,failures,screenshots:screenshots.length,artifacts:directory},null,2));
  if(failures.length)process.exitCode=1;
}
