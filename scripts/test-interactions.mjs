import { pages as pageConfig } from "../src/pages.mjs";

const port = process.env.CDP_PORT || "9222";
const baseUrl = process.env.BASE_URL || "http://127.0.0.1:4173";

const target = await (await fetch(`http://127.0.0.1:${port}/json/new`, { method: "PUT" })).json();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

let seq = 0;
const pending = new Map();
const runtimeErrors = [];
socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.method === "Runtime.exceptionThrown") {
    runtimeErrors.push(message.params.exceptionDetails.text);
  }
  if (!message.id || !pending.has(message.id)) return;
  const { resolve, reject } = pending.get(message.id);
  pending.delete(message.id);
  if (message.error) reject(new Error(message.error.message));
  else resolve(message.result);
});

const send = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", {
  width: 390,
  height: 844,
  deviceScaleFactor: 1,
  mobile: false,
});

const wait = (ms = 120) => new Promise((resolve) => setTimeout(resolve, ms));

async function navigate(path) {
  await send("Page.navigate", { url: `${baseUrl}${path}` });
  await wait(350);
}

async function evaluate(expression) {
  const result = await send("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text);
  }
  return result.result.value;
}

const results = [];
function check(name, condition, detail = "") {
  results.push({ name, ok: Boolean(condition), detail });
}

await navigate("/");
await evaluate(`document.querySelector('[data-pipeline-step="5"]').click()`);
const pipelineTitle = await evaluate(`document.querySelector('[data-pipeline-title]').textContent`);
check("首页流水线步进", pipelineTitle.includes("自动微分"), pipelineTitle);
await evaluate(`document.querySelector('[data-scale="sentence"]').click()`);
const scaleTitle = await evaluate(`document.querySelector('[data-scale-title]').textContent`);
check("首页表示尺度切换", scaleTitle.includes("检索单位"), scaleTitle);
await evaluate(`document.querySelector('[data-theme-toggle]').click()`);
const savedTheme = await evaluate(`document.documentElement.dataset.theme`);
check("主题切换", savedTheme === "dark" || savedTheme === "light", savedTheme);

await navigate("/representation.html");
await evaluate(`document.querySelectorAll('[data-era]')[4].click()`);
const eraYear = await evaluate(`document.querySelector('[data-era-year]').textContent`);
check("表示历史切换", eraYear === "2018", eraYear);
await evaluate(`document.querySelector('[data-objective="retrieval"]').click()`);
const geometryTitle = await evaluate(`document.querySelector('[data-objective-title]').textContent`);
check("目标几何切换", geometryTitle.includes("检索"), geometryTitle);
await evaluate(`(() => {
  const select = document.querySelector('[name="embedding-unit"]');
  select.value = "pair";
  select.dispatchEvent(new Event("change", { bubbles: true }));
})()`);
const pickerTitle = await evaluate(`document.querySelector('[data-picker-title]').textContent`);
check("选型建议切换", pickerTitle.includes("bi-encoder"), pickerTitle);

await navigate("/training.html");
const allTrainingModes = [];
for (const mode of ["word2vec", "masked", "causal", "contrastive"]) {
  await evaluate(`document.querySelector('[data-training-mode="${mode}"]').click()`);
  for (let step = 0; step < 6; step += 1) {
    await evaluate(`document.querySelector('[data-training-step="${step}"]').click()`);
  }
  allTrainingModes.push(await evaluate(`({
    mode: document.querySelector('[data-walker-mode]').textContent,
    step: document.querySelector('[data-walker-step]').textContent,
    title: document.querySelector('[data-walker-title]').textContent,
    code: document.querySelector('[data-walker-code]').textContent
  })`));
}
check(
  "四种训练范式六步均可展开",
  allTrainingModes.every((item) => item.step === "06" && item.title && item.code),
  JSON.stringify(allTrainingModes.map((item) => item.mode)),
);
await evaluate(`document.querySelector('[data-training-mode="contrastive"]').click()`);
await evaluate(`document.querySelector('[data-walker-next]').click()`);
await evaluate(`document.querySelector('[data-walker-next]').click()`);
const trainingState = await evaluate(`({
  mode: document.querySelector('[data-walker-mode]').textContent,
  step: document.querySelector('[data-walker-step]').textContent,
  title: document.querySelector('[data-walker-title]').textContent
})`);
check("训练范式与步进", trainingState.mode.includes("CONTRASTIVE") && trainingState.step === "03", JSON.stringify(trainingState));
const initialLoss = await evaluate(`document.querySelector('[data-lab-loss]').textContent`);
await evaluate(`document.querySelector('[data-lab-step]').click()`);
await evaluate(`document.querySelector('[data-lab-step]').click()`);
const labState = await evaluate(`({
  loss: document.querySelector('[data-lab-loss]').textContent,
  updates: document.querySelector('[data-lab-updates]').textContent
})`);
check("二维负采样 SGD", Number(labState.loss) < Number(initialLoss) && labState.updates === "2", JSON.stringify(labState));

await navigate("/posttraining.html");
await evaluate(`document.querySelector('[data-post-stage="preference"]').click()`);
const postState = await evaluate(`({
  title: document.querySelector('[data-post-title]').textContent,
  direct: document.querySelector('[data-impact="ranking"]').className
})`);
check("后训练阶段切换", postState.title.includes("学习偏好") && postState.direct.includes("is-direct"), JSON.stringify(postState));

await navigate("/lab.html");
const initialResults = await evaluate(`document.querySelectorAll('.retrieval-item').length`);
await evaluate(`(() => {
  const input = document.querySelector('[name="chunking"][value="fine"]');
  input.checked = true;
  input.dispatchEvent(new Event("change", { bubbles: true }));
})()`);
await evaluate(`(() => {
  const input = document.querySelector('[name="topk"]');
  input.value = "7";
  input.dispatchEvent(new Event("input", { bubbles: true }));
})()`);
await evaluate(`(() => {
  const input = document.querySelector('[name="rerank"]');
  input.checked = true;
  input.dispatchEvent(new Event("change", { bubbles: true }));
})()`);
const labResults = await evaluate(`({
  count: document.querySelectorAll('.retrieval-item').length,
  topk: document.querySelector('[data-topk-value]').textContent,
  state: document.querySelector('[data-pipeline-state]').textContent,
  summary: document.querySelector('[data-result-summary]').textContent
})`);
check("检索分块与重排", initialResults === 4 && labResults.count === 7 && labResults.state.includes("重排"), JSON.stringify(labResults));

await navigate("/representation-deep.html");
await evaluate(`document.querySelector('[data-pooling="cls"]').click()`);
const poolingTitle = await evaluate(`document.querySelector('[data-pooling-title]').textContent`);
const rawAniso = await evaluate(`document.querySelector('[data-aniso-cos]').textContent`);
await evaluate(`document.querySelector('[data-aniso-mode="center"]').click()`);
const centeredAniso = await evaluate(`document.querySelector('[data-aniso-cos]').textContent`);
await evaluate(`document.querySelector('[data-dim="2"]').click()`);
const dimTwoTop = await evaluate(`document.querySelector('.mrl-row strong').textContent`);
await evaluate(`document.querySelector('[data-dim="4"]').click()`);
const dimFourTop = await evaluate(`document.querySelector('.mrl-row strong').textContent`);
check("表示几何池化实验", poolingTitle.includes("CLS"), poolingTitle);
check("各向异性去均值", Number(rawAniso) > Number(centeredAniso), `${rawAniso} -> ${centeredAniso}`);
check("Matryoshka 截断改变排序", dimTwoTop === "B" && dimFourTop === "A", `2D:${dimTwoTop}, 4D:${dimFourTop}`);

await navigate("/training-deep.html");
await evaluate(`document.querySelector('[data-row-case="target"]').click()`);
await evaluate(`document.querySelector('[data-row-tie]').click()`);
const tiedTarget = await evaluate(`document.querySelector('[data-row="77"] .row-state:nth-child(4)').textContent`);
await evaluate(`document.querySelector('[data-row-case="padding"]').click()`);
const tiedPad = await evaluate(`document.querySelector('[data-row="PAD"] .row-state:nth-child(4)').textContent`);
await evaluate(`document.querySelector('[data-row-case="frozen"]').click()`);
const frozenOutputs = await evaluate(`Array.from(document.querySelectorAll('.row-state:nth-child(5)')).every(e => e.textContent === '绑定冻结' && !e.classList.contains('is-update'))`);
await evaluate(`document.querySelector('[data-loss-mode="sgns"]').click()`);
const sgnsFormula = await evaluate(`document.querySelector('[data-loss-equation="sgns"]:not([hidden]) annotation')?.textContent || ''`);
check("权重绑定改变行梯度", tiedTarget === "输出梯度" && tiedPad === "输出梯度", `${tiedTarget} / PAD:${tiedPad}`);
check("冻结绑定权重不显示更新", frozenOutputs, String(frozenOutputs));
check("SGNS 排版公式切换", sgnsFormula.includes("sigma"), sgnsFormula);
const sgnsTauDisabled = await evaluate(`document.querySelector('[data-math-tau-input]').disabled`);
check("SGNS 禁用无作用的温度", sgnsTauDisabled, String(sgnsTauDisabled));
await evaluate(`document.querySelector('[data-loss-mode="infonce"]').click()`);
const mathState = await evaluate(`({p: Number(document.querySelector('[data-math-p]').textContent), g: Number(document.querySelector('[data-math-grad]').textContent), n: Number(document.querySelector('[data-math-negative-grad]').textContent)})`);
check("InfoNCE 显示原始分数梯度", Math.abs(mathState.g - (mathState.p - 1) / 0.2) < 0.006 && mathState.n > 0, JSON.stringify(mathState));
const activeEquation = await evaluate(`({
  mode: document.querySelector('[data-loss-equation]:not([hidden])').dataset.lossEquation,
  math: document.querySelectorAll('[data-loss-equation]:not([hidden]) math').length,
  source: document.querySelector('[data-loss-equation]:not([hidden]) annotation').textContent
})`);
check("InfoNCE 数学排版随模式恢复",activeEquation.mode==='infonce' && activeEquation.math===1 && activeEquation.source.includes('tau'),JSON.stringify(activeEquation));

await navigate("/posttraining-deep.html");
await evaluate(`document.querySelector('[data-scope="lora"]').click()`);
const loraState = await evaluate(`({
  title: document.querySelector('[data-scope-title]').textContent,
  encoder: document.querySelector('[data-scope-block="encoder"]').className
})`);
const dpoBefore = await evaluate(`document.querySelector('[data-dpo-loss]').textContent`);
await evaluate(`(() => {
  const input = document.querySelector('[data-dpo-beta-input]');
  input.value = "50";
  input.dispatchEvent(new Event("input", { bubbles: true }));
})()`);
const dpoAfter = await evaluate(`document.querySelector('[data-dpo-loss]').textContent`);
const dpoGradient = await evaluate(`Number(document.querySelector('[data-dpo-gradient]').textContent)`);
check("LoRA 参数范围", loraState.title.includes("低秩") && loraState.encoder.includes("is-partial"), JSON.stringify(loraState));
check("DPO beta 数值实验", Number(dpoAfter) < Number(dpoBefore), `${dpoBefore} -> ${dpoAfter}`);
check("DPO 带符号与 beta 的链式梯度", Math.abs(dpoGradient + 0.5 / (1 + Math.exp(0.6))) < 0.001, String(dpoGradient));

await navigate("/engineering.html");
const engModes = [];
for (const mode of ["pretrain", "embedding", "sft", "dpo"]) {
  await evaluate(`document.querySelector('[data-eng-mode="${mode}"]').click()`);
  for (let step = 0; step < 8; step += 1) {
    await evaluate(`document.querySelector('[data-eng-step-index="${step}"]').click()`);
  }
  engModes.push(await evaluate(`({
    step: document.querySelector('[data-eng-step]').textContent,
    title: document.querySelector('[data-eng-title]').textContent,
    code: document.querySelector('[data-eng-code]').textContent
  })`));
}
const initialBudget = await evaluate(`document.querySelector('[data-budget-tokens]').textContent`);
await evaluate(`(() => {
  const input = document.querySelector('[data-budget-gpus]');
  input.value = "16";
  input.dispatchEvent(new Event("input", { bubbles: true }));
})()`);
const changedBudget = await evaluate(`document.querySelector('[data-budget-tokens]').textContent`);
const initialMemory = await evaluate(`document.querySelector('[data-memory-per-gpu]').textContent`);
await evaluate(`document.querySelector('[data-zero-stage="3"]').click()`);
const shardedMemory = await evaluate(`document.querySelector('[data-memory-per-gpu]').textContent`);
check(
  "四种训练工程流水线均可走完",
  engModes.every((item) => item.step === "08" && item.title && item.code),
  engModes.map((item) => item.title).join(" / "),
);
check("tokens / optimizer step 计算", initialBudget === "524.3K" && changedBudget === "1.0M", `${initialBudget} -> ${changedBudget}`);
check("静态显存与 ZeRO-3 分片", initialMemory === "112.0 GB" && Number.parseFloat(shardedMemory) < Number.parseFloat(initialMemory), `${initialMemory} -> ${shardedMemory}`);

await navigate("/simulator.html");
await evaluate(`document.querySelector('[data-sim-step]').click()`);
const firstSimStep = await evaluate(`({
  steps: document.querySelector('[data-sim-steps]').textContent,
  loss: document.querySelector('[data-sim-loss]').textContent,
  recall: document.querySelector('[data-sim-recall]').textContent,
  chart: document.querySelector('[data-sim-chart] path')?.getAttribute("d") || "",
  batchRows: document.querySelectorAll('.sim-query-row').length
})`);
await evaluate(`(() => {
  const button = document.querySelector('[data-sim-step]');
  for (let i = 0; i < 40; i += 1) button.click();
})()`);
const fortySteps = await evaluate(`({
  steps: document.querySelector('[data-sim-steps]').textContent,
  loss: document.querySelector('[data-sim-loss]').textContent,
  recall: document.querySelector('[data-sim-recall]').textContent
})`);
await evaluate(`document.querySelector('[data-sim-preset="noisy"]').click()`);
await evaluate(`(() => {
  const button = document.querySelector('[data-sim-step]');
  for (let i = 0; i < 30; i += 1) button.click();
})()`);
const noisyRun = await evaluate(`({
  steps: document.querySelector('[data-sim-steps]').textContent,
  noise: document.querySelector('[data-sim-noise-value]').textContent,
  note: document.querySelector('[data-sim-error-note]').textContent
})`);
await evaluate(`document.querySelector('[data-sim-reset]').click()`);
const resetSteps = await evaluate(`document.querySelector('[data-sim-steps]').textContent`);
check("训练模拟器真实单步更新", firstSimStep.steps === "1" && firstSimStep.loss !== "—" && firstSimStep.chart && firstSimStep.batchRows > 0, JSON.stringify(firstSimStep));
check("训练模拟器连续 40 步", fortySteps.steps === "41" && Number(fortySteps.loss) > 0, JSON.stringify(fortySteps));
check("错误标签注入与重置", noisyRun.steps === "30" && noisyRun.noise === "20%" && noisyRun.note.includes("累计注入") && resetSteps === "0", JSON.stringify(noisyRun));

await navigate("/transformer-evolution.html");
await evaluate(`document.querySelector('[data-evo="3"]').click()`);
const evolutionState = await evaluate(`({
  title: document.querySelector('[data-evo-title]').textContent,
  position: document.querySelector('[data-evo-position]').textContent
})`);
check("Transformer 演化阶段切换", evolutionState.title.includes("RoPE") && evolutionState.position.includes("RoPE"), JSON.stringify(evolutionState));

await navigate("/transformer-anatomy.html");
const denseParams = await evaluate(`({
  total: document.querySelector('[data-arch-total]').textContent,
  active: document.querySelector('[data-arch-active]').textContent,
  rows: document.querySelectorAll('[data-weight-table] tr').length
})`);
await evaluate(`document.querySelector('[data-arch-preset="moe"]').click()`);
const moeParams = await evaluate(`({
  total: document.querySelector('[data-arch-total]').textContent,
  active: document.querySelector('[data-arch-active]').textContent,
  ratio: document.querySelector('[data-arch-ratio]').textContent,
  caption: document.querySelector('[data-arch-caption]').textContent
})`);
await evaluate(`document.querySelector('[data-moe-route]').click()`);
const routedTokens = await evaluate(`document.querySelectorAll('.moe-token').length`);
await evaluate(`document.querySelector('[data-arch-preset="mixtral"]').click()`);
const mixtralParams = await evaluate(`({
  total: document.querySelector('[data-arch-total]').textContent,
  active: document.querySelector('[data-arch-active]').textContent
})`);
check("权重矩阵与参数量计算", denseParams.total === "1.26B" && denseParams.active === "1.26B" && denseParams.rows === 11, JSON.stringify(denseParams));
check("MoE 总参数与激活参数分离", moeParams.total === "7.08B" && moeParams.active === "2.09B" && moeParams.ratio === "30%" && moeParams.caption.includes("top-2"), JSON.stringify(moeParams));
check("MoE token 路由可视化", routedTokens === 8, String(routedTokens));
check("公开 MoE 结构量级", mixtralParams.total === "46.70B" && mixtralParams.active === "12.88B", JSON.stringify(mixtralParams));

await navigate("/inference-engine.html");
await evaluate(`document.querySelector('[data-infer-prefill]').click()`);
const prefillState = await evaluate(`({
  phase: document.querySelector('[data-infer-phase]').textContent,
  compute: document.querySelector('[data-infer-compute]').textContent,
  cache: document.querySelector('[data-infer-cache-tokens]').textContent,
  batchCache: document.querySelector('[data-infer-batch-cache]').textContent
})`);
await evaluate(`document.querySelector('[data-infer-ten]').click()`);
const decodeState = await evaluate(`({
  generated: document.querySelector('[data-infer-generated]').textContent,
  compute: document.querySelector('[data-infer-compute]').textContent,
  cache: document.querySelector('[data-infer-cache-tokens]').textContent
})`);
const kvGqa = await evaluate(`({
  perToken: document.querySelector('[data-kv-per-token]').textContent,
  request: document.querySelector('[data-kv-request]').textContent,
  total: document.querySelector('[data-kv-total]').textContent
})`);
await evaluate(`document.querySelector('[data-kv-preset="mqa"]').click()`);
const kvMqa = await evaluate(`document.querySelector('[data-kv-per-token]').textContent`);
check("prefill 采样首 token，decode 缓存落后一位", prefillState.phase.includes("首 token") && prefillState.compute === "4,096" && prefillState.cache === "1,024" && prefillState.batchCache === "4,096" && decodeState.generated === "11" && decodeState.compute === "4" && decodeState.cache === "1,034", JSON.stringify({ prefillState, decodeState }));
check("GQA / MQA KV Cache 计算", kvGqa.perToken === "128 KiB" && kvGqa.request === "512 MiB" && kvGqa.total === "2 GiB" && kvMqa === "16 KiB", JSON.stringify({ kvGqa, kvMqa }));

await navigate("/data-tokenizer.html");
await evaluate(`document.querySelector('[data-token-mode="char"]').click()`);
const charTokens = await evaluate(`document.querySelector('[data-tokenizer-count]').textContent`);
await evaluate(`document.querySelector('[data-token-mode="byte"]').click()`);
const byteTokens = await evaluate(`document.querySelector('[data-tokenizer-count]').textContent`);
await evaluate(`document.querySelector('[data-data-step="3"]').click()`);
const dataStep = await evaluate(`document.querySelector('[data-data-title]').textContent`);
const mixtureBars = await evaluate(`document.querySelectorAll('.mixture-row').length`);
check("Tokenizer 单位切换", charTokens === "15" && byteTokens === "45", `${charTokens} / ${byteTokens}`);
check("数据流水线与混合视图", dataStep.includes("训练集与评估集") && mixtureBars === 5, `${dataStep} / ${mixtureBars}`);

await navigate("/model-evaluation.html");
await evaluate(`document.querySelector('[data-eval-lens="sft"]').click()`);
const evalLens = await evaluate(`document.querySelector('[data-eval-title]').textContent`);
await evaluate(`(() => {
  for (const input of document.querySelectorAll('[data-slice]')) {
    input.checked = input.dataset.slice === "zh";
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }
})()`);
const sliceState = await evaluate(`({
  macro: document.querySelector('[data-slice-macro]').textContent,
  micro: document.querySelector('[data-slice-micro]').textContent,
  weak: document.querySelector('[data-slice-weak]').textContent
})`);
check("评估指标切换", evalLens.includes("token loss"), evalLens);
check("评估切片宏/微平均", sliceState.macro === "0.860" && sliceState.micro === "0.860" && sliceState.weak.includes("中文短问"), JSON.stringify(sliceState));

await navigate("/model-artifacts.html");
await evaluate(`document.querySelector('[data-artifact="optimizer"]').click()`);
const artifactState = await evaluate(`({
  title: document.querySelector('[data-artifact-title]').textContent,
  deploy: document.querySelector('[data-artifact-deploy]').textContent
})`);
check("模型工件检查器", artifactState.title.includes("继续训练") && artifactState.deploy === "通常不需要", JSON.stringify(artifactState));

await navigate("/training-systems-deep.html");
await evaluate(`document.querySelector('[data-par-mode="tp"]').click()`);
const parallelState = await evaluate(`({
  title: document.querySelector('[data-par-title]').textContent,
  comm: document.querySelector('[data-par-comm]').textContent,
  cards: document.querySelectorAll('.par-gpu-card').length
})`);
const attentionBefore = await evaluate(`document.querySelector('[data-act-attention]').textContent`);
await evaluate(`(() => {
  const input = document.querySelector('[data-act-t]');
  input.value = "16384";
  input.dispatchEvent(new Event("input", { bubbles: true }));
})()`);
const attentionAfter = await evaluate(`document.querySelector('[data-act-attention]').textContent`);
check("训练并行方式切换", parallelState.title.includes("单层矩阵") && parallelState.comm.includes("all-reduce") && parallelState.cards === 4, JSON.stringify(parallelState));
check("Activation 随序列增长", attentionBefore !== attentionAfter, `${attentionBefore} -> ${attentionAfter}`);

await navigate("/architecture-variants.html");
await evaluate(`document.querySelector('[data-variant="mla"]').click()`);
const variantState = await evaluate(`({
  title: document.querySelector('[data-variant-title]').textContent,
  structure: document.querySelector('[data-variant-structure]').textContent
})`);
await evaluate(`document.querySelector('[data-arch-goal="long"]').click()`);
const goalState = await evaluate(`({
  title: document.querySelector('[data-goal-title]').textContent,
  stackItems: document.querySelectorAll('[data-goal-stack] > div').length
})`);
check("架构变体切换", variantState.title.includes("隐变量") && variantState.structure.includes("latent"), JSON.stringify(variantState));
check("目标导向结构组合", goalState.title.includes("位置") && goalState.stackItems === 5, JSON.stringify(goalState));

await navigate("/frontier.html");
const frontierRoutes = await evaluate(`document.querySelectorAll('.frontier-route-grid > a').length`);
check("现代 AI 专题入口", frontierRoutes === 11, String(frontierRoutes));

await navigate("/scaling-laws.html");
const scalingDefault = await evaluate(`({
  model: document.querySelector('[data-scaling-model-value]').textContent,
  tokens: document.querySelector('[data-scaling-tokens]').textContent,
  ratio: document.querySelector('[data-scaling-ratio]').textContent,
  epochs: document.querySelector('[data-scaling-epochs]').textContent
})`);
await evaluate(`(() => {
  const input = document.querySelector('[data-scaling-compute]');
  input.value = "1e23";
  input.dispatchEvent(new Event("change", { bubbles: true }));
})()`);
const scalingChanged = await evaluate(`document.querySelector('[data-scaling-tokens]').textContent`);
check("Scaling 预算分配", scalingDefault.model === "7.1B" && scalingDefault.ratio === "33" && scalingDefault.epochs === "0.79×" && scalingChanged !== scalingDefault.tokens, JSON.stringify({ scalingDefault, scalingChanged }));

await navigate("/rl-posttraining.html");
await evaluate(`document.querySelector('[data-reward-source="verifier"]').click()`);
const rewardState = await evaluate(`({
  title: document.querySelector('[data-reward-title]').textContent,
  algos: document.querySelector('[data-reward-algos]').textContent
})`);
check("RL 奖励来源切换", rewardState.title.includes("自动判分") && rewardState.algos.includes("RLVR"), JSON.stringify(rewardState));

await navigate("/model-case-studies.html");
await evaluate(`document.querySelector('[data-case="mixtral"]').click()`);
const caseState = await evaluate(`({
  title: document.querySelector('[data-case-title]').textContent,
  params: document.querySelector('[data-case-params]').textContent,
  ffn: document.querySelector('[data-case-ffn]').textContent
})`);
await evaluate(`document.querySelector('[data-case="deepseek"]').click()`);
const deepseekState = await evaluate(`document.querySelector('[data-case-title]').textContent`);
check("代表模型案例切换", caseState.params.includes("46.7B") && caseState.ffn.includes("top-2") && deepseekState.includes("DeepSeek"), JSON.stringify({ caseState, deepseekState }));

await navigate("/training-engineer-guide.html");
await evaluate(`document.querySelector('[data-problem-category="throughput"]').click()`);
const filteredProblems = await evaluate(`document.querySelectorAll('[data-problem-id]').length`);
await evaluate(`document.querySelector('[data-problem-id="low-util"]').click()`);
const problemState = await evaluate(`({
  title: document.querySelector('[data-problem-title]').textContent,
  observe: document.querySelector('[data-problem-observe]').textContent,
  next: document.querySelector('[data-problem-next]').textContent
})`);
const contracts = await evaluate(`document.querySelector('.engineer-contracts').children.length`);
await evaluate(`(() => {
  const input = document.querySelector('[data-site-search]');
  input.value = "GQA";
  input.dispatchEvent(new Event("input", { bubbles: true }));
})()`);
const searchResults = await evaluate(`({
  count: document.querySelectorAll('[data-site-index-results] > a').length,
  first: document.querySelector('[data-site-index-results] > a strong')?.textContent || ""
})`);
check("训练工程师问题雷达", filteredProblems === 1 && problemState.title.includes("利用率") && problemState.observe.includes("dataloader") && contracts === 7, JSON.stringify({ filteredProblems, problemState, contracts }));
check("概念索引搜索", searchResults.count >= 2 && searchResults.first, JSON.stringify(searchResults));

await navigate("/reasoning-compute.html");
await evaluate(`document.querySelector('.math-derivation summary').click()`);
const reasoningMath = await evaluate(`({open:document.querySelector('.math-derivation').open,count:document.querySelectorAll('.math-derivation math').length,source:document.querySelector('.math-derivation annotation').textContent})`);
check("CoT 边缘概率推导可展开",reasoningMath.open && reasoningMath.count>=1 && reasoningMath.source.includes('sum_r'),JSON.stringify(reasoningMath));
await evaluate(`document.querySelector('[data-open-search]').click()`);
await evaluate(`(() => {
  const input=document.querySelector('[data-global-search]');
  input.value='J-space'; input.dispatchEvent(new Event('input',{bubbles:true}));
})()`);
const globalSearch = await evaluate(`({
  open:document.querySelector('[data-search-dialog]').open,
  focused:document.activeElement===document.querySelector('[data-global-search]'),
  links:[...document.querySelectorAll('[data-global-search-results] a')].map(a=>a.getAttribute('href'))
})`);
check("全站搜索覆盖新增 J-space 专题",globalSearch.open && globalSearch.focused && globalSearch.links.includes('reasoning-compute.html'),JSON.stringify(globalSearch));
await send('Page.bringToFront');
await send('Emulation.setFocusEmulationEnabled',{enabled:true});
await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
await wait(100);
const searchEscape=await evaluate(`!document.querySelector('[data-search-dialog]').open && document.activeElement===document.querySelector('[data-open-search]')`);
check("搜索有内容时 Esc 关闭并返回焦点",searchEscape,String(searchEscape));
await evaluate(`document.querySelector('[data-open-library]').click()`);
const mobileLibrary=await evaluate(`({open:document.querySelector('[data-library-dialog]').open, links:document.querySelectorAll('[data-library-dialog] nav a').length})`);
check("手机知识目录覆盖全部专题",mobileLibrary.open && mobileLibrary.links===pageConfig.length,JSON.stringify(mobileLibrary));
await evaluate(`document.querySelector('[data-library-dialog] [data-close-dialog]').click()`);
await evaluate(`document.querySelector('[data-wiki-toc] summary').click()`);
const toc=await evaluate(`({open:document.querySelector('[data-wiki-toc]').open,links:document.querySelectorAll('[data-toc-links] a').length})`);
check("页内章节可展开",toc.open && toc.links>=5,JSON.stringify(toc));
await evaluate(`(() => {
  const input=document.querySelector('[data-reason-n]');input.value='1';input.dispatchEvent(new Event('input',{bubbles:true}));
})()`);
const singleCandidate=await evaluate(`({coverage:document.querySelector('[data-reason-coverage]').textContent,selected:document.querySelector('[data-reason-selected]').textContent})`);
check("单候选正确率不被选择器再次打折",singleCandidate.coverage==='30.0%' && singleCandidate.selected==='30.0%',JSON.stringify(singleCandidate));
await evaluate(`(() => {
  for(const [selector,value] of [['[data-reason-n]','4'],['[data-reason-q]','0']]) {
    const input=document.querySelector(selector);input.value=value;input.dispatchEvent(new Event('input',{bubbles:true}));
  }
})()`);
const weakSelector=await evaluate(`Number.parseFloat(document.querySelector('[data-reason-selected]').textContent)`);
check("候选正确不保证选择正确",weakSelector<1,String(weakSelector));

await navigate("/generation-mechanisms.html");
await evaluate(`document.querySelector('[data-gen-next]').click()`);
const arState=await evaluate(`({known:document.querySelectorAll('[data-gen-tape] .pending').length,title:document.querySelector('[data-gen-title]').textContent})`);
await evaluate(`document.querySelector('[data-gen-mode="diffusion"]').click()`);
await evaluate(`document.querySelector('[data-gen-next]').click()`);
const diffusionState=await evaluate(`({known:document.querySelectorAll('[data-gen-tape] .accepted').length,title:document.querySelector('[data-gen-title]').textContent})`);
check("自回归与离散去噪的状态区别",arState.known===1 && diffusionState.known===3,JSON.stringify({arState,diffusionState}));

await navigate("/multimodal.html");
const vision=await evaluate(`(() => {
  const canvas=document.querySelector('[data-mm-canvas]');
  const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
  const colors=new Set();for(let i=0;i<pixels.length;i+=64)colors.add([pixels[i],pixels[i+1],pixels[i+2]].join(','));
  return {patches:document.querySelector('[data-mm-patches]').textContent,tokens:document.querySelector('[data-mm-tokens]').textContent,colors:colors.size};
})()`);
await evaluate(`(() => {const input=document.querySelector('[data-mm-frames]');input.value='8';input.dispatchEvent(new Event('input',{bubbles:true}));})()`);
const videoTokens=await evaluate(`document.querySelector('[data-mm-tokens]').textContent`);
check("视觉 patch 预算与非空画布",vision.patches==='1,024' && vision.tokens==='256' && vision.colors>12 && videoTokens==='2,048',JSON.stringify({vision,videoTokens}));

await navigate("/distillation-data.html");
const teacherBefore=await evaluate(`Number.parseFloat(document.querySelector('[data-distill-max]').textContent)`);
await evaluate(`(() => {const input=document.querySelector('[data-distill-temperature]');input.value='8';input.dispatchEvent(new Event('input',{bubbles:true}));})()`);
const teacherAfter=await evaluate(`Number.parseFloat(document.querySelector('[data-distill-max]').textContent)`);
check("温度软化教师分布",teacherAfter<teacherBefore,`${teacherBefore} -> ${teacherAfter}`);

await navigate("/efficient-inference.html");
const quantBefore=await evaluate(`Number(document.querySelector('[data-quant-mse]').textContent)`);
await evaluate(`(() => {const input=document.querySelector('[data-quant-bits]');input.value='8';input.dispatchEvent(new Event('input',{bubbles:true}));})()`);
const quantAfter=await evaluate(`Number(document.querySelector('[data-quant-mse]').textContent)`);
check("量化位宽改变数值误差",quantAfter<quantBefore,`${quantBefore} -> ${quantAfter}`);
await evaluate(`(() => {for(let i=0;i<3;i++)document.querySelector('[data-spec-next]').click();})()`);
const speculative=await evaluate(`({accepted:document.querySelectorAll('[data-spec-tape] .accepted').length,rejected:document.querySelectorAll('[data-spec-tape] .rejected').length,corrected:document.querySelectorAll('[data-spec-tape] .corrected').length,done:document.querySelector('[data-spec-next]').disabled})`);
check("投机解码拒绝后丢弃后续草稿",speculative.accepted===2 && speculative.rejected===2 && speculative.corrected===1 && speculative.done,JSON.stringify(speculative));

await navigate("/agents-context.html");
await evaluate(`(() => {for(let i=0;i<3;i++)document.querySelector('[data-agent-next]').click();})()`);
const agentState=await evaluate(`document.querySelector('[data-agent-copy]').textContent`);
check("工具执行与模型更新分开显示",agentState.includes('未执行训练反传') && agentState.includes('12'),agentState);

await navigate("/data-tokenizer.html");
const cleanBefore=await evaluate(`({kept:document.querySelector('[data-curation-kept]').textContent,leaks:document.querySelector('[data-curation-leaks]').textContent})`);
await evaluate(`document.querySelector('[data-curation-holdout]').click()`);
const cleanAfter=await evaluate(`document.querySelector('[data-curation-leaks]').textContent`);
check("质量过滤与评测污染分别生效",cleanBefore.kept==='4/8' && cleanBefore.leaks==='0' && cleanAfter==='1',JSON.stringify({cleanBefore,cleanAfter}));
await evaluate(`(() => {for(const x of document.querySelectorAll('[data-mixture]')) {x.value='0';x.dispatchEvent(new Event('input',{bubbles:true}));}})()`);
check("全零数据混合不显示虚假预算分配",await evaluate(`document.querySelector('[data-mixture-bars]').textContent.includes('无法分配')`));
await navigate("/model-evaluation.html");
await evaluate(`(() => {for(const x of document.querySelectorAll('[data-slice]')) {x.checked=false;x.dispatchEvent(new Event('change',{bubbles:true}));}})()`);
check("空评估切片显示无数据",await evaluate(`document.querySelector('[data-slice-macro]').textContent==='无数据' && document.querySelector('[data-slice-micro]').textContent==='无数据'`));

await navigate("/optimization-dynamics.html");
await evaluate(`(() => { for(let i=0;i<6;i++)document.querySelector('[data-opt-next]').click(); })()`);
const optimizerState = await evaluate(`({step:document.querySelector('[data-opt-step]').textContent,rows:document.querySelectorAll('[data-opt-rows] tr').length,note:document.querySelector('[data-opt-note]').textContent})`);
check("优化器历史状态与零梯度",optimizerState.step==='6/7' && optimizerState.rows===6 && optimizerState.note.includes('g=0'),JSON.stringify(optimizerState));
await evaluate(`document.querySelector('[data-opt-reset]').click()`);
check("优化器账本可重置",await evaluate(`document.querySelector('[data-opt-theta]').textContent==='1.0000'`));

await navigate("/parallel-correctness.html");
const correctGradient = await evaluate(`document.querySelector('[data-reduce-target]').textContent`);
await evaluate(`(() => { const x=document.querySelector('[data-reduce-mode]');x.value='mean';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
const wrongGradient = await evaluate(`document.querySelector('[data-reduce-actual]').textContent`);
check("多卡等权平均可反转梯度",correctGradient==='0.400' && wrongGradient==='-0.500',`${correctGradient} / ${wrongGradient}`);
await evaluate(`(() => {for(const s of ['[data-reduce-a]','[data-reduce-b]']){const x=document.querySelector(s);x.value='0';x.dispatchEvent(new Event('input',{bubbles:true}));}})()`);
const emptyWindow = await evaluate(`({target:document.querySelector('[data-reduce-target]').textContent,note:document.querySelector('[data-reduce-note]').textContent})`);
check("零有效 token 不伪造均值",emptyWindow.target==='未定义' && emptyWindow.note.includes('协调跳过'),JSON.stringify(emptyWindow));
await wait(100);
const tableRegion = await evaluate(`(() => { const region=document.querySelector('.experiment .compare-wrap');region.focus();region.scrollLeft=120;return {focus:document.activeElement===region,scroll:region.scrollLeft,label:region.getAttribute('aria-label')};})()`);
check("手机宽表可聚焦并横向查看",tableRegion.focus && tableRegion.scroll>0 && tableRegion.label,JSON.stringify(tableRegion));

await navigate("/performance-analysis.html");
const initialBound = await evaluate(`document.querySelector('[data-perf-bound]').textContent`);
await evaluate(`(() => {const x=document.querySelector('[data-perf-m]');x.value='4096';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
const finalBound = await evaluate(`({bound:document.querySelector('[data-perf-bound]').textContent,path:document.querySelector('[data-perf-chart] path:last-of-type').getAttribute('d')})`);
check("Roofline 随 GEMM 形状改变限制",initialBound==='带宽' && finalBound.bound==='算力' && finalBound.path.includes('L'),JSON.stringify({initialBound,finalBound:finalBound.bound}));

await navigate("/engineering-cases.html");
const caseRuns=[];
for(const key of ['parallel','data','moe','reward']) {
  await evaluate(`(() => {const x=document.querySelector('[data-diagnosis-case]');x.value='${key}';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  const locked = await evaluate(`document.querySelector('[data-diagnosis-next]').disabled`);
  await evaluate(`document.querySelector('[name=diagnosis-evidence]').click()`);
  await evaluate(`(() => {for(let i=0;i<3;i++)document.querySelector('[data-diagnosis-next]').click();})()`);
  const state=await evaluate(`({count:document.querySelectorAll('[data-diagnosis-evidence] li').length,done:document.querySelector('[data-diagnosis-next]').disabled,conclusion:!document.querySelector('[data-diagnosis-conclusion]').hidden,link:document.querySelector('[data-diagnosis-conclusion] a')?.getAttribute('href')})`);
  caseRuns.push({key,locked,...state});
}
check("四个案例逐步揭示证据与结论",caseRuns.every(x=>x.locked && x.count===3 && x.done && x.conclusion && x.link),JSON.stringify(caseRuns));
await evaluate(`document.querySelector('[data-diagnosis-reset]').click()`);
check("案例重置清除证据与选择",await evaluate(`document.querySelectorAll('[data-diagnosis-evidence] li').length===0 && document.querySelector('[data-diagnosis-conclusion]').hidden && document.querySelector('[data-diagnosis-next]').disabled`));

await navigate("/finetuning-recipe.html");
const recipeState = () => evaluate(`({processed:document.querySelector('[data-ft-processed]').textContent,supervised:document.querySelector('[data-ft-supervised]').textContent,updates:document.querySelector('[data-ft-updates]').textContent,note:document.querySelector('[data-ft-note]').textContent})`);
const ftDefault = await recipeState();
check("微调记录、监督 token 与计划更新", ftDefault.processed==='30,000,000' && ftDefault.supervised==='12,000,000' && ftDefault.updates==='471',JSON.stringify(ftDefault));
await evaluate(`document.querySelector('[data-ft-packing]').click()`);
const ftPacked = await recipeState();
check("整条 packing 保持监督量并改变更新次数",ftPacked.supervised===ftDefault.supervised && ftPacked.updates==='237',JSON.stringify(ftPacked));
await evaluate(`(() => {const x=document.querySelector('[data-ft-limit]');x.value='512';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
const ftTruncated = await recipeState();
check("截断全部回答后提示跳过无监督更新",ftTruncated.supervised==='0' && ftTruncated.note.includes('跳过'),JSON.stringify(ftTruncated));
await evaluate(`(() => {const x=document.querySelector('[data-ft-records]');x.value='';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
check("微调空输入清除旧数值",(await recipeState()).updates==='-');
await evaluate(`(() => {const x=document.querySelector('[data-ft-records]');x.value='0';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
check("零条记录没有计划窗口",(await recipeState()).updates==='0');
await evaluate(`(() => {const x=document.querySelector('[data-ft-target]');x.value='all';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
const ftLora = await evaluate(`({params:document.querySelector('[data-ft-params]').textContent,rows:document.querySelectorAll('[data-ft-lora-rows] tr').length,cols:document.querySelector('[data-ft-lora-rows] tr').children.length,heads:document.querySelector('[data-ft-lora-rows]').previousElementSibling.querySelectorAll('th').length})`);
check("LoRA 目标模块参数与表格列对齐",ftLora.params==='36,831,232' && ftLora.rows===7 && ftLora.cols===ftLora.heads,JSON.stringify(ftLora));

await navigate("/rl-posttraining.html");
for (const value of ['allpass','allfail']) {
  await evaluate(`(() => {const x=document.querySelector('[data-rl-group]');x.value='${value}';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  check(`RL ${value} 组没有相对优势`,await evaluate(`document.querySelector('[data-rld-signal]').textContent==='无' && [...document.querySelectorAll('[data-rld-rows] tr')].every(row=>row.lastElementChild.textContent==='0.000')`));
}
await evaluate(`(() => {for(const [selector,value] of [['[data-rld-old]','80'],['[data-rld-current]','20'],['[data-rld-sign]','-1']]) {const x=document.querySelector(selector);x.value=value;x.dispatchEvent(new Event('input',{bubbles:true}));}})()`);
const negativeClip = await evaluate(`({ratio:document.querySelector('[data-rld-ratio]').textContent,objective:document.querySelector('[data-rld-objective]').textContent,note:document.querySelector('[data-rld-clip-note]').textContent})`);
check("负优势在低概率比侧形成裁剪平台",negativeClip.ratio==='0.250' && negativeClip.objective==='-0.800' && negativeClip.note.includes('低于 0.8'),JSON.stringify(negativeClip));

await navigate("/distillation-data.html");
await evaluate(`(() => {const x=document.querySelector('[data-opd-mix]');x.value='0';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
const opdState = await evaluate(`({coverage:document.querySelector('[data-opd-recovery]').textContent,teacher:document.querySelector('[data-opd-teacher-loss]').textContent,student:document.querySelector('[data-opd-student-loss]').textContent,cols:document.querySelector('[data-opd-rows] tr').children.length,heads:document.querySelector('[data-opd-rows]').previousElementSibling.querySelectorAll('th').length})`);
check("学生前缀权重与 KL 表格对齐",opdState.coverage==='100%' && opdState.teacher==='0.226' && opdState.student==='0.832' && opdState.cols===opdState.heads,JSON.stringify(opdState));

await navigate("/agents-context.html");
await evaluate(`(() => {const x=document.querySelector('[data-memory-mode]');x.value='compact';x.dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[data-memory-verify]').click();})()`);
check("摘要丢失约束且无法回读缺失来源",await evaluate(`document.querySelector('[data-memory-constraints]').textContent==='缺失' && document.querySelector('[data-memory-grounding]').textContent==='未回读' && document.querySelectorAll('[data-memory-items] li').length===2`));
await evaluate(`(() => {const x=document.querySelector('[data-memory-mode]');x.value='structured';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("结构化记忆保留新日期、约束和模拟回读",await evaluate(`document.querySelector('[data-memory-constraints]').textContent==='已保留' && document.querySelector('[data-memory-date]').textContent==='周六' && document.querySelector('[data-memory-grounding]').textContent==='模拟已回读' && document.querySelectorAll('[data-memory-items] li').length===4`));

await navigate("/model-evaluation.html");
const repeatState = await evaluate(`({any:document.querySelector('[data-er-any]').textContent,all:document.querySelector('[data-er-all]').textContent,observed:document.querySelector('[data-er-observed]').textContent,bias:document.querySelector('[data-er-bias]').textContent})`);
check("重复可靠性与判分器偏差",repeatState.any==='99.97%' && repeatState.all==='32.77%' && repeatState.observed==='58.00%' && repeatState.bias==='-2.0 pp',JSON.stringify(repeatState));
await evaluate(`(() => {for(const selector of ['[data-er-sensitivity]','[data-er-specificity]']){const x=document.querySelector(selector);x.value='100';x.dispatchEvent(new Event('input',{bubbles:true}));}})()`);
check("完美判分器恢复真实成功率",await evaluate(`document.querySelector('[data-er-observed]').textContent==='60.00%' && document.querySelector('[data-er-bias]').textContent==='0.0 pp'`));

await navigate("/multimodal.html");
await evaluate(`(() => {const x=document.querySelector('[data-stream-interval]');x.value='100';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
const streamState = await evaluate(`({first:document.querySelector('[data-stream-first]').textContent,end:document.querySelector('[data-stream-end]').textContent,offline:document.querySelector('[data-stream-offline]').textContent,backlog:document.querySelector('[data-stream-backlog]').textContent,bars:[...document.querySelectorAll('[data-stream-timeline] i')].map(el=>({left:parseFloat(el.style.left),width:parseFloat(el.style.width)})),cols:document.querySelector('[data-stream-rows] tr').children.length})`);
check("流式队列积压与实际处理条",streamState.first==='280 ms' && streamState.end==='820 ms' && streamState.offline==='1120 ms' && streamState.backlog==='240 ms' && streamState.cols===5 && streamState.bars.length===4 && streamState.bars.every(b=>b.width>0 && b.left+b.width<=100.001),JSON.stringify(streamState));

await navigate("/model-case-studies.html");
check("新版架构案例明确版本和存储范围",await evaluate(`document.querySelector('#frontier-cache-case').textContent.includes('跨层复用') && document.querySelectorAll('.architecture-modes > div').length===3 && document.body.textContent.includes('552B backbone') && document.body.textContent.includes('196B Engram')`));

await navigate("/uncertainty-calibration.html");
const calibration = await evaluate(`({coverage:document.querySelector('[data-uc-coverage]').textContent,risk:document.querySelector('[data-uc-risk]').textContent,brier:document.querySelector('[data-uc-brier]').textContent,rows:document.querySelectorAll('[data-uc-rows] tr').length})`);
check("回答门槛与全体校准分开计算",calibration.coverage==='70.00%' && calibration.risk==='28.57%' && calibration.rows===10,JSON.stringify(calibration));
await evaluate(`(() => {const x=document.querySelector('[data-uc-threshold]');x.value='100';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
check("无人作答不显示零风险",await evaluate(`document.querySelector('[data-uc-risk]').textContent==='无作答' && document.querySelector('[data-uc-brier]').textContent===${JSON.stringify(calibration.brier)}`));
await evaluate(`document.querySelector('[data-uc-stable]').click()`);
check("高分错误改变校准误差",await evaluate(`Number(document.querySelector('[data-uc-brier]').textContent)>Number(${JSON.stringify(calibration.brier)})`));
await evaluate(`(() => {const x=document.querySelector('[data-uc-semantic]');x.value='stablewrong';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("稳定错误仍可具有零语义熵",await evaluate(`document.querySelector('[data-uc-entropy]').textContent==='0.0000' && document.querySelector('[data-uc-semantic-accuracy]').textContent==='0.00%'`));
await evaluate(`(() => {const x=document.querySelector('[data-uc-semantic]');x.value='conflict';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("含义分歧提高经验熵",await evaluate(`Number(document.querySelector('[data-uc-entropy]').textContent)>1 && document.querySelectorAll('[data-uc-meanings] tr').length===3`));

await navigate("/continual-learning.html");
const canvasBefore = await evaluate(`document.querySelector('[data-cl-canvas]').toDataURL()`);
await evaluate(`(() => {for(const [selector,value] of [['[data-cl-angle]','0'],['[data-cl-lr]','250']]) {const x=document.querySelector(selector);x.value=value;x.dispatchEvent(new Event('input',{bubbles:true}));}})()`);
const overshoot = await evaluate(`({a:document.querySelector('[data-cl-loss-a]').textContent,b:document.querySelector('[data-cl-loss-b]').textContent,note:document.querySelector('[data-cl-note]').textContent,canvas:document.querySelector('[data-cl-canvas]').toDataURL()})`);
check("梯度一致时过大步长仍会增加损失",overshoot.a==='1.1250' && overshoot.b==='1.1250' && overshoot.note.includes('实际损失上升') && overshoot.canvas!==canvasBefore,JSON.stringify({a:overshoot.a,b:overshoot.b,note:overshoot.note}));
await evaluate(`(() => {for(const [selector,value] of [['[data-cl-angle]','180'],['[data-cl-mode]','projected']]) {const x=document.querySelector(selector);x.value=value;x.dispatchEvent(new Event('input',{bubbles:true}));}})()`);
check("完全冲突投影后停在原点",await evaluate(`document.querySelector('[data-cl-loss-a]').textContent==='0.5000' && document.querySelector('[data-cl-loss-b]').textContent==='0.5000'`));
const canvasColors = await evaluate(`(() => {const c=document.querySelector('[data-cl-canvas]');const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;const colors=new Set();for(let i=0;i<d.length;i+=16)colors.add(d[i]+','+d[i+1]+','+d[i+2]);return colors.size;})()`);
check("任务曲面画布非空",canvasColors>20,String(canvasColors));
const preThemeCanvas=await evaluate(`document.querySelector('[data-cl-canvas]').toDataURL()`);
await evaluate(`document.querySelector('[data-theme-toggle]').click()`);
await wait(100);
check("任务曲面随主题重绘",await evaluate(`document.querySelector('[data-cl-canvas]').toDataURL()!==${JSON.stringify(preThemeCanvas)}`));

await navigate("/learning-mechanisms.html");
check("模式复制读取历史后继",await evaluate(`document.querySelector('[data-icl-copy-result]').textContent==='B' && document.querySelector('[data-icl-prob-b]').textContent==='88.1%' && document.querySelectorAll('[data-icl-tokens] span').length===4`));
await evaluate(`(() => {const x=document.querySelector('[data-icl-example]');x.value='D';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("换上下文示例改变复制结果",await evaluate(`document.querySelector('[data-icl-copy-result]').textContent==='D' && document.querySelector('[data-icl-prob-d]').textContent==='88.1%'`));
for(const selector of ['[data-icl-previous]','[data-icl-matching]','[data-icl-copy]']) {
  await evaluate(`document.querySelector('${selector}').click()`);
  check(`电路消融 ${selector}`,await evaluate(`document.querySelector('[data-icl-prob-b]').textContent==='50.0%' && document.querySelector('[data-icl-prob-d]').textContent==='50.0%'`));
  await evaluate(`document.querySelector('${selector}').click()`);
}

await navigate("/training-environments.html");
await evaluate(`(() => {const x=document.querySelector('[data-env-scenario]');x.value='sideeffect';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("只看数量会奖励带副作用的轨迹",await evaluate(`document.querySelector('[data-env-reward]').textContent==='1' && document.querySelector('[data-env-checks] tr:last-child').lastElementChild.textContent==='不满足'`));
await evaluate(`(() => {const x=document.querySelector('[data-env-grader]');x.value='strict';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("完整验证器拒绝副作用",await evaluate(`document.querySelector('[data-env-reward]').textContent==='0'`));
await evaluate(`(() => {const x=document.querySelector('[data-env-scenario]');x.value='good';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("切换轨迹从相同初始快照恢复",await evaluate(`document.querySelector('[data-env-stock]').textContent==='8' && document.querySelector('[data-env-reserved]').textContent==='2' && document.querySelector('[data-env-reward]').textContent==='1'`));

await navigate("/reasoning-compute.html");
const reBase = await evaluate(`({pass1:document.querySelector('[data-re-pass1]').textContent,passk:document.querySelector('[data-re-passk]').textContent,sound:document.querySelector('[data-re-sound]').textContent})`);
await evaluate(`(() => {const x=document.querySelector('[data-re-mode]');x.value='process';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("过程正确性可在答案命中率不变时提高",await evaluate(`document.querySelector('[data-re-pass1]').textContent===${JSON.stringify(reBase.pass1)} && document.querySelector('[data-re-passk]').textContent===${JSON.stringify(reBase.passk)} && parseFloat(document.querySelector('[data-re-sound]').textContent)>parseFloat(${JSON.stringify(reBase.sound)})`));
await evaluate(`(() => {const x=document.querySelector('[data-re-mode]');x.value='efficient';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("单次成功提升可伴随有限多次覆盖下降",await evaluate(`parseFloat(document.querySelector('[data-re-pass1]').textContent)>parseFloat(${JSON.stringify(reBase.pass1)}) && parseFloat(document.querySelector('[data-re-passk]').textContent)<parseFloat(${JSON.stringify(reBase.passk)})`));

await navigate("/optimization-dynamics.html");
const matrixBefore = await evaluate(`([...document.querySelectorAll('[data-mu-rows] tr')].map(row=>[...row.children].map(x=>x.textContent)))`);
await evaluate(`(() => {const x=document.querySelector('[data-mu-ratio]');x.value='32';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
const matrixAfter = await evaluate(`([...document.querySelectorAll('[data-mu-rows] tr')].map(row=>[...row.children].map(x=>x.textContent)))`);
check("改变奇异值比保持极因子方向",matrixAfter.length===4 && matrixAfter.every((row,i)=>row.length===5 && row[4]===matrixBefore[i][4]) && matrixAfter.some((row,i)=>row[1]!==matrixBefore[i][1]),JSON.stringify(matrixAfter));

await navigate("/backpropagation.html");
const bpStages = [];
for (let i = 0; i < 13; i++) {
  bpStages.push(await evaluate(`({step:document.querySelector('[data-bp-step]').textContent,title:document.querySelector('[data-bp-title]').textContent,rows:document.querySelectorAll('[data-bp-matrix] tr').length,hasBad:/NaN|undefined/.test(document.querySelector('[data-bp-matrix]').textContent)})`));
  if(i<12) await evaluate(`document.querySelector('[data-bp-next]').click()`);
}
check("整层反传十三个节点均有数值和梯度",bpStages.every(x=>x.rows>0&&!x.hasBad)&&bpStages.at(-1).step==='13/13',JSON.stringify(bpStages.map(x=>x.title)));
check("反传末节点停止且输入前位置有梯度",await evaluate(`document.querySelector('[data-bp-next]').disabled && [...document.querySelectorAll('[data-bp-matrix] tr')].slice(0,2).some(row=>Number(row.lastElementChild.textContent)!==0)`));
await evaluate(`document.querySelector('[data-bp-prev]').click()`);
check("反传可向前回看",await evaluate(`document.querySelector('[data-bp-step]').textContent==='12/13'`));
await evaluate(`document.querySelector('[data-bp-reset]').click()`);
check("反传复位到 loss",await evaluate(`document.querySelector('[data-bp-step]').textContent==='1/13' && document.querySelector('[data-bp-prev]').disabled`));
const bpChecks=[];
for(const name of ['q','k','v','gate','up','down','norm1','norm2','out','X']) {
  await evaluate(`(() => {const x=document.querySelector('[data-bp-parameter]');x.value=${JSON.stringify(name)};x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  bpChecks.push(await evaluate(`Number(document.querySelector('[data-bp-error]').textContent)`));
}
check("各参数组有限差分核验",bpChecks.every(x=>Number.isFinite(x)&&x<1e-7),JSON.stringify(bpChecks));

await navigate("/attention-kernels.html");
await evaluate(`(() => {for(let i=0;i<6;i++)document.querySelector('[data-ak-next]').click();})()`);
const ak=await evaluate(`({error:Number(document.querySelector('[data-ak-error]').textContent),rows:document.querySelectorAll('[data-ak-rows] tr').length,out:document.querySelector('[data-ak-out]').textContent,done:document.querySelector('[data-ak-next]').disabled})`);
check("分块 attention 完成后等于整行参考",ak.error<1e-12&&ak.rows===3&&ak.done,JSON.stringify(ak));
await evaluate(`(() => {const b=document.querySelector('[data-ak-block]');b.value='1';b.dispatchEvent(new Event('change',{bubbles:true}));const o=document.querySelector('[data-ak-offset]');o.value='1000';o.dispatchEvent(new Event('input',{bubbles:true}));for(let i=0;i<6;i++)document.querySelector('[data-ak-next]').click();})()`);
check("改变分块和平移仍得到相同 attention",await evaluate(`Number(document.querySelector('[data-ak-error]').textContent)<1e-12 && document.querySelectorAll('[data-ak-rows] tr').length===6`));

await navigate("/numerical-computing.html");
const ncBefore=await evaluate(`document.querySelector('[data-nc-sum]').textContent`);
await evaluate(`(() => {const x=document.querySelector('[data-nc-order]');x.value='smallfirst';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
const ncAfter=await evaluate(`document.querySelector('[data-nc-sum]').textContent`);
check("舍入导致求和顺序改变结果",Number(ncBefore)===0&&Number(ncAfter)===1,`${ncBefore} -> ${ncAfter}`);
const ncSoftmax=await evaluate(`({naive:document.querySelector('[data-nc-naive]').textContent,stable:document.querySelector('[data-nc-stable]').textContent})`);
check("大偏移稳定 softmax 保持有限概率",/NaN|未定义|溢出|无效/.test(ncSoftmax.naive)&&!/NaN|Infinity/.test(ncSoftmax.stable),JSON.stringify(ncSoftmax));
await evaluate(`(() => {const x=document.querySelector('[data-nc-value]');x.value='';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
check("清空数值输入不会保留旧舍入结果",await evaluate(`!/^(1|1\.0+)$/.test(document.querySelector('[data-nc-rounded]').textContent)`));

await navigate("/distributed-execution.html");
const dxForward=await evaluate(`Number(document.querySelector('[data-dx-error]').textContent)`);
await evaluate(`(() => {const x=document.querySelector('[data-dx-stage]');x.value='backward';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("两卡前后向均与未分片参考一致",dxForward<1e-12&&await evaluate(`Number(document.querySelector('[data-dx-error]').textContent)<1e-12 && document.querySelector('[data-dx-reference]').textContent===document.querySelector('[data-dx-combined]').textContent`));
await evaluate(`(() => {const x=document.querySelector('[data-dx-collective]');x.value='alltoall';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("AllToAll 按目标片发送并按来源排列",await evaluate(`document.querySelector('[data-dx-collective-rows] tr').lastElementChild.textContent==='[1, 2, 10, 20]'`));

await navigate("/activation-checkpointing.html");
await evaluate(`(() => {for(let i=0;i<60;i++)document.querySelector('[data-ac-next]').click();})()`);
const acStore=await evaluate(`({peak:document.querySelector('[data-ac-peak]').textContent,forward:document.querySelector('[data-ac-forward]').textContent,error:document.querySelector('[data-ac-error]').textContent,loss:document.querySelector('[data-ac-loss]').textContent})`);
await evaluate(`(() => {const x=document.querySelector('[data-ac-mode]');x.value='checkpoint';x.dispatchEvent(new Event('change',{bubbles:true}));for(let i=0;i<60;i++)document.querySelector('[data-ac-next]').click();})()`);
const acRecompute=await evaluate(`({peak:document.querySelector('[data-ac-peak]').textContent,forward:document.querySelector('[data-ac-forward]').textContent,error:document.querySelector('[data-ac-error]').textContent,loss:document.querySelector('[data-ac-loss]').textContent,done:document.querySelector('[data-ac-next]').disabled})`);
check("重算降低本例峰值并保持 loss 和梯度",acStore.peak==='7'&&acStore.forward==='6'&&acRecompute.peak==='5'&&acRecompute.forward==='12'&&acRecompute.error==='0'&&acStore.loss===acRecompute.loss&&acRecompute.done,JSON.stringify({acStore,acRecompute}));
await evaluate(`document.querySelector('[data-ac-reset]').click()`);
check("激活账本可重置",await evaluate(`document.querySelector('[data-ac-forward]').textContent==='0' && document.querySelectorAll('[data-ac-events] tr').length===1`));

await navigate("/decoding-algorithms.html");
await evaluate(`document.querySelector('[data-dec-grammar]').click()`);
const decAllowed=[];
for(const [prefix,expected] of [['','{"ok":'],['{"ok":true','}'],['{"ok":true}','<eos>']]) {
  await evaluate(`(() => {const x=document.querySelector('[data-dec-prefix]');x.value=${JSON.stringify(prefix)};x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  decAllowed.push(await evaluate(`document.querySelector('[data-dec-sampled]').textContent===${JSON.stringify(expected)}`));
}
check("有限语言约束逐前缀只允许合法延伸",decAllowed.every(Boolean));
check("EOS 作为文字显示不被 HTML 吞掉",await evaluate(`[...document.querySelectorAll('[data-dec-rows] code')].some(el=>el.textContent==='<eos>')`));
await evaluate(`(() => {const x=document.querySelector('[data-dec-beam]');x.value='2';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("宽度二搜索保留更优完整路径",await evaluate(`document.querySelector('[data-dec-beam-result]').textContent.includes('Bx') && document.querySelector('[data-dec-beam-result]').textContent.includes('0.396')`));

await navigate("/finetuning-recipe.html");
check("十样本三卡补齐为十二次读取",await evaluate(`document.querySelector('[data-do-reads]').textContent==='12' && document.querySelector('[data-do-unique]').textContent==='10' && document.querySelectorAll('[data-do-rows] tr').length===3`));
await evaluate(`document.querySelector('[data-do-sampler-drop]').click();document.querySelector('[data-do-loader-drop]').click();`);
check("Sampler 和 Loader 丢尾分别生效",await evaluate(`document.querySelector('[data-do-reads]').textContent==='6' && document.querySelector('[data-do-unique]').textContent==='6'`));
await evaluate(`document.querySelector('[data-do-shard]').click()`);
check("未分片 Iterable worker 重复读取全数据",await evaluate(`[...document.querySelectorAll('[data-do-worker-rows] tr')].every(row=>JSON.parse(row.lastElementChild.textContent).length===10)`));
await evaluate(`(() => {const x=document.querySelector('[data-do-count]');x.value='';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
check("数据分发空输入清除旧结果",await evaluate(`document.querySelector('[data-do-reads]').textContent==='-' && document.querySelectorAll('[data-do-rows] tr').length===0`));

await navigate("/data-tokenizer.html");
const encodedBefore=await evaluate(`document.querySelector('[data-bpe-encoded]').textContent`);
await evaluate(`(() => {const x=document.querySelector('[data-bpe-steps]');x.value='0';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
const bpeZero=await evaluate(`({text:document.querySelector('[data-bpe-encoded]').textContent,decoded:document.querySelector('[data-bpe-decoded]').textContent,learned:document.querySelector('[data-bpe-learned]').textContent})`);
check("BPE 零合并保留字符且解码还原",bpeZero.text!==encodedBefore&&bpeZero.decoded==='lowest'&&bpeZero.learned==='11',JSON.stringify(bpeZero));
await evaluate(`(() => {const x=document.querySelector('[data-bpe-steps]');x.value='10';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
check("累计 BPE 词表保留全部合并产物",await evaluate(`document.querySelector('[data-bpe-learned]').textContent==='21' && document.querySelectorAll('[data-bpe-rounds] tr').length===10 && document.querySelector('[data-bpe-rounds]').textContent.includes('</w>')`));

await navigate("/tensor-layout.html");
check("转置共享存储但无法一维无拷贝展平",await evaluate(`document.querySelector('[data-tl-shared]').textContent==='共享' && document.querySelector('[data-tl-view]').textContent.includes('不可行') && document.querySelectorAll('[data-tl-storage] .storage-cell').length===12`));
await evaluate(`(() => {const x=document.querySelector('[data-tl-mode]');x.value='slice';x.dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[data-tl-write]').click();})()`);
check("非连续切片可 stride2 展平，写入影响原存储",await evaluate(`document.querySelector('[data-tl-view]').textContent.includes('步幅 2') && document.querySelector('[data-tl-storage] strong').textContent==='值 99'`));
await evaluate(`document.querySelector('[data-tl-materialize]').click()`);
check("独立复制后写入不改变原存储",await evaluate(`document.querySelector('[data-tl-shared]').textContent==='独立副本' && document.querySelector('[data-tl-storage] strong').textContent==='值 0' && document.querySelector('[data-tl-matrix] span').textContent==='99'`));

await navigate("/distributed-execution.html");
const vocabBefore=await evaluate(`document.querySelector('[data-vp-loss]').textContent`);
await evaluate(`(() => {const s=document.querySelector('[data-vp-shards]');s.value='3';s.dispatchEvent(new Event('change',{bubbles:true}));const o=document.querySelector('[data-vp-offset]');o.value='1000';o.dispatchEvent(new Event('input',{bubbles:true}));})()`);
const vocabAfter=await evaluate(`({loss:document.querySelector('[data-vp-loss]').textContent,error:Number(document.querySelector('[data-vp-error]').textContent),owner:document.querySelector('[data-vp-target-owner]').textContent,rows:document.querySelectorAll('[data-vp-shards-rows] tr').length})`);
check("词表分片和平移保持 CE，目标归属更新",vocabAfter.loss===vocabBefore&&vocabAfter.error<1e-12&&vocabAfter.owner==='进程 2'&&vocabAfter.rows===3,JSON.stringify(vocabAfter));

await navigate("/training-systems-deep.html");
const ppBefore=await evaluate(`({finish:document.querySelector('[data-pp-finish]').textContent,peak:document.querySelector('[data-pp-peak]').textContent})`);
await evaluate(`(() => {const x=document.querySelector('[data-pp-mode]');x.value='1f1b';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
const ppAfter=await evaluate(`({finish:document.querySelector('[data-pp-finish]').textContent,peak:document.querySelector('[data-pp-peak]').textContent,forward:document.querySelectorAll('[data-pp-grid] .pp-forward').length,backward:document.querySelectorAll('[data-pp-grid] .pp-backward').length})`);
check("流水线同工作量改变激活峰值",ppBefore.finish==='12'&&ppBefore.peak==='4 / 4 / 4'&&ppAfter.finish==='12'&&ppAfter.peak==='3 / 2 / 1'&&ppAfter.forward===12&&ppAfter.backward===12,JSON.stringify({ppBefore,ppAfter}));

await navigate("/inference-engine.html");
await evaluate(`(() => {for(let i=0;i<80;i++)document.querySelector('[data-ss-next]').click();})()`);
const ss=await evaluate(`({computed:document.querySelector('[data-ss-computed]').textContent,reused:document.querySelector('[data-ss-reused]').textContent,used:document.querySelectorAll('[data-ss-blocks] .is-used').length,done:document.querySelector('[data-ss-next]').disabled,rows:document.querySelectorAll('[data-ss-rows] tr').length})`);
check("多请求完成并仅保留预热共享块",ss.computed==='18'&&ss.reused==='8'&&ss.used===1&&ss.done&&ss.rows===3,JSON.stringify(ss));
await evaluate(`document.querySelector('[data-ss-prefix]').click();(() => {for(let i=0;i<80;i++)document.querySelector('[data-ss-next]').click();})()`);
check("关闭前缀复用增加计算并释放所有块",await evaluate(`document.querySelector('[data-ss-computed]').textContent==='26' && document.querySelector('[data-ss-reused]').textContent==='0' && document.querySelectorAll('[data-ss-blocks] .is-used').length===0`));
await evaluate(`document.querySelector('[data-ss-reset]').click()`);
check("调度可回到初始并逐步前后查看",await evaluate(`document.querySelector('[data-ss-computed]').textContent==='0' && document.querySelector('[data-ss-prev]').disabled`));

await navigate("/engineering-check.html");
check("综合题须先作答才能展开",await evaluate(`document.querySelector('[data-check-reveal]').disabled && document.querySelector('[data-check-next]').disabled`));
for(const [i,answer] of [1,0,1,1,2,1,1].entries()) {
  await evaluate(`document.querySelector('[name=engineering-answer][value="${answer}"]').click();document.querySelector('[data-check-reveal]').click();`);
  check(`综合题 ${i+1} 答案与依据`,await evaluate(`!document.querySelector('[data-check-explanation]').hidden && document.querySelector('[data-check-explanation]').textContent.includes('这项判断正确') && !!document.querySelector('[data-check-explanation] a')`));
  if(i<6) await evaluate(`document.querySelector('[data-check-next]').click()`);
}
check("综合题结束与累计成绩",await evaluate(`document.querySelector('[data-check-next]').disabled && document.querySelector('[data-check-score]').textContent.includes('其中 7 项')`));
await evaluate(`document.querySelector('[data-check-reset]').click()`);
check("综合题重置清除选择与解答",await evaluate(`document.querySelector('[data-check-progress]').textContent==='1/7' && document.querySelector('[data-check-explanation]').hidden && !document.querySelector('[name=engineering-answer]:checked')`));

await navigate("/position-cache.html");
check("位置与缓存正确路径逐位置一致",await evaluate(`document.querySelector('[data-rc-first]').textContent==='全序列一致' && Number(document.querySelector('[data-rc-error]').textContent)<1e-12`));
await evaluate(`(() => {const x=document.querySelector('[data-rc-fault]');x.value='reset';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
check("后缀位置重置产生实际 logits 偏差",await evaluate(`document.querySelector('[data-rc-first]').textContent==='位置 2' && Number(document.querySelector('[data-rc-error]').textContent)>0.01`));
await evaluate(`(() => {const p=document.querySelector('[data-rc-padding]');p.value='2';const f=document.querySelector('[data-rc-fault]');f.value='mask';f.dispatchEvent(new Event('input',{bubbles:true}));})()`);
check("全屏蔽 query 明确显示无定义",await evaluate(`document.querySelector('[data-rc-error]').textContent==='存在无定义行' && document.querySelector('[data-rc-logits-rows]').textContent.includes('无有效 key')`));

await navigate("/numerical-computing.html");
await evaluate(`(() => {const x=document.querySelector('[data-mp-order]');x.value='clip-first';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("先裁剪再还原暴露错误梯度尺度",await evaluate(`document.querySelector('[data-mp-applied]').textContent==='[0.075, 0.1]'`));
await evaluate(`(() => {const x=document.querySelector('[data-mp-mode]');x.value='overflow';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("非有限梯度跳过一次更新",await evaluate(`document.querySelector('[data-mp-updates]').textContent==='3 / 4' && document.querySelector('[data-mp-update-rows]').textContent.includes('跳过更新')`));

await navigate("/model-artifacts.html");
check("完整 checkpoint 恢复的参数轨迹一致",await evaluate(`document.querySelector('[data-cr-first]').textContent==='轨迹一致' && Number(document.querySelector('[data-cr-error]').textContent)===0`));
for(const omission of ["optimizer","scheduler","rng","data","weights-only"]) {
  await evaluate(`(() => {const x=document.querySelector('[data-cr-omission]');x.value=${JSON.stringify(omission)};x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  check(`遗漏 ${omission} 状态产生恢复偏差`,await evaluate(`document.querySelector('[data-cr-first]').textContent==='第 4 步' && Number(document.querySelector('[data-cr-error]').textContent)>0`));
}

await navigate("/training-systems-deep.html");
check("FSDP 初始账本不预报未来峰值",await evaluate(`document.querySelector('[data-fl-live]').textContent==='8 元素' && document.querySelector('[data-fl-peak]').textContent==='8 元素'`));
await evaluate(`document.querySelector('[data-fl-next]').click()`);
check("FSDP 聚合后增加临时完整参数",await evaluate(`document.querySelector('[data-fl-live]').textContent==='12 元素' && document.querySelector('[data-fl-collectives]').textContent==='1 / 0'`));
await evaluate(`(() => {const x=document.querySelector('[data-fl-event]');x.value=x.lastElementChild.value;x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("FSDP 完整更新保持参考结果并释放梯度",await evaluate(`document.querySelector('[data-fl-live]').textContent==='8 元素' && Number(document.querySelector('[data-fl-error]').textContent)===0 && document.querySelector('[data-fl-next]').disabled`));
check("动态流水线宽表能获得键盘焦点",await evaluate(`document.querySelector('[data-pp-grid]').getAttribute('tabindex')==='0'`));

await navigate("/transformer-anatomy.html");
await evaluate(`(() => {const p=document.querySelector('[data-md-pattern]');p.value='skewed';const c=document.querySelector('[data-md-capacity]');c.value='1';c.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("MoE 容量丢弃后按相同规则合并",await evaluate(`document.querySelector('[data-md-kept]').textContent==='2 / 8' && document.querySelector('[data-md-load]').textContent==='1 / 1 / 0 / 0' && Number(document.querySelector('[data-md-error]').textContent)===0`));

await navigate("/efficient-inference.html");
await evaluate(`(() => {const x=document.querySelector('[data-sp-scenario]');x.value='equal';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("相同草稿分布全部接受且无需修正",await evaluate(`document.querySelector('[data-sp-correction]').textContent==='不需要修正' && document.querySelector('[data-sp-round]').textContent.includes('全部接受')`));
await evaluate(`(() => {const x=document.querySelector('[data-sp-scenario]');x.value='disjoint';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("分离支持集依靠修正恢复目标分布",await evaluate(`Number(document.querySelector('[data-sp-error]').textContent)===0 && document.querySelector('[data-sp-round]').textContent.includes('第 1 个草稿被拒绝')`));

await navigate("/packing-correctness.html");
check("Packing 正确路径保持九个标签与参考梯度", await evaluate(`document.querySelector('[data-pk-count]').textContent==='9' && Number(document.querySelector('[data-pk-gradient]').textContent)<1e-12 && document.querySelectorAll('[data-pk-tokens] tr').length===24`));
await evaluate(`(() => {const x=document.querySelector('[data-pk-mode]');x.value='leaky';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("跨样本可见产生前向和梯度偏差", await evaluate(`Number(document.querySelector('[data-pk-gradient]').textContent)>0.001 && !document.querySelector('[data-pk-isolation]').textContent.includes('0.00e+0')`));
await evaluate(`(() => {const x=document.querySelector('[data-pk-mode]');x.value='double-shift';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("重复 shift 数量相同但标签和梯度不同", await evaluate(`document.querySelector('[data-pk-count]').textContent==='9' && Number(document.querySelector('[data-pk-gradient]').textContent)>0.1 && document.querySelector('[data-pk-note]').textContent.includes('t+2')`));
await evaluate(`(() => {const x=document.querySelector('[data-pk-limit]');x.value='3';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("截断至无有效监督时不显示伪零 loss", await evaluate(`document.querySelector('[data-pk-loss]').textContent==='未定义' && document.querySelector('[data-pk-gradient]').textContent==='无有效目标' && document.querySelectorAll('[data-pk-tokens] tr').length===6`));
await evaluate(`(() => {const m=document.querySelector('[data-pk-mode]');m.value='padded';const l=document.querySelector('[data-pk-limit]');l.value='32';l.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("Padding 保留独立边界与额外空槽", await evaluate(`document.querySelectorAll('[data-pk-tokens] tr').length===26 && Number(document.querySelector('[data-pk-gradient]').textContent)<1e-12 && document.querySelector('[data-pk-isolation]').textContent.includes('0.00e+0')`));

await navigate("/lora-mechanics.html");
check("LoRA 首步梯度与显式参考相同", await evaluate(`document.querySelector('[data-lm-output]').textContent==='[1.8, -1.3]' && document.querySelector('[data-lm-gradients]').textContent.includes('2.25')`));
await evaluate(`(() => {const x=document.querySelector('[data-lm-steps]');x.value='1';x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
check("LoRA 一步更新与合并输出一致", await evaluate(`document.querySelector('[data-lm-output]').textContent==='[1.51875, -0.940625]' && Number(document.querySelector('[data-lm-merge]').textContent)<1e-12`));
await evaluate(`(() => {const x=document.querySelector('[data-lm-precision]');x.value='int4';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("量化底座与再次量化误差分别统计", await evaluate(`Number(document.querySelector('[data-lm-base-error]').textContent)>0 && Number(document.querySelector('[data-lm-requantize]').textContent)>0 && Number(document.querySelector('[data-lm-merge]').textContent)<1e-12`));
await evaluate(`(() => {const x=document.querySelector('[data-lm-graph]');x.value='detach-all';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("切断整层后无 A/B 梯度且未执行更新", await evaluate(`document.querySelector('[data-lm-updates]').textContent==='0' && document.querySelector('[data-lm-gradients]').textContent.includes('无计算图')`));

await navigate("/parallel-correctness.html");
const globalContrastiveLoss = await evaluate(`document.querySelector('[data-cd-loss]').textContent`);
await evaluate(`(() => {const x=document.querySelector('[data-cd-mode]');x.value='detached';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("跨卡 detach 保持前向但丢失远端梯度", await evaluate(`document.querySelector('[data-cd-loss]').textContent===${JSON.stringify(globalContrastiveLoss)} && document.querySelector('[data-cd-gradients]').textContent.includes('-0.059601') && Number(document.querySelector('[data-cd-error]').textContent)>0.9`));
await evaluate(`(() => {const x=document.querySelector('[data-cd-mode]');x.value='micro';x.dispatchEvent(new Event('change',{bubbles:true}));})()`);
check("单样本 microbatch 累积不能恢复全局负例", await evaluate(`document.querySelector('[data-cd-loss]').textContent==='0' && document.querySelector('[data-cd-scores]').textContent.includes('未进入候选')`));

await navigate("/attention-kernels.html");
check("分块反传完成前不显示最终一致", await evaluate(`document.querySelector('[data-ab-error]').textContent==='反向尚未完成' && document.querySelector('[data-ab-prev]').disabled`));
await evaluate(`(() => {const x=document.querySelector('[data-ab-next]');for(let i=0;i<30&&!x.disabled;i++)x.click();})()`);
check("分块反传完整 dQ/dK/dV 与参考一致", await evaluate(`Number(document.querySelector('[data-ab-error]').textContent)<1e-12 && document.querySelectorAll('[data-ab-events] tr').length===9`));
await evaluate(`(() => {const f=document.querySelector('[data-ab-fault]');f.value='overwrite-kv';f.dispatchEvent(new Event('change'));const x=document.querySelector('[data-ab-next]');for(let i=0;i<30&&!x.disabled;i++)x.click();})()`);
check("覆盖共享 K/V 梯度造成实际偏差", await evaluate(`Number(document.querySelector('[data-ab-error]').textContent)>0.1 && Number(document.querySelector('[data-ab-forward]').textContent)<1e-12`));
await evaluate(`(() => {const f=document.querySelector('[data-ab-fault]');f.value='none';const m=document.querySelector('[data-ab-mask]');m.value='empty-first';m.dispatchEvent(new Event('change'));const x=document.querySelector('[data-ab-next]');for(let i=0;i<30&&!x.disabled;i++)x.click();})()`);
check("首个 query 全屏蔽时采用显式零行约定", await evaluate(`Number(document.querySelector('[data-ab-error]').textContent)<1e-12 && document.querySelector('[data-ab-stats]').textContent.includes('空行') && !document.querySelector('[data-ab-gradients]').textContent.includes('NaN')`));

await navigate("/activation-checkpointing.html");
check("梯度缓存第一阶段未产生全局 loss", await evaluate(`document.querySelector('[data-gc-loss]').textContent==='尚未计算全局 loss'`));
await evaluate(`(() => {const x=document.querySelector('[data-gc-next]');for(let i=0;i<30&&!x.disabled;i++)x.click();})()`);
const gradientCacheLoss = await evaluate(`document.querySelector('[data-gc-loss]').textContent`);
check("梯度缓存重放完成保持相同梯度", await evaluate(`Number(document.querySelector('[data-gc-error]').textContent)<1e-12 && document.querySelector('[data-gc-work]').textContent.includes('已编码 16 行')`));
await evaluate(`(() => {const f=document.querySelector('[data-gc-fault]');f.value='fresh-rng';f.dispatchEvent(new Event('change'));const x=document.querySelector('[data-gc-next]');for(let i=0;i<30&&!x.disabled;i++)x.click();})()`);
check("新的 dropout 随机轨迹改变梯度但保留缓存 loss", await evaluate(`Number(document.querySelector('[data-gc-error]').textContent)>0.01 && document.querySelector('[data-gc-loss]').textContent===${JSON.stringify(gradientCacheLoss)}`));
await evaluate(`(() => {const f=document.querySelector('[data-gc-dropout]');f.value='0';f.dispatchEvent(new Event('change'));const x=document.querySelector('[data-gc-next]');for(let i=0;i<30&&!x.disabled;i++)x.click();})()`);
check("关闭 dropout 后随机重放故障不再改变梯度", await evaluate(`Number(document.querySelector('[data-gc-error]').textContent)<1e-12`));

await navigate("/parallel-correctness.html");
check("分片 [3,4] 使用全局范数五与统一系数", await evaluate(`document.querySelector('[data-sc-norm]').textContent==='5' && document.querySelector('[data-sc-output]').textContent==='1'`));
await evaluate(`(() => {const x=document.querySelector('[data-sc-mode]');x.value='local';x.dispatchEvent(new Event('change'));})()`);
check("逐卡裁剪违反全局阈值", await evaluate(`document.querySelector('[data-sc-output]').textContent==='1.414214'`));
await evaluate(`(() => {document.querySelector('[data-sc-mode]').value='global';const x=document.querySelector('[data-sc-scenario]');x.value='mixed';x.dispatchEvent(new Event('change'));})()`);
check("复制参数按唯一拥有者计数", await evaluate(`document.querySelector('[data-sc-norm]').textContent==='13' && document.querySelector('[data-sc-squares]').textContent.includes('独立元素 3')`));
await evaluate(`(() => {const x=document.querySelector('[data-sc-scenario]');x.value='overflow';x.dispatchEvent(new Event('change'));})()`);
check("单卡非有限梯度导致两卡统一跳步", await evaluate(`document.querySelector('[data-sc-decision]').textContent==='两卡均跳步'`));
await evaluate(`(() => {const x=document.querySelector('[data-sc-mode]');x.value='local-skip';x.dispatchEvent(new Event('change'));})()`);
check("局部跳步明确显示更新分叉", await evaluate(`document.querySelector('[data-sc-decision]').textContent.includes('分叉')`));

await navigate("/performance-analysis.html");
await evaluate(`(() => {const transfer=new DataTransfer();transfer.items.add(new File(['{}'],'invalid-report.json',{type:'application/json'}));const input=document.querySelector('[data-mr-file]');input.files=transfer.files;input.dispatchEvent(new Event('change'));})()`);
await wait(100);
check("报告读取拒绝缺失配置且不展示旧数据", await evaluate(`document.querySelector('[data-mr-output]').hidden && document.querySelector('[data-mr-status]').textContent.includes('无法读取')`));

const overflow = [];
for (const path of pageConfig.map(page=>`/${page.file}`)) {
  await navigate(path);
  const layout = await evaluate(`({
    width: window.innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    theme: document.documentElement.dataset.theme
  })`);
  if (layout.scrollWidth > layout.width) overflow.push({ path, ...layout });
}
check(`${pageConfig.length} 页 390px 无横向溢出`, overflow.length === 0, JSON.stringify(overflow));
check("浏览器运行无未捕获异常", runtimeErrors.length === 0, JSON.stringify(runtimeErrors));

for (const result of results) {
  console.log(`${result.ok ? "PASS" : "FAIL"}  ${result.name}${result.detail ? `  ${result.detail}` : ""}`);
}

socket.close();
await fetch(`http://127.0.0.1:${port}/json/close/${target.id}`);

if (results.some((result) => !result.ok)) process.exitCode = 1;
