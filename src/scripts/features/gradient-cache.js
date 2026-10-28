import { $ } from "../core.js";
import { zeros, matmul, transpose, matrixMap, matrixAdd, softmax, crossEntropy, maxDifference, formatValue, formatVector } from "./objective-math.js";

export function gradientCacheInputs() {
  return { query: [[.2, .5, -.1], [.6, -.3, .4], [-.2, .8, .7], [.3, .1, -.5]], document: [[.1, .7, -.2], [.3, -.1, .6], [-.3, .5, .9], [.7, .1, -.4]],
    W1: [[.3, -.2, .5, .1], [-.4, .6, .2, -.3], [.1, .3, -.5, .4]], W2: [[.5, -.2], [.1, .6], [-.4, .3], [.2, .5]], seed: 31 };
}
function uniform(state) { state.value = (Math.imul(state.value, 1664525) + 1013904223) >>> 0; return state.value / 4294967296; }
export function cachedEncoder(X, parameters, dropout, state) {
  const H = matrixMap(matmul(X, parameters.W1), Math.tanh);
  const mask = matrixMap(H, () => dropout === 0 ? 1 : uniform(state) >= dropout ? 1 / (1 - dropout) : 0);
  const Z = matrixMap(H, (value, i, j) => value * mask[i][j]);
  return { X, H, mask, Z, output: matmul(Z, parameters.W2) };
}
function encoderBackward(cache, upstream, parameters) {
  const dZ = matmul(upstream, transpose(parameters.W2));
  const dPre = matrixMap(cache.H, (value, i, j) => dZ[i][j] * cache.mask[i][j] * (1 - value * value));
  return { W1: matmul(transpose(cache.X), dPre), W2: matmul(transpose(cache.Z), upstream) };
}
export function cachedContrastive(query, document) {
  const scores = matmul(query, transpose(document)), P = scores.map(softmax), n = query.length;
  const dS = P.map((row, i) => row.map((value, j) => (value - Number(i === j)) / n));
  return { loss: scores.reduce((sum, row, i) => sum + crossEntropy(row, i) / n, 0), scores, P, dq: matmul(dS, document), dd: matmul(transpose(dS), query) };
}
export function gradientCacheExperiment({ chunk = 2, dropout = .25, fault = "none", inputs = gradientCacheInputs() } = {}) {
  if (!Number.isInteger(chunk) || chunk < 1 || chunk > 4 || ![0, .25, .5].includes(dropout) || !["none", "fresh-rng"].includes(fault)) throw new RangeError("分块、dropout 或重算模式无效。");
  const n = inputs.query.length, chunks = [], state = { value: inputs.seed }, features = { query: [], document: [] }, events = [];
  for (const role of ["query", "document"]) for (let start = 0; start < n; start += chunk) {
    const X = inputs[role].slice(start, start + chunk), rngBefore = state.value, cache = cachedEncoder(X, inputs, dropout, state);
    chunks.push({ role, start, rngBefore, cache }); features[role].push(...cache.output);
    events.push({ stage: "encode", role, start, count: X.length, rngBefore, forwardRows: features.query.length + features.document.length });
  }
  const rngAfterFirstPass = state.value, global = cachedContrastive(features.query, features.document);
  const reference = { W1: zeros(3, 4), W2: zeros(4, 2) }, actual = { W1: zeros(3, 4), W2: zeros(4, 2) };
  events.push({ stage: "global", forwardRows: 2 * n });
  let replayError = 0, forwardRows = 2 * n;
  for (const part of chunks) {
    const upstream = (part.role === "query" ? global.dq : global.dd).slice(part.start, part.start + part.cache.X.length);
    const ref = encoderBackward(part.cache, upstream, inputs);
    const replayState = fault === "none" ? { value: part.rngBefore } : state;
    const before = replayState.value, replay = cachedEncoder(part.cache.X, inputs, dropout, replayState), contribution = encoderBackward(replay, upstream, inputs);
    replayError = Math.max(replayError, maxDifference(replay.output, part.cache.output)); forwardRows += part.cache.X.length;
    for (const name of ["W1", "W2"]) { reference[name] = matrixAdd(reference[name], ref[name]); actual[name] = matrixAdd(actual[name], contribution[name]); }
    events.push({ stage: "replay", role: part.role, start: part.start, count: part.cache.X.length, rngBefore: before, expectedRng: part.rngBefore, forwardRows, contribution, gradients: structuredClone(actual) });
  }
  const gradientError = Math.max(...["W1", "W2"].map(name => maxDifference(actual[name], reference[name])));
  // Count H, dropout mask and Z as three logical hidden tensors per row.
  // Input/weight tensors, copies, debugger snapshots and workspaces are excluded.
  return { inputs, chunk, dropout, fault, chunks, features, global, actual, reference, events, replayError, gradientError, rngAfterFirstPass,
    rngAfterReplay: fault === "none" ? rngAfterFirstPass : state.value,
    storage: { baselineHidden: 2 * n * 4 * 3, cachedHidden: Math.min(chunk, n) * 4 * 3, representations: 2 * n * 2, representationGradients: 2 * n * 2, scoreMatrix: n * n, rngStates: chunks.length }, baselineForwardRows: 2 * n, cachedForwardRows: 4 * n };
}
export function initGradientCache() {
  const chunk = $("[data-gc-chunk]"); if (!chunk) return;
  const dropout = $("[data-gc-dropout]"), fault = $("[data-gc-fault]"); let step = 0;
  const set = (name, value) => { $(`[data-gc-${name}]`).textContent = value; };
  const render = () => {
    const r = gradientCacheExperiment({ chunk: Number(chunk.value), dropout: Number(dropout.value), fault: fault.value }); step = Math.min(step, r.events.length);
    const shown = r.events.slice(0, step), current = shown.at(-1), lastReplay = shown.findLast(event => event.gradients), complete = step === r.events.length;
    set("step", `${step}/${r.events.length}`); set("loss", step > r.chunks.length ? formatValue(r.global.loss) : "尚未计算全局 loss"); set("error", complete ? r.gradientError.toExponential(2) : "反向尚未完成");
    set("storage", `编码器隐藏量：完整保留 ${r.storage.baselineHidden}，分块反传最多 ${r.storage.cachedHidden} 元素。另有表示 ${r.storage.representations}、表示梯度 ${r.storage.representationGradients}、全局分数 ${r.storage.scoreMatrix} 元素及 ${r.storage.rngStates} 份 RNG 状态。`);
    set("work", `已编码 ${current?.forwardRows || 0} 行；完整保留图 ${r.baselineForwardRows} 行，完整缓存流程 ${r.cachedForwardRows} 行。`);
    $("[data-gc-events]").innerHTML = shown.map((e, i) => `<tr><td>${i + 1}</td><td>${{ encode: "无图编码", global: "全局损失 / 表示梯度", replay: "重算 / 参数反传" }[e.stage]}</td><td>${e.role ? `${e.role} ${e.start}…${e.start + e.count - 1}` : "全部 4×4 候选"}</td><td>${e.rngBefore ?? "—"}</td></tr>`).join("");
    $("[data-gc-gradients]").innerHTML = ["W1", "W2"].flatMap(name => r.actual[name].map((_, i) => `<tr><th scope="row">${name}[${i}]</th><td>${formatVector(lastReplay ? lastReplay.gradients[name][i] : Array(r.actual[name][i].length).fill(0))}</td><td>${formatVector(r.reference[name][i])}</td></tr>`)).join("");
    set("note", complete ? `重算表示最大差 ${r.replayError.toExponential(2)}；参数梯度最大差 ${r.gradientError.toExponential(2)}。${r.fault === "fresh-rng" ? "缓存阶段 loss 没变，重算却使用了新的随机轨迹；关闭 dropout 时这个故障不会造成差异。" : "每块恢复自己的前向随机状态，结束后保留第一遍编码完成时的全局 RNG，避免重算额外推进下一步随机序列。"}` : "先编码全部样本，再统一计算候选损失与表示梯度，最后逐块重算并累加参数梯度；中间没有 optimizer.step。");
    $("[data-gc-next]").disabled = complete; $("[data-gc-prev]").disabled = step === 0;
  };
  [chunk, dropout, fault].forEach(input => input.addEventListener("change", () => { step = 0; render(); }));
  $("[data-gc-next]").addEventListener("click", () => { step++; render(); }); $("[data-gc-prev]").addEventListener("click", () => { step = Math.max(0, step - 1); render(); }); $("[data-gc-reset]").addEventListener("click", () => { step = 0; render(); }); render();
}
