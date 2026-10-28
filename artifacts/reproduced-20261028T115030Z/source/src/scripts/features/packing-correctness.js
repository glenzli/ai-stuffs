import { $ } from "../core.js";
import { zeros, transpose, matmul, matrixMap, matrixAdd, dot, softmax, crossEntropy, maxDifference, formatValue, escapeText } from "./objective-math.js";

export const packingVocabulary = ["PAD", "BOS", "USER", "ASSISTANT", "EOS", "问一", "答一", "问二", "答二", "问三", "答三", "改写"];
export function packingRecords(changeA = false) {
  return [
    { id: "A", messages: [{ role: "user", ids: [changeA ? 11 : 5] }, { role: "assistant", ids: [6] }, { role: "user", ids: [7] }, { role: "assistant", ids: [8] }] },
    { id: "B", messages: [{ role: "user", ids: [9, 5] }, { role: "assistant", ids: [10] }, { role: "user", ids: [7] }, { role: "assistant", ids: [8, 10] }] },
  ];
}
export function packingParameters() {
  return {
    embedding: [[0, 0], [.1, .2], [-.2, .3], [.4, -.1], [.1, -.3], [.5, .2], [-.4, .6], [.2, -.5], [.7, .1], [-.3, -.2], [.3, .5], [-.7, .8]],
    q: [[.3, -.2], [.4, .5]], k: [[.2, .6], [-.3, .4]], v: [[.7, .1], [-.2, .5]],
    out: [[.2, -.1, .4, .1, -.3, .5, -.2, .3, -.4, .6, .2, -.5], [-.3, .4, .1, -.2, .5, -.1, .6, -.4, .2, .1, -.5, .3]],
  };
}
export function packingTokens({ limit = 32, supervise = "all", changeA = false } = {}) {
  if (!Number.isInteger(limit) || limit < 1 || limit > 32 || !["all", "last"].includes(supervise) || typeof changeA !== "boolean") throw new RangeError("序列上限、监督范围或改写开关无效。");
  return packingRecords(changeA).map(record => {
    const last = record.messages.map(message => message.role).lastIndexOf("assistant");
    const tokens = [{ id: 1, role: "boundary", supervised: false }];
    record.messages.forEach((message, index) => {
      tokens.push({ id: message.role === "user" ? 2 : 3, role: "boundary", supervised: false });
      const supervised = message.role === "assistant" && (supervise === "all" || index === last);
      message.ids.forEach(id => tokens.push({ id, role: message.role, supervised }));
      if (message.role === "assistant") tokens.push({ id: 4, role: "assistant", supervised });
    });
    return tokens.slice(0, limit).map((token, position) => ({ ...token, record: record.id, position, valid: true, key: `${record.id}:${position}` }));
  });
}

function runSequence(rows, parameters, { crossRecord = false, shift = 1, denominator } = {}) {
  const { embedding, q, k, v, out } = parameters, count = rows.length;
  const X = rows.map(row => row.valid ? [embedding[row.id][0] + .04 * row.position, embedding[row.id][1] - .03 * row.position] : [0, 0]);
  const Q = matmul(X, q), K = matmul(X, k), V = matmul(X, v);
  const mask = rows.map((row, i) => rows.map((key, j) => row.valid && key.valid && j <= i && (crossRecord || row.record === key.record)));
  const probabilities = rows.map((row, i) => row.valid ? softmax(K.map((key, j) => mask[i][j] ? dot(Q[i], key) / Math.sqrt(2) : -Infinity)) : Array(count).fill(0));
  const H = matrixAdd(X, matmul(probabilities, V)), logits = matmul(H, out);
  const labels = rows.map((row, i) => {
    const target = rows[i + shift];
    return row.valid && target?.valid && row.record === target.record && target.supervised ? target.id : null;
  });
  const supervisedCount = labels.filter(label => label !== null).length;
  const N = denominator ?? supervisedCount;
  const lossSum = labels.reduce((sum, label, i) => sum + (label === null ? 0 : crossEntropy(logits[i], label)), 0);
  const result = { rows, X, mask, probabilities, logits, labels, supervisedCount, lossSum, loss: N ? lossSum / N : null, gradients: null };
  if (!N) return result;
  const dLogits = logits.map((row, i) => labels[i] === null ? Array(out[0].length).fill(0) : softmax(row).map((value, j) => (value - Number(j === labels[i])) / N));
  const gradients = { out: matmul(transpose(H), dLogits) };
  const dH = matmul(dLogits, transpose(out)), dP = matmul(dH, transpose(V)), dV = matmul(transpose(probabilities), dH);
  const dScores = probabilities.map((row, i) => { const weighted = dot(row, dP[i]); return row.map((value, j) => value * (dP[i][j] - weighted) / Math.sqrt(2)); });
  const dQ = matmul(dScores, K), dK = matmul(transpose(dScores), Q);
  gradients.q = matmul(transpose(X), dQ); gradients.k = matmul(transpose(X), dK); gradients.v = matmul(transpose(X), dV);
  const dX = matrixAdd(dH, matrixAdd(matmul(dQ, transpose(q)), matrixAdd(matmul(dK, transpose(k)), matmul(dV, transpose(v)))));
  gradients.embedding = zeros(embedding.length, 2);
  rows.forEach((row, i) => { if (row.valid) for (let j = 0; j < 2; j++) gradients.embedding[row.id][j] += dX[i][j]; });
  return { ...result, gradients, inputGradients: dX };
}

export function packingExperiment({ mode = "packed", limit = 32, supervise = "all", changeA = false, parameters = packingParameters() } = {}) {
  if (!["packed", "padded", "separate", "leaky", "double-shift"].includes(mode)) throw new RangeError("未知的布局或故障。");
  const records = packingTokens({ limit, supervise, changeA });
  const shift = mode === "double-shift" ? 2 : 1;
  const denominator = records.reduce((sum, record) => sum + record.filter((token, index) => index >= shift && token.supervised).length, 0);
  let groups;
  if (mode === "separate") groups = records;
  else if (mode === "padded") {
    const width = Math.max(...records.map(record => record.length));
    groups = [records.flatMap(record => [...record, ...Array.from({ length: width - record.length }, (_, i) => ({ id: 0, role: "padding", record: record[0].record, position: record.length + i, valid: false, supervised: false, key: `${record[0].record}:pad${i}` }))])];
  } else groups = [records.flat()];
  const runs = groups.map(rows => runSequence(rows, parameters, { denominator, shift, crossRecord: mode === "leaky" }));
  const outputs = runs.flatMap(run => run.rows.map((row, i) => ({ ...row, label: run.labels[i], logits: run.logits[i] }))).filter(row => row.valid);
  const gradients = denominator ? Object.fromEntries(Object.keys(parameters).map(name => [name, parameters[name].map((row, i) => row.map((_, j) => runs.reduce((sum, run) => sum + run.gradients[name][i][j], 0)))])) : null;
  return { mode, limit, supervise, changeA, records, runs, denominator, outputs, gradients, loss: denominator ? runs.reduce((sum, run) => sum + run.lossSum, 0) / denominator : null };
}

export function packingComparison(options = {}) {
  const reference = packingExperiment({ ...options, mode: "separate" });
  const variants = ["separate", "padded", "packed", "leaky", "double-shift"].map(mode => {
    const result = packingExperiment({ ...options, mode });
    return { ...result, logitError: maxDifference(result.outputs.map(row => row.logits), reference.outputs.map(row => row.logits)),
      gradientError: result.gradients && reference.gradients ? Math.max(...Object.keys(result.gradients).map(name => maxDifference(result.gradients[name], reference.gradients[name]))) : null };
  });
  return { reference, variants };
}

const modeNames = { separate: "分别处理", padded: "右侧 padding", packed: "独立样本 packing", leaky: "错误：跨样本可见", "double-shift": "错误：重复 shift" };
export function initPackingCorrectness() {
  const mode = $("[data-pk-mode]"); if (!mode) return;
  const limit = $("[data-pk-limit]"), supervise = $("[data-pk-supervise]"), change = $("[data-pk-change]"), query = $("[data-pk-query]");
  const set = (name, value) => { $(`[data-pk-${name}]`).textContent = value; };
  const render = () => {
    const options = { limit: Number(limit.value), supervise: supervise.value, changeA: change.checked };
    const { variants } = packingComparison(options), result = variants.find(item => item.mode === mode.value);
    set("count", result.denominator); set("loss", formatValue(result.loss)); set("gradient", result.gradientError === null ? "无有效目标" : result.gradientError.toExponential(2));
    $("[data-pk-comparisons]").innerHTML = variants.map(item => `<tr${item.mode === mode.value ? ' class="objective-selected"' : ""}><th scope="row">${modeNames[item.mode]}</th><td>${item.denominator}</td><td>${formatValue(item.loss)}</td><td>${item.logitError.toExponential(2)}</td><td>${item.gradientError === null ? "—" : item.gradientError.toExponential(2)}</td></tr>`).join("");
    const unchanged = packingExperiment({ ...options, mode: mode.value, changeA: !change.checked });
    const bError = maxDifference(result.outputs.filter(row => row.record === "B").map(row => row.logits), unchanged.outputs.filter(row => row.record === "B").map(row => row.logits));
    set("isolation", `只替换 A 第一轮的一个 user token，B 的最大 logits 差为 ${bError.toExponential(2)}。${options.limit < 3 ? "当前截断未保留被改写位置。" : ""}`);
    const visible = result.runs.flatMap((run, runIndex) => run.rows.map((row, index) => ({ ...row, run, runIndex, index })));
    query.max = String(visible.length - 1); query.value = String(Math.min(Number(query.value), visible.length - 1));
    const selected = visible[Number(query.value)];
    set("query-value", `${selected.record}:${selected.position}${selected.valid ? "" : " PAD"}`);
    $("[data-pk-tokens]").innerHTML = visible.map((row, slot) => `<tr><td>${slot}</td><td>${row.key}</td><td>${row.position}</td><td>${row.id} / ${escapeText(packingVocabulary[row.id])}</td><td>${row.supervised ? "是" : "否"}</td><td>${row.run.labels[row.index] === null ? "—" : `${row.run.labels[row.index]} / ${escapeText(packingVocabulary[row.run.labels[row.index]])}`}</td></tr>`).join("");
    $("[data-pk-attention]").innerHTML = selected.run.rows.map((row, j) => `<tr><td>${row.key}</td><td>${escapeText(packingVocabulary[row.id])}</td><td class="${selected.run.mask[selected.index][j] ? "objective-allowed" : "objective-muted"}">${selected.run.mask[selected.index][j] ? "可见" : "屏蔽"}</td><td>${formatValue(selected.run.probabilities[selected.index][j])}</td></tr>`).join("");
    set("note", !result.denominator ? "本次截断与监督规则没有保留有效标签。平均 loss 未定义，应跳过优化器更新；表格仍可检查前向与可见范围。" : mode.value === "leaky" ? "位置编号已在每条记录内重置，B 仍可读取 A：位置规则不能代替 Attention 隔离。" : mode.value === "double-shift" ? "这里用位置 t 预测 t+2。前向 logits 与正确 packing 相同，监督落点和梯度发生变化。" : "三个正确布局使用相同的全局有效标签分母；分别处理时，每条记录的 loss sum 按同一个分母累加。比较的是有效位置，不统计 padding 输出。");
  };
  [mode, limit, supervise, change, query].forEach(input => { input.addEventListener("input", render); input.addEventListener("change", render); });
  render();
}
