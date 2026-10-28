import { $ } from "../core.js";

export function vocabParallel({ shards = 2, target = 4, offset = 0 } = {}) {
  if (!Number.isInteger(shards) || ![1, 2, 3].includes(shards)) {
    throw new RangeError("分片数须为 1、2 或 3。");
  }
  if (!Number.isInteger(target) || target < 0 || target > 5) {
    throw new RangeError("目标词元须为 0 到 5 的整数。");
  }
  if (!Number.isFinite(offset) || Math.abs(offset) > 1000) {
    throw new RangeError("偏移须为 -1000 到 1000 之间的有限数值。");
  }
  const scores = [1.2, -0.5, 2, 0.1, 1.4, -1].map(value => value + offset);
  const width = scores.length / shards;
  const localRows = Array.from({ length: shards }, (_, rank) => {
    const start = rank * width, end = start + width, logits = scores.slice(start, end);
    return { rank, start, end, logits, localMax: Math.max(...logits) };
  });
  const globalMax = Math.max(...localRows.map(row => row.localMax));
  const rows = localRows.map(row => ({
    ...row,
    localSum: row.logits.reduce((sum, logit) => sum + Math.exp(logit - globalMax), 0),
    targetContribution: target >= row.start && target < row.end ? scores[target] - globalMax : 0,
  }));
  const denominator = rows.reduce((sum, row) => sum + row.localSum, 0);
  const targetShifted = rows.reduce((sum, row) => sum + row.targetContribution, 0);
  const loss = Math.log(denominator) - targetShifted;
  for (const row of rows) {
    row.probabilities = row.logits.map(logit => Math.exp(logit - globalMax) / denominator);
    row.gradients = row.probabilities.map((probability, index) => probability - (row.start + index === target ? 1 : 0));
  }
  const probabilities = rows.flatMap(row => row.probabilities);
  const gradients = rows.flatMap(row => row.gradients);
  const referenceMax = Math.max(...scores);
  const referenceSum = scores.reduce((sum, score) => sum + Math.exp(score - referenceMax), 0);
  const reference = Math.log(referenceSum) - (scores[target] - referenceMax);
  const error = Math.abs(loss - reference);
  return { globalMax, denominator, targetShifted, loss, reference, error, rows, probabilities, gradients };
}

export function initVocabParallel() {
  const shards = $("[data-vp-shards]");
  if (!shards) return;
  const target = $("[data-vp-target]"), offset = $("[data-vp-offset]");
  const format = value => value.toFixed(6);
  const set = (name, value) => {
    const node = $(`[data-vp-${name}]`);
    if (node) node.textContent = String(value);
  };
  const render = () => {
    const targetIndex = Number(target?.value ?? 4), shift = Number(offset?.value ?? 0);
    const result = vocabParallel({ shards: Number(shards.value), target: targetIndex, offset: shift });
    const width = 6 / result.rows.length, owner = Math.floor(targetIndex / width);
    set("offset-value", shift);
    set("max", format(result.globalMax));
    set("denom", format(result.denominator));
    set("loss", format(result.loss));
    set("target-owner", `进程 ${owner}`);
    set("error", result.error === 0 ? "0" : result.error.toExponential(3));
    const shardRows = $("[data-vp-shards-rows]");
    if (shardRows) shardRows.innerHTML = result.rows.map(row =>
      `<tr><td>进程 ${row.rank}</td><td>${row.start}-${row.end - 1}</td><td>[${row.logits.map(format).join(", ")}]</td><td>${format(row.localMax)}</td><td>${format(row.localSum)}</td><td>${format(row.targetContribution)}</td></tr>`
    ).join("");
    const gradRows = $("[data-vp-grad-rows]");
    if (gradRows) gradRows.innerHTML = result.probabilities.map((probability, token) =>
      `<tr><td>${token}</td><td>进程 ${Math.floor(token / width)}</td><td>${format(probability)}</td><td>${format(result.gradients[token])}</td></tr>`
    ).join("");
    set("note", `各分片先归约全局最大值，再求平移后的局部指数和与目标项；全局分母用于所有词元的概率和梯度。目标词元 ${targetIndex} 属于进程 ${owner}，其平移目标项为 ${format(result.targetShifted)}，其他进程贡献 0。损失为对数全局分母减去平移目标项，梯度为概率减去目标指示值。这里只执行固定数值的数组运算，未执行 NCCL 通信或训练。`);
  };
  for (const control of [shards, target, offset]) {
    for (const event of ["input", "change"]) control?.addEventListener(event, render);
  }
  render();
}
