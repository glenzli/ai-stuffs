import { $ } from "../core.js";
import { softmax, crossEntropy, formatValue } from "./objective-math.js";

export function contrastiveCalculation({ mode = "global", temperature = 1, query = [1, 2], document = [1, -1] } = {}) {
  if (!["global", "detached", "micro"].includes(mode) || !Number.isFinite(temperature) || temperature <= 0 || query.length !== 2 || document.length !== 2 || ![...query, ...document].every(Number.isFinite)) throw new RangeError("需要两个有限表示、正温度与已知梯度方案。");
  const scores = query.map(q => document.map(d => q * d / temperature));
  const probabilities = scores.map((row, i) => mode === "micro" ? row.map((_, j) => Number(i === j)) : softmax(row));
  const localLosses = scores.map((row, i) => mode === "micro" ? 0 : crossEntropy(row, i)), loss = (localLosses[0] + localLosses[1]) / 2;
  const scoreGradients = probabilities.map((row, i) => row.map((p, j) => (p - Number(i === j)) / (2 * temperature)));
  const queryGradients = scoreGradients.map(row => row.reduce((sum, gradient, j) => sum + gradient * document[j], 0));
  const contributions = scoreGradients.map((row, i) => row.map(gradient => gradient * query[i]));
  const documentGradients = document.map((_, j) => contributions.reduce((sum, row, i) => sum + (mode === "detached" && i !== j ? 0 : row[j]), 0));
  // q_i = wq * source_q_i and d_i = wd * source_d_i, evaluated at wq=wd=1.
  // Each rank differentiates its unscaled local loss. Gather backward sums,
  // then DDP averages shared parameter gradients across the two owners.
  const ranks = query.map((source, rank) => ({ rank, localLoss: localLosses[rank], rawQueryParameter: 2 * queryGradients[rank] * source, rawDocumentParameter: 2 * documentGradients[rank] * document[rank] }));
  const parameterGradients = { query: ranks.reduce((sum, rank) => sum + rank.rawQueryParameter, 0) / 2, document: ranks.reduce((sum, rank) => sum + rank.rawDocumentParameter, 0) / 2 };
  return { mode, temperature, query, document, scores, probabilities, localLosses, loss, contributions, queryGradients, documentGradients, parameterGradients, ranks };
}
export function initContrastiveDistributed() {
  const mode = $("[data-cd-mode]"); if (!mode) return;
  const temperature = $("[data-cd-temperature]");
  const set = (name, value) => { $(`[data-cd-${name}]`).textContent = value; };
  const render = () => {
    const result = contrastiveCalculation({ mode: mode.value, temperature: Number(temperature.value) }), reference = contrastiveCalculation({ temperature: Number(temperature.value) });
    set("loss", formatValue(result.loss)); set("reference", formatValue(reference.loss));
    set("error", Math.max(...result.documentGradients.map((value, i) => Math.abs(value - reference.documentGradients[i]))).toExponential(2));
    $("[data-cd-scores]").innerHTML = result.scores.map((row, i) => `<tr><th scope="row">q${i}=${result.query[i]}</th>${row.map((value, j) => `<td>${mode.value === "micro" && i !== j ? "未进入候选" : `${formatValue(value)} / ${formatValue(result.probabilities[i][j])}`}${i === j ? "（正例）" : ""}</td>`).join("")}</tr>`).join("");
    $("[data-cd-gradients]").innerHTML = [0, 1].map(j => `<tr><th scope="row">d${j}</th><td>${formatValue(result.contributions[0][j])}${mode.value === "detached" && j !== 0 ? "（被丢弃）" : ""}</td><td>${formatValue(result.contributions[1][j])}${mode.value === "detached" && j !== 1 ? "（被丢弃）" : ""}</td><td>${formatValue(result.documentGradients[j])}</td><td>${formatValue(reference.documentGradients[j])}</td></tr>`).join("");
    $("[data-cd-ranks]").innerHTML = result.ranks.map(rank => `<tr><th scope="row">Rank ${rank.rank}</th><td>${formatValue(rank.localLoss)}</td><td>${formatValue(rank.rawQueryParameter)}</td><td>${formatValue(rank.rawDocumentParameter)}</td></tr>`).join("");
    set("parameters", `DDP 平均后的共享参数梯度：wq=${formatValue(result.parameterGradients.query)}，wd=${formatValue(result.parameterGradients.document)}。完整目标参考：wq=${formatValue(reference.parameterGradients.query)}，wd=${formatValue(reference.parameterGradients.document)}。`);
    set("note", mode.value === "detached" ? "每卡仍看见相同的两个候选，所以前向 loss 相同；但远端表示的梯度没有送回拥有者。这里没有加入任何补偿，文档梯度与共享参数更新发生变化。" : mode.value === "micro" ? "每个 microbatch 只有本地一对样本，候选集合只剩自己的正例，概率为 1、loss 和梯度为 0。累积这两次梯度不会自动恢复另一个候选。" : "每卡只计算自己的 query loss。文档梯度包含两个 query 的贡献，按文档拥有者汇合；共享编码器参数梯度再按声明的 DDP 平均规则归约。");
  };
  [mode, temperature].forEach(input => { input.addEventListener("input", render); input.addEventListener("change", render); }); render();
}
