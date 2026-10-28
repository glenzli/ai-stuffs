import { $ } from "../core.js";
import { formatValue } from "./objective-math.js";

export function clippingLayout(scenario = "shards") {
  const layouts = {
    shards: [[{ id: "a", gradient: 3 }], [{ id: "b", gradient: 4 }]],
    mixed: [[{ id: "a", gradient: 3 }, { id: "r", gradient: 12 }], [{ id: "b", gradient: 4 }, { id: "r", gradient: 12 }]],
    replicated: [[{ id: "a", gradient: 3 }, { id: "b", gradient: 4 }], [{ id: "a", gradient: 3 }, { id: "b", gradient: 4 }]],
    zero: [[{ id: "a", gradient: 0 }], [{ id: "b", gradient: 0 }]],
    overflow: [[{ id: "a", gradient: 3 }], [{ id: "b", gradient: Infinity }]],
  };
  if (!layouts[scenario]) throw new RangeError("未知的参数布局。");
  return structuredClone(layouts[scenario]);
}
export function shardedClipping({ scenario = "shards", mode = "global", threshold = 1, scale = 8 } = {}) {
  if (!["global", "local", "duplicate", "local-skip"].includes(mode) || !Number.isFinite(threshold) || threshold < 0 || !Number.isFinite(scale) || scale <= 0) throw new RangeError("裁剪模式、阈值或 scale 无效。");
  const layout = clippingLayout(scenario), owners = new Map();
  layout.forEach((rows, rank) => rows.forEach(row => { if (!owners.has(row.id)) owners.set(row.id, rank); }));
  const ranks = layout.map((rows, rank) => ({ rank, rows: rows.map(row => ({ ...row, scaled: row.gradient * scale, owner: owners.get(row.id) })), finite: rows.every(row => Number.isFinite(row.gradient * scale)), localSquare: rows.reduce((sum, row) => sum + row.gradient ** 2, 0), ownedSquare: rows.filter(row => owners.get(row.id) === rank).reduce((sum, row) => sum + row.gradient ** 2, 0) }));
  const finite = ranks.every(rank => rank.finite), norm = finite ? Math.sqrt(ranks.reduce((sum, rank) => sum + rank.ownedSquare, 0)) : null;
  const countedNorm = finite ? Math.sqrt(ranks.reduce((sum, rank) => sum + (mode === "duplicate" ? rank.localSquare : rank.ownedSquare), 0)) : null;
  const coefficient = value => value === 0 ? 1 : Math.min(1, threshold / value);
  for (const rank of ranks) {
    rank.skip = mode === "local-skip" ? !rank.finite : !finite;
    rank.usedNorm = rank.skip ? null : mode === "local" || mode === "local-skip" ? Math.sqrt(rank.localSquare) : countedNorm;
    rank.coefficient = rank.skip ? null : coefficient(rank.usedNorm);
    rank.output = rank.rows.map(row => ({ id: row.id, gradient: rank.skip ? null : row.gradient * rank.coefficient, parameterBefore: 1, parameterAfter: rank.skip ? 1 : 1 - .1 * row.gradient * rank.coefficient }));
  }
  const uniqueOutput = ranks.flatMap(rank => rank.output.filter(row => owners.get(row.id) === rank.rank));
  const outputNorm = ranks.some(rank => rank.skip) ? null : Math.hypot(...uniqueOutput.map(row => row.gradient));
  const consistentDecision = ranks.every(rank => rank.skip === ranks[0].skip);
  return { scenario, mode, threshold, scale, ranks, finite, norm, countedNorm, correctCoefficient: finite ? coefficient(norm) : null, outputNorm, consistentDecision,
    uniqueCount: owners.size, physicalCount: layout.flat().length,
    error: finite ? Math.max(0, ...ranks.flatMap(rank => rank.output.map((row, i) => Math.abs(row.gradient - rank.rows[i].gradient * coefficient(norm))))) : null };
}
const finiteText = value => value === Infinity ? "∞" : formatValue(value);
export function initShardedClipping() {
  const scenario = $("[data-sc-scenario]"); if (!scenario) return;
  const mode = $("[data-sc-mode]"), threshold = $("[data-sc-threshold]");
  const set = (name, value) => { $(`[data-sc-${name}]`).textContent = value; };
  const render = () => {
    const r = shardedClipping({ scenario: scenario.value, mode: mode.value, threshold: Number(threshold.value) });
    set("norm", r.finite ? formatValue(r.norm) : "存在非有限梯度"); set("output", r.outputNorm === null ? "未完成统一更新" : formatValue(r.outputNorm)); set("decision", r.consistentDecision ? r.ranks[0].skip ? "两卡均跳步" : "两卡均更新" : "错误：更新步数分叉");
    $("[data-sc-rows]").innerHTML = r.ranks.flatMap(rank => rank.rows.map((row, i) => `<tr><td>${rank.rank}</td><td>${row.id}</td><td>${finiteText(row.scaled)} → ${finiteText(row.gradient)}</td><td>${row.owner === rank.rank ? "计入" : "副本不重复计入"}</td><td>${rank.coefficient === null ? "—" : formatValue(rank.coefficient)}</td><td>${rank.output[i].gradient === null ? "跳步" : formatValue(rank.output[i].gradient)}</td><td>${formatValue(rank.output[i].parameterAfter)}</td></tr>`)).join("");
    set("squares", `Rank 0/1 的唯一拥有平方和：${r.ranks.map(rank => finiteText(rank.ownedSquare)).join(" / ")}；物理元素 ${r.physicalCount} 个，独立元素 ${r.uniqueCount} 个。`);
    set("note", !r.consistentDecision ? "仅在本地检查有限性导致 Rank 0 更新、Rank 1 跳过。下一步参数版本、计数和通信流程可能不再一致。" : !r.finite ? "先全局归约非有限标志，所有 rank 协调跳步。参数、动量和成功更新计数应保持；本例只显示无动量 SGD 参数。" : mode.value === "local" ? "每卡按本地范数计算系数；分片布局中会改变全局裁剪。全部参数完整复制时，本地范数可以与全局范数相等。" : mode.value === "duplicate" ? "错误对照把所有物理副本都计入平方和。没有复制参数时它恰好正确，混合或完整复制布局才会暴露重复计数。" : "先 unscale，再全局检查有限性并汇总每个独立梯度一次。所有副本使用同一个系数，然后统一进行一次更新。");
  };
  [scenario, mode, threshold].forEach(input => input.addEventListener("change", render)); render();
}
