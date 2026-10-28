import { $ } from "../core.js";
import { attentionBackwardInputs, denseAttentionReference } from "./attention-backward.js";
import { zeros, dot, softmax, maxDifference, formatValue, formatVector } from "./objective-math.js";

export function contextParallel({ split = 2, block = 2, mask = "causal", fault = "none", inputs = attentionBackwardInputs() } = {}) {
  const referenceInputs = structuredClone(inputs);
  denseAttentionReference(inputs, mask); // also validates the four matrices
  const { Q, K, V, G } = inputs, n = Q.length, d = Q[0].length;
  if (!Number.isInteger(split) || split < 1 || split > n || !Number.isInteger(block) || block < 1 || block > n) throw new RangeError("分片和块大小必须在 1 与序列长度之间。");
  if (!["none", "local-softmax", "local-position", "drop-remote", "local-mean"].includes(fault)) throw new RangeError("未知的上下文并行故障。");
  referenceInputs.G = G.map(row => row.map(x => x / n));
  const reference = denseAttentionReference(referenceInputs, mask);
  const ranges = split === n ? [[0, n]] : [[0, split], [split, n]], count = ranges.length;
  const owner = i => ranges.findIndex(([start, end]) => i >= start && i < end);
  const visible = (i, j) => !(mask === "empty-first" && i === 0) && (mask === "full" || (fault === "local-position" ? j - ranges[owner(j)][0] <= i - ranges[owner(i)][0] : j <= i));
  const tiles = ranges.flatMap(([start, end], rank) => Array.from({ length: Math.ceil((end - start) / block) }, (_, t) => ({ owner: rank, start: start + t * block, end: Math.min(end, start + (t + 1) * block) })));
  const O = zeros(n, d), P = zeros(n, n), lse = Array(n).fill(null), events = [], localParts = Array.from({ length: n }, () => []);
  let pairs = 0;
  for (let rank = 0; rank < count; rank++) {
    const [start, end] = ranges[rank], states = Array.from({ length: end - start }, () => ({ m: -Infinity, sum: 0, weighted: Array(d).fill(0) }));
    // Local KV first, followed by remote owners. The global normalization is order independent in real arithmetic.
    for (const tile of [...tiles.filter(t => t.owner === rank), ...tiles.filter(t => t.owner !== rank)]) {
      const rows = [];
      for (let i = start; i < end; i++) {
        const ids = Array.from({ length: tile.end - tile.start }, (_, k) => tile.start + k).filter(j => visible(i, j));
        const scores = ids.map(j => dot(Q[i], K[j]) / Math.sqrt(d)); pairs += ids.length;
        const state = states[i - start];
        if (ids.length) {
          const maximum = Math.max(state.m, ...scores), rescale = state.m === -Infinity ? 0 : Math.exp(state.m - maximum);
          state.sum *= rescale; state.weighted = state.weighted.map(x => x * rescale);
          ids.forEach((j, k) => { const e = Math.exp(scores[k] - maximum); state.sum += e; state.weighted = state.weighted.map((x, c) => x + e * V[j][c]); });
          state.m = maximum;
          localParts[i].push({ ids, probabilities: softmax(scores) });
        }
        rows.push({ query: i, visible: ids, maximum: Number.isFinite(state.m) ? state.m : null, denominator: state.sum, output: state.sum ? state.weighted.map(x => x / state.sum) : Array(d).fill(0) });
      }
      events.push({ stage: "forward", rank, owner: tile.owner, start: tile.start, end: tile.end, remote: rank !== tile.owner, rows });
    }
    for (let i = start; i < end; i++) {
      const state = states[i - start];
      if (!state.sum) continue;
      lse[i] = state.m + Math.log(state.sum);
      if (fault === "local-softmax") {
        const parts = localParts[i];
        for (const part of parts) part.ids.forEach((j, k) => { P[i][j] = part.probabilities[k] / parts.length; });
        O[i] = V[0].map((_, c) => P[i].reduce((sum, p, j) => sum + p * V[j][c], 0));
      } else {
        O[i] = state.weighted.map(x => x / state.sum);
        for (let j = 0; j < n; j++) if (visible(i, j)) P[i][j] = Math.exp(dot(Q[i], K[j]) / Math.sqrt(d) - lse[i]);
      }
    }
  }
  const dQ = zeros(n, d), dK = zeros(n, d), dV = zeros(n, d);
  const contributions = ranges.map(() => ({ dK: zeros(n, d), dV: zeros(n, d) }));
  let loss = 0;
  for (let i = 0; i < n; i++) {
    const rank = owner(i), factor = fault === "local-mean" ? 1 / (count * (ranges[rank][1] - ranges[rank][0])) : 1 / n;
    const upstream = G[i].map(x => x * factor); loss += dot(O[i], upstream);
    const parts = fault === "local-softmax" ? localParts[i] : [{ ids: Array.from({ length: n }, (_, j) => j), probabilities: P[i] }];
    for (const part of parts) {
      const baseline = part.ids.reduce((sum, j, k) => sum + part.probabilities[k] * dot(upstream, V[j]), 0);
      part.ids.forEach((j, k) => {
        const p = part.probabilities[k] / parts.length, ds = p * (dot(upstream, V[j]) - baseline) / Math.sqrt(d);
        for (let c = 0; c < d; c++) {
          dQ[i][c] += ds * K[j][c]; contributions[rank].dK[j][c] += ds * Q[i][c]; contributions[rank].dV[j][c] += p * upstream[c];
        }
      });
    }
  }
  for (let rank = 0; rank < count; rank++) for (let dest = 0; dest < count; dest++) {
    const [start, end] = ranges[dest], omitted = fault === "drop-remote" && rank !== dest;
    for (let j = start; j < end; j++) for (let c = 0; c < d; c++) if (!omitted) { dK[j][c] += contributions[rank].dK[j][c]; dV[j][c] += contributions[rank].dV[j][c]; }
    events.push({ stage: "backward", rank, owner: dest, start, end, remote: rank !== dest, omitted, dK: contributions[rank].dK.slice(start, end), dV: contributions[rank].dV.slice(start, end) });
  }
  const ranks = ranges.map(([start, end], rank) => {
    const local = end - start, remote = n - local;
    // This declared protocol replays KV in backward and returns one dK/dV block to its owner.
    return { rank, start, end, kvForwardSend: 2 * local * d * (count - 1), kvReplaySend: 2 * local * d * (count - 1), gradientSend: 2 * remote * d,
      sendBytes: (4 * local * d * (count - 1) + 2 * remote * d) * 8, receiveBytes: (4 * remote * d + 2 * local * d * (count - 1)) * 8,
      residentQKV: 3 * local * d, remoteTile: count > 1 ? 2 * Math.min(block, remote) * d : 0,
      softmaxState: local * (d + 2), outputs: local * d, lse: local };
  });
  return { inputs, split, block, mask, fault, ranges, O, P, lse, dQ, dK, dV, loss, reference, contributions, events, ranks, pairs,
    outputError: maxDifference(O, reference.O), gradientError: Math.max(...["dQ", "dK", "dV"].map(name => maxDifference({ dQ, dK, dV }[name], reference[name]))), lossError: Math.abs(loss - reference.loss) };
}

export function initContextParallel() {
  const split = $("[data-cp-split]"); if (!split) return;
  const block = $("[data-cp-block]"), mask = $("[data-cp-mask]"), fault = $("[data-cp-fault]"); let step = 0;
  const set = (name, value) => { $(`[data-cp-${name}]`).textContent = value; };
  const render = () => {
    const r = contextParallel({ split: +split.value, block: +block.value, mask: mask.value, fault: fault.value }); step = Math.min(step, r.events.length - 1);
    set("output", r.outputError.toExponential(2)); set("gradient", r.gradientError.toExponential(2)); set("loss", `${formatValue(r.loss)} / ${formatValue(r.reference.loss)}`);
    set("step", `${step + 1} / ${r.events.length}`); const e = r.events[step];
    set("event", e.stage === "forward" ? `前向：rank ${e.rank} ${e.remote ? "接收" : "读取本地"} rank ${e.owner} 的 K/V ${e.start}…${e.end - 1}` : `反向：rank ${e.rank} 将 ${e.start}…${e.end - 1} 的 dK/dV ${e.remote ? "返回" : "留在"}拥有者 rank ${e.owner}${e.omitted ? "（故障：贡献被丢弃）" : ""}`);
    $("[data-cp-event-rows]").innerHTML = e.stage === "forward" ? e.rows.map(row => `<tr><td>Q${row.query}</td><td>${row.visible.join(", ") || "空"}</td><td>${row.maximum === null ? "−∞" : formatValue(row.maximum)}</td><td>${formatValue(row.denominator)}</td><td>${formatVector(row.output)}</td></tr>`).join("") : e.dK.map((row, j) => `<tr><td>K/V${j + e.start}</td><td>${e.omitted ? "未归还" : `rank ${e.owner}`}</td><td colspan="2">dK ${formatVector(row)}</td><td>dV ${formatVector(e.dV[j])}</td></tr>`).join("");
    $("[data-cp-results]").innerHTML = r.O.map((row, i) => `<tr><th scope="row">${i}</th><td>${formatVector(row)}</td><td>${formatVector(r.reference.O[i])}</td><td>${formatVector(r.dQ[i])}</td><td>${formatVector(r.dK[i])}</td><td>${formatVector(r.dV[i])}</td></tr>`).join("");
    $("[data-cp-ranks]").innerHTML = r.ranks.map(rank => `<tr><th scope="row">${rank.rank}</th><td>${rank.start}…${rank.end - 1}</td><td>${rank.kvForwardSend} / ${rank.kvReplaySend} / ${rank.gradientSend}</td><td>${rank.sendBytes} / ${rank.receiveBytes}</td><td>${rank.residentQKV} / ${rank.remoteTile} / ${rank.softmaxState}</td></tr>`).join("");
    $("[data-cp-inputs]").textContent = JSON.stringify(r.inputs, null, 2);
    const notes = { none: "输出与梯度均按全局位置和 1/N 目标计算；事件表显示当前交换，顶部数值显示全部事件完成后的结果。", "local-softmax": "故障：每个非空 K/V 块单独 softmax 后等权平均。事件中的在线状态仍作为正确路径对照；最终输出与梯度使用错误块平均。只有一个有效块时可能恰好相同。", "local-position": "故障：因果 mask 用分片内位置，可能同时屏蔽过去位置和看到未来位置。全部可见或单分片时该故障不改变结果。", "drop-remote": "故障：前向完整，dQ 完整，但丢弃远端 query 对本地 K/V 的梯度。单分片时没有远端贡献可丢。", "local-mean": "故障：先按本地 query 数取平均，再等权平均各 rank。本例 2+3 分片改变了每个 query 的权重。" };
    set("note", notes[fault.value]); $("[data-cp-prev]").disabled = step === 0; $("[data-cp-next]").disabled = step === r.events.length - 1;
  };
  [split, block, mask, fault].forEach(el => el.addEventListener("change", () => { step = 0; render(); }));
  $("[data-cp-next]").addEventListener("click", () => { step++; render(); }); $("[data-cp-prev]").addEventListener("click", () => { step--; render(); }); $("[data-cp-reset]").addEventListener("click", () => { step = 0; render(); }); render();
}
