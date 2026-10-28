import { $ } from "../core.js";
import { zeros, dot, softmax, maxDifference, formatVector, formatValue } from "./objective-math.js";

export function attentionBackwardInputs() {
  return {
    Q: [[.2, -.3], [.5, .1], [-.4, .7], [.6, -.2], [.1, .8]],
    K: [[.3, .5], [-.2, .4], [.7, -.1], [.1, -.6], [-.5, .2]],
    V: [[1, 0], [0, 2], [-1, 1], [2, -1], [.5, .3]],
    G: [[.2, -.1], [.3, .5], [-.4, .2], [.1, -.3], [.6, .4]],
  };
}
const visible = (i, j, mask) => !(mask === "empty-first" && i === 0) && (mask === "full" || j <= i);
function validate(inputs, mask) {
  const n = inputs.Q?.length, d = inputs.Q?.[0]?.length;
  if (!n || !d || !["causal", "full", "empty-first"].includes(mask)) throw new RangeError("Attention 输入或 mask 无效。");
  for (const name of ["Q", "K", "V", "G"]) if (inputs[name]?.length !== n || !inputs[name].every(row => row.length === d && row.every(Number.isFinite))) throw new RangeError("本例 Q/K/V/G 必须为相同形状的有限矩阵。");
}
export function denseAttentionReference(inputs = attentionBackwardInputs(), mask = "causal") {
  validate(inputs, mask);
  const { Q, K, V, G } = inputs, n = Q.length, d = Q[0].length;
  const P = Q.map((q, i) => {
    const scores = K.map((k, j) => visible(i, j, mask) ? dot(q, k) / Math.sqrt(d) : -Infinity);
    return scores.some(Number.isFinite) ? softmax(scores) : Array(n).fill(0);
  });
  const O = Q.map((_, i) => V[0].map((_, c) => P[i].reduce((s, p, j) => s + p * V[j][c], 0)));
  const dQ = zeros(n, d), dK = zeros(n, d), dV = zeros(n, d);
  for (let i = 0; i < n; i++) {
    const dP = V.map(v => dot(G[i], v)), baseline = dot(P[i], dP);
    for (let j = 0; j < n; j++) {
      const dScore = P[i][j] * (dP[j] - baseline) / Math.sqrt(d);
      for (let c = 0; c < d; c++) {
        dQ[i][c] += dScore * K[j][c]; dK[j][c] += dScore * Q[i][c]; dV[j][c] += P[i][j] * G[i][c];
      }
    }
  }
  return { O, P, dQ, dK, dV, loss: O.reduce((sum, row, i) => sum + dot(row, G[i]), 0) };
}

export function tiledAttentionBackward({ queryBlock = 2, keyBlock = 2, mask = "causal", fault = "none", inputs = attentionBackwardInputs() } = {}) {
  validate(inputs, mask);
  for (const value of [queryBlock, keyBlock]) if (!Number.isInteger(value) || value < 1 || value > inputs.Q.length) throw new RangeError("块大小必须在 1 与序列长度之间。");
  if (!["none", "local-softmax", "overwrite-kv"].includes(fault)) throw new RangeError("未知的反传故障。");
  const { Q, K, V, G } = inputs, n = Q.length, d = Q[0].length, O = zeros(n, d), lse = Array(n).fill(-Infinity);
  let forwardPairs = 0;
  for (let i = 0; i < n; i++) {
    let maximum = -Infinity, denominator = 0, weighted = Array(d).fill(0);
    for (let start = 0; start < n; start += keyBlock) {
      const ids = Array.from({ length: Math.min(keyBlock, n - start) }, (_, j) => start + j).filter(j => visible(i, j, mask));
      if (!ids.length) continue;
      const scores = ids.map(j => dot(Q[i], K[j]) / Math.sqrt(d)); forwardPairs += ids.length;
      const nextMax = Math.max(maximum, ...scores), correction = maximum === -Infinity ? 0 : Math.exp(maximum - nextMax);
      denominator *= correction; weighted = weighted.map(x => x * correction);
      ids.forEach((j, index) => { const p = Math.exp(scores[index] - nextMax); denominator += p; weighted = weighted.map((x, c) => x + p * V[j][c]); });
      maximum = nextMax;
    }
    if (denominator) { O[i] = weighted.map(x => x / denominator); lse[i] = maximum + Math.log(denominator); }
  }
  const dQ = zeros(n, d), dK = zeros(n, d), dV = zeros(n, d), D = O.map((row, i) => dot(row, G[i])), events = [];
  let reconstructedPairs = 0;
  for (let qStart = 0; qStart < n; qStart += queryBlock) for (let kStart = 0; kStart < n; kStart += keyBlock) {
    const qEnd = Math.min(n, qStart + queryBlock), kEnd = Math.min(n, kStart + keyBlock);
    const localDK = zeros(kEnd - kStart, d), localDV = zeros(kEnd - kStart, d), probability = zeros(qEnd - qStart, kEnd - kStart);
    for (let i = qStart; i < qEnd; i++) {
      const ids = Array.from({ length: kEnd - kStart }, (_, j) => j + kStart).filter(j => visible(i, j, mask));
      const scores = ids.map(j => dot(Q[i], K[j]) / Math.sqrt(d)); reconstructedPairs += ids.length;
      const localP = scores.length && fault === "local-softmax" ? softmax(scores) : null;
      ids.forEach((j, index) => {
        const p = localP ? localP[index] : Math.exp(scores[index] - lse[i]); probability[i - qStart][j - kStart] = p;
        const dScore = p * (dot(G[i], V[j]) - D[i]) / Math.sqrt(d);
        for (let c = 0; c < d; c++) {
          dQ[i][c] += dScore * K[j][c]; localDK[j - kStart][c] += dScore * Q[i][c]; localDV[j - kStart][c] += p * G[i][c];
        }
      });
    }
    for (let j = kStart; j < kEnd; j++) for (let c = 0; c < d; c++) {
      dK[j][c] = (fault === "overwrite-kv" ? 0 : dK[j][c]) + localDK[j - kStart][c];
      dV[j][c] = (fault === "overwrite-kv" ? 0 : dV[j][c]) + localDV[j - kStart][c];
    }
    // Snapshots exist only for the teaching UI; excluded from algorithm storage.
    events.push({ qStart, qEnd, kStart, kEnd, probability, reconstructedPairs, dQ: structuredClone(dQ), dK: structuredClone(dK), dV: structuredClone(dV) });
  }
  const reference = denseAttentionReference(inputs, mask);
  return { inputs, queryBlock, keyBlock, mask, fault, O, lse, D, dQ, dK, dV, events, reference, forwardPairs, reconstructedPairs,
    error: Math.max(...["dQ", "dK", "dV"].map(name => maxDifference({ dQ, dK, dV }[name], reference[name]))), outputError: maxDifference(O, reference.O),
    storage: { sharedInputsOutput: 4 * n * d, denseProbabilities: n * n, savedStatistics: n, tileCapacity: Math.min(queryBlock, n) * Math.min(keyBlock, n), upstreamAndGradients: 4 * n * d } };
}

export function initAttentionBackward() {
  const query = $("[data-ab-query]"); if (!query) return;
  const key = $("[data-ab-key]"), mask = $("[data-ab-mask]"), fault = $("[data-ab-fault]"); let step = 0;
  const set = (name, value) => { $(`[data-ab-${name}]`).textContent = value; };
  const render = () => {
    const r = tiledAttentionBackward({ queryBlock: Number(query.value), keyBlock: Number(key.value), mask: mask.value, fault: fault.value });
    step = Math.min(step, r.events.length); const event = r.events[step - 1], complete = step === r.events.length;
    set("step", `${step}/${r.events.length}`); set("error", complete ? r.error.toExponential(2) : "反向尚未完成"); set("forward", r.outputError.toExponential(2));
    set("storage", `概率矩阵 ${r.storage.denseProbabilities} → LSE ${r.storage.savedStatistics} 元素；当前块容量最多 ${r.storage.tileCapacity}。`);
    set("pairs", `前向有效 QK 点积 ${r.forwardPairs} 次；已重建 ${event?.reconstructedPairs || 0} 次，完整反向共 ${r.reconstructedPairs} 次。`);
    $("[data-ab-stats]").innerHTML = r.lse.map((value, i) => `<tr><th scope="row">${i}</th><td>${value === -Infinity ? "−∞（空行）" : formatValue(value)}</td><td>${formatVector(r.O[i])}</td><td>${formatValue(r.D[i])}</td></tr>`).join("");
    $("[data-ab-events]").innerHTML = r.events.slice(0, step).map((e, i) => `<tr><td>${i + 1}</td><td>${e.qStart}…${e.qEnd - 1}</td><td>${e.kStart}…${e.kEnd - 1}</td><td>${e.probability.map(formatVector).join(" / ")}</td></tr>`).join("");
    $("[data-ab-gradients]").innerHTML = ["dQ", "dK", "dV"].flatMap(name => r[name].map((_, i) => `<tr><th scope="row">${name}[${i}]</th><td>${formatVector(event ? event[name][i] : [0, 0])}</td><td>${formatVector(r.reference[name][i])}</td></tr>`)).join("");
    set("note", fault.value === "local-softmax" ? "故障只作用于反向：按当前 key 块重新 softmax，丢失整行归一化。一个块覆盖全部 key 时可能恰好不出错。" : fault.value === "overwrite-kv" ? "故障只作用于反向：后一个 query 块覆盖 dK/dV，丢掉更早 query 的贡献。只有一个 query 块时，这个错误不会暴露。" : complete ? "全部块完成，输出与全部 Q/K/V 梯度可与完整矩阵参考比较。空行的输出与梯度按本例约定为零。" : "前向已经完成；推进反向块观察局部概率重建，以及多个 query 对同一个 K/V 的梯度累加。");
    $("[data-ab-next]").disabled = complete; $("[data-ab-prev]").disabled = step === 0;
  };
  [query, key, mask, fault].forEach(input => input.addEventListener("change", () => { step = 0; render(); }));
  $("[data-ab-next]").addEventListener("click", () => { step++; render(); }); $("[data-ab-prev]").addEventListener("click", () => { step = Math.max(0, step - 1); render(); }); $("[data-ab-reset]").addEventListener("click", () => { step = 0; render(); }); render();
}
