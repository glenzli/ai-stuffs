import { $ } from "../core.js";
import { groupAdvantages } from "./rl-dynamics.js";
import { zeros, matmul, softmax, dot, formatValue, formatVector } from "./objective-math.js";

export function policyInputs() {
  return {
    weights: [[.25, -.15, .1], [-.2, .3, .05]],
    reference: [[.1, -.05, .15], [.05, .1, -.1]],
    rewards: [1, 0, .5, 1],
    responses: [[0, 1], [2], [1, 0, 2], [0, 2]],
  };
}
export function policyTokens(inputs = policyInputs()) {
  return inputs.responses.flatMap((actions, sample) => [
    { sample, position: -1, kind: "prompt", action: 1, x: [1, -.5] },
    ...actions.map((action, position) => ({ sample, position, kind: "response", action, x: [1 + sample * .1, .2 + position * .35 - sample * .1] })),
    { sample, position: actions.length, kind: "padding", action: 2, x: [0, 1] },
  ]);
}
const distribution = (weights, x) => {
  const logits = matmul([x], weights)[0], p = softmax(logits), max = Math.max(...logits);
  const lse = max + Math.log(logits.reduce((s, z) => s + Math.exp(z - max), 0));
  return { logits, p, logp: logits.map(z => z - lse) };
};

export function policyStep({ weights = policyInputs().weights, oldWeights = policyInputs().weights, reference = policyInputs().reference, inputs = policyInputs(), normalization = "token", fault = "none", limit = 3, beta = .08, rate = .2, epsilon = .2, rewards = inputs.rewards } = {}) {
  for (const w of [weights, oldWeights, reference]) if (w?.length !== 2 || !w.every(row => row.length === 3 && row.every(Number.isFinite))) throw new RangeError("策略参数必须是 2×3 有限矩阵。");
  if (!["token", "sequence"].includes(normalization) || !["none", "old-graph", "include-controls", "refresh-old", "wrong-denominator"].includes(fault)) throw new RangeError("未知的策略目标或故障。");
  if (!Number.isInteger(limit) || limit < 0 || limit > 3 || !Number.isFinite(beta) || beta < 0 || !Number.isFinite(rate) || rate < 0 || !Number.isFinite(epsilon) || epsilon <= 0 || epsilon >= 1) throw new RangeError("策略实验参数越界。");
  if (rewards.length !== inputs.responses.length) throw new RangeError("每个回答必须有一个奖励。");
  const advantages = groupAdvantages(rewards).advantages;
  const tokens = policyTokens(inputs), responseCount = tokens.filter(t => t.kind === "response" && t.position < limit).length;
  const active = tokens.filter(t => (t.kind === "response" && t.position < limit) || (fault === "include-controls" && t.kind !== "response"));
  const lengths = inputs.responses.map((_, sample) => active.filter(t => t.sample === sample).length), nonempty = lengths.filter(Boolean).length;
  const gradient = zeros(2, 3), rows = []; let surrogate = 0, kl = 0, clipped = 0;
  for (const t of tokens) {
    const now = distribution(weights, t.x), old = distribution(oldWeights, t.x), ref = distribution(reference, t.x), enabled = active.includes(t);
    const ratio = fault === "old-graph" ? 1 : Math.exp(now.logp[t.action] - old.logp[t.action]);
    const advantage = advantages[t.sample], clamp = Math.max(1 - epsilon, Math.min(1 + epsilon, ratio));
    const isClipped = advantage > 0 ? ratio > 1 + epsilon : advantage < 0 && ratio < 1 - epsilon;
    const objective = Math.min(ratio * advantage, clamp * advantage);
    const divergences = now.logp.map((l, k) => l - ref.logp[k]), tokenKL = dot(now.p, divergences);
    let factor = enabled ? normalization === "token" ? 1 / active.length : 1 / (nonempty * lengths[t.sample]) : 0;
    if (fault === "wrong-denominator" && enabled) factor = normalization === "token" ? 1 / (nonempty * lengths[t.sample]) : 1 / active.length;
    const policyDerivative = fault === "old-graph" || isClipped ? 0 : -advantage * ratio;
    const dLogits = now.p.map((p, k) => factor * (policyDerivative * ((k === t.action ? 1 : 0) - p) + beta * p * (divergences[k] - tokenKL)));
    if (enabled) {
      surrogate += factor * objective; kl += factor * tokenKL; clipped += Number(isClipped);
      for (let h = 0; h < 2; h++) for (let k = 0; k < 3; k++) gradient[h][k] += t.x[h] * dLogits[k];
    }
    rows.push({ ...t, enabled, factor, logits: now.logits, referenceP: ref.p, logp: now.logp[t.action], oldLogp: fault === "old-graph" ? now.logp[t.action] : old.logp[t.action], referenceLogp: ref.logp[t.action], ratio, advantage, clipped: isClipped, kl: tokenKL, dLogits, p: now.p });
  }
  const updated = weights.map((row, i) => row.map((w, j) => w - rate * gradient[i][j]));
  return { weights, oldWeights, reference, normalization, fault, limit, beta, rate, epsilon, rewards, advantages, rows, gradient, updated,
    loss: active.length ? -surrogate + beta * kl : null, surrogate, kl, count: active.length, responseCount, clipped, skipped: !active.length,
    after: rows.map(row => distribution(updated, row.x).p) };
}

export function policyTrajectory({ steps = 1, ...options } = {}) {
  if (!Number.isInteger(steps) || steps < 0 || steps > 30) throw new RangeError("更新次数必须为 0…30。");
  let weights = structuredClone(options.weights || policyInputs().weights);
  const behavior = structuredClone(options.oldWeights || policyInputs().weights), history = [];
  for (let i = 0; i < steps; i++) {
    const oldWeights = options.fault === "refresh-old" ? structuredClone(weights) : behavior;
    const result = policyStep({ ...options, weights, oldWeights }); history.push(result); weights = result.updated;
  }
  return { weights, behavior, history, next: policyStep({ ...options, weights, oldWeights: options.fault === "refresh-old" ? weights : behavior }) };
}

export function enumeratedPolicyGradient(weights = policyInputs().weights) {
  // All 3×3 trajectories. Second-step state depends on the first sampled action.
  const root = distribution(weights, [1, .2]), rewards = [[1, -.3, .2], [0, .8, -.1], [-.2, .1, 1.2]], gradient = zeros(2, 3), rows = [];
  let expectation = 0;
  for (let a = 0; a < 3; a++) {
    const x = [1, .3 + a * .4], second = distribution(weights, x);
    for (let b = 0; b < 3; b++) {
      const probability = root.p[a] * second.p[b], reward = rewards[a][b]; expectation += probability * reward;
      for (let h = 0; h < 2; h++) for (let k = 0; k < 3; k++) gradient[h][k] += probability * reward * ((h === 0 ? 1 : .2) * ((k === a ? 1 : 0) - root.p[k]) + x[h] * ((k === b ? 1 : 0) - second.p[k]));
      rows.push({ a, b, probability, reward });
    }
  }
  return { expectation, gradient, rows };
}

export function initRlUpdate() {
  const fault = $("[data-ru-fault]"); if (!fault) return;
  const normalization = $("[data-ru-normalization]"), limit = $("[data-ru-limit]"), beta = $("[data-ru-beta]"), rewards = $("[data-ru-rewards]"); let steps = 0;
  const set = (name, value) => { $(`[data-ru-${name}]`).textContent = value; };
  const render = () => {
    const trajectory = policyTrajectory({ steps, fault: fault.value, normalization: normalization.value, limit: +limit.value, beta: +beta.value, rewards: rewards.value === "equal" ? [1, 1, 1, 1] : policyInputs().rewards }); const r = trajectory.next;
    set("steps", steps); set("loss", formatValue(r.loss)); set("count", `${r.count}（有效回答 ${r.responseCount}）`); set("kl", formatValue(r.kl)); set("clip", `${r.clipped} / ${r.count}`);
    $("[data-ru-rows]").innerHTML = r.rows.map((t, i) => `<tr><td>${t.sample + 1}/${t.kind === "prompt" ? "提示" : t.kind === "padding" ? "填充" : t.position + 1}</td><td>${t.action}</td><td>${t.enabled ? formatValue(t.factor) : "屏蔽"}</td><td>${formatValue(t.advantage)}</td><td>${formatValue(t.oldLogp)} / ${formatValue(t.logp)}</td><td>${formatValue(t.ratio)}${t.clipped ? " · 裁剪" : ""}</td><td>${formatValue(t.p[t.action])} → ${formatValue(r.after[i][t.action])}</td></tr>`).join("");
    $("[data-ru-parameters]").innerHTML = r.weights.flatMap((row, i) => row.map((w, j) => `<tr><th scope="row">W[${i},${j}]</th><td>${formatValue(w)}</td><td>${formatValue(r.gradient[i][j])}</td><td>${formatValue(r.updated[i][j])}</td><td>${formatValue(r.oldWeights[i][j])}</td></tr>`)).join("");
    $("[data-ru-history]").innerHTML = trajectory.history.map((item, i) => `<tr><td>${i + 1}</td><td>${formatValue(item.loss)}</td><td>${formatValue(item.kl)}</td><td>${item.clipped}</td><td>${formatVector(item.updated.flat())}</td></tr>`).join("");
    $("[data-ru-distributions]").innerHTML = r.rows.filter(t => t.kind === "response").map(t => `<tr><td>${t.sample + 1}/${t.position + 1}</td><td>${formatVector(t.x)}</td><td>${formatVector(t.logits)}</td><td>${formatVector(t.p)}</td><td>${formatVector(t.referenceP)}</td><td>${formatVector(t.dLogits)}</td></tr>`).join("");
    $("[data-ru-inputs]").textContent = JSON.stringify({ initialWeights: policyInputs().weights, savedBehaviorWeights: r.oldWeights, referenceWeights: r.reference, rewards: r.rewards, advantages: r.advantages, responseActions: policyInputs().responses }, null, 2);
    const messages = { none: "同批回答与行为概率固定。顶部展示当前参数下的下一步目标，表格右列预览下一次 SGD；点击更新才推进参数。", "old-graph": "故障：分母取同一计算图中的当前概率，概率比恒为 1，策略项导数抵消。KL 项仍可更新参数。", "include-controls": "故障：prompt 和 padding 也进入目标，被错误赋予该回答的优势与 KL 权重。", "refresh-old": "故障：每次更新前改写 old 参数，同批 rollout 的生成概率被丢失。第一步可能与正确路径相同，继续更新观察分叉。", "wrong-denominator": "故障：实际使用另一种归一化。两种目标都可合法，但实现必须与所声明的 token/序列权重一致。" };
    set("note", r.skipped ? "没有有效 token：目标未定义，参数不更新。" : messages[fault.value]); $("[data-ru-update]").disabled = r.skipped || steps >= 30;
    const exact = enumeratedPolicyGradient(); set("expectation", formatValue(exact.expectation)); set("exact-gradient", formatVector(exact.gradient.flat()));
    $("[data-ru-enumeration]").innerHTML = exact.rows.map(row => `<tr><td>${row.a} → ${row.b}</td><td>${formatValue(row.probability)}</td><td>${formatValue(row.reward)}</td><td>${formatValue(row.probability * row.reward)}</td></tr>`).join("");
  };
  [fault, normalization, limit, beta, rewards].forEach(el => el.addEventListener("change", () => { steps = 0; render(); }));
  $("[data-ru-update]").addEventListener("click", () => { steps++; render(); }); $("[data-ru-reset]").addEventListener("click", () => { steps = 0; render(); }); render();
}
