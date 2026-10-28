import { $ } from "../core.js";

export function groupAdvantages(rewards) {
  if (!Array.isArray(rewards)) throw new TypeError("rewards must be an array of finite numbers.");
  if (!rewards.length) throw new RangeError("rewards must be nonempty.");
  for (const reward of rewards) {
    if (!Number.isFinite(reward)) throw new TypeError("rewards must contain only finite numbers.");
  }
  if (rewards.every(reward => reward === rewards[0])) {
    return { mean: rewards[0], std: 0, advantages: rewards.map(() => 0), signal: false };
  }

  // Offset before rescaling to preserve small differences between large rewards.
  let origin = rewards[0];
  const offsets = rewards.map(reward => reward - origin);
  let scale = offsets.reduce((largest, reward) => Math.max(largest, Math.abs(reward)), 0);
  let scaled;
  if (Number.isFinite(scale)) {
    scaled = offsets.map(reward => reward / scale);
  } else {
    origin = 0;
    scale = rewards.reduce((largest, reward) => Math.max(largest, Math.abs(reward)), 0);
    scaled = rewards.map(reward => reward / scale);
  }
  const mean = scaled.reduce((average, reward, index) => average + (reward - average) / (index + 1), 0);
  const centered = scaled.map(reward => reward - mean);
  const std = Math.sqrt(centered.reduce((sum, reward) => sum + reward ** 2 / rewards.length, 0));
  return {
    mean: origin + mean * scale,
    std: std * scale,
    advantages: centered.map(reward => reward / std),
    signal: std > 0,
  };
}

export function clippedSurrogate({ old = 0.2, current = 0.4, advantage = 1, clip = 0.2 } = {}) {
  for (const probability of [old, current]) {
    if (!Number.isFinite(probability) || probability <= 0 || probability > 1) {
      throw new RangeError("old and current must be finite probabilities in (0, 1].");
    }
  }
  if (!Number.isFinite(advantage)) throw new TypeError("advantage must be finite.");
  if (!Number.isFinite(clip) || clip <= 0 || clip >= 1) {
    throw new RangeError("clip must be finite and in (0, 1).");
  }
  const ratio = current / old;
  const clamped = Math.max(1 - clip, Math.min(1 + clip, ratio));
  const objective = advantage === 0 ? 0 : Math.min(ratio * advantage, clamped * advantage);
  const clipped = advantage > 0 ? ratio > clamped : advantage < 0 && ratio < clamped;
  return { ratio, clamped, objective, clipped };
}

export function initRlDynamics() {
  const group = $("[data-rl-group]");
  if (!group) return;
  const old = $("[data-rld-old]");
  const current = $("[data-rld-current]");
  const sign = $("[data-rld-sign]");
  const groups = {
    mixed: [0, 0, 1, 1],
    allpass: [1, 1, 1, 1],
    allfail: [0, 0, 0, 0],
    graded: [0, 0.2, 0.7, 1],
  };
  const renderGroup = () => {
    const rewards = groups[group.value];
    const result = groupAdvantages(rewards);
    $("[data-rld-mean]").textContent = result.mean.toFixed(3);
    $("[data-rld-std]").textContent = result.std.toFixed(3);
    $("[data-rld-signal]").textContent = result.signal ? "有" : "无";
    $("[data-rld-rows]").innerHTML = rewards.map((reward, index) =>
      `<tr><td>${index + 1}</td><td>${reward.toFixed(2)}</td><td>${result.advantages[index].toFixed(3)}</td></tr>`
    ).join("");
    $("[data-rld-note]").textContent = result.signal
      ? "优势由本组奖励减去均值，再除以总体标准差。这里只计算组内结果奖励项，不包含 KL 正则、熵项或其他组的贡献。"
      : "本组奖励全部相同，优势全为 0，本组结果奖励项没有区分信号。完整目标还需计入 KL、熵项及其他组的贡献。";
  };
  const renderClip = () => {
    for (const [name, input] of [["old", old], ["current", current]]) {
      const output = $(`[data-rld-${name}-value]`);
      if (output) output.textContent = `${input.value}%`;
    }
    const advantage = Number(sign.value);
    const result = clippedSurrogate({ old: Number(old.value) / 100, current: Number(current.value) / 100, advantage });
    $("[data-rld-ratio]").textContent = result.ratio.toFixed(3);
    $("[data-rld-objective]").textContent = result.objective.toFixed(3);
    const explanation = result.clipped
      ? `当前优势为${advantage > 0 ? "正" : "负"}且概率比${advantage > 0 ? "超过 1.2" : "低于 0.8"}，min 选择裁剪项，代理目标进入平台。`
      : "当前 min 保留原始代理目标；正优势仅在比值超过 1.2 时、负优势仅在比值低于 0.8 时形成裁剪平台。";
    $("[data-rld-clip-note]").textContent = `行为概率 ${old.value}%，当前概率 ${current.value}%。${explanation}全模型 KL 和其他样本的更新需单独检查。`;
  };
  for (const event of ["input", "change"]) {
    group.addEventListener(event, renderGroup);
    sign.addEventListener(event, renderClip);
  }
  old.addEventListener("input", renderClip);
  current.addEventListener("input", renderClip);
  renderGroup();
  renderClip();
}
