import { $ } from "../core.js";

export const gradientSequences = {
  reverse: [1, 1, 1, -1, -1, 0, 0],
  spike: [0.1, 0.1, 8, 0.1, 0, 0, 0],
  zero: [0, 0, 0, 0, 0, 0, 0],
};

export function optimizerLedger(gradients, { method = "adamw", lr = 0.02, beta = 0.9, decay = 0.1 } = {}) {
  let theta = 1, m = 0, v = 0;
  return gradients.map((g, index) => {
    const step = index + 1, before = theta;
    let direction = g;
    if (method === "adamw") {
      m = beta * m + (1 - beta) * g;
      v = 0.99 * v + 0.01 * g * g;
      direction = (m / (1 - beta ** step)) / (Math.sqrt(v / (1 - 0.99 ** step)) + 1e-8);
    } else if (method === "momentum") {
      m = beta * m + g;
      direction = m;
    } else {
      m = g;
    }
    const gradientDelta = -lr * direction, decayDelta = -lr * decay * before;
    theta = before + gradientDelta + decayDelta;
    return { step, g, m, v, before, theta, gradientDelta, decayDelta, delta: theta - before };
  });
}

export function initOptimizationDynamics() {
  const method = $("[data-opt-method]");
  if (!method) return;
  const sequence = $("[data-opt-sequence]"), lr = $("[data-opt-lr]"), beta = $("[data-opt-beta]"), decay = $("[data-opt-decay]");
  let steps = 0;
  const number = x => Math.abs(x) < 0.000005 ? "0.00000" : x.toFixed(5);
  const render = () => {
    const config = { method: method.value, lr: Number(lr.value) / 1000, beta: Number(beta.value) / 100, decay: Number(decay.value) / 100 };
    const rows = optimizerLedger(gradientSequences[sequence.value].slice(0, steps), config), last = rows.at(-1);
    $("[data-opt-lr-value]").textContent = config.lr.toFixed(3);
    $("[data-opt-beta-value]").textContent = config.beta.toFixed(2);
    $("[data-opt-decay-value]").textContent = config.decay.toFixed(2);
    beta.disabled = method.value === "sgd";
    $("[data-opt-theta]").textContent = (last?.theta ?? 1).toFixed(4);
    $("[data-opt-delta]").textContent = number(last?.delta ?? 0);
    $("[data-opt-step]").textContent = `${steps}/7`;
    $("[data-opt-rows]").innerHTML = rows.length ? rows.map(row => `<tr><td>${row.step} / ${row.g}</td><td>${number(row.m)} / ${method.value === "adamw" ? number(row.v) : "不使用"}</td><td>${number(row.gradientDelta)}</td><td>${number(row.decayDelta)}</td><td>${number(row.theta)}</td></tr>`).join("") : '<tr><td colspan="5">初始 θ=1，尚未更新。</td></tr>';
    $("[data-opt-note]").textContent = !last ? "当前没有历史状态。开始后逐项检查位移来源。" : last.g === 0 ? "当前 g=0。历史动量可能仍产生梯度驱动位移，衰减也可能继续改变参数；冻结还需排除优化器更新。" : Math.sign(last.gradientDelta) === Math.sign(last.g) ? "本步梯度驱动位移与当前 −g 方向相反：历史动量仍在起作用。是否降低 loss 还取决于目标曲面和实际步长。" : "更新同时受当前梯度、历史状态和参数衰减影响。比较三种模式时，应分别选择合理的 LR 后比较任务结果。";
    $("[data-opt-next]").disabled = steps === 7;
  };
  [method, sequence, lr, beta, decay].forEach(input => input.addEventListener("input", render));
  $("[data-opt-next]").addEventListener("click", () => { steps = Math.min(7, steps + 1); render(); });
  $("[data-opt-reset]").addEventListener("click", () => { steps = 0; render(); });
  render();
}
