import { $ } from "../core.js";

const weights = [0.9, 1.1, 0.8, 1.2, 0.7, 1.05];
const phases = { initial: "初始", forward: "前向", loss: "损失", recompute: "重算", backward: "反向", release: "释放" };

function scalarReference() {
  const values = [0.8];
  for (const weight of weights) values.push(Math.tanh(weight * values.at(-1)));
  const gradients = Array(weights.length);
  let upstream = values.at(-1);
  for (let layer = weights.length; layer > 0; layer--) {
    const derivative = upstream * (1 - values[layer] ** 2);
    gradients[layer - 1] = derivative * values[layer - 1];
    upstream = derivative * weights[layer - 1];
  }
  return gradients;
}

export function activationTrace(mode = "store", segment = 2) {
  if (mode !== "store" && mode !== "checkpoint") throw new RangeError("激活保留模式无效。");
  if (!Number.isInteger(segment) || segment < 1 || segment > weights.length) {
    throw new RangeError("分段层数必须为 1 至 6 的整数。");
  }
  const boundaries = [0];
  for (let layer = segment; layer < weights.length; layer += segment) boundaries.push(layer);
  boundaries.push(weights.length);
  const checkpoints = new Set(boundaries);
  const live = new Set([0]);
  const values = [0.8];
  const gradients = Array(weights.length);
  const events = [];
  let forwardCount = 0, backwardCount = 0, upstream;
  const record = (phase, label) => {
    const ids = Array.from(live).sort((a, b) => a - b).map(layer => `h${layer}`);
    events.push({ phase, label, live: ids, count: ids.length, forwardCount, backwardCount });
  };
  const release = layer => {
    live.delete(layer);
    record("release", `释放 h${layer}`);
  };
  const forward = (layer, phase) => {
    const reused = live.has(layer);
    values[layer] = Math.tanh(weights[layer - 1] * values[layer - 1]);
    live.add(layer);
    forwardCount++;
    record(phase, `${phases[phase]}第 ${layer} 层：计算 h${layer}${reused ? "，复用段末边界" : ""}`);
  };
  const backward = layer => {
    const derivative = upstream * (1 - values[layer] ** 2);
    gradients[layer - 1] = derivative * values[layer - 1];
    upstream = derivative * weights[layer - 1];
    backwardCount++;
    record("backward", `反向第 ${layer} 层：使用 h${layer - 1}、h${layer}`);
    release(layer);
  };

  record("initial", "保留初始激活 h0");
  for (let layer = 1; layer <= weights.length; layer++) {
    forward(layer, "forward");
    if (mode === "checkpoint" && !checkpoints.has(layer - 1)) release(layer - 1);
  }
  const loss = 0.5 * values.at(-1) ** 2;
  upstream = values.at(-1);
  record("loss", "计算损失，保留末层激活 h6");
  if (mode === "store") {
    for (let layer = weights.length; layer > 0; layer--) backward(layer);
  } else {
    // Numeric values support arithmetic; the live set alone models logical activation retention.
    for (let index = boundaries.length - 1; index > 0; index--) {
      const start = boundaries[index - 1], end = boundaries[index];
      for (let layer = start + 1; layer <= end; layer++) forward(layer, "recompute");
      for (let layer = end; layer > start; layer--) backward(layer);
    }
  }
  release(0);
  return {
    events, peak: Math.max(...events.map(event => event.count)),
    totalForward: forwardCount, gradients, loss, referenceGradients: scalarReference(),
  };
}

export function initActivationCheckpointing() {
  const mode = $("[data-ac-mode]");
  if (!mode) return;
  const segment = $("[data-ac-segment]");
  const next = $("[data-ac-next]"), resetButton = $("[data-ac-reset]");
  let step = 0, trace;
  const render = () => {
    const shown = trace.events.slice(0, step + 1), current = shown.at(-1);
    $("[data-ac-step]").textContent = `${step}/${trace.events.length - 1}`;
    $("[data-ac-peak]").textContent = String(Math.max(...shown.map(event => event.count)));
    $("[data-ac-forward]").textContent = String(current.forwardCount);
    $("[data-ac-loss]").textContent = shown.some(event => event.phase === "loss") ? trace.loss.toFixed(8) : "尚未计算";
    const error = Math.max(...trace.gradients.map((value, index) => Math.abs(value - trace.referenceGradients[index])));
    $("[data-ac-error]").textContent = current.backwardCount === weights.length
      ? (error === 0 ? "0" : error.toExponential(3)) : "尚未完成反向";
    $("[data-ac-live]").innerHTML = current.live.length
      ? current.live.map(id => `<span>${id}</span>`).join("") : "无保留激活";
    $("[data-ac-events]").innerHTML = shown.map((event, index) =>
      `<tr><td>${index}：${event.label}</td><td>${phases[event.phase]}</td><td>${event.live.join("、") || "无"}</td><td>${event.count}</td><td>${event.forwardCount}</td></tr>`
    ).join("");
    const explanation = mode.value === "store"
      ? "全部保留模式保存 h0 至 h6。每层反向使用输入和输出，求导后释放输出。"
      : `每 ${segment.value} 层分段，前向仅保留段边界。每段反向前完整重算该段全部前向，这是本例的简化；段末相同值复用同一逻辑编号。反向逐层释放输出，段首留给上一段。分段层数为 1 时所有激活都是边界，峰值与全部保留相同，计算次数仍增加。`;
    $("[data-ac-note]").textContent = `${explanation}计数只含唯一逻辑激活，不含参数、梯度或临时工作区，也不代表脚本实际内存占用。前向和重算计入输入与新输出同时存在的峰值；h0 保留到最后一次反向结束。这里只执行标量数值运算。`;
    segment.disabled = mode.value === "store";
    next.disabled = step === trace.events.length - 1;
  };
  const reset = () => {
    trace = activationTrace(mode.value, Number(segment.value));
    step = 0;
    render();
  };
  for (const event of ["input", "change"]) {
    mode.addEventListener(event, reset);
    segment.addEventListener(event, reset);
  }
  next.addEventListener("click", () => { step = Math.min(step + 1, trace.events.length - 1); render(); });
  resetButton.addEventListener("click", reset);
  reset();
}
