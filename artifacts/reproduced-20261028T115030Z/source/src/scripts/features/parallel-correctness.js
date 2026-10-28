import { $ } from "../core.js";

export function reductionExample(a, b, mode = "weighted") {
  const total = a + b;
  const target = total ? (-2 * a + b) / total : null;
  const actual = !total ? null : mode === "mean" ? (a && b ? -0.5 : null) : mode === "double" ? target / 2 : target;
  return { total, target, actual, theta: actual === null ? null : -0.1 * actual };
}

export function initParallelCorrectness() {
  const aInput = $("[data-reduce-a]");
  if (!aInput) return;
  const bInput = $("[data-reduce-b]"), mode = $("[data-reduce-mode]");
  const fmt = x => x === null ? "未定义" : x.toFixed(3);
  const render = () => {
    const a = Number(aInput.value), b = Number(bInput.value), result = reductionExample(a, b, mode.value);
    $("[data-reduce-a-value]").textContent = a;
    $("[data-reduce-b-value]").textContent = b;
    $("[data-reduce-target]").textContent = fmt(result.target);
    $("[data-reduce-actual]").textContent = fmt(result.actual);
    $("[data-reduce-theta]").textContent = fmt(result.theta);
    $("[data-reduce-rows]").innerHTML = [["A", a, -2], ["B", b, 1]].map(([name, n, g]) => `<tr><td>${name}</td><td>${n}</td><td>${n ? g : "无有效位置"}</td><td>${n * g}</td><td>${result.total ? (100 * n / result.total).toFixed(1) + "%" : "未定义"}</td></tr>`).join("");
    $("[data-reduce-visual]").innerHTML = [["正确", result.target], ["当前", result.actual]].map(([label, g]) => `<div><span>${label}</span><div class="signed-axis"><i style="left:${g === null || g >= 0 ? 50 : 50 + g * 20}%;width:${g === null ? 0 : Math.abs(g) * 20}%;background:var(${label === "正确" ? "--accent" : "--coral"})"></i><b></b></div><output>${fmt(g)}</output></div>`).join("");
    $("[data-reduce-note]").textContent = !result.total ? "本窗口没有有效标签。目标未定义，所有副本需协调跳过，不能单卡自行退出 collective。" : result.actual === null ? "空副本无法计算 local mean。数学上它的 loss sum 为零，但执行还必须保留兼容的计算/通信流程。" : Math.abs(result.actual - result.target) < 1e-10 ? "本例两者数值一致。若采用副本等权平均，应再试不等长样本；还需覆盖其他长度组合。" : Math.sign(result.actual) !== Math.sign(result.target) && result.target !== 0 ? "当前实现甚至改变了梯度方向。局部均值等权把短副本的样本权重放大了。" : "当前实现改变了梯度尺度或目标权重。学习率只缩放整体更新，样本相对权重需从 loss 归一化处修正。";
  };
  [aInput, bInput, mode].forEach(input => input.addEventListener("input", render));
  render();
}
