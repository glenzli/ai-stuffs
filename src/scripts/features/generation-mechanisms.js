import { $, $$ } from "../core.js";

export function initGenerationMechanisms() {
  if (!$("[data-gen-tape]")) return;
  const text = ["先", "列", "条件", "再", "算", "结果"];
  const states = [
    ["?", "?", "?", "?", "?", "?"],
    ["先", "?", "条件", "?", "?", "结果"],
    ["先", "列", "条件", "再", "?", "结果"],
    text,
  ];
  let mode = "ar", step = 0;
  const render = () => {
    const values = mode === "ar" ? text.map((token, i) => i < step ? token : "?") : states[step];
    $("[data-gen-tape]").innerHTML = values.map((token, i) => `<span class="${token === "?" ? "" : mode === "ar" && i === step - 1 ? "pending" : "accepted"}">${token}</span>`).join("");
    $("[data-gen-title]").textContent = mode === "ar" ? `自回归 · 已生成 ${step} / 6 个位置` : `离散去噪 · 第 ${step} / 3 轮`;
    $("[data-gen-copy]").textContent = mode === "ar" ? "本步只决定紧随前缀的一个 token。后续分布依赖已经生成的前缀；新 token 要在下一次 forward 才成为带 KV 的输入位置。" : "每轮依据当前可见序列预测多个未定位置。本例按预设日程揭示 token。实际算法还需定义置信度、重遮与采样规则；状态变化可能令旧 KV 失效。";
    $("[data-gen-next]").disabled = step >= (mode === "ar" ? 6 : 3);
    $$("[data-gen-mode]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.genMode === mode)));
  };
  $$("[data-gen-mode]").forEach(button => button.addEventListener("click", () => { mode = button.dataset.genMode; step = 0; render(); }));
  $("[data-gen-next]").addEventListener("click", () => { step = Math.min(step + 1, mode === "ar" ? 6 : 3); render(); });
  $("[data-gen-reset]").addEventListener("click", () => { step = 0; render(); });
  render();
}
