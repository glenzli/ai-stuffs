import { $ } from "../core.js";
import { roundFormat } from "./numerical-computing.js";

const clip = (v, threshold) => { const norm = Math.hypot(...v), coefficient = norm === 0 ? 1 : Math.min(1, threshold / norm); return { norm, coefficient, gradient: v.map(x => x * coefficient) }; };
export function scaledGradientWindow({ gradients = [[2,4],[4,4]], scale = 8, threshold = 1, order = "correct" } = {}) {
  if (!Array.isArray(gradients) || !gradients.length || !gradients.every(v => Array.isArray(v) && v.length === 2 && v.every(Number.isFinite))
      || !Number.isFinite(scale) || scale <= 0 || !Number.isFinite(threshold) || threshold <= 0
      || !["correct", "clip-first", "change-scale"].includes(order)) throw new RangeError("梯度窗口配置无效。");
  let accumulated = [0,0];
  const rows = gradients.map((gradient, i) => {
    const currentScale = order === "change-scale" && i > 0 ? scale * 2 : scale;
    const weighted = gradient.map(x => x / gradients.length), scaled = weighted.map(x => roundFormat(x * currentScale, "fp16"));
    accumulated = accumulated.map((v, j) => roundFormat(v + scaled[j], "fp32"));
    return { gradient: [...gradient], weighted, scale: currentScale, scaled, accumulated: [...accumulated] };
  });
  const reference = clip(gradients[0].map((_, j) => gradients.reduce((s, v) => s + v[j] / gradients.length, 0)), threshold);
  const finite = accumulated.every(Number.isFinite), finalScale = rows.at(-1).scale;
  if (!finite) return { rows, accumulated, finite, finalScale, reference, unscaled: null, applied: null, norm: null, coefficient: null, error: null };
  const unscaled = accumulated.map(x => x / finalScale), clipped = clip(order === "clip-first" ? accumulated : unscaled, threshold);
  const applied = order === "clip-first" ? clipped.gradient.map(x => x / finalScale) : clipped.gradient;
  return { rows, accumulated, finite, finalScale, reference, unscaled, applied, norm: clipped.norm, coefficient: clipped.coefficient,
    error: Math.max(...applied.map((x, i) => Math.abs(x - reference.gradient[i]))) };
}

export function mixedPrecisionRun({ mode = "steady", scale = 8, order = "correct", windows = 4 } = {}) {
  if (!["steady", "overflow", "tiny"].includes(mode) || !Number.isInteger(windows) || windows < 1 || windows > 8) throw new RangeError("更新序列配置无效。");
  let parameters = [.8,-.5].map(Math.fround), currentScale = scale, updates = 0, good = 0;
  const rows = [];
  for (let i = 0; i < windows; i++) {
    const gradients = mode === "tiny" ? [[2e-8,-1e-8],[2e-8,-1e-8]] : mode === "overflow" && i === 0 ? [[20000,40000],[20000,40000]] : [[2,4],[4,4]];
    const result = scaledGradientWindow({ gradients, scale: currentScale, order }), before = [...parameters], lr = .05 / (1 + .2 * updates);
    if (result.finite) {
      parameters = parameters.map((x, j) => roundFormat(x - lr * result.applied[j], "fp32"));
      updates++; good++;
      currentScale = result.finalScale * (good === 2 ? 2 : 1);
      if (good === 2) good = 0;
    } else { currentScale = result.finalScale / 2; good = 0; }
    rows.push({ window: i + 1, scale: result.finalScale, nextScale: currentScale, result, before, after: [...parameters], lr, updates, skipped: !result.finite });
  }
  return { rows, updates, parameters, scale: currentScale };
}

const fmt = v => Number.isNaN(v) ? "NaN" : Number.isFinite(v) ? (Math.abs(v) > 0 && Math.abs(v) < .0001 ? v.toExponential(3) : Number(v.toPrecision(6)).toString()) : v > 0 ? "+∞" : "−∞";
const vector = v => v ? `[${v.map(fmt).join(", ")}]` : "跳过";
export function initMixedPrecisionStep() {
  const mode = $("[data-mp-mode]"); if (!mode) return;
  const scale = $("[data-mp-scale]"), order = $("[data-mp-order]"), window = $("[data-mp-window]");
  const set = (name, value) => { $(`[data-mp-${name}]`).textContent = value; };
  const render = () => {
    const run = mixedPrecisionRun({ mode: mode.value, scale: Number(scale.value), order: order.value });
    const current = run.rows[Number(window.value) - 1], r = current.result;
    set("updates", `${run.updates} / 4`); set("next-scale", fmt(run.scale)); set("params", vector(run.parameters));
    set("unscaled", vector(r.unscaled)); set("applied", vector(r.applied)); set("reference", vector(r.reference.gradient));
    set("norm", r.norm === null ? "跳过" : fmt(r.norm)); set("coefficient", r.coefficient === null ? "跳过" : fmt(r.coefficient));
    $("[data-mp-micro-rows]").innerHTML = r.rows.map((row, i) => `<tr><td>${i + 1}</td><td>${vector(row.gradient)}</td><td>${fmt(row.scale)}</td><td>${vector(row.scaled)}</td><td>${vector(row.accumulated)}</td></tr>`).join("");
    $("[data-mp-update-rows]").innerHTML = run.rows.map(row => `<tr><td>${row.window}</td><td>${fmt(row.scale)} → ${fmt(row.nextScale)}</td><td>${row.skipped ? "跳过更新" : `更新 ${row.updates}`}</td><td>${fmt(row.lr)}</td><td>${vector(row.after)}</td></tr>`).join("");
    const notes = { correct: "先完成两份梯度累积，再统一 unscale、检查与裁剪。", "clip-first": "错误对照：在缩放后的梯度上使用阈值 1，再除以 scale；实际阈值随 scale 缩小。", "change-scale": "错误对照：第二个 microbatch 的 scale 临时翻倍，最后统一除以新 scale，两个贡献被赋予不同权重。" };
    set("note", `${notes[order.value]} ${r.finite ? `所选窗口与理想算术裁剪结果的最大差：${fmt(r.error)}。` : "所选窗口出现非有限梯度，参数及学习率计数保持原值，降低 scale 后进入下一窗口。"}`);
  };
  for (const control of [mode, scale, order, window]) control.addEventListener("change", render);
  render();
}
