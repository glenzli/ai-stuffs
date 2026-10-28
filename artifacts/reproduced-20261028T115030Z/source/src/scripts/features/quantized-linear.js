import { $ } from "../core.js";
import { matmul, maxDifference, formatValue, formatVector } from "./objective-math.js";
import { roundFormat } from "./numerical-computing.js";

export const roundAway = x => Math.sign(x) * Math.floor(Math.abs(x) + .5);
const clamp = (x, low, high) => Math.max(low, Math.min(high, x));
export function affineCalibration(values, bits = 8) {
  if (!values.length || !values.every(Number.isFinite) || !Number.isInteger(bits) || bits < 2 || bits > 8) throw new RangeError("校准数据或位宽无效。");
  const low = Math.min(0, ...values), high = Math.max(0, ...values), qmax = 2 ** bits - 1;
  const scale = high === low ? 1 : (high - low) / qmax, zero = clamp(roundAway(-low / scale), 0, qmax);
  return { low, high, qmax, scale, zero };
}
const quantize = (value, calibration) => {
  const raw = roundAway(value / calibration.scale) + calibration.zero, q = clamp(raw, 0, calibration.qmax);
  return { q, restored: (q - calibration.zero) * calibration.scale, saturated: raw !== q };
};
export function quantizedLinearInputs(scenario = "standard") {
  const data = { X: [[-.7, .3, 1.2], [1.1, -.5, .1]], W: [[.4, -1.1], [-.8, .2], [.6, .9]], bias: [.07, -.11], calibrationX: [[-1.2, -.2, .7], [.4, .9, 1.6]] };
  if (scenario === "outlier") data.W[2][1] = 9;
  else if (scenario === "saturation") data.X = data.X.map(row => row.map(x => 8 * x));
  else if (scenario === "zero") for (const key of ["X", "W", "calibrationX"]) data[key] = data[key].map(row => row.map(() => 0));
  else if (scenario !== "standard") throw new RangeError("未知的量化输入场景。");
  if (scenario === "zero") data.bias = [0, 0];
  return data;
}

export function quantizedLinear({ scenario = "standard", fault = "none", mode = "w8a8", inputs = quantizedLinearInputs(scenario) } = {}) {
  if (!["none", "zero-point", "scale-axis", "bias-unit"].includes(fault) || !["w8a8", "w4a16"].includes(mode)) throw new RangeError("量化模式或故障无效。");
  const { X, W, bias, calibrationX } = inputs;
  if (X?.length !== 2 || W?.length !== 3 || calibrationX?.length !== 2 || bias?.length !== 2 || ![...X, ...calibrationX].every(row => row.length === 3 && row.every(Number.isFinite)) || !W.every(row => row.length === 2 && row.every(Number.isFinite)) || !bias.every(Number.isFinite)) throw new RangeError("本例固定 X:[2,3]、W:[3,2]、bias:[2]。");
  const floating = matmul(X, W).map(row => row.map((x, j) => x + bias[j]));
  const sx = affineCalibration(calibrationX.flat()), sw = W[0].map((_, j) => affineCalibration(W.map(row => row[j])));
  const calibrationY = matmul(calibrationX, W).map(row => row.map((x, j) => x + bias[j])), sy = affineCalibration(calibrationY.flat());
  if (mode === "w4a16") {
    const scales = W[0].map((_, j) => Math.max(...W.map(row => Math.abs(row[j]))) / 7 || 1);
    const qW = W.map(row => row.map((w, j) => clamp(roundAway(w / scales[j]), -7, 7))), restoredW = qW.map(row => row.map((q, j) => q * scales[j]));
    const restoredX = X.map(row => row.map(x => roundFormat(x, "fp16")));
    const restored = matmul(restoredX, restoredW).map(row => row.map((x, j) => x + bias[j]));
    return { scenario, fault: "none", mode, inputs, floating, restored, integerRestored: restored, output: restored, qW, restoredX, restoredW, scales,
      arithmeticError: 0, approximationError: maxDifference(restored, floating), outputError: maxDifference(restored, floating), weightError: maxDifference(W, restoredW), saturation: 0,
      bytes: { weightPayload: 3, scaleMetadata: 8, zeroMetadata: 0, activationPayload: 12, biasPayload: 8 }, rows: floating.flatMap((row, i) => row.map((x, j) => ({ i, j, floating: x, restored: restored[i][j], actual: restored[i][j], output: restored[i][j] }))) };
  }
  const xRecords = X.map(row => row.map(x => quantize(x, sx))), wRecords = W.map(row => row.map((w, j) => quantize(w, sw[j])));
  const qX = xRecords.map(row => row.map(x => x.q)), qW = wRecords.map(row => row.map(x => x.q));
  const restoredX = xRecords.map(row => row.map(x => x.restored)), restoredW = wRecords.map(row => row.map(x => x.restored));
  const biasScale = sw.map(w => sx.scale * w.scale), qBias = bias.map((b, j) => roundAway(b / biasScale[j])), restoredBias = qBias.map((b, j) => b * biasScale[j]);
  if (qBias.some(b => !Number.isSafeInteger(b) || Math.abs(b) + 3 * 255 ** 2 > 2147483647)) throw new RangeError("累加器或 bias 超出声明的 INT32 范围。");
  const restored = matmul(restoredX, restoredW).map(row => row.map((x, j) => x + restoredBias[j]));
  const rows = floating.flatMap((row, i) => row.map((x, j) => {
    const raw = qX[i].reduce((sum, q, k) => sum + q * qW[k][j], 0), rowSum = qX[i].reduce((a, b) => a + b, 0), colSum = qW.reduce((sum, r) => sum + r[j], 0);
    const correction = -sw[j].zero * rowSum - sx.zero * colSum + 3 * sx.zero * sw[j].zero;
    const accumulator = raw + correction, direct = qX[i].reduce((sum, q, k) => sum + (q - sx.zero) * (qW[k][j] - sw[j].zero), 0);
    const actualAccumulator = fault === "zero-point" ? raw : accumulator;
    const actualScale = fault === "scale-axis" ? biasScale[0] : biasScale[j];
    const actual = fault === "bias-unit" ? actualAccumulator * actualScale + qBias[j] : (actualAccumulator + qBias[j]) * actualScale;
    const correct = (accumulator + qBias[j]) * biasScale[j], output = quantize(actual, sy);
    return { i, j, floating: x, raw, correction, accumulator, direct, qBias: qBias[j], scale: biasScale[j], actualScale, restored: restored[i][j], correct, actual, output: output.restored, qOutput: output.q, outputSaturated: output.saturated };
  }));
  const matrix = key => X.map((_, i) => W[0].map((_, j) => rows.find(row => row.i === i && row.j === j)[key]));
  const integerRestored = matrix("actual"), output = matrix("output");
  return { scenario, fault, mode, inputs, sx, sw, sy, biasScale, qBias, restoredBias, qX, qW, restoredX, restoredW, floating, restored, integerRestored, output, rows,
    arithmeticError: maxDifference(integerRestored, restored), approximationError: maxDifference(restored, floating), outputError: maxDifference(output, floating), weightError: maxDifference(W, restoredW),
    saturation: xRecords.flat().filter(x => x.saturated).length, outputSaturation: rows.filter(row => row.outputSaturated).length,
    bytes: { weightPayload: 6, scaleMetadata: 8, zeroMetadata: 2, activationPayload: 6, biasPayload: 8 } };
}

export function initQuantizedLinear() {
  const scenario = $("[data-ql-scenario]"); if (!scenario) return;
  const fault = $("[data-ql-fault]"), mode = $("[data-ql-mode]");
  const set = (name, value) => { $(`[data-ql-${name}]`).textContent = value; };
  const render = () => {
    const r = quantizedLinear({ scenario: scenario.value, fault: fault.value, mode: mode.value }); fault.disabled = mode.value === "w4a16";
    set("weight-error", r.weightError.toExponential(2)); set("arithmetic", r.arithmeticError.toExponential(2)); set("approximation", r.approximationError.toExponential(2)); set("output-error", r.outputError.toExponential(2));
    set("saturation", `${r.saturation} 个输入${r.mode === "w8a8" ? `，${r.outputSaturation} 个输出` : "；此模式未量化输出"}`);
    $("[data-ql-results]").innerHTML = r.rows.map(row => `<tr><th scope="row">Y[${row.i},${row.j}]</th><td>${formatValue(row.floating)}</td><td>${formatValue(row.restored)}</td><td>${formatValue(row.actual)}</td><td>${formatValue(row.output)}</td></tr>`).join("");
    $("[data-ql-integers]").innerHTML = r.mode === "w8a8" ? r.rows.map(row => `<tr><td>${row.i},${row.j}</td><td>${row.raw}</td><td>${row.correction}</td><td>${row.accumulator}</td><td>${row.qBias}</td><td>${row.scale.toExponential(4)}</td><td>${row.qOutput}</td></tr>`).join("") : '<tr><td colspan="7">W4A16 对照先恢复权重，用舍入到 FP16 的激活计算；没有整数输入点积账本。</td></tr>';
    set("calibration", r.mode === "w8a8" ? `输入 s=${formatValue(r.sx.scale)}, z=${r.sx.zero}；权重 s=${formatVector(r.sw.map(w => w.scale))}, z=${formatVector(r.sw.map(w => w.zero))}；输出 s=${formatValue(r.sy.scale)}, z=${r.sy.zero}。` : `权重逐输出通道对称刻度 −7…7；scale=${formatVector(r.scales)}。输入先按最近偶数舍入为 FP16 可表示值，乘加用 binary64。`);
    set("storage", `6 个权重：原始低位数值 ${r.bytes.weightPayload} B，假定 FP32 scales ${r.bytes.scaleMetadata} B，零点 ${r.bytes.zeroMetadata} B；本例 6 个 FP16 权重为 12 B。小矩阵的元数据可能抵消压缩收益。`);
    const notes = { none: "整数校正与恢复后的矩阵乘应一致；浮点原始层与量化层允许近似误差。输出重定标还会再次舍入和饱和。", "zero-point": "故障：省略零点修正，把原始整数乘积直接恢复。输入与权重的整数零点不为 0 时偏差明显。", "scale-axis": "故障：所有输出都使用第 0 通道的 scale，第 1 通道失去自己的单位。两通道 scale 恰好相同时故障可能被掩盖。", "bias-unit": "故障：把 INT32 bias 整数当成实数直接相加，丢失 sX×sW 的单位转换。bias=0 时此故障不暴露。" };
    set("note", r.mode === "w4a16" ? "该对照核对权重量化与 FP16 激活舍入的机制；没有执行 INT4 kernel，也没有模拟 FP16 乘加吞吐。" : notes[fault.value]);
    $("[data-ql-inputs]").textContent = JSON.stringify({ ...r.inputs, qX: r.qX, qW: r.qW, qBias: r.qBias }, null, 2);
  };
  [scenario, fault, mode].forEach(el => el.addEventListener("change", render)); render();
}
