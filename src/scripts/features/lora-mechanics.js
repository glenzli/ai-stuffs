import { $ } from "../core.js";
import { transpose, matmul, matrixMap, matrixAdd, maxDifference, formatValue, formatVector } from "./objective-math.js";

export function loraParameters(initialization = "zero-b") {
  if (!["zero-b", "both-zero"].includes(initialization)) throw new RangeError("未知的初始化方式。");
  return { W: [[1, .2], [-.3, .7]], A: [initialization === "zero-b" ? [.5, -.25] : [0, 0]], B: [[0], [0]], X: [[2], [-1]], target: [[0], [1]], scale: 1 };
}
export function loraQuantize(matrix) {
  const maximum = Math.max(...matrix.flat().map(Math.abs)), scale = maximum ? maximum / 7 : 1;
  // Explicit nearest with ties away from zero; this example is not NF4.
  const integers = matrixMap(matrix, value => Math.sign(value) * Math.min(7, Math.floor(Math.abs(value) / scale + .5)));
  return { scale, integers, restored: matrixMap(integers, value => value * scale) };
}
export function loraCalculation(parameters = loraParameters(), { precision = "full", graph = "frozen" } = {}) {
  if (!["full", "int4"].includes(precision) || !["frozen", "detach-base", "detach-all"].includes(graph)) throw new RangeError("底座精度或梯度路径无效。");
  const { W, A, B, X, target, scale } = parameters, quantization = loraQuantize(W);
  const base = precision === "int4" ? quantization.restored : W;
  const hidden = matmul(A, X), adapter = matrixMap(matmul(B, hidden), value => value * scale), output = matrixAdd(matmul(base, X), adapter);
  const upstream = matrixMap(output, (value, i, j) => value - target[i][j]);
  const loss = upstream.flat().reduce((sum, value) => sum + value * value / 2, 0);
  const gradients = graph === "detach-all" ? null : {
    A: matrixMap(matmul(matmul(transpose(B), upstream), transpose(X)), value => value * scale),
    B: matrixMap(matmul(upstream, transpose(hidden)), value => value * scale),
    X: matrixAdd(graph === "detach-base" ? [[0], [0]] : matmul(transpose(base), upstream), matrixMap(matmul(transpose(A), matmul(transpose(B), upstream)), value => value * scale)),
  };
  const increment = matrixMap(matmul(B, A), value => value * scale), merged = matrixAdd(base, increment), mergedOutput = matmul(merged, X);
  const requantized = loraQuantize(merged), requantizedOutput = matmul(requantized.restored, X), fullBaseOutput = matrixAdd(matmul(W, X), adapter);
  return { parameters, precision, graph, base, quantization, hidden, adapter, output, upstream, loss, gradients, increment, merged, mergedOutput,
    mergeError: maxDifference(output, mergedOutput), requantized, requantizedOutput, requantizeError: maxDifference(output, requantizedOutput), fullBaseOutput, baseError: maxDifference(output, fullBaseOutput) };
}
export function loraExperiment({ steps = 0, initialization = "zero-b", precision = "full", graph = "frozen", learningRate = .1 } = {}) {
  if (!Number.isInteger(steps) || steps < 0 || steps > 5 || !Number.isFinite(learningRate) || learningRate < 0 || learningRate > .2) throw new RangeError("更新次数或学习率无效。");
  const parameters = loraParameters(initialization), history = [];
  for (let step = 0; step < steps; step++) {
    const calculation = loraCalculation(parameters, { precision, graph });
    history.push({ step, loss: calculation.loss, gradients: structuredClone(calculation.gradients) });
    if (calculation.gradients) for (const key of ["A", "B"]) parameters[key] = matrixMap(parameters[key], (value, i, j) => value - learningRate * calculation.gradients[key][i][j]);
  }
  return { ...loraCalculation(parameters, { precision, graph }), steps, updates: graph === "detach-all" ? 0 : steps, history, learningRate, initialization };
}
export function initLoraMechanics() {
  const steps = $("[data-lm-steps]"); if (!steps) return;
  const precision = $("[data-lm-precision]"), graph = $("[data-lm-graph]"), initialization = $("[data-lm-init]");
  const set = (name, value) => { $(`[data-lm-${name}]`).textContent = value; };
  const render = () => {
    const result = loraExperiment({ steps: Number(steps.value), precision: precision.value, graph: graph.value, initialization: initialization.value });
    set("steps-value", steps.value); set("loss", formatValue(result.loss)); set("output", formatVector(result.output.flat())); set("updates", result.updates);
    const rows = [["A", result.parameters.A, result.gradients?.A], ["B", result.parameters.B, result.gradients?.B], ["输入 x", result.parameters.X, result.gradients?.X]];
    $("[data-lm-gradients]").innerHTML = rows.map(([name, value, gradient]) => `<tr><th scope="row">${name}</th><td>${formatVector(value.flat())}</td><td>${gradient ? formatVector(gradient.flat()) : "无计算图"}</td></tr>`).join("");
    const matrices = [["原始 W", result.parameters.W], ["计算用 W₀", result.base], ["sBA", result.increment], ["合并 W₀+sBA", result.merged], ["合并后重新量化", result.requantized.restored]];
    $("[data-lm-matrices]").innerHTML = matrices.map(([name, matrix]) => `<tr><th scope="row">${name}</th><td>${formatVector(matrix[0])}</td><td>${formatVector(matrix[1])}</td></tr>`).join("");
    set("merge", result.mergeError.toExponential(2)); set("requantize", formatValue(result.requantizeError)); set("base-error", formatValue(result.baseError));
    set("quantization", `底座 scale=${formatValue(result.quantization.scale)}，整数为 ${result.quantization.integers.map(formatVector).join(" / ")}；合并后重新取 max-abs scale=${formatValue(result.requantized.scale)}。`);
    set("note", graph.value === "detach-all" ? "整段输出被切断，loss 数值仍可计算，但没有可回传的计算图。更新按钮对应的尝试不会改变 A/B。" : graph.value === "detach-base" ? "只切断 Wx 分支：A/B 的局部梯度仍存在，输入 x 丢失 W₀ᵀg 这一项；更早的可训练层会收到不同梯度。" : initialization.value === "both-zero" ? "A 和 B 同时为零，二者局部梯度都为零；这个纯 adapter SGD 例子不会离开零增量。" : Number(steps.value) === 0 ? "B=0 时 A 的首步梯度为零，B 可获得非零梯度。W₀ 保持冻结，输入仍接收经过底座的梯度。" : "每次只用该步梯度同时更新 A/B，W₀ 和输入 x 保持固定。表中显示完成这些更新后，当前 loss 对各量的局部梯度。");
  };
  [steps, precision, graph, initialization].forEach(input => { input.addEventListener("input", render); input.addEventListener("change", render); });
  render();
}
