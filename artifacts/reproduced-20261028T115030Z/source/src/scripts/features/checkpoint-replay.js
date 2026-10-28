import { $ } from "../core.js";
import { blockParameters, blockBackward } from "./backpropagation.js";

const seed = 20261027;
const data = [
  { id: "A", X: [[.5,-.2],[.1,.7]], target: 1 },
  { id: "B", X: [[-.3,.6],[.8,.2]], target: 2 },
  { id: "C", X: [[.2,.9],[-.4,.3]], target: 0 },
  { id: "D", X: [[.7,.1],[.3,-.6]], target: 1 },
];
const matrixMap = (matrix, fn) => matrix.map((row, i) => row.map((v, j) => fn(v, i, j)));
const zeros = parameters => Object.fromEntries(Object.entries(parameters).map(([key, matrix]) => [key, matrixMap(matrix, () => 0)]));
const maxDifference = (a, b) => Math.max(...Object.keys(a).flatMap(key => a[key].flat().map((v, i) => Math.abs(v - b[key].flat()[i]))));

export function initialTrainingState() {
  const { X, ...parameters } = blockParameters();
  return { version: 1, config: "tiny-block-adamw-noise-v1", parameters,
    optimizer: { m: zeros(parameters), v: zeros(parameters), t: 0 }, scheduler: 0, rng: seed, cursor: 0, step: 0 };
}

export function validateTrainingState(state) {
  const initial = initialTrainingState();
  if (state?.version !== 1 || state.config !== initial.config) throw new RangeError("Checkpoint 配置或版本不匹配。");
  for (const key of ["scheduler", "cursor", "step", "rng"]) if (!Number.isSafeInteger(state[key]) || state[key] < 0) throw new RangeError(`Checkpoint ${key} 无效。`);
  if (state.rng > 0xffffffff || !Number.isSafeInteger(state.optimizer?.t) || state.optimizer.t < 0) throw new RangeError("Checkpoint 优化器或随机状态无效。");
  for (const group of [state.parameters, state.optimizer.m, state.optimizer.v]) {
    if (!group || Object.keys(group).length !== Object.keys(initial.parameters).length) throw new RangeError("Checkpoint 参数集合不匹配。");
    for (const [key, matrix] of Object.entries(initial.parameters)) {
      const value = group[key];
      if (!Array.isArray(value) || value.length !== matrix.length || !value.every((row, i) => Array.isArray(row) && row.length === matrix[i].length && row.every(Number.isFinite))) throw new RangeError(`Checkpoint ${key} 形状或数值无效。`);
    }
  }
  if (Object.values(state.optimizer.v).flat(2).some(v => v < 0)) throw new RangeError("平方矩不能为负。");
  return state;
}

function uniform(state) {
  state.rng = (Math.imul(state.rng, 1664525) + 1013904223) >>> 0;
  return state.rng / 4294967296;
}

export function trainingStep(state) {
  const sample = data[state.cursor % data.length], rngBefore = state.rng;
  const X = matrixMap(sample.X, v => v + .05 * (2 * uniform(state) - 1));
  const result = blockBackward({ ...state.parameters, X }, sample.target), lr = .015 / (1 + .15 * state.scheduler);
  const gradientNorm = Math.hypot(...Object.keys(state.parameters).flatMap(key => result.gradients[key].flat()));
  state.optimizer.t++;
  for (const [key, matrix] of Object.entries(state.parameters)) {
    state.parameters[key] = matrixMap(matrix, (value, i, j) => {
      const gradient = result.gradients[key][i][j];
      const m = state.optimizer.m[key][i][j] = .9 * state.optimizer.m[key][i][j] + .1 * gradient;
      const v = state.optimizer.v[key][i][j] = .99 * state.optimizer.v[key][i][j] + .01 * gradient ** 2;
      return value * (1 - lr * .01) - lr * (m / (1 - .9 ** state.optimizer.t)) / (Math.sqrt(v / (1 - .99 ** state.optimizer.t)) + 1e-8);
    });
  }
  state.cursor++; state.scheduler++; state.step++;
  return { step: state.step, sample: sample.id, target: sample.target, rngBefore, input: X, lr, loss: result.forward.loss, gradientNorm,
    optimizerStep: state.optimizer.t, parameters: structuredClone(state.parameters) };
}

export function runTraining(state, count) {
  validateTrainingState(state);
  if (!Number.isInteger(count) || count < 0 || count > 1000) throw new RangeError("训练步数必须为 0 到 1000 的整数。");
  return Array.from({ length: count }, () => trainingStep(state));
}

export function restoreTrainingState(checkpoint, omission = "none") {
  validateTrainingState(checkpoint);
  if (!["none", "optimizer", "scheduler", "rng", "data", "weights-only"].includes(omission)) throw new RangeError("恢复模式无效。");
  const state = JSON.parse(JSON.stringify(checkpoint)), initial = initialTrainingState();
  if (omission === "optimizer" || omission === "weights-only") state.optimizer = initial.optimizer;
  if (omission === "scheduler" || omission === "weights-only") state.scheduler = initial.scheduler;
  if (omission === "rng" || omission === "weights-only") state.rng = initial.rng;
  if (omission === "data" || omission === "weights-only") state.cursor = initial.cursor;
  return state;
}

export function checkpointExperiment({ cut = 3, total = 8, omission = "none" } = {}) {
  if (!Number.isInteger(cut) || cut < 1 || !Number.isInteger(total) || total <= cut || total > 20) throw new RangeError("保存点与总步数无效。");
  const baselineState = initialTrainingState(), baseline = runTraining(baselineState, total);
  const interrupted = initialTrainingState(); runTraining(interrupted, cut);
  const checkpoint = JSON.parse(JSON.stringify(interrupted)), restoredState = restoreTrainingState(checkpoint, omission);
  const resumed = runTraining(restoredState, total - cut);
  const rows = resumed.map((row, i) => ({ baseline: baseline[cut + i], resumed: row,
    difference: maxDifference(baseline[cut + i].parameters, row.parameters) }));
  return { checkpoint, baseline, rows, restoredState, baselineState,
    maxError: Math.max(...rows.map(row => row.difference)), firstDivergence: rows.find(row => row.difference > 1e-12)?.resumed.step ?? null };
}

export function initCheckpointReplay() {
  const omission = $("[data-cr-omission]"); if (!omission) return;
  const cut = $("[data-cr-cut]");
  const fmt = v => v.toFixed(6), set = (key, value) => { $(`[data-cr-${key}]`).textContent = value; };
  $("[data-cr-data]").innerHTML = data.map(sample => `<tr><td>${sample.id}</td><td>${sample.X.map(row => `[${row.join(", ")}]`).join("<br>")}</td><td>${sample.target}</td></tr>`).join("");
  const render = () => {
    const result = checkpointExperiment({ cut: Number(cut.value), omission: omission.value });
    set("first", result.firstDivergence === null ? "轨迹一致" : `第 ${result.firstDivergence} 步`);
    set("error", result.maxError.toExponential(2));
    set("saved", `完成 ${cut.value} 次更新`);
    $("[data-cr-rows]").innerHTML = result.rows.map(row => `<tr><td>${row.resumed.step}</td><td>${row.baseline.sample} / ${row.resumed.sample}</td><td>${fmt(row.baseline.lr)} / ${fmt(row.resumed.lr)}</td><td>${fmt(row.baseline.loss)} / ${fmt(row.resumed.loss)}</td><td>${fmt(row.baseline.gradientNorm)} / ${fmt(row.resumed.gradientNorm)}</td><td>${row.difference.toExponential(2)}</td></tr>`).join("");
    const saved = result.checkpoint;
    const restored = restoreTrainingState(saved, omission.value);
    $("[data-cr-state-rows]").innerHTML = [
      ["Adam 步数", saved.optimizer.t, restored.optimizer.t], ["LR 计数", saved.scheduler, restored.scheduler],
      ["RNG 整数状态", saved.rng, restored.rng], ["已读样本数", saved.cursor, restored.cursor],
    ].map(row => `<tr>${row.map(value => `<td>${value}</td>`).join("")}</tr>`).join("");
    const notes = { none: "完整恢复参数、Adam 状态、学习率计数、随机状态和数据游标。", optimizer: "清空 Adam 的一阶矩、二阶矩和偏置校正步数；其余状态恢复。", scheduler: "学习率计数从 0 重新开始；Adam 状态保留。", rng: "随机源从初始种子重新开始，恢复后的输入扰动发生变化。", data: "数据游标回到样本 A；若保存点恰好位于四样本周期边界，本例的后续样本可以相同。", "weights-only": "仅保留权重和显示用的总步数，优化器、学习率、RNG 与数据位置全部重置。" };
    set("note", `${notes[omission.value]} 每格左侧是未中断基线，右侧是恢复运行；loss 和范数取更新前，参数差取更新后。`);
  };
  omission.addEventListener("change", render); cut.addEventListener("change", render); render();
}
