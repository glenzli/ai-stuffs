import { $ } from "../core.js";

const rotation = degrees => {
  const radians = degrees * Math.PI / 180, c = Math.cos(radians), s = Math.sin(radians);
  return [[c, -s], [s, c]];
};
const transpose = matrix => matrix[0].map((_, column) => matrix.map(row => row[column]));
const multiply = (a, b) => a.map(row => b[0].map((_, column) => row.reduce((sum, value, index) => sum + value * b[index][column], 0)));
const frobenius = matrix => Math.hypot(...matrix.flat());
const normalize = matrix => {
  const norm = frobenius(matrix);
  return matrix.map(row => row.map(value => value / norm));
};

export function matrixDirections(ratio = 8) {
  if (!Number.isFinite(ratio) || ratio < 1 || ratio > 32) {
    throw new RangeError("ratio must be a finite number in [1, 32].");
  }
  // The family has a known decomposition; no numerical SVD or iterative polar solver.
  const u = rotation(30), vt = transpose(rotation(20));
  const raw = multiply(multiply(u, [[ratio, 0], [0, 1]]), vt);
  const polar = multiply(u, vt);
  const adam = raw.map(row => row.map(value => value / (Math.abs(value) + 1e-8)));
  const qqt = multiply(polar, transpose(polar));
  const orthogonalityError = frobenius(qqt.map((row, i) => row.map((value, j) => value - (i === j ? 1 : 0))));
  // Equal Frobenius norms compare directions only, not optimizer step sizes.
  return {
    singulars: [ratio, 1], raw, polar, orthogonalityError,
    sgdDirection: normalize(raw), adamDirection: normalize(adam), polarDirection: normalize(polar),
  };
}

export function initMatrixOptimizer() {
  const ratio = $("[data-mu-ratio]");
  if (!ratio) return;
  const number = value => Math.abs(value) < 0.00005 ? "0.0000" : value.toFixed(4);
  const render = () => {
    const result = matrixDirections(Number(ratio.value));
    $("[data-mu-ratio-value]").textContent = String(result.singulars[0]);
    $("[data-mu-rows]").innerHTML = result.raw.flatMap((row, i) => row.map((value, j) =>
      `<tr><th scope="row">(${i + 1}, ${j + 1})</th><td>${number(value)}</td><td>${number(result.sgdDirection[i][j])}</td><td>${number(result.adamDirection[i][j])}</td><td>${number(result.polarDirection[i][j])}</td></tr>`
    )).join("");
    $("[data-mu-note]").textContent = `已知分解 M = R(30°) diag(${result.singulars.join(", ")}) R(20°)ᵀ，极因子 Q = UVᵀ；||QQᵀ - I||_F = ${result.orthogonalityError.toExponential(2)}。逐元素列采用首步 AdamW 的 M/(|M| + 1e-8)，忽略权重衰减。三列方向各自归一化为 Frobenius 范数 1，仅比较方向。极因子列是 Muon 的方向几何示意。`;
  };
  ratio.addEventListener("input", render);
  ratio.addEventListener("change", render);
  render();
}
