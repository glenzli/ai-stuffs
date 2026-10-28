// Small binary64 reference operations. These are not tensor-library kernels.
export const zeros = (rows, columns) => Array.from({ length: rows }, () => Array(columns).fill(0));
export const transpose = a => a[0].map((_, j) => a.map(row => row[j]));
export const matmul = (a, b) => a.map(row => b[0].map((_, j) => row.reduce((sum, x, k) => sum + x * b[k][j], 0)));
export const matrixMap = (a, fn) => a.map((row, i) => row.map((value, j) => fn(value, i, j)));
export const matrixAdd = (a, b) => matrixMap(a, (value, i, j) => value + b[i][j]);
export const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
export function softmax(values) {
  const maximum = Math.max(...values), exp = values.map(value => Math.exp(value - maximum));
  const denominator = exp.reduce((a, b) => a + b, 0);
  return exp.map(value => value / denominator);
}
export function crossEntropy(logits, target) {
  const maximum = Math.max(...logits);
  return maximum + Math.log(logits.reduce((sum, x) => sum + Math.exp(x - maximum), 0)) - logits[target];
}
export function maxDifference(a, b) {
  const left = a.flat(Infinity), right = b.flat(Infinity);
  if (left.length !== right.length) throw new RangeError("比较的元素数量必须相同。");
  return Math.max(0, ...left.map((value, i) => Math.abs(value - right[i])));
}
export const formatValue = value => value === null ? "未定义" : Math.abs(value) < 1e-12 ? "0" : Number(value.toFixed(6)).toString();
export const formatVector = values => `[${values.map(formatValue).join(", ")}]`;
export const escapeText = value => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
