import { $ } from "../core.js";

export function onlineAttention({ block = 2, offset = 0 } = {}) {
  if (!Number.isInteger(block) || block < 1 || block > 6) throw new RangeError("块大小须为 1 到 6 的整数。");
  if (!Number.isFinite(offset) || Math.abs(offset) > 1000) throw new RangeError("偏移须在 −1000 到 1000 之间。");
  const scores = [1, 2, -1, 3, 0, 2.5].map(x => x + offset);
  const values = [[1, 0], [0, 2], [-1, 1], [2, -1], [1, 1], [0, -2]];
  const max = Math.max(...scores), fullExp = scores.map(x => Math.exp(x - max)), total = fullExp.reduce((a, b) => a + b);
  const fullProbs = fullExp.map(x => x / total);
  const reference = [0, 1].map(j => fullProbs.reduce((s, p, i) => s + p * values[i][j], 0));
  let m = -Infinity, l = 0, a = [0, 0];
  const rows = [];
  for (let start = 0; start < scores.length; start += block) {
    const end = Math.min(start + block, scores.length), localScores = scores.slice(start, end), localMax = Math.max(...localScores);
    const newMax = Math.max(m, localMax), scale = m === -Infinity ? 0 : Math.exp(m - newMax);
    l *= scale; a = a.map(x => x * scale);
    for (let i = start; i < end; i++) {
      const e = Math.exp(scores[i] - newMax);
      l += e; a = a.map((x, j) => x + e * values[i][j]);
    }
    m = newMax;
    rows.push({ start, end, localScores, localMax, m, scale, l, a: [...a], output: a.map(x => x / l) });
  }
  const output = rows.at(-1).output, error = Math.max(...output.map((x, i) => Math.abs(x - reference[i])));
  return { rows, reference, fullProbs, output, error, scores, values, block, fullScores: 6, maxBlockScores: block };
}

export function initAttentionKernels() {
  const block = $("[data-ak-block]"); if (!block) return;
  const offset = $("[data-ak-offset]"), next = $("[data-ak-next]");
  let steps = 0;
  const fmt = x => x.toFixed(6), vector = a => `[${a.map(fmt).join(", ")}]`;
  const render = () => {
    const result = onlineAttention({ block: Number(block.value), offset: Number(offset.value) });
    steps = Math.min(steps, result.rows.length);
    const shown = result.rows.slice(0, steps), current = shown.at(-1);
    $("[data-ak-offset-value]").textContent = offset.value;
    $("[data-ak-step]").textContent = `${steps}/${result.rows.length}`;
    $("[data-ak-m]").textContent = current ? fmt(current.m) : "−∞";
    $("[data-ak-l]").textContent = current ? fmt(current.l) : "0";
    $("[data-ak-out]").textContent = current ? vector(current.output) : "尚未读取，输出未定义";
    $("[data-ak-reference]").textContent = vector(result.reference);
    $("[data-ak-error]").textContent = current ? Math.max(...current.output.map((x, i) => Math.abs(x - result.reference[i]))).toExponential(2) : "尚未读取";
    $("[data-ak-rows]").innerHTML = shown.map((row, index) => `<tr><td>${index + 1} / 位置 ${row.start}–${row.end - 1}</td><td>[${row.localScores.join(", ")}]</td><td>${row.localMax}</td><td>${row.m}</td><td>${fmt(row.scale)}</td><td>${fmt(row.l)}</td></tr>`).join("");
    $("[data-ak-values]").innerHTML = result.values.map((v, i) => `<tr><td>${i}</td><td>[${v.join(", ")}]</td><td>${fmt(result.fullProbs[i])}</td></tr>`).join("");
    next.disabled = steps === result.rows.length;
  };
  const reset = () => { steps = 0; render(); };
  block.addEventListener("change", reset); offset.addEventListener("input", reset);
  next.addEventListener("click", () => { steps++; render(); }); $("[data-ak-reset]").addEventListener("click", reset);
  render();
}
