import { $ } from "../core.js";

const transpose = a => a[0].map((_, j) => a.map(row => row[j]));
const mm = (a, b) => a.map(row => b[0].map((_, j) => row.reduce((s, x, k) => s + x * b[k][j], 0)));
const map = (a, fn) => a.map((row, i) => row.map((x, j) => fn(x, i, j)));
const add = (a, b) => map(a, (x, i, j) => x + b[i][j]);
const mul = (a, b) => map(a, (x, i, j) => x * b[i][j]);
const sumRows = a => [a[0].map((_, j) => a.reduce((s, row) => s + row[j], 0))];
const softmax = row => { const m = Math.max(...row), e = row.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b); return e.map(x => x / s); };
const sigmoid = x => 1 / (1 + Math.exp(-x));
const norm = a => Math.hypot(...a.flat());

export function blockParameters() {
  return {
    X: [[0.5, -0.2], [0.1, 0.7]],
    q: [[0.3, -0.2], [0.4, 0.5]], k: [[0.2, 0.6], [-0.3, 0.4]],
    v: [[0.7, 0.1], [-0.2, 0.5]], o: [[0.4, -0.3], [0.2, 0.6]],
    gate: [[0.3, -0.4, 0.2], [0.5, 0.1, -0.3]],
    up: [[0.2, 0.6, -0.1], [-0.4, 0.3, 0.5]],
    down: [[0.4, -0.2], [0.1, 0.3], [-0.5, 0.2]],
    out: [[0.4, -0.3, 0.1], [-0.2, 0.5, 0.6]], norm1: [[1, 0.9]], norm2: [[0.8, 1.1]],
  };
}

function rms(x, gamma) {
  const inverse = x.map(row => 1 / Math.sqrt(row.reduce((s, v) => s + v * v, 0) / row.length + 1e-5));
  return { inverse, y: map(x, (v, i, j) => v * inverse[i] * gamma[0][j]) };
}

function rmsBackward(x, gamma, inverse, dy) {
  const weighted = map(dy, (v, i, j) => v * gamma[0][j]);
  const dots = x.map((row, i) => row.reduce((s, v, j) => s + v * weighted[i][j], 0));
  return {
    dx: map(x, (v, i, j) => weighted[i][j] * inverse[i] - v * inverse[i] ** 3 * dots[i] / x[i].length),
    dg: sumRows(map(dy, (v, i, j) => v * x[i][j] * inverse[i])),
  };
}

export function blockForward(p = blockParameters(), target = 1) {
  if (!Number.isInteger(target) || target < 0 || target > 2) throw new RangeError("目标类别必须为 0、1 或 2。");
  const n1 = rms(p.X, p.norm1), u1 = n1.y;
  const Q = mm(u1, p.q), K = mm(u1, p.k), V = mm(u1, p.v);
  const S = map(mm(Q, transpose(K)), v => v / Math.sqrt(2));
  const P = S.map((row, i) => softmax(row.map((x, j) => j <= i ? x : -Infinity)));
  const ctx = mm(P, V), attn = mm(ctx, p.o), h1 = add(p.X, attn);
  const n2 = rms(h1, p.norm2), u2 = n2.y;
  const gate = mm(u2, p.gate), up = mm(u2, p.up), silu = map(gate, x => x * sigmoid(x));
  const gated = mul(silu, up), ffn = mm(gated, p.down), h2 = add(h1, ffn);
  const logits = mm(h2, p.out), probabilities = logits.map(softmax);
  const row = logits[1], max = Math.max(...row);
  const loss = max + Math.log(row.reduce((s, x) => s + Math.exp(x - max), 0)) - row[target];
  return { X: p.X, n1, u1, Q, K, V, S, P, ctx, attn, h1, n2, u2, gate, up, silu, gated, ffn, h2, logits, probabilities, loss };
}

export function blockBackward(p = blockParameters(), target = 1) {
  const c = blockForward(p, target), g = {}, d = {};
  d.logits = c.probabilities.map((row, i) => row.map((v, j) => i === 1 ? v - Number(j === target) : 0));
  g.out = mm(transpose(c.h2), d.logits);
  d.h2 = mm(d.logits, transpose(p.out));
  d.ffn = d.h2.map(row => [...row]);
  g.down = mm(transpose(c.gated), d.ffn);
  d.gated = mm(d.ffn, transpose(p.down));
  d.up = mul(d.gated, c.silu);
  d.gate = map(c.gate, (x, i, j) => d.gated[i][j] * c.up[i][j] * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))));
  g.gate = mm(transpose(c.u2), d.gate); g.up = mm(transpose(c.u2), d.up);
  d.u2 = add(mm(d.gate, transpose(p.gate)), mm(d.up, transpose(p.up)));
  const rn2 = rmsBackward(c.h1, p.norm2, c.n2.inverse, d.u2);
  g.norm2 = rn2.dg;
  d.h1 = add(d.h2, rn2.dx);
  d.attn = d.h1.map(row => [...row]);
  g.o = mm(transpose(c.ctx), d.attn);
  d.ctx = mm(d.attn, transpose(p.o));
  d.P = mm(d.ctx, transpose(c.V)); d.V = mm(transpose(c.P), d.ctx);
  d.S = c.P.map((row, i) => { const dot = row.reduce((s, v, j) => s + v * d.P[i][j], 0); return row.map((v, j) => v * (d.P[i][j] - dot)); });
  d.Q = map(mm(d.S, c.K), x => x / Math.sqrt(2));
  d.K = map(mm(transpose(d.S), c.Q), x => x / Math.sqrt(2));
  for (const name of ["q", "k", "v"]) g[name] = mm(transpose(c.u1), d[name.toUpperCase()]);
  d.u1 = add(add(mm(d.Q, transpose(p.q)), mm(d.K, transpose(p.k))), mm(d.V, transpose(p.v)));
  const rn1 = rmsBackward(p.X, p.norm1, c.n1.inverse, d.u1);
  g.norm1 = rn1.dg;
  d.X = add(d.h1, rn1.dx); g.X = d.X;
  return { forward: c, gradients: g, activationGradients: d };
}

export function gradientCheck(name = "q", index = 0, target = 1, epsilon = 1e-5) {
  const p = blockParameters();
  if (!Object.hasOwn(p, name) || !Number.isInteger(index) || index < 0 || index >= p[name].flat().length) throw new RangeError("参数索引无效。");
  if (!Number.isFinite(epsilon) || epsilon <= 0) throw new RangeError("差分步长必须为正有限数。");
  const width = p[name][0].length, i = Math.floor(index / width), j = index % width;
  const analytic = blockBackward(p, target).gradients[name][i][j], value = p[name][i][j];
  p[name][i][j] = value + epsilon; const plus = blockForward(p, target).loss;
  p[name][i][j] = value - epsilon; const minus = blockForward(p, target).loss;
  const numeric = (plus - minus) / (2 * epsilon);
  return { value, analytic, numeric, error: Math.abs(analytic - numeric), i, j };
}

const stages = [
  ["输出交叉熵", "logits", "末位置 softmax 减去目标 one-hot；前位置未直接监督，logits 梯度为零。"],
  ["输出投影", "h2", "输出头的转置把词表梯度传回 hidden；权重梯度由 hiddenᵀ × logits 梯度累加。"],
  ["第二个残差", "ffn", "相加操作把同一上游梯度传给 h1 的恒等路径和 FFN 分支。"],
  ["FFN 下投影", "gated", "经过 downᵀ 返回门控中间层；down 的权重梯度汇总全部位置。"],
  ["SwiGLU 门控", "gate", "乘法把梯度分给 gate/up 两支；gate 再乘 SiLU 的局部导数。"],
  ["第二个 RMSNorm", "u2", "gate 与 up 的输入梯度相加，经过归一化导数传回 h1。"],
  ["第一层残差状态", "h1", "来自 FFN/RMSNorm 的梯度与第二残差的恒等路径在 h1 汇合。"],
  ["Attention 输出投影", "ctx", "oᵀ 把梯度传回 P·V；o 的梯度为 ctxᵀ × 上游梯度。"],
  ["Attention 加权求和", "P", "P·V 对 P 和 V 都有梯度，分别是 G·Vᵀ 与 Pᵀ·G。"],
  ["按行 softmax", "S", "每行使用 p ⊙ (g − Σp·g)。被因果 mask 排除的位置梯度为零。"],
  ["QK 点积", "Q", "Q 和 K 都接收分数梯度，包含 1/√d 缩放；K 路径使用转置梯度。"],
  ["Q/K/V 三条投影", "u1", "三条输入梯度在共享 u1 上相加，三组权重各自收到梯度。"],
  ["第一 RMSNorm 与残差", "X", "归一化路径与第一残差的恒等路径相加，得到输入 X 的总梯度。"],
];

export function initBackpropagation() {
  const target = $("[data-bp-target]"); if (!target) return;
  const parameter = $("[data-bp-parameter]"), element = $("[data-bp-element]");
  let step = 0;
  const format = value => Math.abs(value) < 5e-7 ? "0.000000" : value.toFixed(6);
  const fillElements = () => {
    const matrix = blockParameters()[parameter.value];
    element.innerHTML = matrix.flatMap((row, i) => row.map((_, j) => `<option value="${i * row.length + j}">[${i}, ${j}]</option>`)).join("");
  };
  const render = () => {
    const p = blockParameters(), result = blockBackward(p, Number(target.value));
    const [title, key, copy] = stages[step], matrix = result.forward[key], gradient = result.activationGradients[key];
    $("[data-bp-step]").textContent = `${step + 1}/${stages.length}`;
    $("[data-bp-title]").textContent = title; $("[data-bp-note]").textContent = copy;
    $("[data-bp-loss]").textContent = format(result.forward.loss);
    $("[data-bp-shape]").textContent = `[${matrix.length}, ${matrix[0].length}]`;
    $("[data-bp-matrix]").innerHTML = matrix.flatMap((row, i) => row.map((v, j) => `<tr><td>${key}[${i},${j}]</td><td>${format(v)}</td><td>${format(gradient[i][j])}</td></tr>`)).join("");
    $("[data-bp-next]").disabled = step === stages.length - 1;
    $("[data-bp-prev]").disabled = step === 0;
    $("[data-bp-params]").innerHTML = Object.entries(p).map(([name, values]) => `<tr><td>${name}</td><td>${values.length} × ${values[0].length}</td><td>${format(norm(result.gradients[name]))}</td></tr>`).join("");
    const check = gradientCheck(parameter.value, Number(element.value), Number(target.value));
    $("[data-bp-analytic]").textContent = format(check.analytic); $("[data-bp-numeric]").textContent = format(check.numeric);
    $("[data-bp-error]").textContent = check.error.toExponential(2);
  };
  $("[data-bp-next]").addEventListener("click", () => { step = Math.min(stages.length - 1, step + 1); render(); });
  $("[data-bp-prev]").addEventListener("click", () => { step = Math.max(0, step - 1); render(); });
  $("[data-bp-reset]").addEventListener("click", () => { step = 0; render(); });
  parameter.addEventListener("change", () => { fillElements(); render(); });
  element.addEventListener("change", render); target.addEventListener("change", render);
  fillElements(); render();
}
