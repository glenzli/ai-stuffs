import { $ } from "../core.js";

const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
const loss = (theta, target) => 0.5 * theta.reduce((sum, value, index) => sum + (value - target[index]) ** 2, 0);

export function projectConflict(gA, gB) {
  if (!Array.isArray(gA) || !Array.isArray(gB) || !gA.length || gA.length !== gB.length ||
      !Array.from(gA).every(Number.isFinite) || !Array.from(gB).every(Number.isFinite)) {
    throw new RangeError("Gradients must be nonempty, equally sized arrays of finite numbers.");
  }
  const conflict = dot(gA, gB), normA = dot(gA, gA), normB = dot(gB, gB);
  if (![conflict, normA, normB].every(Number.isFinite)) {
    throw new RangeError("Gradient dot products and squared norms must be finite.");
  }
  // Both projections use the original pair, once each; zero vectors need no projection.
  return {
    a: gA.map((value, index) => conflict < 0 && normB > 0 ? value - conflict / normB * gB[index] : value),
    b: gB.map((value, index) => conflict < 0 && normA > 0 ? value - conflict / normA * gA[index] : value),
  };
}

export function interferenceStep({ angle = 120, weight = 0.5, lr = 0.5, mode = "joint" } = {}) {
  for (const [name, value, max] of [["angle", angle, 180], ["weight", weight, 1], ["lr", lr, 3]]) {
    if (!Number.isFinite(value) || value < 0 || value > max) {
      throw new RangeError(`${name} must be a finite number in [0, ${max}].`);
    }
  }
  if (!["joint", "projected"].includes(mode)) throw new RangeError("mode must be joint or projected.");
  const radians = angle * Math.PI / 180;
  const targetA = [1, 0];
  // Cardinal angles remain exact, including cancellation at 180 degrees.
  const targetB = angle === 0 ? [1, 0] : angle === 90 ? [0, 1] : angle === 180 ? [-1, 0] : [Math.cos(radians), Math.sin(radians)];
  const gA = [-1, 0], gB = targetB.map(value => value === 0 ? 0 : -value);
  const pair = mode === "projected" ? projectConflict(gA, gB) : { a: gA, b: gB };
  const g = pair.a.map((value, index) => (1 - weight) * value + weight * pair.b[index]);
  const theta = g.map(value => lr === 0 || value === 0 ? 0 : -lr * value);
  const oldA = 0.5, oldB = 0.5, newA = loss(theta, targetA), newB = loss(theta, targetB);
  return {
    gA, gB, g, theta, targetA, targetB, cosine: dot(gA, gB),
    oldA, oldB, newA, newB, deltaA: newA - oldA, deltaB: newB - oldB,
    linearA: -lr * dot(gA, g), linearB: -lr * dot(gB, g),
  };
}

function drawInterference(canvas, ctx, result) {
  const style = getComputedStyle(canvas);
  const color = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
  const colors = {
    background: color("--surface-strong", "#ffffff"),
    ink: color("--ink", "#242424"),
    muted: color("--ink-faint", "#686868"),
    grid: color("--line", "#dedede"),
    axis: color("--line-strong", "#a3a3a3"),
    a: color("--purple", "#7452ad"),
    b: color("--blue", "#355bc2"),
    update: color("--coral", "#d95f36"),
  };
  const width = canvas.width, height = canvas.height, extent = 2.5, pad = 44;
  const scale = (Math.min(width, height) - 2 * pad) / (2 * extent);
  const point = ([x, y]) => [width / 2 + x * scale, height / 2 - y * scale];
  const line = (from, to) => {
    ctx.beginPath(); ctx.moveTo(...point(from)); ctx.lineTo(...point(to)); ctx.stroke();
  };
  const marker = (position, radius, fill) => {
    ctx.beginPath(); ctx.arc(...point(position), radius, 0, 2 * Math.PI); ctx.fillStyle = fill; ctx.fill();
  };
  ctx.save();
  ctx.fillStyle = colors.background; ctx.fillRect(0, 0, width, height);
  ctx.font = `19px ${style.getPropertyValue("--sans").trim() || "sans-serif"}`;
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  for (const [label, fill, x] of [["任务 A", colors.a, 24], ["任务 B", colors.b, 124], ["更新后", colors.update, 224]]) {
    ctx.fillStyle = fill; ctx.fillText(label, x, 22);
  }
  ctx.fillStyle = colors.ink;
  ctx.fillText(`θ = (${result.theta.map(value => value.toFixed(2)).join(", ")})`, 320, 22);

  ctx.save();
  const [left, top] = point([-extent, extent]);
  ctx.beginPath(); ctx.rect(left, top, 2 * extent * scale, 2 * extent * scale); ctx.clip();
  ctx.strokeStyle = colors.grid; ctx.lineWidth = 0.75;
  for (let tick = -extent; tick <= extent; tick += 0.5) {
    line([tick, -extent], [tick, extent]); line([-extent, tick], [extent, tick]);
  }
  ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
  for (const [target, stroke] of [[result.targetA, colors.a], [result.targetB, colors.b]]) {
    ctx.strokeStyle = stroke;
    for (const radius of [0.5, 1, 1.5]) {
      ctx.beginPath(); ctx.arc(...point(target), radius * scale, 0, 2 * Math.PI); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1; ctx.restore();
  ctx.strokeStyle = colors.axis; ctx.lineWidth = 1.25;
  line([-extent, 0], [extent, 0]); line([0, -extent], [0, extent]);
  ctx.fillStyle = colors.muted;
  for (const tick of [-2, -1, 1, 2]) {
    const [x] = point([tick, 0]);
    ctx.textAlign = "center"; ctx.fillText(String(tick), x, height / 2 + 15);
    ctx.textAlign = "right"; ctx.fillText(String(tick), width / 2 - 9, point([0, tick])[1]);
  }
  ctx.textAlign = "left";
  ctx.fillText("θ₁", point([extent, 0])[0] + 10, height / 2);
  ctx.textAlign = "right";
  ctx.fillText("θ₂", width / 2 - 10, point([0, extent])[1] - 12);
  ctx.textAlign = "left";
  ctx.setLineDash([5, 5]); ctx.lineWidth = 1.5;
  for (const [target, stroke] of [[result.targetA, colors.a], [result.targetB, colors.b]]) {
    ctx.strokeStyle = stroke; line([0, 0], target);
  }
  ctx.setLineDash([]);
  const start = point([0, 0]), end = point(result.theta);
  const length = Math.hypot(end[0] - start[0], end[1] - start[1]);
  if (length > 0) {
    ctx.strokeStyle = colors.update; ctx.lineWidth = 3;
    line([0, 0], result.theta);
    const direction = Math.atan2(end[1] - start[1], end[0] - start[0]);
    const head = Math.min(10, length * 0.4);
    ctx.beginPath(); ctx.moveTo(...end);
    ctx.lineTo(end[0] - head * Math.cos(direction - Math.PI / 6), end[1] - head * Math.sin(direction - Math.PI / 6));
    ctx.lineTo(end[0] - head * Math.cos(direction + Math.PI / 6), end[1] - head * Math.sin(direction + Math.PI / 6));
    ctx.closePath(); ctx.fillStyle = colors.update; ctx.fill();
  }
  marker([0, 0], 4, colors.ink);
  marker(result.targetA, 6, colors.a);
  marker(result.targetB, 4, colors.b);
  marker(result.theta, 3, colors.update);
  ctx.fillStyle = colors.ink; ctx.textAlign = "right";
  ctx.fillText("初始 (0, 0)", start[0] - 12, start[1] + 30);
  ctx.textAlign = "left";
  ctx.fillStyle = colors.a; ctx.fillText("目标 A", point(result.targetA)[0] + 12, point(result.targetA)[1] + 30);
  ctx.fillStyle = colors.b; ctx.fillText("目标 B", point(result.targetB)[0] + 12, point(result.targetB)[1] - 15);
  ctx.restore();
}

export function initContinualLearning() {
  const angle = $("[data-cl-angle]");
  if (!angle) return;
  const weight = $("[data-cl-weight]"), lr = $("[data-cl-lr]"), mode = $("[data-cl-mode]");
  const canvas = $("[data-cl-canvas]");
  if (canvas) { canvas.width = 640; canvas.height = 440; }
  const ctx = canvas?.getContext("2d");
  const number = value => Math.abs(value) < 0.00005 ? "0.0000" : Number(value.toPrecision(14)).toFixed(4);
  const render = () => {
    const options = { angle: Number(angle.value), weight: Number(weight.value) / 100, lr: Number(lr.value) / 100, mode: mode?.value || "joint" };
    const result = interferenceStep(options);
    $("[data-cl-angle-value]").textContent = `${options.angle}°`;
    $("[data-cl-weight-value]").textContent = `${(options.weight * 100).toFixed(0)}%`;
    $("[data-cl-lr-value]").textContent = options.lr.toFixed(2);
    $("[data-cl-cos]").textContent = number(result.cosine);
    $("[data-cl-loss-a]").textContent = number(result.newA);
    $("[data-cl-loss-b]").textContent = number(result.newB);
    $("[data-cl-rows]").innerHTML = ["A", "B"].map(task =>
      `<tr><th scope="row">任务 ${task}</th><td>${number(result[`old${task}`])}</td><td>${number(result[`old${task}`] + result[`linear${task}`])}</td><td>${number(result[`new${task}`])}</td></tr>`
    ).join("");
    const remainder = 0.5 * options.lr ** 2 * dot(result.g, result.g);
    const changes = ["A", "B"].map(task => `ΔL_${task} = ${number(result[`delta${task}`])}`).join("，");
    const scope = options.mode === "projected" ? "PCGrad 双任务对称局部示意；两次投影使用同一对原始梯度。" : "按任务权重合成原始梯度。";
    const overshoot = ["A", "B"].some(task => result[`linear${task}`] < -1e-12 && result[`delta${task}`] > 1e-12);
    $("[data-cl-note]").textContent = `${scope}${changes}；两项损失的二次项均为 ${number(remainder)}。${result.g.every(value => value === 0) ? "合成梯度为零，参数与损失不变。" : overshoot ? "线性预测下降，但步长过大使实际损失上升。" : ""}`;
    if (ctx) drawInterference(canvas, ctx, result);
  };
  [angle, weight, lr, mode].filter(Boolean).forEach(input => {
    input.addEventListener("input", render);
    input.addEventListener("change", render);
  });
  if (ctx && typeof MutationObserver !== "undefined") {
    new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class", "style"] });
  }
  render();
}
