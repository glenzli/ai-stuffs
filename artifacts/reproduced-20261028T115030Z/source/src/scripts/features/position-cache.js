import { $ } from "../core.js";

const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const project = (x, w) => w[0].map((_, j) => x.reduce((s, v, i) => s + v * w[i][j], 0));
const inputs = [[.5,-.2,.3,.7],[.1,.7,-.4,.2],[-.6,.2,.8,-.1],[.3,.9,.1,-.5],[.7,-.3,-.2,.4],[-.2,.5,.6,.3]];
const weights = {
  q: [[.6,-.2,.1,.3],[.2,.5,-.3,.1],[-.1,.4,.7,.2],[.3,.1,-.2,.6]],
  k: [[.3,.6,-.2,.1],[-.4,.2,.5,.3],[.2,-.1,.4,.7],[.5,.3,.1,-.2]],
  v: [[.7,.1,-.2,.4],[.2,.6,.3,-.1],[-.3,.2,.5,.1],[.1,-.2,.4,.8]],
  out: [[.5,-.3,.2],[-.2,.6,.1],[.4,.1,-.5],[.2,-.4,.7]],
};

export function positionCacheParameters() {
  return structuredClone({ inputs, weights });
}

export function ropeRotate(vector, position, base = 10000) {
  if (!Array.isArray(vector) || !vector.length || vector.length % 2 || !vector.every(Number.isFinite)
      || !Number.isFinite(position) || !Number.isFinite(base) || base <= 1) throw new RangeError("旋转需要偶数维有限向量、有限位置和大于 1 的频率基数。");
  const result = [];
  for (let i = 0; i < vector.length; i += 2) {
    const angle = position * base ** (-i / vector.length), c = Math.cos(angle), s = Math.sin(angle);
    result.push(vector[i] * c - vector[i + 1] * s, vector[i] * s + vector[i + 1] * c);
  }
  return result;
}

export function rotaryRelation(m = 2, n = 5, base = 10000) {
  const q = [.8,-.4,.2,.6], k = [-.3,.5,.7,.1];
  const rq = ropeRotate(q, m, base), rk = ropeRotate(k, n, base);
  return { q, k, rq, rk, beforeNorm: Math.hypot(...q), afterNorm: Math.hypot(...rq),
    product: dot(rq, rk), relative: dot(q, ropeRotate(k, n - m, base)) };
}

function outputFor(x, q, keys, allowed) {
  const scores = keys.map((key, j) => allowed[j] ? dot(q, key.k) / 2 : -Infinity);
  const finite = scores.filter(Number.isFinite);
  if (!finite.length) return { logits: null, scores, probabilities: scores.map(() => 0) };
  const max = Math.max(...finite), exponentials = scores.map(s => Math.exp(s - max));
  const denominator = exponentials.reduce((a, b) => a + b, 0), probabilities = exponentials.map(v => v / denominator);
  const context = x.map((_, d) => probabilities.reduce((s, p, j) => s + p * keys[j].v[d], 0));
  return { logits: project(x.map((v, d) => v + context[d]), weights.out), scores, probabilities };
}

export function cacheConsistency({ prefix = 2, chunk = 2, padding = 0, fault = "none", base = 10000 } = {}) {
  if (!Number.isInteger(prefix) || prefix < 1 || prefix >= inputs.length || !Number.isInteger(chunk) || chunk < 1 || chunk > 3
      || !Number.isInteger(padding) || padding < 0 || padding > 3 || !["none", "reset", "double", "mask"].includes(fault)) throw new RangeError("缓存配置无效。");
  ropeRotate(inputs[0], 0, base);
  const pad = Array.from({ length: padding }, (_, slot) => ({ slot, position: null, rotation: null, valid: false, k: [0,0,0,0], v: [0,0,0,0] }));
  const fullKeys = pad.concat(inputs.map((x, i) => ({ slot: padding + i, position: i, rotation: i, valid: true, k: ropeRotate(project(x, weights.k), i, base), v: project(x, weights.v) })));
  const reference = inputs.map((x, i) => outputFor(x, ropeRotate(project(x, weights.q), i, base), fullKeys, fullKeys.map(key => key.valid && key.position <= i)).logits);
  let cache = structuredClone(pad), next = 0;
  const events = [], rows = [];
  while (next < inputs.length) {
    const start = next, count = start === 0 ? prefix : Math.min(chunk, inputs.length - start);
    if (start > 0 && fault === "double") cache = cache.map(key => key.valid ? { ...key, k: ropeRotate(key.k, key.position, base), rotation: key.rotation + key.position } : key);
    const added = inputs.slice(start, start + count).map((x, j) => {
      const position = start + j, rotation = start > 0 && fault === "reset" ? j : position;
      return { slot: padding + position, position, rotation, valid: true, k: ropeRotate(project(x, weights.k), rotation, base), v: project(x, weights.v) };
    });
    cache.push(...added);
    const queries = inputs.slice(start, start + count).map((x, j) => {
      const position = start + j, rotation = added[j].rotation;
      const qRaw = project(x, weights.q), q = ropeRotate(qRaw, rotation, base);
      const allowed = cache.map(key => key.valid && (start > 0 && fault === "mask" ? key.slot <= j : key.position <= position));
      const result = outputFor(x, q, cache, allowed);
      const error = result.logits ? Math.max(...result.logits.map((v, k) => Math.abs(v - reference[position][k]))) : null;
      const row = { position, rotation, qRaw, q, allowed, reference: reference[position], ...result, error };
      rows.push(row);
      return row;
    });
    events.push({ start, count, cache: structuredClone(cache), queries });
    next += count;
  }
  return { events, rows, reference, invalidRows: rows.filter(r => r.logits === null).length,
    maxError: Math.max(...rows.map(r => r.error ?? 0)), firstDivergence: rows.find(r => r.error === null || r.error > 1e-12)?.position ?? null };
}

const fmt = n => n.toFixed(5);
const vector = v => `[${v.map(fmt).join(", ")}]`;
export function initPositionCache() {
  const prefix = $("[data-rc-prefix]"); if (!prefix) return;
  const chunk = $("[data-rc-chunk]"), padding = $("[data-rc-padding]"), fault = $("[data-rc-fault]"), base = $("[data-rc-base]");
  const m = $("[data-rc-m]"), n = $("[data-rc-n]"), eventSelect = $("[data-rc-event]");
  const set = (name, value) => { $(`[data-rc-${name}]`).textContent = value; };
  $("[data-rc-inputs]").innerHTML = inputs.map((x, i) => `<tr><td>${i}</td><td>${vector(x)}</td></tr>`).join("");
  $("[data-rc-weights]").innerHTML = Object.entries(weights).map(([name, matrix]) => `<tr><td>W_${name}</td><td>${matrix.map(vector).join("<br>")}</td></tr>`).join("");
  let selected = 1;
  const render = () => {
    const frequency = Number(base.value), relation = rotaryRelation(Number(m.value), Number(n.value), frequency);
    set("m-value", m.value); set("n-value", n.value);
    set("norm", `${fmt(relation.beforeNorm)} → ${fmt(relation.afterNorm)}`);
    set("dot", fmt(relation.product)); set("relative", fmt(relation.relative));
    $("[data-rc-rotation-rows]").innerHTML = [0, 1].map(i => `<tr><td>${i}</td><td>${frequency ** (-i / 2)}</td><td>${fmt(Number(m.value) * frequency ** (-i / 2))}</td><td>${fmt(Number(n.value) * frequency ** (-i / 2))}</td><td>${vector(relation.rq.slice(i * 2, i * 2 + 2))}</td><td>${vector(relation.rk.slice(i * 2, i * 2 + 2))}</td></tr>`).join("");
    const result = cacheConsistency({ prefix: Number(prefix.value), chunk: Number(chunk.value), padding: Number(padding.value), fault: fault.value, base: frequency });
    selected = Math.min(selected, result.events.length - 1);
    eventSelect.innerHTML = result.events.map((e, i) => `<option value="${i}"${i === selected ? " selected" : ""}>${i === 0 ? "Prefill" : `后缀块 ${i}`} · 位置 ${e.start}–${e.start + e.count - 1}</option>`).join("");
    const event = result.events[selected];
    set("error", result.invalidRows ? "存在无定义行" : result.maxError.toExponential(2));
    set("first", result.firstDivergence === null ? "全序列一致" : `位置 ${result.firstDivergence}`);
    set("cache-size", `${event.cache.length} 槽 / ${event.cache.filter(k => k.valid).length} 有效`);
    $("[data-rc-logits-rows]").innerHTML = event.queries.map(r => `<tr><td>${r.position}</td><td>${r.rotation}</td><td>${vector(r.reference)}</td><td>${r.logits ? vector(r.logits) : "无有效 key"}</td><td>${r.error === null ? "无定义" : r.error.toExponential(2)}</td></tr>`).join("");
    $("[data-rc-cache-rows]").innerHTML = event.cache.map(k => `<tr><td>${k.slot}</td><td>${k.valid ? k.position : "PAD"}</td><td>${k.rotation ?? "—"}</td><td>${vector(k.k)}</td><td>${vector(k.v)}</td></tr>`).join("");
    $("[data-rc-mask-head]").innerHTML = `<th>Query 位置</th>${event.cache.map(k => `<th>${k.valid ? `K${k.position}` : "PAD"}</th>`).join("")}`;
    $("[data-rc-mask-rows]").innerHTML = event.queries.map(r => `<tr><th>${r.position}</th>${r.allowed.map(a => `<td class="${a ? "chain-allowed" : "chain-blocked"}">${a ? "允许" : "屏蔽"}</td>`).join("")}</tr>`).join("");
    const first = event.queries[0];
    set("query", `当前块首个 query：旋转前 ${vector(first.qRaw)}；旋转后 ${vector(first.q)}。`);
    const notes = { none: "位置、旋转和可见范围与完整前向对齐。每个 K 旋转一次后入缓存，V 不做位置旋转。",
      reset: "错误：每个新后缀块的旋转位置从 0 开始，旧缓存仍使用之前的位置。",
      double: "错误：每次追加前，把缓存中已经旋转的 K 按原位置再旋转一次。表中累计旋转位置随之增加。",
      mask: "错误：用块内 query 下标直接截取完整 key 轴；遗漏旧前缀的偏移，含左 PAD 时还可能屏蔽全部有效 key。" };
    set("note", `${notes[fault.value]} 上方误差覆盖全部六个有效位置；下表只展示所选计算块。`);
  };
  for (const control of [prefix, chunk, padding, fault, base, m, n]) control.addEventListener("input", render);
  eventSelect.addEventListener("change", () => { selected = Number(eventSelect.value); render(); });
  render();
}
