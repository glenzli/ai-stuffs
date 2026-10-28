import { $ } from "../core.js";
import { zeros, dot, maxDifference, formatValue, formatVector, escapeText } from "./objective-math.js";
import { chunkedCEReportSnapshot } from "./chunked-cross-entropy-data.js";

// Binary64 teaching arithmetic. The independent CPU experiment lives in Python.
export function chunkedCEInputs({ bias = true, mask = "all", target = 6 } = {}) {
  if (!["all", "mixed", "empty", "ignore-all"].includes(mask) || !Number.isInteger(target) || target < 0 || target > 6) throw new RangeError("监督模式或目标 ID 无效。");
  return {
    H: [[.2, -.3, .5], [.5, .1, -.2], [-.4, .7, .1], [.6, -.2, .3], [.1, .8, -.5]],
    W: [[.3, .5, -.1], [-.2, .4, .6], [.7, -.1, .2], [.1, -.6, .4], [-.5, .2, -.3], [.4, .3, -.7], [-.3, -.2, .8]],
    bias: bias ? [.1, -.2, .05, .3, -.1, .2, -.05] : null,
    labels: mask === "ignore-all" ? Array(5).fill(-100) : [0, 2, -100, target, 4],
    mask: mask === "empty" ? Array(5).fill(false) : mask === "mixed" ? [true, false, true, true, true] : Array(5).fill(true),
    ignoreIndex: -100,
  };
}

function validateInputs(inputs, scoreShift, upstream) {
  const n = inputs?.H?.length, v = inputs?.W?.length, d = inputs?.W?.[0]?.length;
  if (!Number.isInteger(n) || n > 64 || !Number.isInteger(v) || v < 1 || v > 256 || !Number.isInteger(d) || d < 1 || d > 32) throw new RangeError("教学矩阵应为 H[N,D] 与 W[V,D]，N≤64、V≤256、D≤32。");
  for (const key of ["H", "W"]) if (!Array.isArray(inputs[key]) || !inputs[key].every(row => Array.isArray(row) && row.length === d && row.every(Number.isFinite))) throw new RangeError("H/W 必须是同宽的有限数矩阵。");
  if (inputs.bias !== null && inputs.bias !== undefined && (!Array.isArray(inputs.bias) || inputs.bias.length !== v || !inputs.bias.every(Number.isFinite))) throw new RangeError("bias 必须为空或长度为 V 的有限数向量。");
  if (!Number.isInteger(inputs.ignoreIndex) || !Array.isArray(inputs.labels) || inputs.labels.length !== n || !inputs.labels.every(label => Number.isInteger(label) && (label === inputs.ignoreIndex || (label >= 0 && label < v)))) throw new RangeError("标签须为词表 ID 或 ignoreIndex；mask 不掩盖越界标签。");
  if (!Array.isArray(inputs.mask) || inputs.mask.length !== n || !inputs.mask.every(x => typeof x === "boolean")) throw new RangeError("mask 必须是长度为 N 的布尔向量，True 表示参与监督。");
  if (!Number.isFinite(scoreShift) || Math.abs(scoreShift) > 1e9 || !Number.isFinite(upstream) || Math.abs(upstream) > 1e6) throw new RangeError("分数平移或上游标量超出教学范围。");
  return { n, v, d, valid: inputs.labels.map((y, i) => inputs.mask[i] && y !== inputs.ignoreIndex) };
}
const score = (inputs, i, j, shift) => {
  const value = dot(inputs.H[i], inputs.W[j]) + (inputs.bias?.[j] || 0) + shift;
  if (!Number.isFinite(value)) throw new RangeError("输出分数溢出；此页只计算有限 logits 的目标。");
  return value;
};

export function denseCEReference(inputs = chunkedCEInputs(), { scoreShift = 0, upstream = 1 } = {}) {
  const { n, v, d, valid } = validateInputs(inputs, scoreShift, upstream), count = valid.filter(Boolean).length;
  const logits = inputs.H.map((_, i) => inputs.W.map((_, j) => score(inputs, i, j, scoreShift)));
  const P = zeros(n, v), dH = zeros(n, d), dW = zeros(v, d), db = inputs.bias ? Array(v).fill(0) : null;
  const lse = Array(n).fill(null), perPositionLoss = Array(n).fill(null);
  let loss = 0;
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    const maximum = Math.max(...logits[i]), sum = logits[i].reduce((s, z) => s + Math.exp(z - maximum), 0), logSum = Math.log(sum);
    lse[i] = maximum + logSum;
    perPositionLoss[i] = (maximum - logits[i][inputs.labels[i]]) + logSum;
    loss += perPositionLoss[i] / count;
    for (let j = 0; j < v; j++) {
      const p = Math.exp((logits[i][j] - maximum) - logSum), g = upstream * (p - Number(j === inputs.labels[i])) / count;
      P[i][j] = p;
      for (let c = 0; c < d; c++) { dH[i][c] += g * inputs.W[j][c]; dW[j][c] += g * inputs.H[i][c]; }
      if (db) db[j] += g;
    }
  }
  return { loss, validCount: count, skipped: count === 0, perPositionLoss, lse, logits, P, dH, dW, db };
}

export function chunkedCrossEntropy({ inputs = chunkedCEInputs(), tokenBlock = 2, vocabBlock = 3, scoreShift = 0, upstream = 1, fault = "none", snapshots = true } = {}) {
  const { n, v, d, valid } = validateInputs(inputs, scoreShift, upstream), count = valid.filter(Boolean).length;
  for (const b of [tokenBlock, vocabBlock]) if (!Number.isInteger(b) || b < 1 || b > 1024) throw new RangeError("块大小须为 1…1024 的整数；尾块使用实际长度。");
  if (!["none", "local-softmax", "overwrite-dh"].includes(fault) || typeof snapshots !== "boolean") throw new RangeError("未知故障或快照选项。");
  const maximum = Array(n).fill(null), logSum = Array(n).fill(null), lse = Array(n).fill(null), targetLogit = Array(n).fill(null), perPositionLoss = Array(n).fill(null);
  const dH = zeros(n, d), dW = zeros(v, d), db = inputs.bias ? Array(v).fill(0) : null, probabilityMass = Array(n).fill(0), events = [];
  let loss = 0, forwardScores = 0, recomputedScores = 0, largestTile = 0;
  if (count) for (let tStart = 0; tStart < n; tStart += tokenBlock) {
    const tEnd = Math.min(n, tStart + tokenBlock), m = Array(tEnd - tStart).fill(-Infinity), sum = Array(tEnd - tStart).fill(0), target = Array(tEnd - tStart).fill(null);
    for (let vStart = 0; vStart < v; vStart += vocabBlock) {
      const vEnd = Math.min(v, vStart + vocabBlock), scores = zeros(tEnd - tStart, vEnd - vStart);
      largestTile = Math.max(largestTile, scores.length * scores[0].length);
      for (let i = tStart; i < tEnd; i++) {
        if (!valid[i]) continue;
        const r = i - tStart;
        for (let j = vStart; j < vEnd; j++) scores[r][j - vStart] = score(inputs, i, j, scoreShift);
        forwardScores += vEnd - vStart;
        const next = Math.max(m[r], ...scores[r]), correction = m[r] === -Infinity ? 0 : Math.exp(m[r] - next);
        sum[r] = correction * sum[r] + scores[r].reduce((s, z) => s + Math.exp(z - next), 0); m[r] = next;
        if (inputs.labels[i] >= vStart && inputs.labels[i] < vEnd) target[r] = scores[r][inputs.labels[i] - vStart];
      }
      // Retained tile/gradient copies are UI-only. No UI memory is measured as algorithm memory.
      if (snapshots) events.push({ stage: "forward", tStart, tEnd, vStart, vEnd, scores, rows: Array.from({ length: tEnd - tStart }, (_, r) => ({ position: tStart + r, active: valid[tStart + r], maximum: valid[tStart + r] ? m[r] : null, denominator: sum[r], targetLogit: target[r], lse: valid[tStart + r] ? m[r] + Math.log(sum[r]) : null })) });
    }
    for (let i = tStart; i < tEnd; i++) if (valid[i]) {
      const r = i - tStart; maximum[i] = m[r]; logSum[i] = Math.log(sum[r]); lse[i] = m[r] + logSum[i]; targetLogit[i] = target[r];
      perPositionLoss[i] = (m[r] - target[r]) + logSum[i]; loss += perPositionLoss[i] / count;
    }
  }
  if (count) for (let tStart = 0; tStart < n; tStart += tokenBlock) for (let vStart = 0; vStart < v; vStart += vocabBlock) {
    const tEnd = Math.min(n, tStart + tokenBlock), vEnd = Math.min(v, vStart + vocabBlock);
    const probability = zeros(tEnd - tStart, vEnd - vStart), dScores = zeros(tEnd - tStart, vEnd - vStart), localDH = zeros(tEnd - tStart, d);
    for (let i = tStart; i < tEnd; i++) {
      if (!valid[i]) continue;
      const scores = Array.from({ length: vEnd - vStart }, (_, j) => score(inputs, i, vStart + j, scoreShift)); recomputedScores += scores.length;
      const localMax = fault === "local-softmax" ? Math.max(...scores) : maximum[i];
      const localLogSum = fault === "local-softmax" ? Math.log(scores.reduce((s, z) => s + Math.exp(z - localMax), 0)) : logSum[i];
      for (let j = vStart; j < vEnd; j++) {
        const p = Math.exp((scores[j - vStart] - localMax) - localLogSum), g = upstream * (p - Number(j === inputs.labels[i])) / count;
        probability[i - tStart][j - vStart] = p; dScores[i - tStart][j - vStart] = g; probabilityMass[i] += p;
        for (let c = 0; c < d; c++) { localDH[i - tStart][c] += g * inputs.W[j][c]; dW[j][c] += g * inputs.H[i][c]; }
        if (db) db[j] += g;
      }
    }
    for (let i = tStart; i < tEnd; i++) for (let c = 0; c < d; c++) dH[i][c] = (fault === "overwrite-dh" ? 0 : dH[i][c]) + localDH[i - tStart][c];
    if (snapshots) events.push({ stage: "backward", tStart, tEnd, vStart, vEnd, probability, dScores, localDH, dH: structuredClone(dH), dW: structuredClone(dW), db: db ? [...db] : null });
  }
  const reference = denseCEReference(inputs, { scoreShift, upstream });
  const errors = { loss: Math.abs(loss - reference.loss), dH: maxDifference(dH, reference.dH), dW: maxDifference(dW, reference.dW), db: db ? maxDifference(db, reference.db) : 0 };
  return { inputs, tokenBlock, vocabBlock, scoreShift, upstream, fault, validCount: count, skipped: count === 0, loss, maximum, logSum, lse, targetLogit, perPositionLoss, probabilityMass, dH, dW, db, reference, errors, events,
    work: { forwardScores, recomputedScores }, storage: { sharedH: n * d, sharedW: v * d, sharedBias: inputs.bias ? v : 0, effectiveLabels: n, savedRowStatistics: count ? 2 * n : 0, denseLogits: n * v, maximumScoreTile: largestTile, gradientBuffers: n * d + v * d + (inputs.bias ? v : 0), snapshotsExcluded: true, note: "Logical elements by category, not allocator peak. Python's score tile is reused for exp/probability/gradient; additional row vectors and GEMM outputs exist." } };
}

const object = x => x && typeof x === "object" && !Array.isArray(x);
const finiteNonnegative = x => typeof x === "number" && Number.isFinite(x) && x >= 0;
const integer = (x, min = 0, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(x) && x >= min && x <= max;
const shortText = (x, max = 500) => typeof x === "string" && x.length > 0 && x.length <= max;
export function validateChunkedCEReport(report) {
  if (!object(report) || report.schema !== "output-lab-v1" || report.status !== "passed" || !shortText(report.created_at, 100) || !Number.isFinite(Date.parse(report.created_at))) throw new RangeError("请选择 run-output-lab.mjs 生成的成功报告。");
  const runtime = report.runtime, provenance = report.provenance, c = report.correctness;
  if (!object(runtime) || !["python", "torch", "numpy", "platform"].every(k => shortText(runtime[k])) || runtime.device !== "cpu" || runtime.dtype !== "float64" || !integer(runtime.threads, 1, 128)) throw new RangeError("报告须声明 CPU float64 环境与线程数。");
  if (!object(provenance) || !shortText(provenance.run_id, 250) || !shortText(provenance.report_path, 1000) || !/^[a-f0-9]{64}$/.test(provenance.fixture_sha256) || !/^[a-f0-9]{64}$/.test(provenance.python_sha256) || !shortText(report.measurement_scope, 3000)) throw new RangeError("报告缺少实验来源与源码哈希。");
  if (!object(c) || !integer(c.cases, 1, 5000) || !integer(c.comparisons, 1, 100000) || !["max_dense_error", "max_js_error", "finite_difference_max_error"].every(k => finiteNonnegative(c[k])) || !integer(c.fault_cases, 1, 1000) || c.no_full_score_allocation !== true || c.no_score_saved_for_backward !== true) throw new RangeError("报告的正确性与分块保存检查无效。");
  if (!Array.isArray(report.measurements) || !report.measurements.length || report.measurements.length > 24) throw new RangeError("测量方案数量无效。");
  const names = new Set();
  for (const row of report.measurements) {
    const t = row.wall_ms, s = row.saved_tensor_storage, l = row.logical, shape = row.shape;
    if (!object(row) || !shortText(row.name, 200) || names.has(row.name) || !["dense", "chunked"].includes(row.method) || !Array.isArray(shape) || shape.length !== 3 || !shape.every(x => integer(x, 1, 1e6)) || !integer(row.valid_positions, 1, shape[0]) || !integer(row.warmup_runs, 1, 100) || !integer(row.repetitions, 2, 80)) throw new RangeError("测量名称、形状或重复次数无效。");
    names.add(row.name);
    if (!object(t) || ![t.min, t.median, t.max].every(finiteNonnegative) || t.min > t.median || t.median > t.max || !Array.isArray(t.samples) || t.samples.length !== row.repetitions || !t.samples.every(finiteNonnegative)) throw new RangeError("CPU 耗时样本无效。");
    const samples = [...t.samples].sort((a, b) => a - b), median = samples.length % 2 ? samples[(samples.length - 1) / 2] : (samples[samples.length / 2 - 1] + samples[samples.length / 2]) / 2;
    if (Math.abs(median - t.median) > 1e-9 || Math.abs(samples[0] - t.min) > 1e-9 || Math.abs(samples.at(-1) - t.max) > 1e-9) throw new RangeError("耗时汇总与原始样本不一致。");
    if (!object(s) || !integer(s.peak_referenced_bytes) || !integer(s.save_requests) || !integer(s.peak_unique_storages) || !integer(s.live_bytes_after_backward) || s.live_bytes_after_backward !== 0 || !shortText(s.includes, 1000) || !shortText(s.excludes, 1000)) throw new RangeError("保存张量 storage 记录无效。");
    if (!object(l) || !["dense_logits_elements", "maximum_score_tile_elements", "shared_input_parameter_elements", "gradient_buffer_elements", "saved_row_statistics_elements", "score_forward_elements", "score_recomputed_elements"].every(k => integer(l[k])) || l.dense_logits_elements !== shape[0] * shape[1] || l.maximum_score_tile_elements > l.dense_logits_elements) throw new RangeError("逻辑元素账本无效。");
    if (row.method === "chunked" && (!Array.isArray(row.blocks) || row.blocks.length !== 2 || !row.blocks.every(x => integer(x, 1, 1e6)) || l.maximum_score_tile_elements !== Math.min(row.blocks[0], shape[0]) * Math.min(row.blocks[1], shape[1]) || l.saved_row_statistics_elements !== 2 * shape[0] || s.full_score_shape_save_requests !== 0)) throw new RangeError("分块方案的块形状或统计保存量无效。");
    if (!object(row.reference_errors) || !["loss", "dH", "dW", "db"].every(k => finiteNonnegative(row.reference_errors[k]))) throw new RangeError("逐变量参考误差无效。");
  }
  return report;
}

export function chunkedCEReportRows(report, shape = null) {
  validateChunkedCEReport(report);
  return report.measurements.filter(row => !shape || row.shape.join("×") === shape).map(row => `<tr><th scope="row">${escapeText(row.name)}</th><td>${row.shape.join("×")}</td><td>${row.blocks?.join("×") || "完整"}</td><td>${row.wall_ms.median.toFixed(3)}</td><td>${row.wall_ms.min.toFixed(3)}…${row.wall_ms.max.toFixed(3)}</td><td>${row.saved_tensor_storage.peak_referenced_bytes.toLocaleString("en-US")}</td><td>${row.logical.maximum_score_tile_elements.toLocaleString("en-US")}</td></tr>`).join("");
}

export function createChunkedCEReportLoader({ onClear = () => {}, onReport = () => {}, onStatus = () => {} } = {}) {
  let request = 0;
  return {
    async read(file) {
      const version = ++request; onClear();
      if (!file) { onStatus("请选择本机 report.json；文件只在浏览器内读取。"); return; }
      try {
        if (!integer(file.size, 0, 5 * 1024 * 1024) || typeof file.text !== "function") throw new RangeError("报告超过 5 MiB 或文件对象无效。");
        const report = validateChunkedCEReport(JSON.parse(await file.text()));
        if (version !== request) return;
        onReport(report); onStatus(`已读取 ${file.name || "report.json"}；显示该文件记录的实际实验结果。`);
      } catch (error) { if (version === request) onStatus(`无法读取：${error.message}`); }
    },
    cancel() { request++; onClear(); },
  };
}

function renderTile(svg, result, event) {
  const { inputs } = result, w = 54, h = 34, x0 = 94, y0 = 36;
  const cells = inputs.H.flatMap((_, i) => inputs.W.map((_, j) => {
    const active = inputs.mask[i] && inputs.labels[i] !== inputs.ignoreIndex, current = event && i >= event.tStart && i < event.tEnd && j >= event.vStart && j < event.vEnd;
    return `<rect x="${x0 + j * w}" y="${y0 + i * h}" width="${w - 3}" height="${h - 3}" class="ce-cell${current ? " ce-current" : ""}${active ? "" : " ce-ignored"}"/><text x="${x0 + j * w + 25}" y="${y0 + i * h + 21}" text-anchor="middle">${active && inputs.labels[i] === j ? "●" : "·"}</text>`;
  }));
  svg.innerHTML = `<text x="94" y="20">词表 ID →</text>${inputs.W.map((_, j) => `<text x="${x0 + j * w + 25}" y="30" text-anchor="middle">${j}</text>`).join("")}${inputs.H.map((_, i) => `<text x="8" y="${y0 + i * h + 21}">位置 ${i}${inputs.mask[i] && inputs.labels[i] !== inputs.ignoreIndex ? "" : " 忽略"}</text>`).join("")}${cells.join("")}<text x="8" y="231">● 目标类别；高亮为当前块。完整网格只表示索引范围。</text>`;
}

function renderMeasurementsChart(svg, rows, metric) {
  const values = rows.map(row => metric === "time" ? row.wall_ms.median : row.saved_tensor_storage.peak_referenced_bytes / 1024), maximum = Math.max(1e-12, ...values), height = 50 + rows.length * 42;
  svg.setAttribute("viewBox", `0 0 620 ${height}`);
  svg.setAttribute("aria-label", metric === "time" ? "保存报告中的 CPU 前向加反向中位耗时，单位毫秒；完整数值见表格" : "保存报告中的唯一 storage 保存引用峰值，单位 KiB；不是进程或分配器内存峰值");
  svg.innerHTML = `<text x="8" y="20">${metric === "time" ? "实际 CPU 时间 / ms" : "保存引用 / KiB"}</text>${rows.map((row, i) => { const y = 34 + i * 42, value = values[i]; return `<text x="8" y="${y + 17}">${row.method === "dense" ? "dense" : `块 ${row.blocks.join("×")}`}</text><rect x="138" y="${y}" width="${360 * value / maximum}" height="23" class="ce-measure-bar${row.method === "dense" ? " ce-dense-bar" : ""}"/><text x="508" y="${y + 17}">${value.toFixed(metric === "time" ? 3 : 1)}</text>`; }).join("")}`;
}

export function initChunkedCrossEntropy() {
  const token = $("[data-ce-token]"); if (!token) return;
  const vocab = $("[data-ce-vocab]"), mask = $("[data-ce-mask]"), bias = $("[data-ce-bias]"), shift = $("[data-ce-shift]"), upstream = $("[data-ce-upstream]"), fault = $("[data-ce-fault]"), target = $("[data-ce-target]");
  let step = 0;
  const set = (name, value) => { $(`[data-ce-${name}]`).textContent = value; };
  const render = () => {
    const r = chunkedCrossEntropy({ inputs: chunkedCEInputs({ bias: bias.value === "on", mask: mask.value, target: Number(target.value) }), tokenBlock: Number(token.value), vocabBlock: Number(vocab.value), scoreShift: Number(shift.value), upstream: Number(upstream.value), fault: fault.value });
    step = Math.min(step, Math.max(0, r.events.length - 1)); const e = r.events[step];
    set("loss", r.skipped ? "未定义 · 跳步" : `${formatValue(r.loss)} / ${formatValue(r.reference.loss)}`);
    for (const key of ["dH", "dW", "db"]) set(key.toLowerCase() + "-error", key === "db" && !r.db ? "无 bias" : r.errors[key].toExponential(2));
    set("count", `${r.validCount} / ${r.inputs.H.length} 个监督位置`); set("loss-error", r.errors.loss.toExponential(2));
    set("step", r.events.length ? `${step + 1} / ${r.events.length}` : "没有块事件");
    set("event", e ? `${e.stage === "forward" ? "前向：在线归一化与目标捕获" : "反向：重算概率与梯度累加"}；位置 ${e.tStart}…${e.tEnd - 1}，词表 ${e.vStart}…${e.vEnd - 1}` : "无有效监督，返回可微零哨兵并跳过更新。");
    set("note", r.skipped ? "数值 0 是空监督的程序哨兵，平均交叉熵未定义；优化器应跳步。" : fault.value === "local-softmax" ? "故障只改反向：每个词表块单独归一化；单个有效位置的总概率质量变为词表块数量。块覆盖整个词表或上游为 0 时，梯度差可能被隐藏。" : fault.value === "overwrite-dh" ? "故障只改反向：后一个词表块覆盖 dH，丢掉前面的贡献。dW/db 仍累计全部位置；单词表块或上游为 0 时，故障可能被隐藏。" : "顶部展示全部块完成后的 loss 与梯度误差。下方事件展示当前块及此事件后的累加状态；上游标量只缩放梯度。");
    renderTile($("[data-ce-tile]"), r, e);
    $("[data-ce-event-rows]").innerHTML = e ? (e.stage === "forward" ? e.rows.map(row => `<tr><th scope="row">${row.position}</th><td>${row.active ? formatVector(e.scores[row.position - e.tStart]) : "忽略"}</td><td>${row.maximum === null ? "—" : formatValue(row.maximum)}</td><td>${formatValue(row.denominator)}</td><td>${row.targetLogit === null ? "尚未命中 / 忽略" : formatValue(row.targetLogit)}</td></tr>`).join("") : e.probability.map((p, i) => `<tr><th scope="row">${e.tStart + i}</th><td>${formatVector(p)}</td><td colspan="2">dZ ${formatVector(e.dScores[i])}</td><td>本块 dH ${formatVector(e.localDH[i])}</td></tr>`).join("")) : `<tr><td colspan="5">没有有效监督位置。</td></tr>`;
    $("[data-ce-state-rows]").innerHTML = r.inputs.H.map((h, i) => `<tr><th scope="row">${i}</th><td>${formatVector(h)}</td><td>${r.inputs.labels[i]} / ${String(r.inputs.mask[i])}</td><td>${r.lse[i] === null ? "—" : formatValue(r.lse[i])}</td><td>${r.targetLogit[i] === null ? "—" : formatValue(r.targetLogit[i])}</td><td>${r.perPositionLoss[i] === null ? "—" : formatValue(r.perPositionLoss[i])}</td><td>${formatValue(r.probabilityMass[i])}</td></tr>`).join("");
    $("[data-ce-gradient-rows]").innerHTML = ["dH", "dW", ...(r.db ? ["db"] : [])].flatMap(name => r[name].map((row, i) => { const current = e?.stage === "backward" ? e[name][i] : (Array.isArray(row) ? row.map(() => 0) : 0); const fmt = Array.isArray(row) ? formatVector : formatValue; return `<tr><th scope="row">${name}[${i}]</th><td>${fmt(current)}</td><td>${fmt(row)}</td><td>${fmt(r.reference[name][i])}</td></tr>`; })).join("");
    set("logical", `逻辑元素：H ${r.storage.sharedH}，W ${r.storage.sharedW}，bias ${r.storage.sharedBias}；完整 logits ${r.storage.denseLogits}；保存行统计 ${r.storage.savedRowStatistics}；单个 score 块最多 ${r.storage.maximumScoreTile}。前向有效点积 ${r.work.forwardScores} 次，反向重算 ${r.work.recomputedScores} 次。`);
    $("[data-ce-inputs]").textContent = JSON.stringify(r.inputs, null, 2);
    $("[data-ce-prev]").disabled = step === 0; $("[data-ce-next]").disabled = !r.events.length || step === r.events.length - 1;
  };
  [token, vocab, mask, bias, shift, upstream, fault, target].forEach(el => el.addEventListener("change", () => { step = 0; render(); }));
  $("[data-ce-next]").addEventListener("click", () => { step++; render(); }); $("[data-ce-prev]").addEventListener("click", () => { step--; render(); }); $("[data-ce-reset]").addEventListener("click", () => { step = 0; render(); }); render();
  const output = $("[data-ce-report-output]"), status = $("[data-ce-report-status]"), shapeSelect = $("[data-ce-report-shape]"), metric = $("[data-ce-report-metric]"); let report = null;
  const renderReport = () => {
    if (!report) return;
    const selected = shapeSelect.value, rows = report.measurements.filter(row => row.shape.join("×") === selected);
    $("[data-ce-report-rows]").innerHTML = chunkedCEReportRows(report, selected);
    renderMeasurementsChart($("[data-ce-measure-chart]"), rows, metric.value);
    $("[data-ce-report-logical]").textContent = JSON.stringify(rows.map(row => ({ name: row.name, logical: row.logical, saved_tensor_storage: row.saved_tensor_storage, reference_errors: row.reference_errors, wall_ms: row.wall_ms })), null, 2);
  };
  const accept = parsed => {
    report = parsed; const c = report.correctness, r = report.runtime;
    $("[data-ce-report-provenance]").textContent = `${report.created_at} · ${r.platform} · Python ${r.python} / PyTorch ${r.torch} / NumPy ${r.numpy} · ${r.device}/${r.dtype} · ${r.threads} 线程。来源：${report.provenance.report_path}；run ${report.provenance.run_id}。`;
    $("[data-ce-report-checks]").textContent = `${c.cases} 个配置、${c.comparisons} 项比较；dense 最大误差 ${c.max_dense_error.toExponential(2)}，JS 最大误差 ${c.max_js_error.toExponential(2)}，有限差分最大误差 ${c.finite_difference_max_error.toExponential(2)}。${c.fault_cases} 项预期故障已定位；独立检查通过：未生成完整 score 形状、未为反向保存 score/probability。`;
    shapeSelect.innerHTML = [...new Set(report.measurements.map(row => row.shape.join("×")))].map(s => `<option>${s}</option>`).join("");
    output.hidden = false; renderReport();
  };
  const loader = createChunkedCEReportLoader({ onClear: () => { report = null; output.hidden = true; for (const name of ["rows", "provenance", "checks", "logical"]) $(`[data-ce-report-${name}]`).textContent = ""; $("[data-ce-measure-chart]").innerHTML = ""; shapeSelect.innerHTML = ""; }, onReport: accept, onStatus: message => { status.textContent = message; } });
  $("[data-ce-report-file]").addEventListener("change", event => loader.read(event.target.files?.[0]));
  shapeSelect.addEventListener("change", renderReport); metric.addEventListener("change", renderReport);
  $("[data-ce-report-default]").addEventListener("click", () => { loader.cancel(); $("[data-ce-report-file]").value = ""; if (chunkedCEReportSnapshot) { accept(validateChunkedCEReport(chunkedCEReportSnapshot)); status.textContent = "显示从已保存实验报告提取的默认快照。"; } });
  if (chunkedCEReportSnapshot) { accept(validateChunkedCEReport(chunkedCEReportSnapshot)); status.textContent = "默认数值来自已保存的真实 CPU 实验；可读取另一次本机报告。"; }
}
