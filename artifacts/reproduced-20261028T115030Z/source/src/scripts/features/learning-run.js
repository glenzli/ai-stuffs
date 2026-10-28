import { $ } from "../core.js";
import { escapeText } from "./objective-math.js";
import { learningRunSnapshot } from "./learning-run-data.js";

export const LEARNING_REPORT_SCHEMA = "learning-run-v1";
export const LEARNING_REPORT_MAX_BYTES = 16 * 1024 * 1024;
const SPLITS = ["train", "validation", "test", "length"];
const PHASES = ["learned", "initial", "identity", "constant", "oracle"];
const SPLIT_LABELS = { train: "训练集", validation: "验证集", test: "独立测试", length: "长度外推" };
const PHASE_LABELS = { learned: "训练后模型", initial: "初始模型", identity: "复制基线", constant: "训练集最常见符号基线", oracle: "算法 oracle（已知规则）" };
const finite = value => Number.isFinite(value) && value >= 0;
const integer = (value, maximum = 1e12) => Number.isInteger(value) && value >= 0 && value <= maximum;
const probability = value => finite(value) && value <= 1;
const string = (value, maximum = 500) => typeof value === "string" && value.length > 0 && value.length <= maximum;
const near = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
const fail = message => { throw new RangeError(message); };
const percent = value => `${(100 * value).toFixed(2)}%`;
const lossText = value => value === null ? "不适用" : value.toFixed(5);
const numberText = value => value.toLocaleString("en-US");

export function scoreLearningPrediction(predicted, target, maxNewTokens = 20) {
  if (!integer(maxNewTokens, 64) || maxNewTokens < 1 || !Array.isArray(target) || target.some(x => !Number.isInteger(x) || x < 4 || x > 7) ||
      !Array.isArray(predicted) || predicted.length > maxNewTokens || predicted.some(x => !Number.isInteger(x) || x < 0 || x > 7) || predicted.slice(0, -1).includes(3)) fail("生成轨迹或目标 token 无效。");
  const gold = [...target, 3], terminated = predicted.at(-1) === 3;
  const content = terminated ? predicted.slice(0, -1) : predicted;
  const correct = predicted.reduce((sum, token, index) => sum + Number(index < gold.length && token === gold[index]), 0);
  const slots = Math.max(predicted.length, gold.length);
  return { correct_tokens: correct, scored_tokens: slots, token_accuracy: correct / slots,
    exact_match: terminated && predicted.length === gold.length && predicted.every((token, index) => token === gold[index]),
    terminated, termination: terminated ? "eos" : predicted.length === maxNewTokens ? "cap" : "incomplete",
    early_eos: terminated && content.length < target.length,
    missing_tokens: Math.max(0, target.length - content.length), extra_tokens: Math.max(0, content.length - target.length),
    eos_position_correct: terminated && content.length === target.length };
}

function validateMetrics(metrics, count, baseline = false) {
  if (!metrics || metrics.records !== count || !integer(count, 2000) || count < 1) fail("评估题目数量无效。");
  for (const key of ["free_token_accuracy", "free_token_accuracy_macro", "exact_match", "eos_rate", "eos_position_correct"]) if (!probability(metrics[key])) fail(`指标 ${key} 必须为 0…1 的有限值。`);
  for (const key of ["correct_tokens", "scored_tokens", "exact_matches", "early_eos_count", "cap_count", "incomplete_count", "missing_tokens", "extra_tokens"]) if (!integer(metrics[key])) fail(`计数字段 ${key} 无效。`);
  if (!metrics.scored_tokens || metrics.correct_tokens > metrics.scored_tokens || metrics.exact_matches > count ||
      metrics.early_eos_count > count || metrics.cap_count + metrics.incomplete_count > count ||
      !near(metrics.free_token_accuracy, metrics.correct_tokens / metrics.scored_tokens) || !near(metrics.exact_match, metrics.exact_matches / count)) fail("准确率与计数不一致。");
  if (baseline) {
    if (metrics.teacher_forced_loss !== null || metrics.teacher_forced_token_accuracy !== null) fail("规则基线不能提供模型 teacher-forced loss。");
  } else if (!finite(metrics.teacher_forced_loss) || !probability(metrics.teacher_forced_token_accuracy) || !integer(metrics.supervised_tokens) || !metrics.supervised_tokens) fail("模型 teacher-forced 指标无效。");
}

function validateEvaluation(evaluation, split, report, baseline = false) {
  const count = report.task.splits[split].records, cap = report.frozen.generation.max_new_tokens;
  if (!evaluation || !Array.isArray(evaluation.predictions) || !evaluation.predictions.length || evaluation.predictions.length > count) fail("报告缺少逐题预测。");
  if (!report.snapshot && evaluation.predictions.length !== count) fail("完整报告的逐题结果不完整。");
  if (report.snapshot && evaluation.prediction_total !== count) fail("快照须声明完整预测数量。");
  validateMetrics(evaluation.metrics, count, baseline);
  if (evaluation.metrics.max_new_tokens !== cap || !string(evaluation.metrics.generation, 120)) fail("生成上限或方法与配置不一致。");
  const ids = new Set();
  for (const row of evaluation.predictions) {
    if (!row || !string(row.id, 160) || !string(row.group_id, 160) || ids.has(row.id) || !integer(row.length, 10) || !row.length ||
        !Array.isArray(row.content) || row.content.length !== row.length || row.content.some(x => !Number.isInteger(x) || x < 4 || x > 7) ||
        !Array.isArray(row.target) || row.target.length !== row.length || !row.target.every((x, i) => x === row.content[row.length - 1 - i]) ||
        !report.task.splits[split].lengths.includes(row.length)) fail("逐题 ID、长度或反转目标无效。");
    ids.add(row.id);
    const score = scoreLearningPrediction(row.predicted, row.target, cap);
    for (const [key, value] of Object.entries(score)) if (typeof value === "number" ? !near(row[key], value) : row[key] !== value) fail(`逐题 ${key} 与实际生成不一致。`);
    if (!baseline && (!finite(row.teacher_forced_loss) || !Array.isArray(row.teacher_forced_predicted) || row.teacher_forced_predicted.length !== row.length + 1 || row.teacher_forced_predicted.some(x => !integer(x, 7)))) fail("逐题 teacher-forced 结果无效。");
  }
  if (!report.snapshot) {
    const rows = evaluation.predictions, metrics = evaluation.metrics;
    for (const key of ["correct_tokens", "scored_tokens", "missing_tokens", "extra_tokens"]) if (rows.reduce((sum, row) => sum + row[key], 0) !== metrics[key]) fail("逐题计数与总指标不一致。");
    const sums = { exact_matches: rows.filter(r => r.exact_match).length, early_eos_count: rows.filter(r => r.early_eos).length,
      cap_count: rows.filter(r => r.termination === "cap").length, incomplete_count: rows.filter(r => r.termination === "incomplete").length };
    for (const [key, value] of Object.entries(sums)) if (metrics[key] !== value) fail("结束计数与逐题结果不一致。");
    if (!near(metrics.free_token_accuracy_macro, rows.reduce((sum, row) => sum + row.token_accuracy, 0) / count) ||
        !near(metrics.eos_rate, rows.filter(r => r.terminated).length / count) || !near(metrics.eos_position_correct, rows.filter(r => r.eos_position_correct).length / count)) fail("生成统计与逐题结果不一致。");
  }
  return evaluation;
}

function validateCacheCheck(row) {
  if (!row || typeof row.passed !== "boolean" || typeof row.generated_equal !== "boolean" || !integer(row.generated_records, 20000) || !row.generated_records) fail("缓存检查结构无效。");
  if (row.dtype_checks) {
    if (!Array.isArray(row.dtype_checks) || row.dtype_checks.length !== 2 || row.dtype_checks.map(x => x.dtype).sort().join(",") !== "float32,float64") fail("缓存检查须含 float32/float64。");
    for (const value of row.dtype_checks) {
      for (const key of ["atol", "rtol", "max_absolute_error", "max_scaled_error", "max_softmax_probability_error"]) if (!finite(value[key])) fail("缓存精度数值无效。");
      if (!integer(value.top1_disagreements) || !integer(value.checked_logit_elements) || typeof value.passed !== "boolean" ||
          value.passed !== (value.max_scaled_error <= 1 && value.top1_disagreements === 0)) fail("缓存通过状态与实际误差不一致。");
    }
    const expected = row.generated_equal && row.saved_predictions_equal !== false && row.dtype_checks.every(x => x.passed);
    if (row.passed !== expected) fail("缓存综合通过状态不一致。");
  } else if (!finite(row.max_logit_error) || !finite(row.logit_tolerance) || typeof row.passed !== "boolean" || row.passed !== (row.generated_equal && row.max_logit_error <= row.logit_tolerance)) fail("原始绝对阈值缓存检查无效。");
}

export function validateLearningReport(report) {
  if (!report || report.schema !== LEARNING_REPORT_SCHEMA || !["passed", "failed"].includes(report.status) || typeof report.snapshot !== "boolean" ||
      !string(report.created_at, 100) || !Number.isFinite(Date.parse(report.created_at)) || !string(report.scope, 1500) ||
      !report.runtime || !report.frozen || !report.task || !Array.isArray(report.runs) || !report.runs.length || report.runs.length > 10) fail("请选择 learning lab 保存的完整报告或有来源的快照。");
  for (const key of ["python", "torch", "numpy", "platform", "device", "dtype"]) if (!string(report.runtime[key])) fail("运行环境字段无效。");
  if (!integer(report.runtime.threads, 256) || report.runtime.threads < 1) fail("CPU 线程数无效。");
  const cap = report.frozen.generation?.max_new_tokens, training = report.frozen.training;
  if (!integer(cap, 64) || cap < 1 || !training || !integer(training.steps, 1000000) || !training.steps || !integer(training.batch_size, 100000) || !training.batch_size || !finite(training.lr) || training.lr === 0) fail("训练预算或生成上限无效。");
  if (!report.provenance || !string(report.provenance.report_path, 1500) || !string(report.task.split_sha256, 64) || !/^[0-9a-f]{64}$/u.test(report.task.split_sha256) ||
      !Array.isArray(report.task.vocab) || report.task.vocab.join(",") !== "PAD,BOS,SEP,EOS,A,B,C,D") fail("报告来源或词表无效。");
  for (const split of SPLITS) {
    const value = report.task.splits?.[split];
    if (!value || !integer(value.records, 2000) || !value.records || !integer(value.groups, value.records) || !value.groups || !Array.isArray(value.lengths) || !value.lengths.length || value.lengths.some(x => !integer(x, 10) || !x)) fail("拆分数量或长度支持无效。");
  }
  const seeds = new Set();
  for (const run of report.runs) {
    if (!integer(run.seed, 2 ** 32 - 1) || seeds.has(run.seed) || !integer(run.parameter_count, 1e9) || !run.parameter_count ||
        run.progress?.step !== training.steps || !integer(run.progress.supervised_tokens) || !integer(run.progress.input_tokens) ||
        !finite(run.training_seconds_including_periodic_eval) || !Array.isArray(run.curve) || run.curve.length < 2 || run.curve.length > 1000) fail("seed、预算或曲线无效。");
    seeds.add(run.seed);
    let previous = -1, previousTokens = -1;
    for (const point of run.curve) {
      if (!integer(point.step, training.steps) || point.step <= previous || !integer(point.supervised_tokens) || point.supervised_tokens < previousTokens || !integer(point.input_tokens)) fail("曲线更新或 token 计数不递增。");
      previous = point.step; previousTokens = point.supervised_tokens;
      for (const split of ["train", "validation"]) validateMetrics(point[split], report.task.splits[split].records);
    }
    if (run.curve[0].step !== 0 || run.curve.at(-1).step !== training.steps || run.curve.at(-1).supervised_tokens !== run.progress.supervised_tokens) fail("曲线缺少初始或固定预算末端。");
    for (const phase of ["initial", "learned"]) for (const split of SPLITS) validateEvaluation(run[phase]?.[split], split, report);
    if (typeof run.fit_passed !== "boolean" || run.fit_passed !== (run.learned.train.metrics.exact_match >= .95)) fail("训练拟合阈值与实际结果不一致。");
  }
  for (const kind of ["identity", "constant", "oracle"]) for (const split of SPLITS) validateEvaluation(report.baselines?.[kind]?.[split], split, report, true);
  if (!report.checks || !Array.isArray(report.checks.cache) || report.checks.cache.length !== report.runs.length * 2 ||
      !Array.isArray(report.checks.resume) || report.checks.resume.length !== report.runs.length || report.checks.split_integrity !== true ||
      report.checks.all_seeds_fit !== report.runs.every(run => run.fit_passed)) fail("核验记录或拟合汇总无效。");
  for (const row of report.checks.cache) validateCacheCheck(row);
  for (const row of report.checks.resume) {
    if (typeof row.passed !== "boolean" || row.fresh_process !== true || !row.comparisons ||
        Object.values(row.comparisons).some(c => typeof c.equal !== "boolean" || !finite(c.max_error)) ||
        row.passed !== Object.values(row.comparisons).every(c => c.equal)) fail("恢复检查无效。");
  }
  const computationsPassed = report.checks.cache.every(c => c.passed) && report.checks.resume.every(c => c.passed);
  if (computationsPassed !== (report.status === "passed")) fail("报告计算状态与实际检查不一致。");
  return report;
}

function selectedRun(report, seed) {
  const run = report.runs.find(row => String(row.seed) === String(seed));
  if (!run) fail("找不到所选 seed。");
  return run;
}

export function learningEvaluation(report, seed, split = "test", phase = "learned") {
  if (!SPLITS.includes(split) || !PHASES.includes(phase)) fail("未知拆分或模型条件。");
  const run = selectedRun(report, seed);
  return ["initial", "learned"].includes(phase) ? run[phase][split] : report.baselines[phase][split];
}

export function learningCurveSeries(report, { seed = report.runs[0].seed, metric = "teacher_forced_loss", axis = "step" } = {}) {
  if (!["teacher_forced_loss", "free_token_accuracy", "exact_match"].includes(metric) || !["step", "supervised_tokens"].includes(axis)) fail("曲线指标或横轴无效。");
  const run = selectedRun(report, seed), scale = metric === "teacher_forced_loss" ? 1 : 100;
  return ["train", "validation"].map(split => ({ name: split, label: SPLIT_LABELS[split], points: run.curve.map(point => ({ x: point[axis], y: point[split][metric] * scale })) }));
}

export function learningCurveMarkup(report, options = {}) {
  const series = learningCurveSeries(report, options), metric = options.metric || "teacher_forced_loss", axis = options.axis || "step";
  const xmax = Math.max(1, ...series.flatMap(s => s.points.map(p => p.x)));
  const ymax = metric === "teacher_forced_loss" ? Math.max(.1, ...series.flatMap(s => s.points.map(p => p.y))) * 1.08 : 100;
  const x = value => 58 + value / xmax * 630, y = value => 225 - value / ymax * 180;
  const tick = value => axis === "step" ? String(Math.round(value)) : value >= 1000 ? `${(value / 1000).toFixed(0)}k` : String(Math.round(value));
  const ticks = Array.from({ length: 5 }, (_, i) => i / 4);
  return `<title>${metric === "teacher_forced_loss" ? "teacher-forced loss：nats/监督 token" : metric === "exact_match" ? "自由生成完整匹配：百分比" : "自由生成 token 准确率：百分比"}；仅保存的 train/validation 测量</title>` +
    ticks.map(t => `<line x1="58" y1="${y(t*ymax)}" x2="688" y2="${y(t*ymax)}" class="learning-grid"/><text x="47" y="${y(t*ymax)+4}" text-anchor="end">${(t*ymax).toFixed(metric === "teacher_forced_loss" ? 2 : 0)}</text><text x="${x(t*xmax)}" y="247" text-anchor="middle">${tick(t*xmax)}</text>`).join("") +
    `<text x="58" y="25">${metric === "teacher_forced_loss" ? "CE · nats / 监督 token" : metric === "exact_match" ? "自由生成 exact match · %" : "自由生成 token accuracy · %"}</text><text x="370" y="280" text-anchor="middle">${axis === "step" ? "优化器更新次数" : "累计消费监督 token"}</text>` +
    series.map((s, i) => `<path d="${s.points.map((p, j) => `${j ? "L" : "M"}${x(p.x).toFixed(3)},${y(p.y).toFixed(3)}`).join(" ")}" class="learning-line learning-line-${i}"/>${s.points.map(p => `<circle cx="${x(p.x)}" cy="${y(p.y)}" r="3.2" class="learning-dot-${i}"/>`).join("")}<text x="${490+i*104}" y="25" class="learning-legend-${i}">${s.label}</text>`).join("");
}

export function learningSummaryRows(report) {
  return report.runs.map(run => `<tr><th scope="row">${run.seed}</th><td>${lossText(run.learned.train.metrics.teacher_forced_loss)}</td>${SPLITS.map(split => `<td>${percent(run.learned[split].metrics.exact_match)}</td>`).join("")}<td>${run.fit_passed ? "≥95% 训练阈值" : "未达 95% 训练阈值"}</td></tr>`).join("");
}

export function learningPredictionRows(report, { seed = report.runs[0].seed, split = "test", phase = "learned", page = 0, pageSize = 25, errorsOnly = false } = {}) {
  if (!integer(page, 10000) || !integer(pageSize, 100) || !pageSize) fail("分页参数无效。");
  const evaluation = learningEvaluation(report, seed, split, phase);
  const all = evaluation.predictions.filter(row => !errorsOnly || !row.exact_match);
  const shown = all.slice(page * pageSize, (page + 1) * pageSize), vocab = report.task.vocab;
  const tokens = values => values.length ? values.map(x => escapeText(vocab[x])).join(" ") : "∅";
  const termination = row => row.early_eos ? "提前 EOS" : row.termination === "eos" ? "EOS" : row.termination === "cap" ? "达到统一上限" : "未完成";
  return { total: all.length, page, pages: Math.max(1, Math.ceil(all.length / pageSize)), shown: shown.length,
    html: shown.map(row => `<tr><th scope="row">${escapeText(row.id)}<small>${escapeText(row.group_id)}</small></th><td>${tokens(row.content)}</td><td>${tokens([...row.target, 3])}</td><td>${tokens(row.predicted)}</td><td>${row.teacher_forced_predicted ? tokens(row.teacher_forced_predicted) : "不适用"}</td><td>${row.correct_tokens}/${row.scored_tokens}<br>${percent(row.token_accuracy)}</td><td>${row.exact_match ? "完整匹配" : "失败"}<br>${termination(row)}<br>缺 ${row.missing_tokens} / 多 ${row.extra_tokens}</td></tr>`).join("") };
}

export function learningPairedData(report, { split = "test", old = "initial", metric = "exact_match" } = {}) {
  validateLearningReport(report);
  if (report.snapshot) fail("配对导出需要完整 report.json；默认快照只保留部分实际预测。");
  if (!SPLITS.includes(split) || !["initial", "identity"].includes(old) || !["exact_match", "token_accuracy"].includes(metric)) fail("配对比较条件无效。");
  const rows = [];
  for (const run of report.runs) {
    const older = new Map(learningEvaluation(report, run.seed, split, old).predictions.map(row => [row.id, row]));
    for (const row of run.learned[split].predictions) {
      const before = older.get(row.id);
      if (!before || before.group_id !== row.group_id) fail("配对 ID 或反转组不一致。");
      rows.push({ id: row.id, group: row.group_id, seed: String(run.seed), old: Number(before[metric]), new: Number(row[metric]) });
    }
  }
  if (rows.length > 2000) fail("当前拆分超过配对评估页的 2,000 行上限。");
  return { schema: 1, rows, provenance: { report_path: report.provenance.report_path, metric, split, old, new: "learned", grouping: "reversal-equivalence group; analyze each seed separately" } };
}

export function learningExampleMarkup(report, { seed = report.runs[0].seed, split = "test" } = {}) {
  const run = selectedRun(report, seed), before = run.initial[split].predictions, after = run.learned[split].predictions;
  const same = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);
  const paired = after.filter(row => before.some(old => old.id === row.id && old.group_id === row.group_id && same(old.content, row.content) && same(old.target, row.target)));
  const current = paired.find(row => !same(row.content, row.target)) || paired[0];
  if (!current) return '<p>当前保存的片段没有同一道题的训练前后预测。可在下方读取完整报告，继续查看学习曲线与总体结果。</p>';
  const initial = before.find(row => row.id === current.id && row.group_id === current.group_id && same(row.content, current.content) && same(row.target, current.target));
  const tokens = values => escapeText(values.map(id => report.task.vocab[id]).join(" "));
  return `<p class="learning-example-source">seed ${run.seed} · ${escapeText(SPLIT_LABELS[split])} · ${escapeText(current.id)} · 同题的实际保存预测</p>
    <div class="learning-example-input"><span>输入内容</span><code>${tokens(current.content)}</code><span>正确答案</span><code>${tokens([...current.target, 3])}</code></div>
    <div class="learning-example-pair"><div><h3>训练前</h3><code>${tokens(initial.predicted)}</code><p>${initial.exact_match ? "完整匹配" : "未完整匹配"}</p></div><div><h3>训练后</h3><code>${tokens(current.predicted)}</code><p>${current.exact_match ? "完整匹配" : "未完整匹配"}</p></div></div>
    <p class="learning-example-note">这道题用于观察输出变化；总体表现、不同 seed 和未见长度的结果在下方分别展示。</p>`;
}

function downloadJSON(value, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2) + "\n"], { type: "application/json" }));
  const link = document.createElement("a"); link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function initLearningRun() {
  const input = $("[data-lr-file]"); if (!input) return;
  const seed = $("[data-lr-seed]"), split = $("[data-lr-split]"), phase = $("[data-lr-phase]"), metric = $("[data-lr-curve-metric]"), axis = $("[data-lr-axis]");
  const output = $("[data-lr-output]"), chart = $("[data-lr-chart]"), paired = $("[data-lr-paired]"), save = $("[data-lr-download]"), errors = $("[data-lr-errors]");
  let report = null, request = 0, page = 0;
  const set = (key, value) => { const node = $(`[data-lr-${key}]`); if (node) node.textContent = value; };
  const clear = () => {
    report = null; output.hidden = true; seed.replaceChildren(); chart.replaceChildren();
    for (const key of ["rows", "summary-rows", "curve-rows"]) $(`[data-lr-${key}]`).replaceChildren();
    $("[data-lr-preview]")?.replaceChildren();
    for (const key of ["provenance", "computation", "fit", "cache", "model", "ce", "token", "em", "termination", "selection", "page", "paired-note", "outcome"]) set(key, "");
    paired.disabled = true; save.disabled = true;
  };
  const renderPredictions = () => {
    if (!report) return;
    const rows = learningPredictionRows(report, { seed: seed.value, split: split.value, phase: phase.value, page, errorsOnly: errors.checked });
    if (page >= rows.pages) { page = rows.pages - 1; return renderPredictions(); }
    $("[data-lr-rows]").innerHTML = rows.html;
    set("page", `${rows.total} 条${report.snapshot ? "已保存快照片段" : "实际逐题结果"} · 第 ${page+1}/${rows.pages} 页${rows.shown ? ` · 显示 ${page*25+1}–${page*25+rows.shown}` : " · 当前筛选无题目"}`);
    $("[data-lr-table-prev]").disabled = page === 0;
    $("[data-lr-table-next]").disabled = page+1 >= rows.pages;
  };
  const render = () => {
    if (!report) return;
    const run = selectedRun(report, seed.value), evaluation = learningEvaluation(report, seed.value, split.value, phase.value), m = evaluation.metrics;
    output.hidden = false;
    const preview = $("[data-lr-preview]");
    if (preview) preview.innerHTML = learningExampleMarkup(report, { seed: seed.value, split: split.value });
    set("ce", lossText(m.teacher_forced_loss)); set("token", percent(m.free_token_accuracy)); set("em", percent(m.exact_match));
    set("termination", `${m.early_eos_count} 提前 EOS / ${m.cap_count} 上限 / 缺 ${m.missing_tokens} / 多 ${m.extra_tokens}`);
    set("selection", `${PHASE_LABELS[phase.value]} · seed ${run.seed} · ${SPLIT_LABELS[split.value]} · ${m.records} 题 · 自由生成统一 max_new_tokens=${report.frozen.generation.max_new_tokens}`);
    const runtime = report.runtime;
    set("provenance", `${report.snapshot ? "默认真实快照" : "完整本地报告"} · ${report.created_at} · Python ${runtime.python} / PyTorch ${runtime.torch} / NumPy ${runtime.numpy} · ${runtime.device} ${runtime.dtype} · ${runtime.threads} 线程 · ${report.provenance.report_path}${report.provenance.report_sha256 ? ` · SHA-256 ${report.provenance.report_sha256}` : ""}`);
    set("computation", report.status === "passed" ? `计算核验通过：${report.checks.resume.length} 个新进程完整恢复，full/cache 精度与实际生成对照通过。` : "计算核验未通过：这是原始报告的检查结果，下面仍展示该次实际学习测量。");
    set("fit", `${report.runs.filter(r => r.fit_passed).length}/${report.runs.length} 个 seed 达到预设训练 EM ≥95% 的描述性阈值；${report.checks.all_seeds_fit ? "全部达到" : "仍有 seed 未达到"}。此阈值与计算核验、独立测试及长度外推分别判断。`);
    set("outcome", `seed ${run.seed}：训练 EM ${percent(run.learned.train.metrics.exact_match)}；独立测试 EM ${percent(run.learned.test.metrics.exact_match)}；长度 ${report.task.splits.length.lengths.join("/")} 外推 EM ${percent(run.learned.length.metrics.exact_match)}。${run.fit_passed ? "" : "该 seed 在固定预算内没有达到 95% 训练拟合阈值。"}`);
    const revision = report.checks.cache_revalidation;
    if (revision) {
      const floating = report.checks.cache.flatMap(c => c.dtype_checks).filter(c => c.dtype === "float32"), doubles = report.checks.cache.flatMap(c => c.dtype_checks).filter(c => c.dtype === "float64");
      set("cache", `原报告 ${revision.original_status}：float32 绝对阈值 2e-5 未通过。精度标准在诊断后修订：float32 atol=${revision.criterion_float32.atol}, rtol=${revision.criterion_float32.rtol}；float64 atol=${revision.criterion_float64.atol}。全部题目最大 float32 logits 差 ${Math.max(...floating.map(c => c.max_absolute_error)).toExponential(3)}，float64 ${Math.max(...doubles.map(c => c.max_absolute_error)).toExponential(3)}；softmax 概率最大差 ${Math.max(...floating.map(c => c.max_softmax_probability_error)).toExponential(3)}。保留原失败报告、训练源码与复核源码哈希；复核没有更新模型。`);
    } else set("cache", report.checks.cache[0].dtype_checks ? "使用冻结配置中声明的逐元素 float32/float64 精度标准；实际生成与全前向逐题核验。" : `原绝对阈值检查：最大 logits 差 ${Math.max(...report.checks.cache.map(c => c.max_logit_error)).toExponential(3)}；检查状态见计算核验。`);
    set("model", JSON.stringify({ model: report.frozen.model, training: report.frozen.training, generation: report.frozen.generation, selected_seed: run.seed, parameter_count: run.parameter_count, consumed: run.progress, training_seconds_including_periodic_eval: run.training_seconds_including_periodic_eval, source_hashes: report.provenance.original_training_sources || report.provenance.sources, reanalysis_source_hashes: report.provenance.reanalysis_sources || null, original_report: report.provenance.original_report || null, split_sha256: report.task.split_sha256 }, null, 2));
    $("[data-lr-summary-rows]").innerHTML = learningSummaryRows(report);
    chart.innerHTML = learningCurveMarkup(report, { seed: seed.value, metric: metric.value, axis: axis.value });
    chart.setAttribute("aria-label", `seed ${run.seed} 的真实训练与验证曲线，${metric.value === "teacher_forced_loss" ? "nats 每监督 token" : "百分比"}；横轴${axis.value === "step" ? "优化器更新次数" : "累计消费监督 token"}`);
    $("[data-lr-curve-rows]").innerHTML = run.curve.map(p => `<tr><th scope="row">${p.step}</th><td>${numberText(p.supervised_tokens)}</td><td>${lossText(p.train.teacher_forced_loss)}</td><td>${lossText(p.validation.teacher_forced_loss)}</td><td>${percent(p.train.free_token_accuracy)}</td><td>${percent(p.validation.free_token_accuracy)}</td><td>${percent(p.train.exact_match)}</td><td>${percent(p.validation.exact_match)}</td></tr>`).join("");
    paired.disabled = report.snapshot; save.disabled = false;
    set("paired-note", report.snapshot ? "默认快照保留实际测量和少量逐题片段；导入完整 report.json 后可导出当前拆分全部 seed 的配对结果。" : "导出当前拆分全部 seed 的初始/复制 → 训练后成对分数；反转等价输入保留同一 group。配对评估页按 seed 分别检验。");
    renderPredictions();
  };
  const load = (value, status) => {
    report = validateLearningReport(value); page = 0;
    seed.replaceChildren(...report.runs.map(run => { const option = document.createElement("option"); option.value = String(run.seed); option.textContent = String(run.seed); return option; }));
    seed.value = String(report.runs[0].seed); set("status", status); render();
  };
  input.addEventListener("change", async () => {
    const ticket = ++request, file = input.files?.[0]; clear();
    if (!file) { set("status", "未选择文件；可恢复默认真实快照。"); return; }
    set("status", "正在读取本地报告…");
    try {
      if (file.size > LEARNING_REPORT_MAX_BYTES) fail("报告超过 16 MiB，请选择 report.json。");
      const content = await file.text(); if (ticket !== request) return;
      load(JSON.parse(content), `已读取 ${file.name}；文件仅在本地浏览器处理。`);
    } catch (error) { if (ticket === request) { clear(); set("status", `无法读取：${error.message}`); } }
  });
  $("[data-lr-default]").addEventListener("click", () => { request++; input.value = ""; clear(); load(learningRunSnapshot, "已恢复默认真实快照。读取完整报告可查看全部预测。"); });
  [seed, split, phase].forEach(control => control.addEventListener("change", () => { page = 0; render(); }));
  [metric, axis].forEach(control => control.addEventListener("change", render));
  errors.addEventListener("change", () => { page = 0; renderPredictions(); });
  $("[data-lr-table-prev]").addEventListener("click", () => { page = Math.max(0, page-1); renderPredictions(); });
  $("[data-lr-table-next]").addEventListener("click", () => { page++; renderPredictions(); });
  save.addEventListener("click", () => { if (report) downloadJSON(report, report.snapshot ? "learning-run-real-snapshot.json" : "learning-run-report.json"); });
  paired.addEventListener("click", () => {
    if (!report) return;
    try {
      const old = $("[data-lr-paired-old]").value, pairedMetric = $("[data-lr-paired-metric]").value;
      downloadJSON(learningPairedData(report, { split: split.value, old, metric: pairedMetric }), `learning-${split.value}-${old}-to-learned-${pairedMetric}-paired.json`);
      set("paired-note", "配对 JSON 已导出；包含当前拆分的全部 seed。请在配对评估页按 seed 与反转来源组分别检验。");
    } catch (error) { set("paired-note", error.message); }
  });
  try { load(learningRunSnapshot, "已载入默认真实快照；计算核验与学习结果分别展示。"); }
  catch (error) { clear(); set("status", `无法读取默认快照：${error.message}`); }
}
