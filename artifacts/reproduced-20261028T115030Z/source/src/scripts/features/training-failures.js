import { $ } from "../core.js";
import { escapeText } from "./objective-math.js";
import { trainingFailuresSnapshot } from "./training-failures-data.js";

const cases = ["overflow", "high-lr", "labels"], variants = ["baseline", "faulty", "repaired"];
const phases = ["step", "post-update", "next-forward"];
const statuses = ["finite", "nan", "positive_infinity", "negative_infinity"];
const vocabulary = ["PAD", "BOS", "SEP", "EOS", "A", "B", "C", "D"];
const phaseLabel = { step: "本步", "post-update": "更新后", "next-forward": "下一前向" };
const variantLabel = { baseline: "基线", faulty: "故障", repaired: "修复" };
const boundedString = (value, max = 2000) => typeof value === "string" && value.length > 0 && value.length <= max;
const finite = value => typeof value === "number" && Number.isFinite(value);
const nonnegative = value => finite(value) && value >= 0;
const integer = (value, max = 10000000) => Number.isSafeInteger(value) && value >= 0 && value <= max;
const hash = value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const requireValue = (condition, message) => { if (!condition) throw new RangeError(message); };
const equalJson = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function validateScalar(value) {
  requireValue(value && statuses.includes(value.status) && (value.status === "finite" ? finite(value.value) : value.value === null), "非有限标量必须用 null 和明确 status 编码。");
}
function validateStats(stats) {
  requireValue(stats && typeof stats.finite === "boolean" && boundedString(stats.dtype, 30) && Array.isArray(stats.shape) && stats.shape.length <= 8 && stats.shape.every(n => integer(n, 100000)), "张量 shape／dtype 无效。");
  const keys = ["numel", "finite_count", "nan_count", "positive_infinity_count", "negative_infinity_count"];
  requireValue(keys.every(key => integer(stats[key], 100000000)) && stats.shape.reduce((a, b) => a * b, 1) === stats.numel &&
    keys.slice(1).reduce((sum, key) => sum + stats[key], 0) === stats.numel && stats.finite === (stats.finite_count === stats.numel), "张量有限／非有限计数不一致。");
  requireValue(["rms", "max_abs", "l2"].every(key => stats.finite ? nonnegative(stats[key]) : stats[key] === null), "非有限张量的统计值应为 null。");
}
function validateDifference(row) {
  requireValue(row && typeof row.equal === "boolean" && typeof row.finite === "boolean" && ["max_abs", "l2", "relative_l2"].every(key => row.finite ? nonnegative(row[key]) : row[key] === null), "张量差值字段无效。");
  if (row.equal && row.finite) requireValue(row.max_abs === 0 && row.l2 === 0 && row.relative_l2 === 0, "相等张量的差值必须为零。");
}
function validateBatch(run) {
  const batch = run.batch;
  requireValue(batch && Array.isArray(batch.record_ids) && batch.record_ids.length > 0 && batch.record_ids.length <= 128 && batch.record_ids.every(id => boundedString(id, 300)), "batch ID 无效。");
  const rows = batch.record_ids.length;
  requireValue(Array.isArray(batch.group_ids) && batch.group_ids.length === rows && batch.group_ids.every(id => boundedString(id, 300)), "batch group ID 无效。");
  const matrix = (values, predicate) => Array.isArray(values) && values.length === rows && values.every(row => Array.isArray(row) && row.length > 0 && row.length <= 128 && row.length === values[0].length && row.every(predicate));
  const label = value => value === -100 || integer(value, 7);
  requireValue(matrix(batch.input_ids, value => integer(value, 7)) && matrix(batch.labels, label) && matrix(run.actual_labels, label) && matrix(batch.attention_mask, value => typeof value === "boolean") && [batch.labels, run.actual_labels, batch.attention_mask].every(value => value[0].length === batch.input_ids[0].length), "输入、标签和 mask 的矩阵形状／范围无效。");
  let count = 0, changed = 0, input = 0;
  for (let i = 0; i < rows; i++) for (let j = 0; j < batch.labels[i].length; j++) {
    const expected = batch.labels[i][j], actual = run.actual_labels[i][j];
    requireValue((expected === -100) === (actual === -100) && (expected === -100 || batch.attention_mask[i][j]), "实际标签更改了监督 mask。");
    count += Number(expected !== -100); changed += Number(actual !== expected); input += Number(batch.attention_mask[i][j]);
  }
  requireValue(count === batch.supervised_tokens && count === run.supervised_count && changed === run.labels_changed && input === batch.input_tokens, "监督数／标签改动数与矩阵不一致。");
  requireValue(run.next_batch && Array.isArray(run.next_batch.record_ids) && run.next_batch.record_ids.every(id => boundedString(id, 300)), "下一 batch ID 无效。");
}

export function validateTrainingFailuresReport(report) {
  requireValue(report?.schema === "failure-replay-v1" && report.status === "passed" && Array.isArray(report.cases) && report.cases.length === 3, "请选择实际 failure-replay-v1 成功报告或精简快照。");
  requireValue(boundedString(report.created_at, 80) && Number.isFinite(Date.parse(report.created_at)), "运行日期无效。");
  const runtime = report.runtime, provenance = report.provenance;
  requireValue(runtime && ["python", "torch", "numpy", "platform", "device", "dtype"].every(key => boundedString(runtime[key], 500)) && integer(runtime.threads, 256) && runtime.threads > 0, "运行环境字段无效。");
  requireValue(provenance && boundedString(provenance.report_path, 2000) && hash(provenance.checkpoint_sha256) && hash(provenance.pre_step_digest) && hash(provenance.split_sha256) && provenance.core_api === "learning-core-v1", "报告来源／快照哈希无效。");
  if (provenance.report_sha256 !== undefined) requireValue(hash(provenance.report_sha256), "完整报告 SHA-256 无效。");
  requireValue(report.config && integer(report.config.warmup_updates, 10000) && provenance.saved_after_updates === report.config.warmup_updates && provenance.pending_step === report.config.warmup_updates + 1, "保存后更新数和待执行步骤不一致。");
  requireValue(Array.isArray(report.checks) && report.checks.length > 0 && report.checks.length <= 500 && report.checks.every(row => boundedString(row.name, 300) && row.passed === true), "报告包含缺失或失败的 checks。");
  requireValue(Array.isArray(report.history) && report.history.length <= 1000, "预热记录无效。");
  report.history.forEach(row => { requireValue(integer(row.step) && row.step > 0 && nonnegative(row.lr), "预热步骤无效。"); validateScalar(row.loss); });
  const seen = new Set();
  for (const item of report.cases) {
    requireValue(cases.includes(item.id) && !seen.has(item.id), "故障案例缺失或重复。"); seen.add(item.id);
    const probe = item.anomaly_probe;
    requireValue(probe && typeof probe.raised === "boolean" && (probe.error === null || boundedString(probe.error, 10000)) && boundedString(probe.scope, 2000), "anomaly probe 无效。");
    requireValue(probe.optimizer_update_executed === false && probe.next_forward_executed === false, "本报告 probe 的范围必须明确为更新前前向／反向。");
    for (const selection of ["target", "control"]) for (const variant of variants) {
      const run = item.runs?.[`${selection}-${variant}`];
      requireValue(run && run.case === item.id && run.variant === variant && run.selection === selection && run.loaded_pre_step_digest === provenance.pre_step_digest && run.checkpoint_sha256 === provenance.checkpoint_sha256, "回放分支或加载快照不一致。");
      requireValue(run.triggered === (selection === "target") && run.fault_enabled === (selection === "target" && variant === "faulty") && hash(run.final_state_digest) && typeof run.all_finite === "boolean", "故障触发范围／最终状态无效。");
      validateBatch(run); validateScalar(run.loss); validateScalar(run.next_loss); validateStats(run.gradient);
      requireValue(nonnegative(run.lr) && finite(run.shared_logit_shift) && Array.isArray(run.events) && run.events.length > 0 && run.events.length <= 4000, "LR／事件范围无效。");
      let previous = -1; const eventKeys = new Set();
      for (const event of run.events) {
        requireValue(integer(event.index, 100000) && event.index > previous && boundedString(event.key, 1000) && !eventKeys.has(event.key) && boundedString(event.name, 500) && phases.includes(event.phase) && ["input", "activation", "loss-op", "gradient", "parameter-gradient", "parameter", "parameter-update"].includes(event.kind), "张量事件顺序或名称无效。");
        previous = event.index; eventKeys.add(event.key); validateStats(event.stats);
      }
      requireValue(run.all_finite === run.events.every(event => event.stats.finite), "all_finite 与事件记录不一致。");
      for (const [field, predicate] of [["first_forward_nonfinite", event => event.phase === "step" && ["input", "activation", "loss-op"].includes(event.kind)], ["first_backward_nonfinite", event => ["gradient", "parameter-gradient"].includes(event.kind)]]) {
        const first = run.events.find(event => predicate(event) && !event.stats.finite) || null;
        requireValue((run[field] === null && first === null) || (run[field]?.key === first?.key && run[field]?.index === first?.index && run[field]?.name === first?.name), "首个非有限张量与实际事件顺序不一致。");
      }
      const update = run.update;
      requireValue(update && typeof update.applied === "boolean" && ["l2", "relative_l2", "threshold"].every(key => nonnegative(update[key])) && update.above_threshold === (update.relative_l2 > update.threshold) && boundedString(update.threshold_kind, 300), "更新统计／启发式阈值不一致。");
      const lossChecks = run.loss_checks;
      requireValue(lossChecks && lossChecks.recomputed_same_status === true, "缺少实际 loss 重算检查。");
      validateScalar(lossChecks.recomputed_from_saved_logits); validateScalar(lossChecks.correct_labels_stable_loss); validateScalar(lossChecks.unshifted_core_loss);
      requireValue(lossChecks.recomputed_from_saved_logits.status === run.loss.status && (run.loss.status !== "finite" || nonnegative(lossChecks.finite_recomputed_abs_error)) && nonnegative(lossChecks.stable_shift_abs_error_vs_core) && (lossChecks.wrong_label_loss_minus_correct === null || finite(lossChecks.wrong_label_loss_minus_correct)), "loss 重算字段无效。");
      requireValue(Array.isArray(run.parameters) && run.parameters.length <= 300, "参数行数无效。");
      for (const row of run.parameters) requireValue(boundedString(row.name, 300) && row.after && typeof row.after.finite === "boolean" && row.delta && typeof row.delta.finite === "boolean" && (row.delta.finite ? nonnegative(row.delta.l2) : row.delta.l2 === null) && (row.relative_update === null || nonnegative(row.relative_update)) && (row.gradient === null || (typeof row.gradient?.finite === "boolean" && (row.gradient.finite ? nonnegative(row.gradient.l2) : row.gradient.l2 === null))), "参数／梯度摘要无效。");
    }
    for (const name of ["faulty_vs_baseline", "repaired_vs_baseline", "control_faulty_vs_baseline", "control_repaired_vs_baseline"]) {
      const comparison = item.comparisons?.[name];
      requireValue(comparison && Array.isArray(comparison.event_differences) && comparison.event_differences.length <= 4000 && typeof comparison.all_compared_tensors_equal === "boolean" && typeof comparison.final_state_equal === "boolean", "回放差值／证明无效。");
      for (const key of ["gradients", "parameters_after", "next_logits"]) validateDifference(comparison[key]);
      comparison.event_differences.forEach(validateDifference);
      for (const finding of [comparison.first_divergence, comparison.first_next_forward_divergence]) if (finding !== null) {
        requireValue(boundedString(finding.key, 1000) && boundedString(finding.name, 500) && integer(finding.index, 100000), "首个分歧记录无效。"); validateDifference(finding);
      }
    }
    for (const selection of ["target", "control"]) {
      const baseline = item.runs[`${selection}-baseline`], repaired = item.runs[`${selection}-repaired`];
      const proof = item.comparisons[selection === "target" ? "repaired_vs_baseline" : "control_repaired_vs_baseline"];
      requireValue(baseline.final_state_digest === repaired.final_state_digest && proof.final_state_equal && proof.all_compared_tensors_equal && proof.identical_event_keys && equalJson(baseline.loss, repaired.loss) && equalJson(baseline.actual_labels, repaired.actual_labels), "修复未恢复基线张量／最终状态。");
    }
    requireValue(item.runs["control-baseline"].final_state_digest === item.runs["control-faulty"].final_state_digest && item.comparisons.control_faulty_vs_baseline.final_state_equal && item.comparisons.control_faulty_vs_baseline.all_compared_tensors_equal, "正常 batch 对照未通过。");
  }
  return report;
}

export function formatFailureValue(value) {
  if (value === null || value === undefined) return "非有限／未定义";
  if (typeof value === "object") return value.status === "finite" ? formatFailureValue(value.value) : ({ nan: "NaN", positive_infinity: "+∞", negative_infinity: "−∞" }[value.status] || "未定义");
  return value === 0 ? "0" : Math.abs(value) < 0.0001 || Math.abs(value) >= 10000 ? value.toExponential(3) : Number(value.toFixed(6)).toString();
}
const finiteText = stats => stats.finite ? "全部有限" : `NaN ${stats.nan_count} / +∞ ${stats.positive_infinity_count} / −∞ ${stats.negative_infinity_count}`;
const findingText = finding => finding ? `原序号 ${finding.index + 1} · ${finding.name}` : "未观察到";

export function failureReplayView(report, { caseId = "overflow", selection = "target", variant = "faulty", phase = "all" } = {}) {
  requireValue(cases.includes(caseId) && ["target", "control"].includes(selection) && variants.includes(variant) && ["all", ...phases].includes(phase), "回放选择无效。");
  const item = report.cases.find(row => row.id === caseId), run = item.runs[`${selection}-${variant}`];
  const comparison = variant === "baseline" ? null : item.comparisons[selection === "control" ? `control_${variant}_vs_baseline` : `${variant}_vs_baseline`];
  const differences = new Map((comparison?.event_differences || []).map(row => [row.key, row]));
  const events = run.events.filter(event => phase === "all" || event.phase === phase);
  const first = run.first_forward_nonfinite || run.first_backward_nonfinite || comparison?.first_divergence || null;
  return { item, run, comparison, differences, events, first };
}

export function failureEventRows(view, current = 0) {
  return view.events.map((event, index) => {
    const diff = view.differences.get(event.key);
    const difference = view.run.variant === "baseline" ? "参考" : diff ? diff.equal ? "相同" : formatFailureValue(diff.max_abs) : view.run.variant === "repaired" || view.run.selection === "control" ? "已核验相同" : "该路径独有／详见完整 trace";
    return `<tr${index === current ? ' class="tf-current" aria-current="true"' : ""}><td>${event.index + 1}</td><td>${escapeText(phaseLabel[event.phase])} / ${escapeText(event.name)}</td><td${event.stats.finite ? "" : ' class="tf-invalid"'}>${event.stats.finite_count} / ${event.stats.numel}<br>${escapeText(finiteText(event.stats))}</td><td>${formatFailureValue(event.stats.rms)}</td><td>${formatFailureValue(event.stats.max_abs)}</td><td>${escapeText(difference)}</td></tr>`;
  }).join("");
}

export function failureLossChart(item, selection) {
  const rows = variants.map(variant => ({ variant, run: item.runs[`${selection}-${variant}`] }));
  const maximum = Math.max(1, ...rows.map(row => row.run.next_loss.value || 0));
  return rows.map((row, index) => `<text x="18" y="${32 + index * 37}">${variantLabel[row.variant]}</text><rect class="tf-bar-${row.variant}" x="90" y="${18 + index * 37}" width="${Math.max(0, (row.run.next_loss.value || 0) / maximum * 380)}" height="20"/><text x="485" y="${32 + index * 37}">${escapeText(formatFailureValue(row.run.next_loss))} nats/token</text>`).join("");
}

export function initTrainingFailures() {
  const file = $("[data-tf-file]"); if (!file) return;
  const node = name => $(`[data-tf-${name}]`);
  const set = (name, value) => { node(name).textContent = value; };
  let report = null, request = 0, current = 0;
  const clear = () => {
    report = null; node("output").hidden = true;
    for (const name of ["provenance", "loss", "gradient", "update", "next-loss", "diagnosis", "case-description", "verification", "step", "event-name", "event-detail", "chart-note", "batch-summary", "evidence", "proof"]) set(name, "");
    for (const name of ["event-rows", "layer-rows", "parameter-rows", "label-rows", "chart"]) node(name).innerHTML = "";
    for (const name of ["prev", "next", "first", "reset", "download"]) node(name).disabled = true;
  };
  const render = () => {
    if (!report) return;
    const view = failureReplayView(report, { caseId: node("case").value, selection: node("selection").value, variant: node("variant").value, phase: node("phase").value });
    const { item, run, comparison, events } = view;
    current = Math.max(0, Math.min(current, events.length - 1));
    node("output").hidden = false;
    const p = report.provenance, r = report.runtime;
    set("provenance", `${report.created_at} · ${r.platform} · PyTorch ${r.torch} / NumPy ${r.numpy} · ${r.device}/${r.dtype} · ${r.threads} 线程 · ${p.parameter_count} 参数。保存于 ${p.saved_after_updates} 次更新后；待执行步骤 ${p.pending_step}。来源：${p.report_path}${p.report_sha256 ? `；报告 SHA-256 ${p.report_sha256}` : ""}。`);
    set("loss", formatFailureValue(run.loss)); set("gradient", formatFailureValue(run.gradient.l2));
    set("update", `${formatFailureValue(run.update.relative_l2)}${run.update.above_threshold ? " · 超阈值" : ""}`);
    set("next-loss", formatFailureValue(run.next_loss));
    set("diagnosis", `${run.first_forward_nonfinite ? `首个前向非有限：${findingText(run.first_forward_nonfinite)}` : "本步前向未观察到非有限"}。${comparison?.first_divergence ? `与基线首个张量分歧：${findingText(comparison.first_divergence)}。` : ""}${comparison?.first_next_forward_divergence ? `下一前向首个分歧：${findingText(comparison.first_next_forward_divergence)}。` : ""}`);
    const description = item.id === "overflow" ? `${run.selection === "target" ? `三种指定 batch 回放共同平移 +${run.shared_logit_shift} logits；故障分支显式 exp，基线／修复使用稳定 log_softmax。` : "正常 batch 对照不触发指定故障；本次共同平移为 0，三分支使用相同稳定目标。"}当前更新${run.update.applied ? "已执行" : "因非有限 loss／梯度被跳过"}。` : item.id === "high-lr" ? `实际 LR=${run.lr}，相对更新阈值 ${run.update.threshold} 是声明的诊断启发式。当前保存张量${run.all_finite ? "全部有限" : "出现非有限"}。anomaly probe 仅检查更新前反向；未执行高 LR optimizer 更新或下一前向。` : `A→B→C→D→A 的符号 ID 轮换；当前改动 ${run.labels_changed} 个标签，监督数 ${run.supervised_count} 保持相同。EOS／−100／输入历史保持原记录；改动类型为符号轮换。`;
    set("case-description", description);
    const repair = item.comparisons.repaired_vs_baseline, control = item.comparisons.control_faulty_vs_baseline;
    set("verification", `${report.checks.length} 项保存检查通过。修复与基线：全部张量${repair.all_compared_tensors_equal ? "相同" : "存在差异"}，最终状态${repair.final_state_equal ? "相同" : "存在差异"}；正常 batch 对照${control.final_state_equal && control.all_compared_tensors_equal ? "通过" : "未通过"}。当前分支加载相同更新前 digest：${run.loaded_pre_step_digest}。`);
    const event = events[current];
    set("step", events.length ? `当前 ${current + 1} / ${events.length} 条可见记录；完整 trace ${run.event_count_saved || run.events.length} 条` : "这个精简快照没有该阶段的记录，完整 trace 见工件路径。");
    set("event-name", event ? `${phaseLabel[event.phase]} · ${event.kind} · ${event.name}` : "无可见事件");
    set("event-detail", event ? `原序号 ${event.index + 1}；shape [${event.stats.shape.join(", ")}] / ${event.stats.dtype}；RMS ${formatFailureValue(event.stats.rms)}；max |x| ${formatFailureValue(event.stats.max_abs)}；${finiteText(event.stats)}。` : "切换阶段或载入完整 report.json。");
    node("event-rows").innerHTML = failureEventRows(view, current);
    node("prev").disabled = current === 0 || !events.length; node("next").disabled = current >= events.length - 1;
    node("first").disabled = !view.first; node("reset").disabled = false; node("download").disabled = false;
    node("chart").innerHTML = failureLossChart(item, run.selection);
    node("chart").setAttribute("aria-label", "保存运行中基线、故障和修复的下一 batch loss 对照，单位 nats／监督 token");
    set("chart-note", "柱长仅表示本次保存运行的下一 batch loss；本步 loss 与非有限状态显示在上方。三个分支的输入与更新前快照相同。");
    node("layer-rows").innerHTML = run.events.filter(event => event.phase === "step" && event.kind === "activation").map(event => `<tr><td>${escapeText(event.name)}</td><td>[${event.stats.shape.join(", ")}]</td><td>${formatFailureValue(event.stats.rms)}</td><td>${formatFailureValue(event.stats.max_abs)}</td><td>${escapeText(finiteText(event.stats))}</td></tr>`).join("");
    node("parameter-rows").innerHTML = run.parameters.length ? run.parameters.map(row => `<tr><td>${escapeText(row.name)}</td><td>${formatFailureValue(row.gradient?.l2)}</td><td>${formatFailureValue(row.delta.l2)}</td><td>${formatFailureValue(row.relative_update)}</td><td>${row.after.finite ? "有限" : "非有限"}</td></tr>`).join("") : '<tr><td colspan="5">精简正常对照保留全参数测量；逐参数记录见完整 report.json。</td></tr>';
    const token = id => id === -100 ? "−100（忽略）" : `${id} / ${vocabulary[id]}`;
    node("label-rows").innerHTML = run.batch.input_ids.flatMap((tokens, i) => tokens.map((id, j) => `<tr><td>${escapeText(run.batch.record_ids[i])} / ${j}</td><td>${token(id)}</td><td>${run.batch.attention_mask[i][j] ? "True" : "False"}</td><td>${token(run.batch.labels[i][j])}</td><td${run.actual_labels[i][j] === run.batch.labels[i][j] ? "" : ' class="tf-divergent"'}>${token(run.actual_labels[i][j])}</td></tr>`)).join("");
    set("batch-summary", `${run.batch.record_ids.length} 条记录；${run.batch.input_tokens} 个输入 token；${run.supervised_count} 个监督 token；${run.labels_changed} 处标签差异。`);
    const grad = comparison?.gradients;
    set("evidence", `实际 loss 重算：${formatFailureValue(run.loss_checks.recomputed_from_saved_logits)}；有限重算误差 ${formatFailureValue(run.loss_checks.finite_recomputed_abs_error)}。相同 logits 的正确标签稳定 loss：${formatFailureValue(run.loss_checks.correct_labels_stable_loss)}；实际减正确 loss：${formatFailureValue(run.loss_checks.wrong_label_loss_minus_correct)}。${grad ? `梯度差 L2 ${formatFailureValue(grad.l2)}；相对 L2 ${formatFailureValue(grad.relative_l2)}；最大元素差 ${formatFailureValue(grad.max_abs)}。` : "当前为参考基线。"}首个反向非有限：${findingText(run.first_backward_nonfinite)}。`);
    set("proof", JSON.stringify({ provenance: p, loaded_pre_step_digest: run.loaded_pre_step_digest, final_state_digest: run.final_state_digest, repaired_vs_baseline: { tensors_equal: repair.all_compared_tensors_equal, state_equal: repair.final_state_equal }, control: { tensors_equal: control.all_compared_tensors_equal, state_equal: control.final_state_equal }, anomaly_probe: { raised: item.anomaly_probe.raised, error: item.anomaly_probe.error, scope: item.anomaly_probe.scope, optimizer_update_executed: false, next_forward_executed: false } }, null, 2));
  };
  const loadDefault = () => { request++; clear(); current = 0; file.value = ""; report = validateTrainingFailuresReport(trainingFailuresSnapshot); set("status", "已载入默认快照，数字来自保存的实际运行报告。"); render(); };
  for (const name of ["case", "variant", "selection", "phase"]) node(name).addEventListener("change", () => { current = 0; render(); });
  node("prev").addEventListener("click", () => { current--; render(); }); node("next").addEventListener("click", () => { current++; render(); });
  node("reset").addEventListener("click", () => { current = 0; render(); });
  node("first").addEventListener("click", () => { if (!report) return; const view = failureReplayView(report, { caseId: node("case").value, selection: node("selection").value, variant: node("variant").value }); node("phase").value = "all"; current = Math.max(0, view.events.findIndex(event => event.key === view.first?.key)); render(); });
  node("default").addEventListener("click", loadDefault);
  file.addEventListener("change", async () => {
    const ticket = ++request; clear(); set("status", "正在读取本地文件…");
    try {
      const selected = file.files?.[0]; if (!selected) { set("status", "未选择文件；可恢复默认快照。"); return; }
      if (selected.size > 12 * 1024 * 1024) throw new RangeError("文件上限 12 MiB，请选择 report.json 或 snapshot.json。");
      const parsed = validateTrainingFailuresReport(JSON.parse(await selected.text())); if (ticket !== request) return;
      report = parsed; current = 0; set("status", `已读取 ${selected.name || "本地报告"}；${report.checks.length} 项保存检查。文件仅在浏览器处理。`); render();
    } catch (error) { if (ticket === request) { clear(); set("status", `无法读取：${error.message}`); } }
  });
  node("download").addEventListener("click", () => {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "training-failures-saved-evidence.json"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  loadDefault();
}
