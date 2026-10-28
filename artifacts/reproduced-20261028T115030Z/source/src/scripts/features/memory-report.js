import { $ } from "../core.js";
import { escapeText } from "./objective-math.js";

export function validateMemoryReport(report) {
  const finite = value => Number.isFinite(value) && value >= 0;
  if (report?.schema !== "memory-lab-v1" || report.status !== "passed" || !report.runtime || !Array.isArray(report.measurements) || !report.measurements.length || report.measurements.length > 20 || !finite(report.max_error) || !Number.isInteger(report.comparisons) || report.comparisons < 1) throw new RangeError("请选择 lab:memory 生成的完整成功报告。");
  for (const key of ["python", "torch", "device", "dtype", "platform"]) if (typeof report.runtime[key] !== "string" || report.runtime[key].length > 500) throw new RangeError("报告运行环境字段无效。");
  for (const row of report.measurements) {
    if (typeof row.name !== "string" || row.name.length > 200 || ![row.wall_ms?.median, row.wall_ms?.min, row.wall_ms?.max, row.saved_tensor_storage?.peak_referenced_bytes].every(finite) || row.wall_ms.min > row.wall_ms.median || row.wall_ms.median > row.wall_ms.max || !Array.isArray(row.profiler_top_self_cpu) || row.profiler_top_self_cpu.length > 30) throw new RangeError("报告的测量字段无效。");
    for (const event of row.profiler_top_self_cpu) if (typeof event.operation !== "string" || event.operation.length > 500 || !finite(event.self_cpu_us) || !Number.isInteger(event.calls) || !Number.isFinite(event.self_cpu_memory_bytes)) throw new RangeError("报告的 profiler 字段无效。");
    if (row.reference_error !== undefined && !finite(row.reference_error)) throw new RangeError("报告误差字段无效。");
    if (row.logical !== undefined && (!row.logical || typeof row.logical !== "object" || Array.isArray(row.logical) || JSON.stringify(row.logical).length > 10000)) throw new RangeError("报告逻辑账本无效。");
  }
  return report;
}
export function memoryReportRows(report) {
  validateMemoryReport(report);
  return report.measurements.map(row => `<tr><th scope="row">${escapeText(row.name)}</th><td>${row.wall_ms.median.toFixed(3)}</td><td>${row.wall_ms.min.toFixed(3)}…${row.wall_ms.max.toFixed(3)}</td><td>${row.saved_tensor_storage.peak_referenced_bytes.toLocaleString("en-US")}</td></tr>`).join("");
}
export function initMemoryReport() {
  const file = $("[data-mr-file]"); if (!file) return;
  const status = $("[data-mr-status]"), output = $("[data-mr-output]"), select = $("[data-mr-measurement]"); let report = null, request = 0;
  const renderProfile = () => {
    if (!report) return;
    const row = report.measurements[Number(select.value)];
    $("[data-mr-profile]").innerHTML = row.profiler_top_self_cpu.map(event => `<tr><th scope="row">${escapeText(event.operation)}</th><td>${event.self_cpu_us.toFixed(2)}</td><td>${event.calls}</td><td>${event.self_cpu_memory_bytes}</td></tr>`).join("");
    $("[data-mr-logical]").textContent = JSON.stringify(row.logical || {}, null, 2);
    $("[data-mr-error]").textContent = row.reference_error === undefined ? "该报告未记录逐方案误差。" : `本方案与对应参考的最大误差：${row.reference_error.toExponential(2)}。参考方案自身标为 0，不表示机器算术没有误差。`;
  };
  file.addEventListener("change", async () => {
    const version = ++request, selected = file.files?.[0]; report = null; output.hidden = true;
    if (!selected) { status.textContent = "选择本机实验的 report.json，查看该次实际结果。"; return; }
    try {
      if (selected.size > 5 * 1024 * 1024) throw new RangeError("报告超过 5 MiB，请选择 report.json，而非 trace 文件。");
      const parsed = validateMemoryReport(JSON.parse(await selected.text()));
      if (version !== request) return; report = parsed;
      const r = report.runtime;
      status.textContent = `${r.platform} · Python ${r.python} · PyTorch ${r.torch} · ${r.device}/${r.dtype}。${report.comparisons} 项比较通过，最大误差 ${report.max_error.toExponential(2)}。这是文件记录的实验结果。`;
      $("[data-mr-rows]").innerHTML = memoryReportRows(report);
      select.innerHTML = report.measurements.map((row, index) => `<option value="${index}">${escapeText(row.name)}</option>`).join(""); select.value = "0";
      renderProfile(); output.hidden = false;
    } catch (error) { if (version === request) status.textContent = `无法读取：${error.message}`; }
  });
  select.addEventListener("change", renderProfile);
}
