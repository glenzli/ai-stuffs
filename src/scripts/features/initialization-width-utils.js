// Local to the two training-dynamics pages. No data, network or DOM side effects.
import { escapeText } from "./objective-math.js";

export const dynamicsNumber = value => value === null || value === undefined ? "未定义" :
  Math.abs(value) > 0 && (Math.abs(value) < .001 || Math.abs(value) >= 10000) ? value.toExponential(3) :
    value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
export const dynamicsPercent = value => value === null || value === undefined ? "未定义" : `${(100 * value).toFixed(1)}%`;
export const dynamicsText = value => escapeText(String(value));
export const dynamicsNear = (a, b, relative = 1e-8, absolute = 1e-11) =>
  Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= absolute + relative * Math.max(Math.abs(a), Math.abs(b));
export function dynamicsRequire(condition, message) { if (!condition) throw new RangeError(message); }
export function dynamicsString(value, max = 500) { return typeof value === "string" && value.length > 0 && value.length <= max; }
export function dynamicsFinite(value, max = 1e12) { return Number.isFinite(value) && value >= 0 && value <= max; }
export function dynamicsInt(value, min = 1, max = 1000000) { return Number.isInteger(value) && value >= min && value <= max; }
export function dynamicsArray(value, min, max) { return Array.isArray(value) && value.length >= min && value.length <= max; }

export function validateDynamicsEnvelope(data, schema) {
  dynamicsRequire(data && [schema, "training-dynamics-v1"].includes(data.schema) && data.status === "complete", "请选择完整的 training-dynamics 报告或本页已保存快照。");
  const runtime = data.runtime, provenance = data.provenance;
  dynamicsRequire(runtime && provenance, "缺少运行环境或报告来源。");
  for (const key of ["python", "torch", "numpy", "device", "platform"]) dynamicsRequire(dynamicsString(runtime[key]), `运行环境 ${key} 无效。`);
  dynamicsRequire(runtime.device === "cpu" && runtime.torch.split("+")[0] === "2.9.0" && runtime.numpy === "2.3.4", "报告不是此实验的固定 CPU 参考环境。");
  dynamicsRequire(dynamicsInt(runtime.threads, 1, 64), "CPU 线程数无效。");
  dynamicsRequire(dynamicsString(provenance.run_id, 300) && dynamicsString(provenance.created_at, 100)
    && Number.isFinite(Date.parse(provenance.created_at)) && dynamicsString(provenance.report_path, 2000), "报告日期、ID 或本地路径无效。");
  dynamicsRequire(provenance.source_sha256 && Object.entries(provenance.source_sha256).length <= 20, "源码来源无效。");
  for (const name of ["training_dynamics.py", "learning_core.py", "run-dynamics-lab.mjs"]) {
    dynamicsRequire(/^[a-f0-9]{64}$/.test(provenance.source_sha256[name] || ""), `缺少 ${name} 的 SHA-256。`);
  }
  if (provenance.report_sha256 !== undefined) dynamicsRequire(/^[a-f0-9]{64}$/.test(provenance.report_sha256), "报告 SHA-256 无效。");
  if (provenance.kind === "saved-measurement-reanalysis") {
    const original = provenance.original_measurement, analysis = provenance.analysis_source_sha256;
    dynamicsRequire(provenance.no_full_retraining === true && original && analysis,
      "重分析须保留原测量来源、当前分析来源及无完整重训声明。");
    dynamicsRequire(dynamicsString(original.run_id, 300) && dynamicsString(original.report_path, 2000)
      && dynamicsString(original.created_at, 100) && Number.isFinite(Date.parse(original.created_at))
      && /^[a-f0-9]{64}$/.test(original.report_sha256 || ""), "原测量报告来源或 SHA-256 无效。");
    for (const name of ["training_dynamics.py", "learning_core.py", "run-dynamics-lab.mjs"]) {
      dynamicsRequire(/^[a-f0-9]{64}$/.test(original.source_sha256?.[name] || "")
        && original.source_sha256[name] === provenance.source_sha256[name]
        && /^[a-f0-9]{64}$/.test(analysis[name] || ""), "原训练与当前分析的源码来源不能混用。");
    }
  }
  return data;
}

export function dynamicsProvenance(data) {
  const r = data.runtime, p = data.provenance;
  const environment = `${r.platform} · Python ${r.python} / PyTorch ${r.torch} / NumPy ${r.numpy} · CPU ${r.threads} 线程。`;
  if (p.kind === "saved-measurement-reanalysis") {
    const original = p.original_measurement;
    return `重分析时间 ${p.created_at} · ${p.run_id}。未重新完整训练：沿用 ${original.created_at} 的原测量 ${original.run_id}。${environment}原报告：${original.report_path}，SHA-256=${original.report_sha256}。原训练代码 SHA-256=${original.source_sha256["training_dynamics.py"]}；当前分析代码 SHA-256=${p.analysis_source_sha256["training_dynamics.py"]}。只复核保存梯度、checkpoint 和恢复 RNG；原学习曲线、坐标与评估结果保持原测量值。重分析来源：${p.source_report || p.report_path}。${p.report_sha256 ? `重分析报告 SHA-256=${p.report_sha256}。` : "本地文件内容未作外部认证。"}`;
  }
  return `${p.created_at} · ${p.run_id} · ${environment}来源：${p.source_report || p.report_path}。源码 SHA-256：${p.source_sha256["training_dynamics.py"]}。${p.report_sha256 ? `报告 SHA-256：${p.report_sha256}。` : "完整 JSON 由本地文件读取，未对内容作外部认证。"}`;
}

export function dynamicsLineChart(series, { xLabel = "x", yLabel = "y", logarithmic = false } = {}) {
  const colors = ["teal", "blue", "coral", "yellow", "teal", "blue", "coral", "yellow"];
  const points = series.flatMap(s => s.points).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (!points.length || (logarithmic && points.some(p => p.y <= 0))) return `<text x="30" y="60">当前指标没有可绘制的有限数值。</text>`;
  const ys = points.map(p => logarithmic ? Math.log10(p.y) : p.y);
  const minX = Math.min(...points.map(p => p.x)), maxX = Math.max(...points.map(p => p.x));
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  if (!logarithmic) minY = Math.min(0, minY);
  if (maxY === minY) maxY = minY + Math.max(1e-6, Math.abs(minY) * .1);
  const sx = x => 65 + 490 * (x - minX) / (maxX - minX || 1);
  const sy = y => 250 - 195 * ((logarithmic ? Math.log10(y) : y) - minY) / (maxY - minY);
  const tickY = value => dynamicsNumber(logarithmic ? 10 ** value : value);
  return `<line x1="65" y1="250" x2="560" y2="250" class="decision-axis"/><line x1="65" y1="55" x2="65" y2="250" class="decision-axis"/>
    <text x="10" y="65">${dynamicsText(yLabel)}${logarithmic ? " / log₁₀" : ""}</text>
    <text x="10" y="86">${dynamicsText(tickY(maxY))}</text><text x="10" y="249">${dynamicsText(tickY(minY))}</text>
    <text x="65" y="274">${dynamicsNumber(minX)}</text><text x="550" y="274" text-anchor="end">${dynamicsNumber(maxX)}</text>
    <text x="555" y="298" text-anchor="end">${dynamicsText(xLabel)}</text>
    ${series.map((s, i) => `<g class="dynamics-series dynamics-${colors[i % colors.length]}"><polyline points="${s.points.map(p => `${sx(p.x).toFixed(2)},${sy(p.y).toFixed(2)}`).join(" ")}" fill="none" stroke-width="2"/>
    ${s.points.map(p => `<circle cx="${sx(p.x).toFixed(2)}" cy="${sy(p.y).toFixed(2)}" r="3"><title>${dynamicsText(s.name)}：x=${dynamicsNumber(p.x)}，y=${dynamicsNumber(p.y)}</title></circle>`).join("")}</g>`).join("")}
    ${series.map((s, i) => `<text x="${65 + (i % 3) * 170}" y="${18 + 15 * Math.floor(i / 3)}" class="dynamics-legend dynamics-${colors[i % colors.length]}">${dynamicsText(s.name)}</text>`).join("")}`;
}

export function dynamicsOptions(select, values, { labels = new Map(), preferred } = {}) {
  const prior = select.value;
  select.innerHTML = values.map(value => `<option value="${dynamicsText(value)}">${dynamicsText(labels.get(value) || value)}</option>`).join("");
  select.value = values.map(String).includes(prior) ? prior : String(preferred ?? values[0]);
}

export function bindDynamicsImport({ file, reset, status, clear, accept, defaultData, validate }) {
  let request = 0;
  const load = (data, label) => { const valid = validate(data); accept(valid); status.textContent = label; };
  file.addEventListener("change", async () => {
    const version = ++request, selected = file.files?.[0]; clear();
    if (!selected) { status.textContent = "请选择本机完整 report.json，或恢复默认快照。"; return; }
    status.textContent = "正在本地读取并核对报告…";
    try {
      if (!Number.isFinite(selected.size) || selected.size < 1 || selected.size > 32 * 1024 * 1024) throw new RangeError("文件须为 1 B…32 MiB 的 report.json。");
      const text = await selected.text();
      if (version !== request) return;
      const data = validate(JSON.parse(text));
      if (version !== request) return;
      load(data, `已读取 ${selected.name || "report.json"}：使用文件内保存的真实结果；文件没有上传。`);
    } catch (error) {
      if (version === request) { clear(); status.textContent = `无法读取：${error.message}`; }
    }
  });
  reset.addEventListener("click", () => {
    ++request; clear(); file.value = "";
    load(defaultData, "默认快照：从已保存完整报告提取；图中每个点都有实际运行值。");
  });
  load(defaultData, "默认快照：从已保存完整报告提取；图中每个点都有实际运行值。");
}
