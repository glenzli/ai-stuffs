import { $ } from "../core.js";
import { initializationWidthData } from "./initialization-width-data.js";
import { dynamicsNumber as fmt, dynamicsPercent as pct, dynamicsText as esc, dynamicsNear as near,
  dynamicsRequire as require, dynamicsString as string, dynamicsFinite as finite,
  dynamicsInt as integer, dynamicsArray as array, validateDynamicsEnvelope,
  dynamicsProvenance, dynamicsLineChart, dynamicsOptions, bindDynamicsImport } from "./initialization-width-utils.js";

const average = values => values.reduce((sum, x) => sum + x, 0) / values.length;
const unique = values => [...new Set(values)];
const coordinateLayers = ["input_pre", "input_relu", "hidden_pre", "hidden_relu", "readout_logits"];
const coordinateMetrics = ["mean_abs", "rms", "displacement_rms"];

function validateLedger(run, baseWidth) {
  require(array(run.ledger, 6, 6), "MLP 参数账本应包含六个参数。");
  const names = new Set();
  for (const p of run.ledger) {
    require(p && string(p.name, 100) && !names.has(p.name) && ["input", "hidden", "readout"].includes(p.role), "参数名称、角色或重复项无效。"); names.add(p.name);
    require(array(p.shape, 1, 2) && p.shape.every(n => integer(n, 1, 8192)) && integer(p.optimizer_group, 0, 20), "参数形状或 optimizer group 无效。");
    for (const key of ["base_lr", "actual_lr", "weight_decay", "expected_initial_variance", "observed_initial_variance", "observed_initial_rms", "readout_input_factor", "bias_forward_factor"]) require(finite(p[key]), `参数账本 ${key} 无效。`);
    require(near(p.base_lr, run.base_lr) && p.weight_decay === 0, "基础 LR 或权重衰减与实验口径不一致。");
    const multiplier = run.width / baseWidth;
    const hiddenWeight = p.name === "hidden.weight";
    const expectedLR = run.parameterization === "mup" && hiddenWeight ? run.base_lr / multiplier : run.base_lr;
    const expectedFactor = run.parameterization === "mup" && p.role === "readout" ? 1 / multiplier : 1;
    const expectedVariance = p.name.endsWith("bias") ? 0 : p.role === "input" ? 2 / 8 : p.role === "hidden" ? 2 / run.width :
      run.parameterization === "mup" ? 1 / baseWidth : 1 / run.width;
    require(near(p.actual_lr, expectedLR) && near(p.readout_input_factor, expectedFactor) && near(p.expected_initial_variance, expectedVariance), "参数实际 LR/readout/初始化规则与声明不一致。");
    require(p.bias_forward_factor === 1, "MuReadout bias 不乘输入缩放因子。");
    if (run.parameterization === "mup") require(p.infinite_dimensions === (hiddenWeight ? 2 : p.name === "readout.bias" ? 0 : 1), "μP 有限／无限维角色不一致。");
  }
  for (const name of ["input.weight", "input.bias", "hidden.weight", "hidden.bias", "readout.weight", "readout.bias"]) require(names.has(name), `缺少 ${name}。`);
}

export function validateInitializationWidthReport(data) {
  validateDynamicsEnvelope(data, "initialization-width-v1");
  const width = data.width, depth = data.depth;
  require(width && depth && width.hidden_depth === 2 && width.base_width === 16 && width.delta_width === 32, "此报告须使用固定深度和约定 base/delta widths。");
  require(data.runtime.mup === "1.0.0", "此报告须使用已核对的 mup 1.0.0。");
  require(array(width.seeds, 2, 10) && width.seeds.every(n => integer(n, 0)) && unique(width.seeds).length === width.seeds.length, "训练 seed 无效。");
  require(array(width.sweep_widths, 2, 6) && width.sweep_widths.every(n => integer(n, 1, 2048)) && unique(width.sweep_widths).length === width.sweep_widths.length, "诊断宽度无效。");
  require(integer(width.held_width, 1, 2048) && !width.sweep_widths.includes(width.held_width) && width.sweep_widths.includes(width.base_width), "held-out width 不能参加扫描。");
  require(array(width.lr_grid, 2, 12) && width.lr_grid.every(lr => finite(lr, 1) && lr > 0) && unique(width.lr_grid).length === width.lr_grid.length, "学习率网格无效。");
  require(integer(width.steps, 4, 1000) && array(width.runs, 1, 1000) && array(width.selected_runs, 1, 200) && array(width.selection, 2, 2), "宽度运行记录缺失或超出范围。");
  const runIds = new Set(), keys = new Set();
  for (const run of width.runs) {
    require(run && string(run.run_id, 300) && !runIds.has(run.run_id) && ["sp", "mup"].includes(run.parameterization)
      && width.sweep_widths.includes(run.width) && width.seeds.includes(run.seed) && width.lr_grid.includes(run.base_lr), "扫描 run 身份或配置无效。");
    require(finite(run.final_validation?.loss) && finite(run.final_validation?.accuracy, 1), "验证 loss 或 accuracy 无效。");
    const key = `${run.parameterization}/${run.width}/${run.seed}/${run.base_lr}`;
    require(!keys.has(key), "扫描配置重复。"); keys.add(key); runIds.add(run.run_id);
  }
  require(width.runs.length === 2 * width.sweep_widths.length * width.seeds.length * width.lr_grid.length, "扫描网格缺少配置。");
  const selection = new Map();
  for (const row of width.selection) {
    require(["sp", "mup"].includes(row.parameterization) && !selection.has(row.parameterization)
      && row.selection_width === width.base_width && row.held_width_used === false && width.lr_grid.includes(row.selected_lr), "学习率选择只能使用宽度 16。");
    require(array(row.scores, width.lr_grid.length, width.lr_grid.length), "LR 选择账本不完整。");
    const scores = [];
    for (const lr of width.lr_grid) {
      const recorded = row.scores.filter(s => s.base_lr === lr);
      const runs = width.runs.filter(r => r.parameterization === row.parameterization && r.width === width.base_width && r.base_lr === lr);
      require(recorded.length === 1 && runs.length === width.seeds.length && finite(recorded[0].mean_validation_loss), "基础宽度的选择分数缺失。");
      const observed = average(runs.map(r => r.final_validation.loss));
      require(near(observed, recorded[0].mean_validation_loss), "选择分数不等于基础宽度的 seed 平均。");
      scores.push({ lr, loss: observed });
    }
    scores.sort((a, b) => a.loss - b.loss);
    require(row.selected_lr === scores[0].lr, "选定 LR 与已测验证最小值不一致。");
    selection.set(row.parameterization, row.selected_lr);
  }
  const selectedKeys = new Set();
  for (const run of width.selected_runs) {
    require(run && ["sp", "mup"].includes(run.parameterization) && string(run.run_id, 300)
      && [...width.sweep_widths, width.held_width].includes(run.width) && width.seeds.includes(run.seed)
      && run.base_lr === selection.get(run.parameterization), "迁移 run 未使用已冻结的基础 LR。");
    if (run.width === width.held_width) require(run.role === "held-out-transfer", "宽度 128 必须明确标为 held-out。");
    const key = `${run.parameterization}/${run.width}/${run.seed}`;
    require(!selectedKeys.has(key), "迁移 run 重复。"); selectedKeys.add(key);
    require(integer(run.updates, 4, 1000) && run.updates === width.steps && finite(run.final_validation?.loss)
      && finite(run.test?.loss) && finite(run.test?.accuracy, 1), "迁移训练／测试结果无效。");
    require(array(run.coordinates, 4, 4), "须包含初始、1次、4次和最终更新的坐标。");
    for (const point of run.coordinates) {
      require([0, 1, 4, width.steps].includes(point.after_updates) && array(point.rows, 5, 5), "坐标测量点无效。");
      require(unique(point.rows.map(r => r.layer)).length === 5, "坐标层重复。");
      for (const row of point.rows) require(coordinateLayers.includes(row.layer) && coordinateMetrics.every(key => finite(row[key])), "坐标量级无效。");
    }
    require(unique(run.coordinates.map(p => p.after_updates)).length === 4, "坐标更新次数重复。");
    validateLedger(run, width.base_width);
  }
  require(width.selected_runs.length === 2 * (width.sweep_widths.length + 1) * width.seeds.length, "迁移结果缺少宽度或 seed。");
  require(depth.core_api === "learning-core-v1" && depth.width === 32 && depth.heads === 4
    && array(depth.runs, 1, 200) && array(depth.depths, 1, 6) && array(depth.seeds, 2, 10), "深度实验未声明共享 core/固定宽度。");
  const depthKeys = new Set();
  for (const run of depth.runs) {
    require(run && integer(run.depth, 1, 32) && depth.depths.includes(run.depth) && depth.seeds.includes(run.seed)
      && ["pre", "post", "none"].includes(run.norm) && ["one", "inverse-sqrt-depth"].includes(run.residual_rule), "深度配置无效。");
    require(run.config?.width === depth.width && run.config?.heads === depth.heads && run.config?.dropout === 0
      && run.config?.norm === run.norm && run.config?.depth === run.depth, "深度试验改变了固定宽度或 core 配置。");
    require(near(run.residual_scale, run.residual_rule === "one" ? 1 : 1 / Math.sqrt(2 * run.depth)), "残差缩放无效。");
    const key = `${run.depth}/${run.norm}/${run.residual_rule}/${run.seed}`;
    require(!depthKeys.has(key), "深度配置重复。"); depthKeys.add(key);
    require(array(run.probes, 2, 2) && run.probes[0].after_updates === 0 && run.probes[1].after_updates === 4, "深度探针须分开初始化和更新后。");
    for (const probe of run.probes) {
      require(finite(probe.loss) && array(probe.rows, 1, 100) && array(probe.parameters, 1, 500), "逐层探针记录无效。");
      for (const row of probe.rows) require(string(row.layer, 200) && finite(row.activation?.rms) && finite(row.gradient?.rms), "逐层激活／梯度无效。");
      for (let i = 0; i < run.depth; i++) require(probe.rows.some(row => row.layer === `block${i}.output`), "缺少某个 block 的输出测量。");
      for (let i = 0; i < run.depth; i++) require(probe.parameters.some(row => row.name === `blocks.${i}.mlp.2.weight` && finite(row.gradient?.rms)), "缺少 FFN 输出权重梯度。");
    }
  }
  require(depth.runs.length === depth.depths.length * 3 * 2 * depth.seeds.length, "深度网格不完整。");
  return data;
}

export function widthCoordinateRows(data, { layer = "readout_logits", afterUpdates = 1, metric = "mean_abs", seed = "all" } = {}) {
  require(coordinateLayers.includes(layer) && coordinateMetrics.includes(metric), "坐标选择无效。");
  const widths = [...data.width.sweep_widths, data.width.held_width];
  return ["sp", "mup"].flatMap(parameterization => widths.map(width => {
    const runs = data.width.selected_runs.filter(r => r.width === width && r.parameterization === parameterization && (seed === "all" || r.seed === Number(seed)));
    const values = runs.map(run => run.coordinates.find(c => c.after_updates === Number(afterUpdates))?.rows.find(r => r.layer === layer)?.[metric]);
    require(values.length > 0 && values.every(Number.isFinite), "报告没有当前坐标。");
    return { parameterization, width, mean: average(values), min: Math.min(...values), max: Math.max(...values),
      seeds: runs.length, base_lr: runs[0].base_lr, held: width === data.width.held_width };
  }));
}

export function depthProbeRows(data, { depth = 4, norm = "pre", residualRule = "one", afterUpdates = 0, seed = "all" } = {}) {
  const runs = data.depth.runs.filter(run => run.depth === Number(depth) && run.norm === norm && run.residual_rule === residualRule && (seed === "all" || run.seed === Number(seed)));
  require(runs.length > 0, "没有当前深度配置。");
  const names = ["embedding", ...Array.from({ length: Number(depth) }, (_, i) => `block${i}.output`), "final_norm", "readout_logits"];
  const probes = runs.map(run => run.probes.find(p => p.after_updates === Number(afterUpdates)));
  require(probes.every(Boolean), "缺少当前更新时刻。");
  return names.map((layer, i) => {
    const rows = probes.map(probe => probe.rows.find(row => row.layer === layer));
    require(rows.every(Boolean), "缺少当前逐层测量。");
    const gradients = i >= 1 && i <= Number(depth) ? probes.map(probe => probe.parameters.find(row => row.name === `blocks.${i - 1}.mlp.2.weight`)?.gradient?.rms) : [];
    return { layer, activation_rms: average(rows.map(row => row.activation.rms)),
      activation_min: Math.min(...rows.map(row => row.activation.rms)), activation_max: Math.max(...rows.map(row => row.activation.rms)),
      hidden_gradient_rms: average(rows.map(row => row.gradient.rms)),
      ffn_weight_gradient_rms: gradients.length && gradients.every(Number.isFinite) ? average(gradients) : null };
  });
}

export function initInitializationWidth() {
  const file = $("[data-iw-file]"); if (!file) return;
  const node = name => $(`[data-iw-${name}]`), set = (name, value) => { node(name).textContent = value; };
  let report = null;
  const renderWidth = () => {
    if (!report) return;
    const rows = widthCoordinateRows(report, { layer: node("layer").value, afterUpdates: +node("step").value, metric: node("metric").value, seed: node("seed").value });
    node("coordinates").innerHTML = rows.map(row => `<tr><th scope="row">${row.parameterization === "mup" ? "μP" : "SP"} / ${row.width}${row.held ? " · 保留宽度" : ""}</th><td>${fmt(row.mean)}</td><td>${fmt(row.min)}…${fmt(row.max)}</td><td>${row.seeds}</td><td>${fmt(row.base_lr)}</td></tr>`).join("");
    node("chart").innerHTML = dynamicsLineChart(["sp", "mup"].map(p => ({ name: p === "mup" ? "μP" : "SP", points: rows.filter(r => r.parameterization === p).map(r => ({ x: r.width, y: r.mean })) })), { xLabel: "固定深度 MLP 宽度", yLabel: node("metric").value });
    const initial = +node("step").value === 0;
    set("coordinate-note", `${node("layer").value} · 更新次数 ${node("step").value} · ${node("seed").value === "all" ? "三个 seed 的均值，表中同时显示范围" : `seed ${node("seed").value}`}。输入为同一组 16 条验证记录。${initial ? "readout 非零初始化：μP 初始输出随宽度缩小是作者说明的有限宽度暂态。" : "这里是已冻结基础 LR 的实测更新坐标；坐标稳定不能单独证明最优 LR 或任务质量迁移。"}`);
  };
  const renderSweep = () => {
    if (!report) return;
    const p = node("sweep-param").value, choice = report.width.selection.find(r => r.parameterization === p);
    const rows = report.width.sweep_widths.flatMap(width => report.width.lr_grid.map(lr => {
      const selected = report.width.runs.filter(r => r.parameterization === p && r.width === width && r.base_lr === lr);
      return { width, lr, loss: average(selected.map(r => r.final_validation.loss)) };
    }));
    node("sweep-rows").innerHTML = rows.map(r => `<tr><td>${r.width}${r.width === report.width.base_width ? " · 选择" : " · 诊断"}</td><td>${fmt(r.lr)}</td><td>${fmt(r.loss)}</td><td>${r.lr === choice.selected_lr ? "已冻结 LR" : "网格候选"}</td></tr>`).join("");
    node("sweep-chart").innerHTML = dynamicsLineChart(report.width.sweep_widths.map(width => ({ name: `宽度 ${width}${width === 16 ? " 选择" : " 诊断"}`, points: rows.filter(r => r.width === width).map(r => ({ x: Math.log10(r.lr), y: r.loss })) })), { xLabel: "log₁₀(基础 LR)", yLabel: "验证 CE / nats" });
    set("selection", `${p === "mup" ? "μP" : "SP"} 基础 LR=${fmt(choice.selected_lr)}，只由宽度 ${report.width.base_width} 的三个 seed 验证 loss 均值选择。宽度 32／64 只作诊断；宽度 ${report.width.held_width} 未参与任何候选扫描。${choice.grid_boundary ? "已测网格中的最佳值位于边界，最优值可能超出网格。" : "这是已测网格中的最佳值。"}测试结果在选择冻结后读取。`);
  };
  const renderLedger = () => {
    if (!report) return;
    const run = report.width.selected_runs.find(r => r.width === +node("ledger-width").value && r.parameterization === node("ledger-param").value && r.seed === +node("ledger-seed").value);
    if (!run) return;
    node("ledger").innerHTML = run.ledger.map(p => `<tr><th scope="row">${esc(p.name)}</th><td>${esc(p.shape.join("×"))}</td><td>${p.infinite_dimensions ?? "SP"}</td><td>${fmt(p.expected_initial_variance)} / ${fmt(p.observed_initial_variance)}</td><td>${p.optimizer_group} / ${fmt(p.actual_lr)}</td><td>${fmt(p.readout_input_factor)}</td></tr>`).join("");
    set("ledger-note", `${esc(run.parameterization.toUpperCase())} / 宽度 ${run.width} / seed ${run.seed}；基础 LR=${fmt(run.base_lr)}，decay=0。账本显示目标初始化方差与该次随机初始化的总体方差；两者不要求逐数值相等。readout 因子作用于输入，bias 前向因子为 1。`);
  };
  const renderTransfer = () => {
    if (!report) return;
    const rows = ["sp", "mup"].flatMap(p => [...report.width.sweep_widths, report.width.held_width].map(width => {
      const runs = report.width.selected_runs.filter(r => r.parameterization === p && r.width === width);
      return { p, width, loss: average(runs.map(r => r.test.loss)), accuracy: average(runs.map(r => r.test.accuracy)),
        min: Math.min(...runs.map(r => r.test.loss)), max: Math.max(...runs.map(r => r.test.loss)) };
    }));
    node("transfer").innerHTML = rows.map(r => `<tr><th scope="row">${r.p === "mup" ? "μP" : "SP"} / ${r.width}${r.width === report.width.held_width ? " · 保留" : ""}</th><td>${fmt(r.loss)}</td><td>${fmt(r.min)}…${fmt(r.max)}</td><td>${pct(r.accuracy)}</td></tr>`).join("");
  };
  const renderDepth = () => {
    if (!report) return;
    const depth = +node("depth").value, norm = node("norm").value, residualRule = node("residual").value;
    const rows = depthProbeRows(report, { depth, norm, residualRule, afterUpdates: +node("depth-step").value, seed: node("depth-seed").value });
    node("depth-rows").innerHTML = rows.map(r => `<tr><th scope="row">${esc(r.layer)}</th><td>${fmt(r.activation_rms)}</td><td>${fmt(r.activation_min)}…${fmt(r.activation_max)}</td><td>${fmt(r.hidden_gradient_rms)}</td><td>${fmt(r.ffn_weight_gradient_rms)}</td></tr>`).join("");
    node("depth-chart").innerHTML = dynamicsLineChart([{ name: "激活 RMS", points: rows.map((r, i) => ({ x: i, y: r.activation_rms })) }], { xLabel: "embedding→各 block→Norm→logits", yLabel: "激活 RMS" });
    const scale = residualRule === "one" ? 1 : 1 / Math.sqrt(2 * depth);
    set("depth-note", `共享 decoder：宽度 32、${depth} 层、${norm.toUpperCase()} LayerNorm、α=${fmt(scale)}、dropout=0、Adam LR=.003；同八条序列，更新 ${node("depth-step").value} 次。hidden 梯度与 FFN 权重梯度分别计 RMS。固定 0.02 初始化使本次深度差异可能很小；Post-LN 输出 RMS 约为 1 是归一化的直接作用。`);
  };
  const clear = () => {
    report = null; node("output").hidden = true; set("provenance", "");
    for (const name of ["coordinates", "chart", "sweep-rows", "sweep-chart", "ledger", "transfer", "depth-rows", "depth-chart"]) node(name).innerHTML = "";
    for (const name of ["coordinate-note", "selection", "ledger-note", "depth-note"]) set(name, "");
  };
  const accept = data => {
    report = data;
    const seeds = ["all", ...data.width.seeds], widths = [...data.width.sweep_widths, data.width.held_width];
    dynamicsOptions(node("seed"), seeds, { labels: new Map([["all", "全部 seed：均值与范围"]]) });
    dynamicsOptions(node("step"), [0, 1, 4, data.width.steps], { preferred: 1 });
    dynamicsOptions(node("ledger-width"), widths, { preferred: data.width.held_width });
    dynamicsOptions(node("ledger-seed"), data.width.seeds);
    dynamicsOptions(node("depth"), data.depth.depths, { preferred: 4 });
    dynamicsOptions(node("depth-seed"), ["all", ...data.depth.seeds], { labels: new Map([["all", "全部 seed：均值与范围"]]) });
    set("provenance", dynamicsProvenance(data));
    renderWidth(); renderSweep(); renderLedger(); renderTransfer(); renderDepth(); node("output").hidden = false;
  };
  for (const name of ["layer", "step", "metric", "seed"]) node(name).addEventListener("change", renderWidth);
  node("sweep-param").addEventListener("change", renderSweep);
  for (const name of ["ledger-width", "ledger-param", "ledger-seed"]) node(name).addEventListener("change", renderLedger);
  for (const name of ["depth", "norm", "residual", "depth-step", "depth-seed"]) node(name).addEventListener("change", renderDepth);
  bindDynamicsImport({ file, reset: node("default"), status: node("status"), clear, accept,
    defaultData: initializationWidthData, validate: validateInitializationWidthReport });
}
