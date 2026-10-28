import { $ } from "../core.js";
import { batchNoiseData } from "./batch-noise-data.js";
import { dynamicsNumber as fmt, dynamicsPercent as pct, dynamicsText as esc, dynamicsNear as near,
  dynamicsRequire as require, dynamicsString as string, dynamicsFinite as finite,
  dynamicsInt as integer, dynamicsArray as array, validateDynamicsEnvelope,
  dynamicsProvenance, dynamicsLineChart, dynamicsOptions, bindDynamicsImport } from "./initialization-width-utils.js";

const average = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const unique = values => [...new Set(values)];
const schemes = ["with-replacement", "without-replacement", "repeated", "correlated"];
const schemeNames = { "with-replacement": "有放回独立抽样", "without-replacement": "无放回抽样", repeated: "整批重复同一序列", correlated: "ρ=.5 相关混合" };

export function validateGradientGram(matrix) {
  require(array(matrix, 1, 32) && matrix.every(row => array(row, matrix.length, matrix.length) && row.every(Number.isFinite)), "梯度 Gram 须为最多 32×32 的有限方阵。");
  const n = matrix.length, scale = Math.max(1e-30, ...matrix.flat().map(Math.abs));
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) require(Math.abs(matrix[i][j] - matrix[j][i]) <= 1e-10 * scale, "梯度 Gram 不对称。");
  // Jacobi eigenvalue sweeps, normalized to avoid a fixed absolute cutoff.
  // Unlike diagonal-only checks this rejects symmetric indefinite inputs.
  const a = matrix.map(row => row.map(value => value / scale));
  for (let sweep = 0; sweep < 50 * n * n; sweep++) {
    let p = 0, q = 0, maximum = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (Math.abs(a[i][j]) > maximum) { maximum = Math.abs(a[i][j]); p = i; q = j; }
    if (maximum <= 1e-13) break;
    const angle = .5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
    const c = Math.cos(angle), s = Math.sin(angle), app = a[p][p], aqq = a[q][q], apq = a[p][q];
    for (let k = 0; k < n; k++) if (k !== p && k !== q) {
      const kp = a[k][p], kq = a[k][q];
      a[k][p] = a[p][k] = c * kp - s * kq;
      a[k][q] = a[q][k] = s * kp + c * kq;
    }
    a[p][p] = c * c * app - 2 * c * s * apq + s * s * aqq;
    a[q][q] = s * s * app + 2 * c * s * apq + c * c * aqq;
    a[p][q] = a[q][p] = 0;
  }
  const eigenvalues = a.map((row, i) => row[i]);
  const tolerance = 1e-10 * Math.max(1e-30, ...eigenvalues.map(Math.abs));
  require(Math.min(...eigenvalues) >= -tolerance, "梯度 Gram 不是正半定矩阵，不能作为梯度内积。");
  return matrix;
}

export function gradientGramStatistics(gram) {
  validateGradientGram(gram);
  const n = gram.length, scale = Math.max(1e-30, ...gram.flat().map(Math.abs));
  const meanNormSqRaw = gram.flat().reduce((sum, value) => sum + value, 0) / (n * n);
  const secondMoment = gram.reduce((sum, row, i) => sum + row[i], 0) / n;
  require(meanNormSqRaw >= -1e-10 * scale && secondMoment >= -1e-10 * scale && secondMoment - meanNormSqRaw >= -1e-10 * scale, "Gram 导出的方差不合法。");
  const meanNormSq = Math.max(0, meanNormSqRaw), trace = Math.max(0, secondMoment - meanNormSqRaw);
  // sqrt(eps)-scale tolerance handles cancellation in a Gram-only mean.
  const threshold = Math.max(1e-15, 1e-7 * Math.sqrt(Math.max(0, secondMoment)));
  const meanNorm = Math.sqrt(meanNormSq);
  const state = meanNorm === 0 ? "undefined-zero-mean" : meanNorm <= threshold ? "ill-conditioned-mean" : "defined";
  return { population_size: n, covariance_denominator: n, mean_norm_sq: meanNormSq,
    mean_norm: meanNorm, second_moment: Math.max(0, secondMoment), covariance_trace: trace,
    mean_norm_threshold: threshold, noise_ratio_state: state,
    noise_ratio: state === "defined" ? trace / meanNormSq : null };
}

export function batchNoiseVariance(trace, n, batchSize, replacement = true) {
  require(finite(trace) && integer(n, 1, 1000000) && integer(batchSize, 1, 1000000), "总体或 batch 大小无效。");
  if (replacement) return trace / batchSize;
  require(batchSize <= n, "无放回 batch 不能超过总体大小。");
  return n === 1 ? 0 : trace * (n - batchSize) / (batchSize * (n - 1));
}

export function enumerateGradientBatches(gram, { batchSize = 2, scheme = "with-replacement", correlation = .5 } = {}) {
  const stats = gradientGramStatistics(gram), n = gram.length;
  require(integer(batchSize, 1, 12) && schemes.includes(scheme) && finite(correlation, 1), "抽样配置无效。");
  require(scheme !== "without-replacement" || batchSize <= n, "无放回 B 不能超过 N。");
  const count = scheme === "repeated" ? n : scheme === "without-replacement" ?
    Array.from({ length: batchSize }, (_, i) => (n - i) / (i + 1)).reduce((a, b) => a * b, 1) : n ** batchSize;
  require(count <= 50000, "穷举超过 50,000 个结果，请减小 B。");
  const draws = [];
  const recurse = prefix => {
    if (prefix.length === batchSize) { draws.push(prefix); return; }
    const start = scheme === "without-replacement" && prefix.length ? prefix.at(-1) + 1 : 0;
    for (let index = start; index < n; index++) recurse([...prefix, index]);
  };
  if (scheme === "repeated") for (let i = 0; i < n; i++) draws.push(Array(batchSize).fill(i));
  else recurse([]);
  let weights = draws.map(() => 1 / draws.length);
  if (scheme === "correlated") {
    weights = weights.map(weight => weight * (1 - correlation));
    for (let i = 0; i < n; i++) { draws.push(Array(batchSize).fill(i)); weights.push(correlation / n); }
  }
  const quadratic = (x, y) => x.reduce((sum, xi, i) => sum + xi * gram[i].reduce((s, value, j) => s + value * y[j], 0), 0);
  const meanWeights = Array(n).fill(1 / n);
  const rows = draws.map((indices, k) => {
    const coefficients = Array(n).fill(0); for (const index of indices) coefficients[index] += 1 / batchSize;
    const centered = coefficients.map(value => value - 1 / n);
    const norm = Math.sqrt(Math.max(0, quadratic(coefficients, coefficients)));
    const distanceSq = Math.max(0, quadratic(centered, centered));
    const inner = quadratic(coefficients, meanWeights), denominator = norm * stats.mean_norm;
    const directionDefined = stats.noise_ratio_state === "defined" && norm > 1e-12 * Math.sqrt(stats.second_moment) && denominator > 0;
    return { indices, probability: weights[k], gradient_norm: norm, distance_sq: distanceSq,
      cosine_to_mean: directionDefined ? Math.max(-1, Math.min(1, inner / denominator)) : null };
  });
  const trace = stats.covariance_trace;
  const formula = scheme === "without-replacement" ? batchNoiseVariance(trace, n, batchSize, false) :
    scheme === "repeated" ? trace : scheme === "correlated" ? trace * (correlation + (1 - correlation) / batchSize) : trace / batchSize;
  const exact = rows.reduce((sum, row) => sum + row.probability * row.distance_sq, 0);
  return { ...stats, batch_size: batchSize, scheme, enumerated_trace: exact, formula_trace: formula,
    iid_formula_trace: trace / batchSize, absolute_error: Math.abs(exact - formula),
    enumerated_outcomes: rows.length, probability_sum: weights.reduce((a, b) => a + b, 0), rows };
}

function validatePopulation(population) {
  require(population && integer(population.population_size, 1, 32) && population.covariance_denominator === population.population_size, "总体协方差分母须明确为 N。");
  const stats = gradientGramStatistics(population.gram);
  require(stats.population_size === population.population_size, "Gram 与总体大小不一致。");
  for (const key of ["mean_norm_sq", "mean_norm", "second_moment", "covariance_trace", "mean_norm_threshold"]) require(finite(population[key]), `总体 ${key} 无效。`);
  require(near(population.mean_norm_sq, stats.mean_norm_sq) && near(population.covariance_trace, stats.covariance_trace)
    && near(population.mean_norm ** 2, population.mean_norm_sq), "总体统计量与已保存 Gram 不一致。");
  require(["defined", "undefined-zero-mean", "ill-conditioned-mean"].includes(population.noise_ratio_state), "均值边界状态无效。");
  if (population.noise_ratio_state === "defined") require(population.mean_norm > population.mean_norm_threshold && finite(population.noise_ratio) && near(population.noise_ratio, population.covariance_trace / population.mean_norm_sq), "噪声比值不等于 trace/mean²。");
  else require(population.noise_ratio === null && population.mean_norm <= population.mean_norm_threshold, "零均值／病态比值必须显示未定义。");
  return population;
}

function validateEvaluation(evaluation) {
  for (const split of ["train", "validation", "test", "length"]) {
    const value = evaluation?.[split], metrics = value?.metrics;
    require(metrics && integer(metrics.records, 1, 10000) && finite(metrics.teacher_forced_loss)
      && finite(metrics.teacher_forced_token_accuracy, 1) && finite(metrics.free_token_accuracy, 1)
      && finite(metrics.exact_match, 1) && metrics.max_new_tokens === 20, "评估必须分开 teacher-forced、自由 token 和整段指标。");
    if (value.predictions !== undefined) {
      require(array(value.predictions, 0, 10000), "逐题预测超出范围。");
      const ids = new Set();
      for (const row of value.predictions) {
        require(row && string(row.id, 300) && !ids.has(row.id) && array(row.content, 1, 10) && row.content.every(n => integer(n, 4, 7))
          && array(row.target, row.content.length, row.content.length) && row.target.every((n, i) => n === row.content[row.content.length - 1 - i])
          && array(row.predicted, 1, 20) && row.predicted.every(n => integer(n, 0, 7))
          && finite(row.token_accuracy, 1) && finite(row.teacher_forced_loss), "逐题预测违背反转任务或范围约定。");
        ids.add(row.id);
      }
    }
  }
}

export function validateBatchNoiseReport(data) {
  validateDynamicsEnvelope(data, "batch-noise-v1");
  const noise = data.noise, batch = data.batch;
  require(noise?.core_api === "learning-core-v1" && batch?.core_api === "learning-core-v1", "噪声与训练须复用共享 decoder core。");
  validatePopulation(noise.population);
  require(noise.population.population_size === 6 && array(noise.record_ids, 6, 6) && unique(noise.record_ids).length === 6, "固定参数人口须为六条序列。");
  require(array(noise.enumerations, 1, 40) && finite(noise.sequence_gradient_mean_error, 1e-8), "固定参数梯度或穷举证据缺失。");
  for (const row of noise.enumerations) {
    require(schemes.includes(row.scheme) && integer(row.batch_size, 1, 6)
      && finite(row.enumerated_trace) && finite(row.formula_trace) && finite(row.absolute_error, 1e-8), "穷举统计无效。");
    const predicted = row.scheme === "without-replacement" ? batchNoiseVariance(noise.population.covariance_trace, 6, row.batch_size, false) :
      row.scheme === "repeated" ? noise.population.covariance_trace : row.scheme === "correlated" ? noise.population.covariance_trace * (.5 + .5 / row.batch_size) : noise.population.covariance_trace / row.batch_size;
    require(near(row.enumerated_trace, predicted) && near(row.formula_trace, predicted), "报告穷举与声明的抽样公式不一致。");
  }
  if (noise.centered_counterexample) validatePopulation(noise.centered_counterexample.population);
  if (noise.duplicate_population) validatePopulation(noise.duplicate_population.population);
  require(array(batch.seeds, 2, 10) && batch.seeds.every(n => integer(n, 0)) && unique(batch.seeds).length === batch.seeds.length
    && array(batch.batch_sizes, 2, 8) && batch.batch_sizes.every(n => integer(n, 1, 1024))
    && array(batch.lr_grid, 1, 8) && batch.lr_grid.every(n => finite(n, 1) && n > 0), "batch/seed/LR 网格无效。");
  require(integer(batch.record_budget, 32, 100000) && batch.input_tokens_per_record === 8 && batch.supervised_tokens_per_record === 4
    && batch.budget_input_tokens === batch.record_budget * 8 && batch.budget_supervised_tokens === batch.record_budget * 4, "等长记录 token 预算不一致。");
  require(batch.config?.width === 32 && batch.config?.depth === 2 && batch.config?.heads === 4
    && batch.config?.dropout === 0 && batch.config?.norm === "pre", "batch 实验修改了约定的固定模型。");
  require(array(batch.runs, 1, 200), "实际训练 runs 缺失。");
  const keys = new Set(), identity = new Map(); let commonTokenGrid = null;
  for (const run of batch.runs) {
    require(run && string(run.run_id, 300) && batch.seeds.includes(run.seed) && batch.batch_sizes.includes(run.batch_size) && batch.lr_grid.includes(run.base_lr), "训练 run 配置无效。");
    const key = `${run.seed}/${run.batch_size}/${run.base_lr}`; require(!keys.has(key), "训练配置重复。"); keys.add(key);
    require(run.consumed_records === batch.record_budget && run.consumed_input_tokens === batch.budget_input_tokens
      && run.consumed_supervised_tokens === batch.budget_supervised_tokens && run.updates * run.batch_size === batch.record_budget, "真实 batch 消费量或更新次数不符合固定预算。");
    require(run.config?.width === batch.config.width && run.config?.depth === batch.config.depth && run.config?.dropout === 0 && run.config?.norm === "pre", "某训练 run 使用了不同模型。");
    require(run.scheduler?.unit === "input_tokens" && run.scheduler?.budget_tokens === batch.budget_input_tokens
      && run.scheduler?.warmup_tokens === 1024 && run.scheduler?.lr_sampling === "actual batch endpoint", "scheduler 不是明确的 input token 口径。");
    for (const name of ["data_stream_sha256", "evaluation_split_sha256", "initial_model_sha256"]) require(/^[a-f0-9]{64}$/.test(run[name] || ""), "缺少同样本流／评估／初始化 hash。");
    const currentIdentity = [run.data_stream_sha256, run.evaluation_split_sha256, run.initial_model_sha256].join("/");
    require(!identity.has(run.seed) || identity.get(run.seed) === currentIdentity, "同 seed 的 batch 比较用了不同流、评估或初始参数。"); identity.set(run.seed, currentIdentity);
    require(array(run.curve, 2, 1000), "实际 loss 曲线无效。");
    let prior = -1;
    for (const point of run.curve) {
      require(integer(point.consumed_input_tokens, 0, batch.budget_input_tokens) && point.consumed_input_tokens > prior
        && point.consumed_supervised_tokens === point.consumed_input_tokens / 2
        && integer(point.updates, 0, run.updates) && point.updates * run.batch_size * 8 === point.consumed_input_tokens
        && finite(point.train?.teacher_forced_loss) && finite(point.validation?.teacher_forced_loss), "曲线计数或实际 loss 无效。");
      prior = point.consumed_input_tokens;
    }
    require(run.curve[0].consumed_input_tokens === 0 && prior === batch.budget_input_tokens, "曲线缺少初始或最终测量。");
    const tokenGrid = run.curve.map(point => point.consumed_input_tokens).join(",");
    require(commonTokenGrid === null || commonTokenGrid === tokenGrid, "各 seed／batch 的曲线必须使用共同的 token 测量网格。");
    commonTokenGrid = tokenGrid;
    validateEvaluation(run.initial_evaluation); validateEvaluation(run.final_evaluation);
    require(near(run.curve.at(-1).validation.teacher_forced_loss, run.final_evaluation.validation.metrics.teacher_forced_loss, 1e-6), "曲线终值与最终评估不一致。");
    require(finite(run.cached_logits_max_error, 1e-5), "完整／增量 logits 核对未通过。");
    if (run.resume) require(run.resume.status === "passed" && run.resume.fresh_python_process === true
      && ["model", "optimizer", "next_steps", "evaluation", "rng"].every(name => run.resume.max_errors?.[name] === 0), "新进程恢复证据须包含实际 RNG 核对。");
    for (const snapshot of [run.initial_noise, run.final_noise]) require(snapshot && finite(snapshot.covariance_trace) && finite(snapshot.mean_norm)
      && (snapshot.noise_ratio === null || finite(snapshot.noise_ratio)), "随训练变化的噪声快照无效。");
  }
  require(batch.runs.length === batch.seeds.length * batch.batch_sizes.length * batch.lr_grid.length, "真实训练网格不完整。");
  return data;
}

export function batchTrainingRows(data, { baseLR = data.batch.lr_grid.at(-1), seed = "all", split = "test" } = {}) {
  require(["train", "validation", "test", "length"].includes(split), "评估集无效。");
  return data.batch.batch_sizes.map(batchSize => {
    const runs = data.batch.runs.filter(run => run.batch_size === batchSize && run.base_lr === Number(baseLR) && (seed === "all" || run.seed === Number(seed)));
    require(runs.length > 0, "没有当前 batch 训练配置。");
    const metrics = runs.map(run => run.final_evaluation[split].metrics);
    return { batch_size: batchSize, runs, seeds: runs.length, updates: runs[0].updates,
      teacher_forced_loss: average(metrics.map(m => m.teacher_forced_loss)),
      loss_min: Math.min(...metrics.map(m => m.teacher_forced_loss)), loss_max: Math.max(...metrics.map(m => m.teacher_forced_loss)),
      free_token_accuracy: average(metrics.map(m => m.free_token_accuracy)), exact_match: average(metrics.map(m => m.exact_match)),
      initial_noise_ratio: runs[0].initial_noise.noise_ratio,
      final_noise_ratio: runs.every(r => r.final_noise.noise_ratio !== null) ? average(runs.map(r => r.final_noise.noise_ratio)) : null };
  });
}

export function initBatchNoise() {
  const file = $("[data-bn-file]"); if (!file) return;
  const node = name => $(`[data-bn-${name}]`), set = (name, value) => { node(name).textContent = value; };
  let report = null;
  const renderNoise = () => {
    if (!report) return;
    const mode = node("population").value;
    const source = mode === "centered" ? report.noise.centered_counterexample?.population : mode === "duplicate" ? report.noise.duplicate_population?.population : report.noise.population;
    if (!source) return;
    const scheme = node("scheme").value, n = source.population_size;
    const fullPopulationAllowed = scheme === "without-replacement" || scheme === "repeated";
    const fullPopulationOption = node("batch").querySelector('option[value="6"]');
    if (fullPopulationOption) fullPopulationOption.disabled = !fullPopulationAllowed;
    if (!fullPopulationAllowed && +node("batch").value > 4) node("batch").value = "4";
    const size = +node("batch").value;
    if (scheme === "without-replacement" && size > n) node("batch").value = String(n);
    const r = enumerateGradientBatches(source.gram, { batchSize: +node("batch").value, scheme });
    set("mean", fmt(source.mean_norm)); set("trace", fmt(source.covariance_trace));
    set("ratio", source.noise_ratio === null || r.noise_ratio_state !== "defined" ? "未定义／病态" : fmt(source.noise_ratio));
    set("variance", fmt(r.enumerated_trace)); set("formula", fmt(r.formula_trace)); set("noise-error", fmt(r.absolute_error));
    const shown = r.rows.slice(0, 30);
    node("draws").innerHTML = shown.map(row => `<tr><td>${row.indices.map(i => i + 1).join(", ")}</td><td>${fmt(row.probability)}</td><td>${fmt(row.gradient_norm)}</td><td>${row.cosine_to_mean === null ? "未定义" : fmt(row.cosine_to_mean)}</td><td>${fmt(row.distance_sq)}</td></tr>`).join("");
    const reference = [1, 2, 3, 4].map(batchSize => enumerateGradientBatches(source.gram, { batchSize, scheme }));
    node("noise-chart").innerHTML = dynamicsLineChart([
      { name: "穷举方差", points: reference.map(row => ({ x: row.batch_size, y: row.enumerated_trace })) },
      { name: "iid Σ/B 预测", points: reference.map(row => ({ x: row.batch_size, y: row.iid_formula_trace })) },
    ], { xLabel: "完整序列 batch B", yLabel: "E‖ḡ−μ‖²" });
    set("noise-note", `${schemeNames[scheme]}，N=${n}，B=${r.batch_size}；精确枚举 ${r.enumerated_outcomes} 个带概率结果，表仅展示前 ${shown.length} 项。${mode === "centered" ? "这是从实测梯度中减去总体均值的代数反例，不是新模型训练。" : mode === "duplicate" ? "总体项重复但这里仍独立抽取索引；新的总体方差仍须单独计算。" : "六条独立完整序列在固定初始化 θ 下的实测梯度。"}${scheme === "repeated" ? "整批共享同一个随机索引，方差保持 Σ，不能按 token 或重复份数除 B。" : scheme === "correlated" ? "ρ=.5：一半批次独立，一半整批重复；方差为 [ρ+(1−ρ)/B]Σ。" : scheme === "without-replacement" ? "无放回使用有限总体修正；B=N 时方差为 0。" : "Σ 的分母为 N；方差法则只描述固定 θ 的抽样。"}`);
    node("records").innerHTML = report.noise.records.map((row, i) => `<tr><th scope="row">${i + 1}</th><td>${esc(row.id)}</td><td>${esc(row.content.map(n => ["PAD", "BOS", "SEP", "EOS", "A", "B", "C", "D"][n]).join(""))}</td><td>${fmt(report.noise.losses[i])}</td></tr>`).join("");
  };
  const renderTraining = () => {
    if (!report) return;
    const lr = +node("lr").value, seed = node("seed").value, split = node("split").value;
    const rows = batchTrainingRows(report, { baseLR: lr, seed, split });
    node("training-rows").innerHTML = rows.map(r => `<tr><th scope="row">${r.batch_size}</th><td>${r.updates}</td><td>${fmt(r.teacher_forced_loss)}</td><td>${fmt(r.loss_min)}…${fmt(r.loss_max)}</td><td>${pct(r.free_token_accuracy)}</td><td>${pct(r.exact_match)}</td><td>${fmt(r.final_noise_ratio)}</td></tr>`).join("");
    const lossSplit = node("curve-split").value;
    const series = rows.map(row => ({ name: `B=${row.batch_size}`, points: row.runs[0].curve.map((point, i) => ({ x: point.consumed_input_tokens,
      y: average(row.runs.map(run => run.curve[i][lossSplit].teacher_forced_loss)) })) }));
    node("training-chart").innerHTML = dynamicsLineChart(series, { xLabel: "实际消费非 padding input token", yLabel: `${lossSplit} CE / nats` });
    set("budget", `${report.batch.budget_input_tokens.toLocaleString("en-US")} input token / ${report.batch.budget_supervised_tokens.toLocaleString("en-US")} 监督 token / ${report.batch.record_budget} 条记录。三个 batch 分别 ${rows.map(r => r.updates).join(" / ")} 次更新；Adam 状态也按各自更新次数推进。`);
    set("training-note", `固定基础 LR=${fmt(lr)}，${seed === "all" ? "三个训练 seed 的均值与范围" : `seed ${seed}`}，表中评估集=${split}。曲线为 ${lossSplit} 每隔 4,096 input token 的实际 teacher-forced loss。scheduler 使用同一 token 公式并在真实 batch 末端取 LR；大 batch 的更新次数更少。初始固定 θ 噪声比值不能生成这些训练曲线，末次 θ 的噪声已发生变化。未单独为各 B 调至最优 LR。`);
    const runs = rows.flatMap(r => r.runs), resume = report.batch.runs.find(run => run.resume)?.resume;
    set("checks", `同 seed 的数据流、初始参数、评估 split 和消费量 SHA-256 一致。完整／增量 logits 最大误差=${fmt(Math.max(...runs.map(r => r.cached_logits_max_error)))}。${resume ? `另有 ${resume.resume_steps} 次更新的新 Python 进程恢复：参数、optimizer、数据、评估与 RNG 比较均为 0。${resume.rng_reference_kind?.startsWith("derived") ? "RNG 参考为 derived：此次从原 mid checkpoint 实际回放四步后取得，原 run 未保存该参考终态 RNG。" : ""}` : "此报告未附新进程恢复对照。"}`);
  };
  const renderPredictions = () => {
    if (!report) return;
    const run = report.batch.runs.find(r => r.batch_size === +node("pred-batch").value && r.seed === +node("pred-seed").value && r.base_lr === +node("lr").value);
    const split = node("split").value, rows = run?.final_evaluation[split].predictions || [];
    const symbols = tokens => tokens.map(n => ["PAD", "BOS", "SEP", "EOS", "A", "B", "C", "D"][n]).join(" ");
    node("predictions").innerHTML = rows.slice(0, 100).map(row => `<tr><th scope="row">${esc(row.id)}</th><td>${esc(symbols(row.content))}</td><td>${esc(symbols([...row.target, 3]))}</td><td>${esc(symbols(row.predicted))}</td><td>${row.exact_match ? "完全正确" : "有错误"}</td></tr>`).join("");
    set("prediction-note", `当前 ${run ? `B=${run.batch_size} / seed=${run.seed} / LR=${fmt(run.base_lr)}` : "配置无效"}，${split} 集。默认只载入每个 split 的前八条真实自由生成轨迹；完整报告导入后显示前 100 条，指标仍计算于全部题目。贪心生成遇 EOS 或统一 20-token cap 停止，缺失、多余和提前结束均计错。`);
  };
  const clear = () => {
    report = null; node("output").hidden = true; set("provenance", "");
    for (const name of ["draws", "noise-chart", "records", "training-rows", "training-chart", "predictions"]) node(name).innerHTML = "";
    for (const name of ["mean", "trace", "ratio", "variance", "formula", "noise-error", "noise-note", "budget", "training-note", "checks", "prediction-note"]) set(name, "");
  };
  const accept = data => {
    report = data; set("provenance", dynamicsProvenance(data));
    dynamicsOptions(node("lr"), data.batch.lr_grid, { preferred: data.batch.lr_grid.at(-1) });
    dynamicsOptions(node("seed"), ["all", ...data.batch.seeds], { labels: new Map([["all", "全部 seed：均值与范围"]]) });
    dynamicsOptions(node("pred-seed"), data.batch.seeds); dynamicsOptions(node("pred-batch"), data.batch.batch_sizes);
    const populations = ["measured", ...(data.noise.centered_counterexample ? ["centered"] : []), ...(data.noise.duplicate_population ? ["duplicate"] : [])];
    dynamicsOptions(node("population"), populations, { labels: new Map([["measured", "实测六条序列"], ["centered", "零均值代数反例"], ["duplicate", "重复总体项"]]) });
    renderNoise(); renderTraining(); renderPredictions(); node("output").hidden = false;
  };
  for (const name of ["population", "scheme", "batch"]) node(name).addEventListener("change", renderNoise);
  for (const name of ["lr", "seed", "split", "curve-split"]) node(name).addEventListener("change", () => { renderTraining(); renderPredictions(); });
  for (const name of ["pred-batch", "pred-seed"]) node(name).addEventListener("change", renderPredictions);
  bindDynamicsImport({ file, reset: node("default"), status: node("status"), clear, accept,
    defaultData: batchNoiseData, validate: validateBatchNoiseReport });
}
