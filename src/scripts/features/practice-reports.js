import { practiceSnapshot } from "./practice-data.js";
import { esc, pct, num, table, lineChart, metrics } from "../experiment-ui.js";
import { bindExperimentView } from "../experiment-view.js";

const finite = x => Number.isFinite(x) && x >= 0;
const probability = x => finite(x) && x <= 1;
const require = (condition, message) => { if (!condition) throw new RangeError(message); };
const seedRun = (runs, seed) => { const run = runs.find(run => String(run.seed) === String(seed)); require(run, "未知 seed"); return run; };
const policies = { standard: "原始自由生成", eos_diagnostic: "禁止过早 EOS · 诊断策略", shifted_positions: "位置平移 +4 · 扰动探针", extended: "加入长度 6–8 训练样本" };
const symbols = values => values.map(id => ["PAD", "BOS", "SEP", "EOS", "A", "B", "C", "D"][id]).join(" ");
const median = values => { const a = [...values].sort((a, b) => a - b); return (a[Math.floor((a.length - 1) / 2)] + a[Math.floor(a.length / 2)]) / 2; };

export function validatePracticeReport(data) {
  require(data?.schema === "model-practice-v1" && data.status === "passed" && /^[a-f0-9]{64}$/.test(data.provenance?.report_sha256 || ""), "请选择有完整来源的本轮实战快照。");
  const { generalization: g, text: t, reproduction: r, profile: p } = data.reports || {};
  require(g?.schema === "generalization-diagnostics-v1" && g.group_overlap === 0 && g.baseline_predictions_reproduced && g.baselines?.length === 3 && g.extended?.length === 3, "泛化对照或分组证据缺失。");
  for (const run of g.baselines) {
    require(g.extended.some(other => other.seed === run.seed), "训练干预缺少 seed。");
    for (const key of ["standard", "eos_diagnostic", "shifted_positions"]) {
      require(Array.isArray(run[key]) && run.summaries?.[key]?.length === 9, "长度切片不完整。");
      for (const row of run.summaries[key]) require(Number.isInteger(row.records) && row.records > 0 && probability(row.exact_match) && probability(row.content_accuracy_macro) && row.early_eos <= row.records, "长度指标超出范围。");
    }
    require(run.position_summary?.every(row => probability(row.teacher_content_accuracy) && probability(row.mean_premature_eos_probability)), "逐位置观测无效。");
  }
  require(t?.schema === "natural-text-project-v1" && t.runs?.length === 3 && t.resume?.passed && Object.values(t.resume.errors).every(x => x === 0), "自然语言项目的恢复核验缺失。");
  const documents = Object.values(t.corpus.documents).flat(); require(new Set(documents).size === documents.length, "文档跨划分重复。");
  require(JSON.stringify(t.tokenizer.training_documents) === JSON.stringify(t.corpus.documents.train) && t.tokenizer.example.text === t.tokenizer.example.decoded, "Tokenizer 不满足训练集隔离或往返还原。");
  for (const run of t.runs) {
    require(run.curve.length === 9 && run.curve[0].update === 0 && run.curve.at(-1).update === t.training.updates, "曲线缺少固定预算端点。");
    for (const phase of ["initial", "learned"]) for (const split of ["train", "validation", "test"]) {
      const row = run.evaluation[phase][split]; require(finite(row.ce_nats_per_token) && probability(row.teacher_forced_accuracy), "文本评估数值无效。");
    }
  }
  require(r?.schema === "reproduction-differences-v1" && r.matched_repeat_exact && r.cases?.length === 3 && r.historical_difference.status === "unresolved", "复现探针或历史不确定性缺失。");
  for (const row of r.cases) require(row.rows.length === 16 && row.rows.every(step => Object.values(step.errors).every(finite)), "复现误差记录无效。");
  require(p?.schema === "full-step-profile-v1" && p.numerical_comparison?.passed && p.rows?.length === 2, "完整 step 的数值对照缺失。");
  for (const row of p.rows) for (const stage of Object.values(row.stages_ms)) {
    require(stage.samples.length === 12 && stage.samples.every(finite) && Math.abs(median(stage.samples) - stage.median) < 1e-9 && stage.min === Math.min(...stage.samples) && stage.max === Math.max(...stage.samples), "实测时间摘要与原始样本不一致。");
  }
  return data;
}

export function generalizationRows(data, seed = 11, policy = "eos_diagnostic") {
  require(Object.hasOwn(policies, policy), "未知诊断策略");
  const original = seedRun(data.reports.generalization.baselines, seed), extended = seedRun(data.reports.generalization.extended, seed);
  return { original, extended, rows: policy === "extended" ? extended.summary : original.summaries[policy],
    predictions: policy === "extended" ? extended.predictions : original[policy] };
}
export function textProjectView(data, seed = 11, split = "test") {
  require(["train", "validation", "test"].includes(split), "未知文本划分");
  const run = seedRun(data.reports.text.runs, seed);
  return { run, initial: run.evaluation.initial[split], learned: run.evaluation.learned[split] };
}
export function reproductionRows(data, name = "cast-float64") {
  const row = data.reports.reproduction.cases.find(row => row.case === name); require(row, "未知复现探针"); return row;
}
export function profileRows(data, name = "dense") {
  const row = data.reports.profile.rows.find(row => row.method === name); require(row, "未知 step 方案"); return row;
}

function select(name, label, options, value) { return `<label><span>${esc(label)}</span><select data-practice-control="${name}">${options.map(([key, title]) => `<option value="${esc(key)}"${String(value) === String(key) ? ' selected' : ''}>${esc(title)}</option>`).join('')}</select></label>`; }

function generalizationView(data, state) {
  const view = generalizationRows(data, state.seed, state.policy), rows = view.rows;
  const source = new Map(view.original.standard.map(row => [row.id, row]));
  const examples = view.predictions.filter(row => row.length === +state.length);
  const result = lineChart([{ name: '原始自由生成', points: view.original.summaries.standard.map(row => ({ x: row.length, y: row.exact_match * 100 })) }, { name: policies[state.policy], points: rows.map(row => ({ x: row.length, y: row.exact_match * 100 })) }], '输入长度', '完整匹配 / %');
  return `${metrics([['当前对照', policies[state.policy]], ['训练监督 token · 原始', view.extended.baseline_supervised_tokens.toLocaleString('en-US')], ['训练监督 token · 扩展支持', view.extended.supervised_tokens.toLocaleString('en-US')]])}<p class="practice-note">EOS 诊断利用任务规定的答案长度改变解码策略；位置平移是推理扰动；扩展训练只对齐 1,200 次更新。三者回答不同问题，不能混成同一能力提升。</p>${result}${table(['长度', '与当前训练长度的关系', '题数', '完整匹配', '内容位置准确率', '提前 EOS'], rows.map(row => [String(row.length), row.length <= (state.policy === 'extended' ? 8 : 5) ? '支持内 · 未见题目' : '长度外推', row.records, pct(row.exact_match), pct(row.content_accuracy_macro), row.early_eos]), '每个长度的实际生成指标')}${table(['同题 ID / 输入', '正确答案', '原始生成', '当前对照'], examples.map(row => [`${esc(row.id)}<br><code>${esc(symbols(row.content))}</code>`, `<code>${esc(symbols([...row.target, 3]))}</code>`, `<code>${esc(symbols(source.get(row.id).predicted))}</code>`, `<code>${esc(symbols(row.predicted))}</code>`]), '相同题目的实际保存预测')}<details><summary>正确答案前缀下的逐位置观测</summary><p>这是 teacher forcing 下的观测；自由生成会接收自己此前的输出。</p>${table(['长度', '内容 top-1 准确率', '内容位置平均 P(EOS)', '正确结束位置平均 P(EOS)'], view.original.position_summary.map(row => [row.length, pct(row.teacher_content_accuracy), num(row.mean_premature_eos_probability), num(row.mean_gold_eos_probability)]), '正确前缀下的 EOS 概率')}</details>`;
}

function textView(data, state) {
  const { run, initial, learned } = textProjectView(data, state.seed, state.split), report = data.reports.text, index = +state.prompt;
  const generation = phase => { const row = run.generations[phase][index]; return `<article><h3>${phase === 'initial' ? '训练前' : '训练后'}</h3><p>提示：<code>${esc(row.prompt)}</code></p><pre>${esc(row.text)}</pre><p class="practice-note">${row.tokens.length} 个生成 token；${row.terminated ? '模型生成 EOS' : '达到 48-token 上限'}；替换字符 ${row.replacement_characters}，非 EOS 特殊 token ${row.special_non_eos}。</p></article>`; };
  return `${metrics([['初始 CE · nats/token', num(initial.ce_nats_per_token)], ['训练后 CE · nats/token', num(learned.ce_nats_per_token)], ['Teacher-forced top-1', pct(learned.teacher_forced_accuracy)]])}<p class="practice-note">评估使用完整的 ${learned.windows} 个 ${esc(state.split)} 窗口、${learned.supervised_tokens.toLocaleString('en-US')} 个监督 token。预测更准不等于文本连贯或事实可靠；这里保留未经修饰的 greedy 输出。</p><div class="practice-pair">${generation('initial')}${generation('learned')}</div>${lineChart([{ name: 'train', points: run.curve.map(point => ({ x: point.update, y: point.train.ce_nats_per_token })) }, { name: 'validation', points: run.curve.map(point => ({ x: point.update, y: point.validation.ce_nats_per_token })) }], '优化器更新次数', 'CE / nats 每 token')}${table(['更新', '累计监督 token', '训练 CE', '验证 CE'], run.curve.map(row => [row.update, row.supervised_tokens.toLocaleString('en-US'), num(row.train.ce_nats_per_token), num(row.validation.ce_nats_per_token)]), '实际训练与验证曲线数值')}<details><summary>只在训练文档学习的 tokenizer 与可逆编码</summary><p>${report.tokenizer.vocab_size} 个 token；256 个字节、3 个特殊 token、${report.tokenizer.merges} 次实际学习的合并。</p><pre>${esc(report.tokenizer.example.text)}\n\n${esc(report.tokenizer.example.tokens.join(' '))}\n\n解码：${esc(report.tokenizer.example.decoded)}</pre></details>`;
}

const caseNames = { 'repeat-float32': '同环境、同输入重复', 'cast-float64': '同初值与输入，转换 float64', 'changed-data-order': '保持 seed，改变数据顺序' };
function reproductionView(data, state) {
  const row = reproductionRows(data, state.case), first = row.first_difference;
  return `${metrics([['首次可观察差异', first ? `第 ${first.update} 次更新 · ${first.stage}` : '16 次更新均逐值一致'], ['同一输入序列', row.same_inputs ? '是' : '否'], ['历史变化原因', '仍未定位']])}<p class="practice-note">这里的精度转换与数据顺序变化是新做的受控探针，不能据此认定旧快照与新运行差异的根因。</p>${lineChart([{ name: '梯度最大绝对差', points: row.rows.map(r => ({ x: r.update, y: r.errors.gradient })) }, { name: '更新后参数最大绝对差', points: row.rows.map(r => ({ x: r.update, y: r.errors.parameters })) }], '更新次数', '相对于 float32 参考的最大绝对差')}${table(['更新', '输入 ID', 'Embedding＋位置', '隐藏状态', 'logits', 'loss', '梯度', '更新后参数'], row.rows.map(r => [r.update, ...['input_ids', 'embedding', 'hidden', 'logits', 'loss', 'gradient', 'parameters'].map(key => num(r.errors[key]))]), '按执行边界记录的首次差异与后续轨迹')}`;
}

const stageNames = { prepare_and_zero_grad: '整理预分词 batch＋清空梯度', model_forward: 'Embedding / Attention / MLP', output_and_loss: '输出投影＋交叉熵', backward: '整条反向传播', clip_and_update: '裁剪＋AdamW', total: '完整 step' };
function profileView(data, state) {
  const row = profileRows(data, state.method), all = data.reports.profile, i = +state.iteration - 1;
  const stages = Object.entries(row.stages_ms).filter(([key]) => key !== 'total');
  return `${metrics([['完整 step 中位数', `${num(row.stages_ms.total.median)} ms`], ['监督 token / 秒', num(row.supervised_tokens_per_second)], ['反向保存引用', `${(row.saved_storage.peak_referenced_bytes / 1048576).toFixed(2)} MiB`]])}<p class="practice-note">CPU float64；3 次预热、12 次实测，两个方案交替运行。时间包含模型前向、反向和更新；不包含磁盘读入、分词、评估和保存。保存引用量不等于物理内存峰值。</p>${table(['阶段', '中位 ms', 'min…max ms', `第 ${i + 1} 次实测 ms`], Object.entries(row.stages_ms).map(([key, stats]) => [esc(stageNames[key]), num(stats.median), `${num(stats.min)}…${num(stats.max)}`, num(stats.samples[i])]), '同一完整训练 step 的阶段与原始时间样本')}<p>本次各阶段之和 ${num(stages.reduce((sum, [, s]) => sum + s.samples[i], 0))} ms；本次完整 step ${num(row.stages_ms.total.samples[i])} ms。各阶段的中位数不必相加等于总中位数。</p>${table(['方案', '完整 step 中位 ms', '反向保存引用 B', '监督 token/s'], all.rows.map(r => [esc(r.method), num(r.stages_ms.total.median), r.saved_storage.peak_referenced_bytes.toLocaleString('en-US'), num(r.supervised_tokens_per_second)]), '相同模型和 batch 序列的两方案对照')}<p class="practice-note">两方案 loss 最大差 ${num(all.numerical_comparison.loss_max_error)}，15 次更新后的参数最大差 ${num(all.numerical_comparison.final_parameter_max_error)}；先过数值门槛，再解释性能差异。</p><details><summary>单独测量的评估与写 checkpoint 开销</summary>${table(['操作', '中位 ms', 'min…max ms'], Object.entries(all.separate_overheads_ms).map(([name, s]) => [name === 'evaluate' ? '完整验证集评估' : '写入 checkpoint（未承诺 fsync）', num(s.median), `${num(s.min)}…${num(s.max)}`]), '不计入训练 step 的独立开销')}</details>`;
}

export function initPracticeReports() {
  const root = document.querySelector('[data-practice-kind]'); if (!root) return;
  const data = validatePracticeReport(practiceSnapshot), kind = root.dataset.practiceKind;
  const state = { seed: '11', policy: 'eos_diagnostic', length: '6', split: 'test', prompt: '0', case: 'cast-float64', method: 'dense', iteration: '1' };
  const seeds = [[11, '11'], [29, '29'], [47, '47']];
  const controls = kind === 'generalization' ? select('seed', '训练 seed', seeds, state.seed) + select('policy', '对照问题', Object.entries(policies), state.policy) + select('length', '查看相同题目的长度', [2, 3, 4, 5, 6, 7, 8, 9, 10].map(x => [x, x]), 6)
    : kind === 'text' ? select('seed', '训练 seed', seeds, state.seed) + select('split', '完整评估拆分', [['train', '训练文档'], ['validation', '验证文档'], ['test', '独立测试文档']], 'test') + select('prompt', '训练前后相同提示', [[0, '模型训练'], [1, '梯度'], [2, '数据']], 0)
    : kind === 'reproduction' ? select('case', '对照条件', Object.entries(caseNames), state.case)
    : select('method', '完整 step 方案', [['dense', '完整输出投影＋CE'], ['chunked-exact', '精确分块输出层']], state.method) + select('iteration', '查看原始实测次数', Array.from({ length: 12 }, (_, i) => [i + 1, i + 1]), 1);
  root.innerHTML = `<div class="practice-controls">${controls}</div><div data-practice-output aria-live="polite"></div>`;
  const render = () => { root.querySelector('[data-practice-output]').innerHTML = ({ generalization: generalizationView, text: textView, reproduction: reproductionView, profile: profileView })[kind](data, state); };
  bindExperimentView(root, { attribute: "practiceControl", onChange: value => { Object.assign(state, value); render(); } });
  root.dataset.practiceReady = 'true';
}
