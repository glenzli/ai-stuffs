import { matchedBudget as report } from './matched-budget-data.js';
import { num, pct, table, lineChart, metrics } from '../experiment-ui.js';
import { bindExperimentView } from '../experiment-view.js';

export function matchedRuns(data, seed) {
  const rows = data.runs.filter(row => String(row.seed) === String(seed));
  if (rows.length !== 2 || rows.some(row => row.supervised_tokens !== data.frozen.supervised_token_budget) || rows[0].initial_sha256 !== rows[1].initial_sha256) throw new RangeError('等预算或同初值证据缺失');
  return rows;
}
export function initMatchedBudget() {
  const root = document.querySelector('[data-matched-budget]'); if (!root) return;
  root.innerHTML = `<div class="practice-controls"><label><span>等预算实验 seed</span><select data-view-key="seed"><option>11</option><option>29</option><option>47</option></select></label></div><div data-budget-results aria-live="polite"></div>`;
  bindExperimentView(root, { prefix: 'budget_', onChange: ({ seed }) => {
    const [short, extended] = matchedRuns(report, seed), names = ['只训练长度 2–5', '训练覆盖长度 2–8'];
    root.querySelector('[data-budget-results]').innerHTML = metrics([['每个方案的监督 token', '410,000'], ['短序列模型更新次数', String(short.updates)], ['扩展支持模型更新次数', String(extended.updates)]]) +
      lineChart([short, extended].map((row, i) => ({ name: names[i], points: row.summary.map(s => ({ x: s.length, y: s.exact_match * 100 })) })), '输入长度', '完整匹配 / %') +
      table(['长度', '题数', '短序列完整匹配', '扩展支持完整匹配', '扩展模型的长度关系'], short.summary.map((s, i) => [s.length, s.records, pct(s.exact_match), pct(extended.summary[i].exact_match), s.length <= 8 ? '支持内 · 未见题目' : '新长度外推']), '同 seed 同初值等监督 token 的完整评估') +
      table(['方案', '监督 token', '更新数', '有效输入位置', '实际计算位置（含 padding）', '末批舍弃标签'], [short, extended].map((r, i) => [names[i], r.supervised_tokens.toLocaleString('en-US'), r.updates, r.input_tokens.toLocaleString('en-US'), r.computed_positions.toLocaleString('en-US'), r.discarded_final_labels]), '相等的监督预算与仍然不同的执行量') +
      `<p class="practice-note">每次都恰好使用 ${report.frozen.supervised_token_budget.toLocaleString('en-US')} 个标签；末批按保存顺序保留剩余标签，其余设为 −100。初值一致、评估组无训练重叠。更新次数、输入位置与 FLOPs 没有对齐，不能把这项对照解释成等计算预算。</p>`;
  } });
  root.dataset.budgetReady = 'true';
}
