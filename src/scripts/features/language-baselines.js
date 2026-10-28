import { languageBaselines as report } from './language-baselines-data.js';
import { esc, num, pct, table } from '../experiment-ui.js';
import { bindExperimentView } from '../experiment-view.js';

export function baselineComparison(data, split) {
  if (!['train', 'validation', 'test'].includes(split)) throw new RangeError('未知评估划分');
  return [...data.models.map(row => ({ name: row.model, alpha: row.selected_alpha, ...row.evaluation[split] })),
    ...data.transformer.map(row => ({ name: `Transformer · seed ${row.seed}`, alpha: null, ...row.evaluation.learned[split] }))];
}
export function initLanguageBaselines() {
  const root = document.querySelector('[data-language-baselines]'); if (!root) return;
  root.innerHTML = `<div class="practice-controls"><label><span>基线比较的完整划分</span><select data-view-key="split"><option value="test">独立测试文档</option><option value="validation">验证文档</option><option value="train">训练文档</option></select></label></div><div data-baseline-results aria-live="polite"></div>`;
  bindExperimentView(root, { prefix: 'baseline_', onChange: ({ split }) => {
    const rows = baselineComparison(report, split);
    root.querySelector('[data-baseline-results]').innerHTML = table(['模型', '选定平滑 α', 'CE · nats/token', 'Teacher-forced top-1', '有效标签'], rows.map(row => [esc(row.name), row.alpha === null ? '不适用' : num(row.alpha), num(row.ce_nats_per_token), pct(row.teacher_forced_accuracy), row.supervised_tokens.toLocaleString('en-US')]), '相同 tokenizer 与所有有效标签上的计数模型和 Transformer 对照') +
      `<details><summary>平滑参数怎样选出？</summary><p>两个计数模型只拟合训练集；各自在预先固定的 0.01、0.1、1 中选择验证 CE 最低的 α。测试数据不参与选择。</p>${table(['模型', 'α', '验证 CE'], report.models.flatMap(row => row.selection_trace.map(candidate => [esc(row.model), num(candidate.alpha), num(candidate.validation.ce_nats_per_token)])), '全部平滑参数的验证记录')}</details>`;
  } });
  root.dataset.baselinesReady = 'true';
}
