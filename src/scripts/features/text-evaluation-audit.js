import { textEvaluationAudit as report } from './text-evaluation-audit-data.js';
import { esc, num, pct, table, metrics } from '../experiment-ui.js';
import { bindExperimentView } from '../experiment-view.js';

export function textAuditView(data, seed, split) {
  if (!['train', 'validation', 'test'].includes(split)) throw new RangeError('未知文档划分');
  const run = data.runs.find(row => String(row.seed) === String(seed) && row.split === split);
  if (!run || !run.documents.length || run.documents.reduce((n, row) => n + row.tokens, 0) !== run.supervised_tokens) throw new RangeError('逐文档评价证据不完整');
  return run;
}
const gain = x => `<span class="${x > 0 ? 'audit-positive' : x < 0 ? 'audit-negative' : ''}">${x > 0 ? '+' : ''}${num(x)}</span>`;
export function initTextEvaluationAudit() {
  const root = document.querySelector('[data-text-evaluation-audit]'); if (!root) return;
  const titles = new Map([...document.querySelectorAll('nav[aria-label="全部专题"] a')].map(link => [link.getAttribute('href'), link.textContent.trim()]));
  root.innerHTML = `<div class="practice-controls"><label><span>逐文档检查的模型 seed</span><select data-view-key="seed"><option>11</option><option>29</option><option>47</option></select></label><label><span>逐文档检查的数据集</span><select data-view-key="split"><option value="test">8 篇独立测试文档</option><option value="validation">6 篇验证文档</option><option value="train">24 篇训练文档</option></select></label></div><div data-document-output aria-live="polite"></div>`;
  bindExperimentView(root, { prefix: 'document_', onChange: ({ seed, split }) => {
    const run = textAuditView(report, seed, split);
    root.querySelector('[data-document-output]').innerHTML = metrics([['逐 token 微平均 CE', num(run.micro_ce)], ['逐文档宏平均 CE', num(run.macro_ce)], ['优于 bigram 的文档', `${run.improved_documents} / ${run.documents.length}`]]) +
      table(['文档', '有效标签 / 微平均权重', 'bigram CE', 'Transformer CE', 'CE 改善 · 正数更好'], run.documents.map(row => [`<a href="${esc(row.document)}.html">${esc(titles.get(row.document + '.html') || row.document)}</a>`, `${row.tokens.toLocaleString('en-US')} / ${pct(row.token_share)}`, num(row.bigram_ce), num(row.ce_nats_per_token), gain(row.gain_nats_per_token)]), '全部文档的有效标签、损失和固定基线差值') +
      table(['模型', 'token 微平均 CE', '文档宏平均 CE'], [['bigram', num(run.bigram_micro_ce), num(run.bigram_macro_ce)], [`Transformer seed ${esc(seed)}`, num(run.micro_ce), num(run.macro_ce)]], '同一批预测在两种文档权重下的总体损失') +
      `<p>逐文档配对后，保留 token 分母的平均 CE 改善为 ${gain(run.paired_gain.estimate)} nats/token；按文档整组重采样的 95% percentile 区间为 ${run.paired_gain.interval.map(num).join(' … ')}。本次只有 ${run.documents.length} 个来源文档，不能把 ${run.supervised_tokens.toLocaleString('en-US')} 个相互依赖的标签当成同样数量的独立样本。</p>` +
      `<p class="practice-note">全部 seed 分开报告，未把它们拼成更多独立文档。窗口、位置、padding 和 EOS 标签沿用原评估；本次仅从同一批保存权重重新评分。与原整体 CE 的最大差为 ${num(run.prior_aggregate_error)}，来自逐位置累计与分批归约的浮点差异，核验阈值为 2e−6。</p>`;
  } });
  root.dataset.documentAuditReady = 'true';
}
