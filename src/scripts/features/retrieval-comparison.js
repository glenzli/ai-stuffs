import { retrievalComparison as report } from './retrieval-comparison-data.js';
import { esc, num, pct, table, metrics } from '../experiment-ui.js';
import { bindExperimentView } from '../experiment-view.js';

const names = { bm25: 'BM25 单路', flat: 'BGE Flat 单路', hybrid: '两路并集 · 截断 RRF' };
const metricNames = { hit_at_10: '指定证据 Hit@10', mrr_at_10: '指定证据 MRR@10' };
const pp = x => `${x > 0 ? '+' : ''}${(100 * x).toFixed(2)} pp`;
export function retrievalComparisonView(data, split, baseline, metric, id) {
  if (!['test', 'validation'].includes(split) || !['bm25', 'flat'].includes(baseline) || !Object.hasOwn(metricNames, metric)) throw new RangeError('未知混合检索比较');
  const query = data.queries.find(row => row.id === id);
  const summary = data.summaries.find(row => row.split === split && row.kind === 'all');
  const comparison = data.comparisons.find(row => row.split === split && row.baseline === baseline && row.metric === metric);
  if (!query || !summary || !comparison) throw new RangeError('混合检索证据缺失');
  return { query, summary, comparison };
}
export function initRetrievalComparison() {
  const root = document.querySelector('[data-retrieval-comparison]'); if (!root) return;
  const select = (key, label, options) => `<label><span>${esc(label)}</span><select data-view-key="${key}">${options.map(([value, text]) => `<option value="${esc(value)}">${esc(text)}</option>`).join('')}</select></label>`;
  root.innerHTML = `<div class="practice-controls">${select('split', '混合比较的数据集', [['test', '测试：24 题 / 12 来源组'], ['validation', '验证：12 题 / 6 来源组']])}${select('baseline', '混合方案的比较基线', [['flat', names.flat], ['bm25', names.bm25]])}${select('metric', '配对比较指标', [['mrr_at_10', metricNames.mrr_at_10], ['hit_at_10', metricNames.hit_at_10]])}${select('query', '查看混合候选的原始问题', report.queries.map(row => [row.id, `${row.split === 'test' ? '测试' : '验证'} · ${row.query}`]))}</div><div data-hybrid-output aria-live="polite"></div>`;
  root.querySelector('[data-view-key="query"]').value = report.queries.find(row => row.split === 'test').id;
  const passages = new Map(report.passages.map(row => [row.id, row]));
  bindExperimentView(root, { prefix: 'hybrid_', onChange: ({ split, baseline, metric, query: id }) => {
    const view = retrievalComparisonView(report, split, baseline, metric, id), s = view.summary, q = view.query, c = view.comparison;
    root.querySelector('[data-hybrid-output]').innerHTML = metrics([['去重候选覆盖上限', pct(s.candidate_coverage)], ['混合排序后 Hit@10', pct(s.methods.hybrid.hit_at_10)], ['每题候选平均数', num(s.mean_candidates)]]) +
      table(['方案', 'Hit@1', 'Hit@5', 'Hit@10', 'MRR@10'], Object.entries(s.methods).map(([key, row]) => [names[key], pct(row.hit_at_1), pct(row.hit_at_5), pct(row.hit_at_10), num(row.mrr_at_10)]), '同一批问题的单路与混合召回结果') +
      `<p>仅 BM25 找到指定证据的题目 ${s.bm25_only_evidence} 道；仅 Flat 找到的 ${s.flat_only_evidence} 道。证据已进入并集、却在最终前十条中丢失的题目 ${s.coverage_lost_after_ranking} 道。并集覆盖是本题集上的上限，排序后未必全部保留。</p><h3>${esc(metricNames[metric])}：混合方案相对 ${esc(names[baseline])}</h3>` +
      table(['重采样单位', '单位数', '同题平均差', '95% percentile 区间', '赢 / 输 / 平'], ['group', 'row'].map(mode => { const a = c.analyses[mode]; return [mode === 'group' ? '来源组 · 整组共同抽取' : '逐题 · 额外假设各题独立', a.units, pp(a.estimate), a.interval.map(pp).join(' … '), `${a.wins} / ${a.losses} / ${a.ties}`]; }), '相同配对数据按题和按来源组重采样的比较') +
      `<p class="practice-note">固定 2,000 次、LCG seed=73、线性分位。本题集已在此前查看，本轮是探索性机制比较；区间不覆盖调参、标注偏差或部署分布变化。来源组更少时区间可能更宽或更窄，不能靠选择抽样方式追求较窄的结果。</p><p><a href="paired-evaluation.html?dataset=retrieval-comparison#pe-lab">在配对评估工具中复算这批真实测试题 →</a>（固定 Flat → 混合，MRR@10）</p>` +
      `<details><summary>查看本题全部 ${q.candidate_count} 条去重候选及两路名次</summary><p>${esc(q.query)} · ${q.split === 'test' ? '测试' : '验证'}问题；名次超过 10 的候选只用于核对并集，最终输出取前十。</p>` +
      table(['混合名次', 'BM25 / Flat 名次', '融合分数', '段落与原文', '指定证据'], q.candidates.map((row, index) => { const p = passages.get(row.passage_id); return [index + 1, `${row.bm25_rank ?? '未入前十'} / ${row.flat_rank ?? '未入前十'}`, num(row.score), `<details><summary>${esc(p.source_document)} · ${esc(p.id.slice(0, 12))}</summary><p>${esc(p.text)}</p></details>`, row.designated_evidence ? '<strong class="audit-positive">是</strong>' : '未标注']; }), '混合召回的完整候选、实际名次和原始段落') + `</details><p class="practice-note">每路各取十条；融合只使用进入该路前十的名次，未进入贡献为零。两路请求和候选数量均不同于单路，尚未测编码成本或服务延迟。</p>`;
  } });
  root.dataset.hybridReady = 'true';
}
