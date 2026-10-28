import { retrievalSnapshot as data } from './retrieval-data.js';
import { esc, num, pct, table, metrics } from '../experiment-ui.js';
import { bindExperimentView } from '../experiment-view.js';

const names = { bm25: '字符 / 英文词 BM25', flat: '真实 BGE · Flat', 'ivf-1': '真实 BGE · IVF nprobe 1', 'ivf-4': '真实 BGE · IVF nprobe 4', 'ivf-16': '真实 BGE · IVF nprobe 16', 'rrf-rerank': 'Flat top-10 内 RRF 重排' };
export function retrievalView(report, id, method, split, kind) {
  const query = report.queries.find(row => row.id === id);
  if (!query || !Object.hasOwn(names, method) || !['test', 'validation'].includes(split) || !['all', 'keyword', 'paraphrase'].includes(kind)) throw new RangeError('未知检索实验选项');
  return { query, result: query.methods[method], summary: report.summaries.filter(row => row.split === split && row.slice === kind) };
}
export function initRetrievalProject() {
  const root = document.querySelector('[data-retrieval-project]'); if (!root) return;
  root.innerHTML = `<div class="practice-controls"><label class="retrieval-question"><span>查看某一道已冻结的问题</span><select data-view-key="query">${data.queries.map(row => `<option value="${esc(row.id)}">${row.split === 'test' ? '测试' : '验证'} · ${esc(row.query)}</option>`).join('')}</select></label><label><span>逐题检索方法</span><select data-view-key="method">${Object.entries(names).map(([key, name]) => `<option value="${key}">${name}</option>`).join('')}</select></label><label><span>汇总数据集</span><select data-view-key="split"><option value="test">24 个测试问题</option><option value="validation">12 个验证问题</option></select></label><label><span>汇总问题类型</span><select data-view-key="kind"><option value="all">全部问题</option><option value="keyword">术语型提问</option><option value="paraphrase">换一种说法</option></select></label></div><div data-retrieval-results aria-live="polite"></div>`;
  root.querySelector('[data-view-key="query"]').value = data.queries.find(row => row.split === 'test').id;
  const byId = new Map(data.passages.map(passage => [passage.id, passage]));
  bindExperimentView(root, { onChange: ({ query: id, method, split, kind }) => {
    const view = retrievalView(data, id, method, split, kind), query = view.query;
    root.querySelector('[data-retrieval-results]').innerHTML = `<h3>完整切片汇总</h3>` +
      table(['方法', '问题数', '指定证据 Hit@1', 'Hit@5', 'Hit@10', 'MRR@10', 'Flat 近邻 Recall@10'], view.summary.map(row => [esc(names[row.method]), row.questions, pct(row.metrics.hit_at_1), pct(row.metrics.hit_at_5), pct(row.metrics.hit_at_10), num(row.metrics.mrr_at_10), row.flat_neighbor_recall_at_10 === null ? '不适用' : pct(row.flat_neighbor_recall_at_10)]), '完整问题集合上的实际检索结果') +
      `<h3>逐题核对：${esc(query.query)}</h3><p>${query.split === 'test' ? '测试问题' : '验证问题'} · ${query.kind === 'keyword' ? '术语型提问' : '换一种说法'} · ${esc(names[method])}</p>` +
      metrics([['指定证据 Hit@5', view.result.metrics.hit_at_5 ? '命中' : '未命中'], ['本题 MRR@10', num(view.result.metrics.mrr_at_10)], ['Flat 近邻 Recall@10', view.result.flat_neighbor_recall_at_10 === null ? '不适用' : pct(view.result.flat_neighbor_recall_at_10)]]) +
      `<details open><summary>在排名之前指定的直接证据</summary>${query.gold.map(gold => `<blockquote><p>${esc(gold.evidence)}</p><footer>来源：<a href="${esc(gold.source_document)}.html">${esc(gold.source_document)}</a> · 段落 ${esc(gold.passage_id.slice(0, 12))}</footer></blockquote>`).join('')}<p class="practice-note">标注给出一个可核对的目标证据段落；没有声称其他段落全不相关。表格统计的是找到指定证据的能力，不是穷尽标注下的通用相关性准确率。</p></details>` +
      `<div class="retrieval-passages">` + table(['排名', '保存的实际得分', '段落来源与 ID', '段落原文', '指定证据'], view.result.results.map((row, i) => { const p = byId.get(row.passage_id); return [i + 1, num(row.score), `<a href="${esc(p.source_document)}.html">${esc(p.source_document)}</a><br><code>${esc(p.id.slice(0, 12))}</code>`, esc(p.text), row.designated_evidence ? '是' : '未标注']; }), '实际检索候选及可回读的原始段落') + `</div>` +
      `<p class="practice-note">BM25、余弦和 RRF 分数使用不同单位，不能横向比较分数大小。RRF 只重排 Flat 已召回的 10 个候选，不能找回候选集之外的证据。</p>`;
  } });
  root.dataset.retrievalReady = 'true';
}
