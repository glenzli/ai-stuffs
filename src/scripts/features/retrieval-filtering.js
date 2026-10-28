import { retrievalFilteringReport as report } from './retrieval-filtering-data.js';
import { esc, num, pct, table, metrics } from '../experiment-ui.js';
import { bindExperimentView } from '../experiment-view.js';

export function filteringView(data, scope, queryId, depth) {
  const condition = data.frozen.filters.find(row => row.id === scope), queryIndex = data.queries.findIndex(row => row.id === queryId);
  if (!condition || queryIndex < 0 || !data.frozen.pool_depths.includes(depth)) throw new RangeError('未知过滤条件、问题或候选深度');
  const request = data.requests.find(row => row.filter_id === scope && row.query_index === queryIndex), selected = request?.post.find(row => row.depth === depth);
  if (!selected || request.reference.length !== Math.min(data.frozen.k, request.allowed_count)) throw new RangeError('过滤参考或实际结果缺失');
  return { condition, query: data.queries[queryIndex], request, selected, summaries: data.summaries.filter(row => row.filter_id === scope) };
}

const evidence = value => value == null ? '无定义：指定证据在范围外' : value ? '命中指定证据' : '未命中指定证据';
const ratio = value => value == null ? '无定义：参考集合为空' : pct(value);

export function initRetrievalFiltering() {
  const root = document.querySelector('[data-retrieval-filtering]'); if (!root) return;
  root.innerHTML = `<div class="practice-controls"><label><span>对所有问题使用同一来源过滤</span><select data-view-key="scope">${report.frozen.filters.map(row => `<option value="${esc(row.id)}"${row.id === 'artifacts' ? ' selected' : ''}>${esc(row.title)}</option>`).join('')}</select></label><label class="retrieval-question"><span>查看原始问题</span><select data-view-key="query">${report.queries.map(row => `<option value="${esc(row.id)}"${row.id === 'resume-keyword' ? ' selected' : ''}>${row.split === 'test' ? '测试' : '验证'} · ${esc(row.query)}</option>`).join('')}</select></label><label><span>后过滤之前读取全库前 L 条</span><select data-view-key="depth">${report.frozen.pool_depths.map(n => `<option value="${n}">${n === report.passages.length ? `${n}（全库完整对照）` : n}</option>`).join('')}</select></label></div><div data-filtering-output aria-live="polite"></div>`;
  bindExperimentView(root, { prefix: 'filter_', onChange: ({ scope, query, depth }) => {
    const view = filteringView(report, scope, query, Number(depth)), { condition, request, selected, summaries } = view;
    const target = request.reference.length, returned = new Set(selected.results.map(row => row.i));
    const gold = new Set(view.query.gold.map(row => row.passage_id));
    root.querySelector('[data-filtering-output]').innerHTML = metrics([
      ['当前范围的实际段落数 M', String(request.allowed_count)], ['范围内可返回的目标 min(10,M)', String(target)], ['后过滤的实际返回 / 可返回目标', `${selected.count} / ${target}`],
    ]) + `<p><strong>${esc(view.query.query)}</strong></p><p>当前范围：${esc(condition.title)}。该题预先指定 ${view.query.gold.length} 段证据，其中 ${request.eligible_gold_count} 段在当前范围内。${request.eligible_gold_count ? '下面的命中只核对这份指定证据。' : '原标注已被范围排除，指定证据命中率显示无定义；这不等于证明其他段落都不能回答。'}</p>` +
      `<p class="practice-verdict">对同一范围精确近邻的覆盖：${ratio(selected.neighbor_recall)}。${evidence(selected.evidence_hit)}。${request.allowed_count === 0 ? '空范围不伪造候选或填成 0% 召回。' : selected.count < target ? '当前候选池过滤后不足可返回数量；范围本身仍有可补回的段落。' : request.allowed_count < report.frozen.k ? '范围本身少于 10 段，本次已经返回全部可用段落。' : '本条件已覆盖范围内前十；指定证据命中仍由标注和排序决定。'}</p>` +
      table(['全库候选 L', '候选池内符合范围', '实际返回 / 可返回', '范围内精确近邻覆盖', '本题指定证据'], request.post.map(row => [row.depth, row.pool_eligible, `${row.count} / ${target}`, ratio(row.neighbor_recall), evidence(row.evidence_hit)]), '同一个问题在四种候选深度下的实际过滤结果') +
      (target ? `<div class="retrieval-passages">${table(['范围内名次', '范围内精确参考：来源与内积', '全库原名次与后过滤结果', '保存的真实原文'], request.reference.map((row, i) => {
        const p = report.passages[row.i], kept = returned.has(row.i);
        return [i + 1, `${esc(p.source_document)} · 第 ${p.paragraph + 1} 段<br>${num(row.score)}${gold.has(p.id) ? '<br><strong>预先指定证据</strong>' : ''}`, `全库第 ${row.global_rank} 名<br><span class="${kept ? 'audit-positive' : 'audit-negative'}">${kept ? '后过滤已返回' : '候选池外，未返回'}</span>`, `<details><summary>展开原文与完整 ID</summary><p>${esc(p.text)}</p><code>${esc(p.id)}</code></details>`];
      }), '范围内前十参考、原始全库名次和后过滤是否保留')}</div>` : '<p>当前范围没有任何段落，两种方案均返回空集合。</p>') +
      `<details><summary>同一范围下全部 36 道问题的结果与分母</summary><p>过滤条件不随问题或标注改变。${summaries[0].evidence_eligible_queries} / ${summaries[0].queries} 道题在此范围内仍有指定证据；条件命中率只使用这些题。全部 36 题都进入候选数量检查。补足比例的分母为 min(10,M)，空范围时无定义。</p>${table(['全库候选 L', '结果不足目标的题数 / 全部题', '平均补足比例', '命中题 / 有可用证据的题', '条件指定证据 Hit@10'], summaries.map(row => [row.depth, `${row.shortfall_queries} / ${row.queries}`, ratio(row.mean_fill), `${row.evidence_hits} / ${row.evidence_eligible_queries}`, row.evidence_hit_rate == null ? '无定义：没有可用指定证据的题' : pct(row.evidence_hit_rate)]), '同一固定来源过滤下的全部问题，不混用可用证据分母')}</details>` +
      '<p class="practice-note">内积来自原 BGE 编码后的真实向量，统一转 float64，按分数降序、完整段落哈希升序排列。这里使用全库精确顺序隔离过滤时机；没有运行新的 ANN 过滤接口或测延迟。扩大 L 在本实验中逐步补回范围内近邻，不能据此保证任意 ANN 实现的召回或吞吐。</p>';
  } });
  root.dataset.filteringReady = 'true';
}
