import { paddingInferenceReport as report } from './padding-inference-data.js';
import { esc, num, table, metrics } from '../experiment-ui.js';
import { bindExperimentView } from '../experiment-view.js';

export function paddingView(data, caseId, side, sample, event) {
  const selected = data.frozen.cases.find(row => row.id === caseId);
  if (!selected || !data.frozen.sides.includes(side) || !Number.isInteger(sample) || sample < 0 || sample >= selected.lengths.length || !Number.isInteger(event) || event < 0 || event > data.frozen.suffix_steps) throw new RangeError('未知填充条件、样本或计算阶段');
  const timeline = data.rows.filter(row => row.case === caseId && row.side === side && row.sample === sample), rows = timeline.filter(row => row.event === event);
  if (rows.length !== data.frozen.methods.length || !rows.some(row => row.method === 'correct')) throw new RangeError('完整模型的比较证据缺失');
  return { selected, side, sample, event, rows, timeline, input: data.inputs.find(row => row.case === caseId).records[sample] };
}

const stage = n => n === 0 ? 'Prefill' : `固定后缀续算 ${n}`;
const error = row => Math.max(row.valid_logits_max_error, row.next_logits_max_error, row.kv_max_error);

export function initPaddingInference() {
  const root = document.querySelector('[data-padding-inference]'); if (!root) return;
  root.innerHTML = `<div class="practice-controls"><label><span>同一批原始窗口的切分</span><select data-view-key="case">${report.frozen.cases.map(row => `<option value="${esc(row.id)}">${esc(row.title)}</option>`).join('')}</select></label><label><span>填充方向</span><select data-view-key="side"><option value="right">右填充</option><option value="left">左填充</option></select></label><label><span>查看批内哪条输入</span><select data-view-key="sample"><option value="0">原窗口 1</option><option value="1">原窗口 2</option><option value="2">原窗口 3</option></select></label><label><span>本次计算阶段</span><select data-view-key="event">${Array.from({ length: report.frozen.suffix_steps + 1 }, (_, n) => `<option value="${n}">${stage(n)}</option>`).join('')}</select></label></div><div data-padding-output aria-live="polite"></div>`;
  bindExperimentView(root, { prefix: 'padding_', onChange: ({ case: caseId, side, sample, event }) => {
    const view = paddingView(report, caseId, side, Number(sample), Number(event)), { selected, rows, timeline, input } = view, baseline = rows.find(row => row.method === 'correct');
    const width = Math.max(...selected.lengths) + selected.extra_pad, length = selected.lengths[view.sample];
    const start = side === 'left' ? width - length : 0, marks = Array.from({ length: width }, (_, i) => i >= start && i < start + length ? '实' : '·').join('');
    root.querySelector('[data-padding-output]').innerHTML = metrics([
      ['原始有效前缀 / 本次已喂入的后缀', `${length} / ${view.event}`], ['每层实际占用缓存槽', String(baseline.physical_cache_slots)], ['本条有效缓存位置', String(baseline.valid_cache_slots)],
    ]) + `<p><strong>${stage(view.event)}</strong>：${view.event ? `本次喂入原窗口中的 token ID ${input.continuation[view.event - 1]}，新输入占一个物理槽；使用原始后缀固定所有方案的后续输入。` : `同一 batch 的有效长度为 ${selected.lengths.join(' / ')}，实际宽度 ${width}。PAD 查询行的输出不作为正确目标。`}固定正弦位置 buffer 与原权重一致。</p>` +
      table(['方案', '读 logits 的物理列 / 正确末列', '末有效 token 位置 ID / 预期', '有效位置 logits 最大差', '读取 next logits 最大差', '有效 K/V 最大差', 'Top-1 ID：实际 / 参考', '三项误差判定'], rows.map(row => [
        esc(report.frozen.methods.find(item => item.id === row.method).title), `${row.selected_column} / ${row.last_valid_column}`, `${row.last_valid_position_id} / ${row.expected_logical_position}`, num(row.valid_logits_max_error), num(row.next_logits_max_error), num(row.kv_max_error), `${row.actual_argmax} / ${row.reference_argmax}`, `<span class="${row.within_tolerance ? 'audit-positive' : 'audit-negative'}">${row.within_tolerance ? '本条件容差内一致' : '已出现数值偏差'}</span>`,
      ]), '完整模型在同一输入下的位置、读取列、logits 和每层有效缓存数值对照') +
      '<p class="practice-note">所有列号和位置 ID 均从 0 开始。第一项误差比较本次全部有效 query，第二项比较实际读出的下一位置分布，第三项比较各层全部有效 K/V。取错列可以在前向与 K/V 仍正确时读错 logits；Top-1 相同也可能已有明显数值差异。</p>' +
      `<details><summary>这条输入在哪个阶段首次偏离？</summary>${table(['方案', '首次超过 2e−10 的阶段', '五阶段最大误差', 'Top-1 首次不同'], report.frozen.methods.map(method => { const values = timeline.filter(row => row.method === method.id), first = values.find(row => !row.within_tolerance), argmax = values.find(row => row.actual_argmax !== row.reference_argmax); return [esc(method.title), first ? stage(first.event) : '五阶段均在容差内', num(Math.max(...values.map(error))), argmax ? stage(argmax.event) : '五阶段均相同']; }), '数值首次偏差与 Top-1 首次变化分别核对')}</details>` +
      `<details><summary>核对原窗口、有效列和固定的后缀 token</summary><p>窗口 ID：<code>${esc(input.id)}</code>。实表示有效输入，点号表示填充；续算输入追加在整批已占用的物理槽之后。</p><pre class="decision-code">批宽 ${width}，有效列 ${start}…${start + length - 1}\n${marks}\n原始前缀 ID：${input.prefix.join(' ')}\n固定后缀 ID：${input.continuation.join(' ')}</pre><p>当前正确路径：读取物理列 ${baseline.last_valid_column}，末有效 token 的逻辑位置为 ${baseline.expected_logical_position}；缓存占 ${baseline.physical_cache_slots} 槽，其中本条 ${baseline.valid_cache_slots} 槽有效。物理缓存保留填充槽，未将其压缩掉。</p></details>` +
      '<p class="practice-note">298,176 参数、2 层、4 头、width 96；CPU float64，无 dropout 或参数更新。参考逐条调用原始 TextDecoder 完整前向，并记录实际层 K/V；批处理适配器单独提供逻辑位置、key mask 和缓存。完整模型使用固定正弦位置，不把本结果当作 RoPE 实现验证。四次续算使用固定真实后缀，没有执行自由采样、EOS 停止或上下文滑窗。</p>';
  } });
  root.dataset.paddingReady = 'true';
}
