import { accumulationReport as report } from './accumulation-data.js';
import { esc, num, table, metrics } from '../experiment-ui.js';
import { bindExperimentView } from '../experiment-view.js';

const names = { reference: '完整 batch 参考', 'global-count': '累积：全局标签分母', 'micro-means': '累积：各微批均值等权', 'planned-divisor': '尾窗口：仍除计划步数', 'zero-step': '空监督：零损失仍更新', 'ddp-global': 'DDP：局部总和 × P/N', 'ddp-local-mean': 'DDP：本地均值再平均' };
export function accumulationView(data, caseId, optimizer) {
  const selected = data.frozen.cases.find(row => row.id === caseId);
  if (!selected || !['adamw', 'sgd'].includes(optimizer)) throw new RangeError('未知累积实验条件');
  const rows = data.rows.filter(row => row.case === caseId && row.optimizer === optimizer), reference = rows.find(row => row.method === 'reference');
  if (!reference || reference.global_labels !== selected.keep.reduce((n, x) => n + x, 0)) throw new RangeError('监督分母或完整 batch 证据缺失');
  return { selected, rows, reference };
}
function verdict(row, tolerance) {
  if (!row.global_mean_defined) return row.optimizer_step ? '<strong class="accumulation-warning">无有效目标，仍发生更新</strong>' : '无有效目标，跳过更新';
  const matches = Math.max(row.gradient_max_error, row.parameter_max_error, row.optimizer_state_max_error) <= tolerance;
  return matches ? '<span class="accumulation-match">本条件下容差内一致</span>' : '<strong class="accumulation-warning">已偏离完整 batch 目标</strong>';
}
export function initAccumulation() {
  const root = document.querySelector('[data-accumulation]'); if (!root) return;
  root.innerHTML = `<div class="practice-controls"><label><span>完整模型的比较条件</span><select data-view-key="case">${report.frozen.cases.map(row => `<option value="${esc(row.id)}">${esc(row.title)}</option>`).join('')}</select></label><label><span>一次更新使用的优化器</span><select data-view-key="optimizer"><option value="adamw">AdamW · 恢复第 800 步状态</option><option value="sgd">SGD · 相同权重、全新状态</option></select></label></div><div data-accumulation-output aria-live="polite"></div>`;
  bindExperimentView(root, { prefix: 'accum_', onChange: ({ case: caseId, optimizer }) => {
    const { selected, rows, reference } = accumulationView(report, caseId, optimizer), labels = reference.global_labels;
    const rankCounts = [0, 1].map(rank => selected.keep.filter((_, i) => i % 2 === rank).reduce((n, value) => n + value, 0));
    const single = rows.some(row => row.execution === 'single' && row.method !== 'reference');
    const countMetric = single ? ['单进程实际 / 计划微批', `${selected.chunks.length} / ${selected.planned}`] : ['DDP 每副本实际微批', String(rows.find(row => row.execution === 'ddp').backward_calls)];
    root.querySelector('[data-accumulation-output]').innerHTML = metrics([['本次更新的有效标签', String(labels)], ['完整 batch 的 token 平均 CE', reference.global_mean_defined ? num(reference.scaled_loss) : '无定义 · 跳过'], countMetric]) +
      `<div class="accumulation-masks" aria-label="输入位置与监督标签的数量">${selected.keep.map((count, i) => `<div><span>记录 ${i + 1} · 96 个输入 / ${count} 个标签</span><div class="accumulation-mask" role="img" aria-label="记录 ${i + 1}：${96 - count} 个目标被忽略，末 ${count} 个位置提供监督"><i style="width:${(count / 96 * 100).toFixed(6)}%"></i></div></div>`).join('')}</div><p class="practice-note">灰色为不计 loss 的位置，蓝色为监督后缀。输入仍完整参与计算；标签在原始 shifted targets 上屏蔽，不再移位，也不是根据输入长度计算分母。</p>` +
      table(['方案', '用于反传的标量', '梯度最大差', '更新后参数最大差', '优化器状态最大差', '实际更新 / Adam 步数', '判定'], rows.map(row => [esc(names[row.method]), row.scaled_loss == null ? row.execution === 'ddp' ? '各 rank 独立记录' : '未计算' : `${num(row.scaled_loss)}${!row.global_mean_defined ? '（零替代量，非均值）' : ''}`, row.backward_calls ? num(row.gradient_max_error) : '未反传', num(row.parameter_max_error), num(row.optimizer_state_max_error), `${row.optimizer_step ? '更新一次' : '跳过'}${row.adam_step_before == null ? '' : `<br>${row.adam_step_before} → ${row.adam_step_after}`}`, verdict(row, report.frozen.tolerance)]), '完整 Transformer 在不同归一化和执行路径下的真实梯度、参数与优化器状态对照') +
      (selected.ddp ? `<p>实际运行两个 CPU Gloo 进程，本条件两副本分别含 ${rankCounts[0]} 和 ${rankCounts[1]} 个监督标签。DDP 默认对梯度平均；正确路径在局部总和上乘 2/${labels}。空副本仍参与相同顺序的通信，其局部损失总和为零；全局分母由所有副本共同统计。</p><details><summary>展开两进程的实际计数、系数与同步记录</summary>${table(['方案', 'rank 0 微批标签', 'rank 0 的局部系数', '是否使用 no_sync', '两 rank 梯度差'], rows.filter(row => row.execution === 'ddp').map(row => [esc(names[row.method]), row.micro_counts.join(' / '), row.loss_scales.map(num).join(' / '), row.contexts.map(esc).join('<br>'), num(row.peer_gradient_max_error)]), '真实 Gloo 记录的计数和默认 DDP 缩放')}<p class="practice-note">两个 rank 各自生成梯度、参数与优化器状态的哈希，再通过 Gloo 交换核对。梯度向量分别保存；参数/状态字节一致时只存一份，并保留两边的实际哈希见证。前向和反向都放在 no_sync 范围内，最后微批触发同步。</p></details>` : `<p>本条件仅做单进程累积对照。双进程记录覆盖“不等量监督”和“一个副本没有监督”，可在上方切换。</p>`) +
      `<details><summary>核对原始窗口、监督后缀和真实梯度比例</summary>${table(['保存的窗口 ID', '末 12 个原始目标 ID', '末 12 个实际标签', '保留标签数'], report.inputs.records.slice(0, selected.keep.length).map((r, i) => { const n = selected.keep[i], masked = r.target.map((id, pos) => pos >= r.target.length - n ? id : -100); return [esc(r.id), `<code>${esc(r.target.slice(-12).join(' '))}</code>`, `<code>${esc(masked.slice(-12).join(' '))}</code>`, n]; }), '同一批原始目标和未重复移位的监督 mask')}${table(['方案', '相对参考的梯度 L2 范数比', '实际微批标签计数', '用于反传的缩放系数'], rows.map(row => [esc(names[row.method]), row.gradient_l2_ratio == null ? '参考梯度范数为零' : num(row.gradient_l2_ratio), row.micro_counts.join(' / '), row.loss_scales.map(num).join(' / ') || '跳过']), '归一化系数与已保存完整梯度的范数比')}</details>` +
      `<p class="practice-note">2 层、width 96、298,176 参数；CPU float64、无 dropout/AMP/梯度裁剪。误差覆盖全部参数，正确路径容差 2e−10。梯度差比较 CE 的导数，衰减与历史状态纳入更新检查，梯度比例不能直接当成参数位移比例。等量条件下“均值等权”也可能与参考一致；这不保证不等量监督或尾窗口仍正确。N=0 时没有定义 token 均值，跳过路径保存的零梯度向量只作比较占位。</p>`;
  } });
  root.dataset.accumulationReady = 'true';
}
