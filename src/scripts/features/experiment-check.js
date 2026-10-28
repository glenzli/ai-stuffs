import { esc } from '../experiment-ui.js';

export const experimentQuestions = {
  eos: { prompt: '如果只禁止提前输出 EOS，是否就能保证更长输入的答案正确？', choices: ['能，输出长度正确就等于内容正确', '不能，内容与顺序仍可能错误', '能直接证明位置编码是唯一原因'], answer: 1, why: '结束策略只约束某个 token 的出现时机。本页保存的诊断消除了提前结束，却没有恢复长序列的完整匹配；还需要检查内容和训练支持。' },
  baseline: { prompt: '测试 CE 优于 bigram 基线，这份结果能直接支持哪项判断？', choices: ['模型已能流畅、可靠地写中文', '在同一 tokenizer 和测试标签上，预测损失更低', '模型已经通过指令微调评估'], answer: 1, why: '比较固定了预测单位和评估集，支持下一 token 预测更好。自由生成、事实可靠性和指令遵循仍需各自的证据。' },
  budget: { prompt: '两次训练都使用 410,000 个监督 token，能否称为相同计算预算？', choices: ['能，所有 token 的计算成本相同', '能，只要 seed 也相同', '不能，还要核对上下文长度、padding 和更新次数'], answer: 2, why: '标签数量只固定了监督分母。本页同时保存更新数、有效输入和含 padding 的计算位置；这些量仍不同，也不能据它们直接当作相同 FLOPs。' },
  reproduction: { prompt: '初始化 seed 相同，但数据顺序变了，还应要求逐步参数完全一致吗？', choices: ['不应该，先对齐实际输入和已保存状态', '应该，seed 决定全部后续状态', '只要最终 loss 接近就算逐值复现'], answer: 0, why: '实际输入已经是不同的计算条件。应先对齐初值、数据和优化器状态，再逐边界比较；最终指标接近也不等于相同数值轨迹。' },
  profile: { prompt: '某个算法减少反向图保存的引用，能否据此断言完整 step 更快？', choices: ['不能，要实测整步并检查数值目标一致', '能，少保存一定减少总时间', '能，把局部阶段的中位数相加即可'], answer: 0, why: '保存引用、物理内存峰值和时间是不同指标。重算、循环与其他阶段仍有成本；同一次 step 的阶段可相加，来自不同样本的中位数不必相加。' },
  retrieval: { prompt: '相关证据没有进入召回的前 10 条，随后只重排这 10 条能找回它吗？', choices: ['能，重排会搜索整个语料库', '不能，候选集的覆盖率限制了重排上限', '只要换成神经网络重排就能保证找回'], answer: 1, why: '重排改变候选的次序，不产生候选集外的新文档。本实验保存 RRF 前后的完整候选 ID，可以直接验证这个边界。' },
};
export function gradeExperimentAnswer(id, choice) {
  if (!Object.hasOwn(experimentQuestions, id) || !Number.isInteger(choice) || choice < 0 || choice >= experimentQuestions[id].choices.length) throw new RangeError('未知预测题或选项');
  return { correct: choice === experimentQuestions[id].answer, explanation: experimentQuestions[id].why };
}
export function initExperimentCheck() {
  for (const root of document.querySelectorAll('[data-experiment-check]')) {
    const id = root.dataset.experimentCheck, question = experimentQuestions[id]; if (!question) throw new RangeError('未知预测题');
    root.classList.add('experiment-question');
    root.innerHTML = `<p class="micro-label">先做一个判断，再看结果</p><fieldset><legend>${esc(question.prompt)}</legend>${question.choices.map((text, i) => `<label><input type="radio" name="predict-${esc(id)}" value="${i}"><span>${esc(text)}</span></label>`).join('')}</fieldset><button type="button" data-reveal-prediction disabled>查看判断依据</button><button type="button" data-reset-prediction hidden>重新作答</button><div data-prediction-feedback role="status" aria-live="polite" hidden></div>`;
    let choice = null;
    const reveal = root.querySelector('[data-reveal-prediction]'), reset = root.querySelector('[data-reset-prediction]'), feedback = root.querySelector('[data-prediction-feedback]');
    for (const input of root.querySelectorAll('input')) input.addEventListener('change', () => { choice = +input.value; reveal.disabled = false; });
    reveal.addEventListener('click', () => {
      const result = gradeExperimentAnswer(id, choice); feedback.hidden = false; feedback.dataset.correct = String(result.correct);
      feedback.innerHTML = `<strong>${result.correct ? '这项判断正确。' : '再核对比较条件。'}</strong><p>${esc(result.explanation)}</p>`;
      reveal.disabled = true; reset.hidden = false; for (const input of root.querySelectorAll('input')) input.disabled = true;
    });
    reset.addEventListener('click', () => { choice = null; feedback.hidden = true; reset.hidden = true; reveal.disabled = true; for (const input of root.querySelectorAll('input')) { input.checked = false; input.disabled = false; } });
    root.dataset.predictionReady = 'true';
  }
}
