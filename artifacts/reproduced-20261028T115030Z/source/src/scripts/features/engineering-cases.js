import { $, $$ } from "../core.js";

export const diagnosticCases = {
  parallel: {
    domain: "OPTIMIZATION / DISTRIBUTED", title: "同一 batch 拆成两份，梯度从 +0.4 变成 −0.5",
    trigger: "权重、样本集合、mask 和精度固定，关闭随机性；单卡与两个 DP 副本的第一步方向不同。尚无任何长程收敛证据。",
    options: [
      { title: "按副本列出有效 token 数与 loss 分母", feedback: "这能直接验证目标权重是否随切分改变，优先级最高。" },
      { title: "把学习率调低十倍", feedback: "能缩小位移，却不能解释或修复同一目标的梯度反号。先核对分母。" },
      { title: "更换通信硬件", feedback: "通信错误是候选假设，但应先核对可精确验证的数学结果；换硬件引入更多变量。" },
    ],
    evidence: [
      ["有效数量", "A 有 2 个有效 token，每 token 梯度 −2；B 有 8 个，每 token 梯度 +1。两卡都用 local mean。"],
      ["规约语义", "DDP 对两卡梯度求平均，得到 (−2+1)/2=−0.5。单卡全局 token 平均为 (−4+8)/10=+0.4。"],
      ["区分验证", "只把局部 loss 改成 P·Sᵣ/N，其他配置不变，聚合梯度恢复 +0.4；再测试相等长度与空标签副本。"],
    ],
    conclusion: "给定证据支持“副本等权替代了 token 等权”。修正归一化后验证梯度、参数更新与真实训练趋势；后续运行仍需监控数值稳定性。",
    link: "parallel-correctness.html#ddp-scaling", next: "查看全局分母推导",
  },
  data: {
    domain: "DATA / GENERALIZATION", title: "筛选后平均质量分提高，少数语言与代码却退化",
    trigger: "新配方提高质量阈值，固定训练 token 预算后主验证集略涨，但长尾测试明显下滑。还需核验评分器对事实性和数据覆盖的判断。",
    options: [
      { title: "对比各来源保留率、唯一量与重复次数", feedback: "这能区分提纯与覆盖损失，并发现高阈值是否让少数剩余样本反复出现。" },
      { title: "再提高过滤阈值", feedback: "可能进一步缩窄分布；没有分来源证据前，不能把质量分当作能力代理。" },
      { title: "只看整体平均 loss", feedback: "平均值可能被主流来源主导；长尾退化需要切片和数据保留证据。" },
    ],
    evidence: [
      ["过滤分布", "构造记录：教育数据保留 82%，代码 35%，少数语言 12%。同一总 token 预算下，教育来源的重复次数增加。"],
      ["抽样复核", "部分被删代码因符号密度高被评分器误判；少数语言评分不可靠。保留数据的来源覆盖明显变窄。"],
      ["消融设计", "固定预算/初始化族，比较旧配方、新配方、按语言/来源校准阈值并保留重复上限的配方。对目标和长尾切片做配对评估，根据切片结果判断修正效果。"],
    ],
    conclusion: "证据支持过滤改变了训练分布，但收益和损害的大小仍需训练消融。应控制唯一量、采样权重和重复预算，并独立验证；以独立任务结果检验评分器带来的收益。",
    link: "data-tokenizer.html#mixture-optimization", next: "查看数据配方与消融",
  },
  moe: {
    domain: "ARCHITECTURE / SYSTEMS", title: "MoE 激活参数更少，step 却从 80 ms 变成 100 ms",
    trigger: "两次配置的工作负载和计时窗口固定。这里比较时间分解，不声称模型质量相等；激活参数只描述一部分计算成本。",
    options: [
      { title: "查看每个 rank 的路由负载和关键路径", feedback: "可以区分 expert 计算、all-to-all、尾部等待与小矩阵效率，避免只看平均算力。" },
      { title: "根据激活参数比例直接推导加速比", feedback: "这种估算遗漏了通信、路由、负载和矩阵形状，需要补全执行时间分析。" },
      { title: "增加专家总数", feedback: "可能进一步分散每个 expert 的 token 并增加状态成本；先定位时间消耗。" },
    ],
    evidence: [
      ["关键路径账本", "人为记录：dense 的计算 70 ms + 暴露等待 10 ms；MoE 的计算 45 ms + 暴露等待 55 ms。时间项按关键路径分段，未把重叠通信重复相加。"],
      ["rank 与形状", "最忙 expert 的 token 是均值的 2.8 倍；部分 expert 的 GEMM 很小，all-to-all 和尾部等待暴露。平均 expert 负载掩盖了最慢 rank。"],
      ["区分实验", "固定模型和 token，分别检查更均衡路由、grouped GEMM 与拓扑/通信重叠。若改变路由容量或丢弃规则，还要验证训练目标与质量。"],
    ],
    conclusion: "该记录中，减少的计算被更多暴露等待抵消。需要优化实际关键路径，并在相同工作负载下测量；不同 batch、拓扑和质量约束下结论可能变化。",
    link: "performance-analysis.html#execution-path", next: "查看关键路径与性能口径",
  },
  reward: {
    domain: "POST-TRAINING / EVALUATION", title: "奖励更高，思考更长，隐藏测试的正确率没涨",
    trigger: "新策略在训练 judge 上表现更好，但固定题集的独立验证无改善，输出 token 增加。还不能据此判断模型“只学会了格式”或“完全没提升”。",
    options: [
      { title: "按长度/任务切片，用独立验证器比较", feedback: "可以区分长度偏好、判分漏洞和真实能力收益；同时控制总推理预算。" },
      { title: "继续奖励更长的推理", feedback: "奖励长度可能使策略偏向更长的表述，任务正确性需另外验证。" },
      { title: "仅检查训练 reward 曲线是否平滑", feedback: "平滑曲线有助于观察优化过程，收益还要由独立验证和成本比较确认。" },
    ],
    evidence: [
      ["奖励切片", "控制答案正确性后，训练 judge 仍偏爱更长的格式化解释；自动测试发现部分高分轨迹有错误中间步骤。"],
      ["预算对齐", "分别固定输出 token 上限和总采样/评分预算评估，新策略的总体正确率差值区间覆盖零；某些题族仍可能有不同表现。"],
      ["后续验证", "用独立且不泄漏答案的检查器复核，做长度偏置消融、校准评分规则，并在新数据上重测。改变 reward 后不沿用原评分器作为唯一最终裁判。"],
    ],
    conclusion: "当前证据不足以声称泛化推理提高，并支持调查奖励偏差。应继续报告各题族的结果、区间与预算，再决定训练调整。",
    link: "reasoning-compute.html#reasoning-evidence", next: "查看推理能力的证据边界",
  },
};

export function initEngineeringCases() {
  const select = $("[data-diagnosis-case]"); if (!select) return;
  let revealed = 0, choice = null;
  const render = () => {
    const data = diagnosticCases[select.value];
    $("[data-diagnosis-domain]").textContent = data.domain;
    $("[data-diagnosis-title]").textContent = data.title;
    $("[data-diagnosis-trigger]").textContent = data.trigger;
    $("[data-diagnosis-feedback]").textContent = choice === null ? "尚未选择第一项观测。" : data.options[choice].feedback;
    $("[data-diagnosis-evidence]").innerHTML = data.evidence.slice(0, revealed).map(([title, text]) => `<li><strong>${title}</strong><p>${text}</p></li>`).join("");
    const done = revealed === data.evidence.length;
    $("[data-diagnosis-conclusion]").hidden = !done;
    $("[data-diagnosis-conclusion]").innerHTML = done ? `<strong>当前能支持的判断</strong><p>${data.conclusion}</p><a href="${data.link}">${data.next}</a>` : "";
    $("[data-diagnosis-next]").disabled = choice === null || done;
    $("[data-diagnosis-next]").textContent = done ? "证据已展开" : revealed ? "查看下一条证据" : "查看第一条证据";
  };
  const reset = () => {
    revealed = 0; choice = null;
    $("[data-diagnosis-options]").innerHTML = diagnosticCases[select.value].options.map((option, i) => `<label><input type="radio" name="diagnosis-evidence" value="${i}"><span>${option.title}</span></label>`).join("");
    $$("[name=diagnosis-evidence]").forEach(input => input.addEventListener("change", () => { choice = Number(input.value); render(); }));
    render();
  };
  select.addEventListener("change", reset);
  $("[data-diagnosis-reset]").addEventListener("click", reset);
  $("[data-diagnosis-next]").addEventListener("click", () => { revealed = Math.min(diagnosticCases[select.value].evidence.length, revealed + 1); render(); });
  reset();
}
