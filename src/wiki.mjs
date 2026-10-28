export const wikiGroups = [
  { title: "从这里开始", shortTitle: "指南", nav: "training-engineer-guide.html", pages: [
    ["training-engineer-guide", "学习指南与问题定位"], ["deep-dive", "六步学习路线"], ["index", "一次学习的全貌"], ["engineering-check", "自测与定位"],
  ] },
  { title: "数据与表示", shortTitle: "数据与表示", nav: "data-tokenizer.html", pages: [
    ["data-tokenizer", "数据与 Tokenizer"], ["packing-correctness", "样本打包与监督边界"],
    ["representation-deep", "表示几何"], ["representation", "Embedding 演化"],
  ] },
  { title: "模型结构", shortTitle: "模型结构", nav: "transformer-lab.html", pages: [
    ["transformer-lab", "从 token 到 logits"], ["transformer-anatomy", "权重与参数"], ["architecture-variants", "架构变体"],
    ["transformer-evolution", "Transformer 演化"], ["model-case-studies", "公开模型案例"],
  ] },
  { title: "训练目标与优化", shortTitle: "目标与优化", nav: "training.html", pages: [
    ["training", "训练回路"], ["learning-run", "小模型学习与泛化"], ["generalization-analysis", "长度泛化失败诊断"],
    ["text-training-project", "自然语言小模型项目"], ["training-deep", "损失与梯度"], ["backpropagation", "整层反向传播"],
    ["optimization-dynamics", "优化与训练稳定性"], ["initialization-width", "初始化与宽度迁移"], ["batch-noise", "Batch 与梯度噪声"],
    ["simulator", "对比学习二维模拟"], ["continual-learning", "持续学习与任务干扰"], ["scaling-laws", "Scaling 与预算"],
  ] },
  { title: "训练系统与性能", shortTitle: "系统与性能", nav: "engineering.html", pages: [
    ["engineering", "工程全流程"], ["numerical-computing", "浮点与数值稳定"], ["tensor-layout", "张量布局与存储视图"],
    ["training-systems-deep", "并行与数值精度"], ["distributed-execution", "多卡张量与通信"], ["parallel-correctness", "并行计算正确性"],
    ["context-parallel", "上下文并行与梯度归还"], ["attention-kernels", "分块 Attention 与反传"],
    ["activation-checkpointing", "激活重算与梯度缓存"], ["chunked-cross-entropy", "分块输出投影与交叉熵"], ["performance-analysis", "性能与瓶颈分析"], ["training-step-profile", "完整训练 step 实测"],
  ] },
  { title: "微调与后训练", shortTitle: "后训练", nav: "posttraining.html", pages: [
    ["posttraining", "后训练概览"], ["finetuning-recipe", "开源模型微调配方"], ["posttraining-deep", "后训练目标"],
    ["lora-mechanics", "LoRA 梯度与合并"], ["rl-posttraining", "RL 后训练"], ["rl-update", "RL 参数更新"], ["distillation-data", "蒸馏与合成数据"],
  ] },
  { title: "生成与检索服务", shortTitle: "生成与检索", nav: "inference-engine.html", pages: [
    ["inference-engine", "生成执行与 KV Cache"], ["position-cache", "位置与缓存一致性"], ["decoding-algorithms", "采样与约束解码"],
    ["efficient-inference", "量化与投机解码"], ["quantized-linear", "量化矩阵乘"], ["lab", "检索沙盘"], ["ann-index", "ANN 索引与召回"],
  ] },
  { title: "评估、诊断与复现", shortTitle: "评估与复现", nav: "model-evaluation.html", pages: [
    ["model-evaluation", "模型评估"], ["paired-evaluation", "配对统计与结果导入"], ["uncertainty-calibration", "不确定性与校准"],
    ["model-artifacts", "工件与恢复"], ["reproducibility-lab", "复现差异定位"], ["training-failures", "训练故障与确定性回放"], ["engineering-cases", "工程诊断案例"],
  ] },
  { title: "模型能力与应用", shortTitle: "能力与应用", nav: "frontier.html", pages: [
    ["frontier", "能力与应用导航"], ["generation-mechanisms", "自回归、扩散与潜在空间"], ["multimodal", "多模态模型工程"],
    ["reasoning-compute", "CoT 与推理计算"], ["agents-context", "工具、Agent 与上下文"],
    ["training-environments", "训练环境与验证器"], ["learning-mechanisms", "上下文学习与机制解释"],
  ] },
];

export const pageLabel = (slug) => wikiGroups.flatMap(group => group.pages).find(([id]) => id === slug)?.[1] || slug;

// Vendored paths from Lucide, see src/assets/lucide-LICENSE.txt.
export const icon = (name) => {
  const paths = {
    search: '<path d="m21 21-4.34-4.34"/><circle cx="11" cy="11" r="8"/>',
    panel: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/>',
    theme: '<path d="M12 2v2"/><path d="M14.837 16.385a6 6 0 1 1-7.223-7.222c.624-.147.97.66.715 1.248a4 4 0 0 0 5.26 5.259c.589-.255 1.396.09 1.248.715"/><path d="M16 12a4 4 0 0 0-4-4"/><path d="m19 5-1.256 1.256"/><path d="M20 12h2"/>',
    close: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    up: '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
  };
  return `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || ""}</svg>`;
};
