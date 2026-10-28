// The guide, detailed route and lesson continuation share this reading order.
// These are reading suggestions, not mathematical or runtime dependencies.
export const learningPath = [
  {
    id: "inputs", title: "输入与监督", href: "index.html", label: "一次学习的全貌",
    question: "模型收到什么，又被要求预测什么？",
    focus: "先看六步训练回路，分清 token ID、向量、预测与目标。",
    prerequisite: "从一个输入和预期答案开始。",
    checkpoint: "能指出输入、监督目标和一次参数更新各是什么。",
    further: [["data-tokenizer.html#tokenizer-title", "Tokenizer 与输入单位"], ["packing-correctness.html", "监督位置与样本隔离"]],
  },
  {
    id: "forward", title: "模型前向", href: "transformer-lab.html", label: "从 token 到 logits",
    question: "输入怎样经过模型变成预测？",
    focus: "先读计算路径和一个 decoder block，再按需展开权重与结构变体。",
    prerequisite: "知道 token ID、向量和目标的区别。",
    checkpoint: "能沿 Embedding、Attention、MLP 和输出头说清张量的去向。",
    further: [["transformer-anatomy.html", "权重与形状"], ["architecture-variants.html", "结构变体"]],
  },
  {
    id: "update", title: "损失与更新", href: "training.html", label: "走完一次训练回路",
    question: "预测误差怎样改变下一次计算？",
    focus: "先走下一词预测的六个步骤，再比较掩码预测与对比学习。",
    prerequisite: "能追踪一次前向的输入和输出。",
    checkpoint: "能区分 loss、梯度与优化器更新，并说明 labels 和 mask 的作用。",
    further: [["training-deep.html#tie-title", "词表交叉熵与梯度"], ["backpropagation.html", "整层反向传播核验"]],
  },
  {
    id: "evidence", title: "学习与泛化", href: "learning-run.html", label: "观察小模型实际学到了什么",
    question: "loss 下降之后，模型真的会做题了吗？",
    focus: "先比较同一道题训练前后的生成，再看学习曲线、独立测试与未见长度。",
    prerequisite: "知道模型按监督目标更新参数。",
    checkpoint: "能分别解释 teacher-forced loss、自由生成和长度外推的结果。",
    further: [["generalization-analysis.html", "诊断长度外推失败"], ["text-training-project.html", "完成自然语言小模型项目"], ["model-evaluation.html#generalization", "训练、验证与泛化"], ["paired-evaluation.html", "配对比较结果"]],
  },
  {
    id: "reproduce", title: "稳定与复现", href: "model-artifacts.html", label: "保存、恢复并定位故障",
    question: "怎样让一次训练能够被检查和接续？",
    focus: "先看 checkpoint 包含什么，再对照完整恢复和遗漏状态后的轨迹。",
    prerequisite: "能区分学习结果和计算核验。",
    checkpoint: "能解释为什么权重、优化器、随机状态和数据位置需要一起保存。",
    further: [["reproducibility-lab.html", "找到复现的首个差异"], ["training-failures.html", "相同输入的故障回放"], ["numerical-computing.html", "数值稳定与混合精度"], ["optimization-dynamics.html", "优化器与更新尺度"]],
  },
  {
    id: "systems", title: "系统与扩展", href: "engineering.html", label: "把训练放进资源约束",
    question: "规模扩大后，计算、显存和通信怎样配合？",
    focus: "先看训练 step、token 预算和显存账本，再按瓶颈进入并行与性能专题。",
    prerequisite: "能追踪一次更新，也知道如何保存和核验它。",
    checkpoint: "能区分数据并行副本、协作分片、显存占用和实际等待。",
    further: [["training-step-profile.html", "比较完整训练 step"], ["training-systems-deep.html", "并行与数值精度"], ["performance-analysis.html", "测量性能瓶颈"], ["initialization-width.html", "初始化与宽度迁移"], ["batch-noise.html", "Batch 与学习预算"], ["chunked-cross-entropy.html", "输出层保存量与精确梯度"]],
  },
];

export const learningBranches = [
  { title: "微调与后训练", href: "posttraining.html", description: "沿 SFT、LoRA、偏好优化、RL 与蒸馏继续改变模型。", links: [["finetuning-recipe.html", "拆解微调配方"], ["rl-posttraining.html", "RL 后训练"]] },
  { title: "生成与检索服务", href: "inference-engine.html", description: "沿 prefill、decode、缓存、解码与检索组织运行过程。", links: [["retrieval-project.html", "真实中文检索实战"], ["lab.html", "检索沙盘"], ["efficient-inference.html", "量化与投机解码"]] },
  { title: "模型能力与应用", href: "frontier.html", description: "按生成机制、多模态、推理解题或 Agent 的具体问题选读。", links: [["reasoning-compute.html", "推理计算"], ["multimodal.html", "多模态"], ["agents-context.html", "工具与上下文"]] },
];

const introductory = new Set([
  "training-engineer-guide", "deep-dive", "index", "engineering-check", "representation", "training",
  "transformer-lab", "posttraining", "engineering", "model-artifacts", "model-evaluation", "lab", "frontier", "learning-run",
]);
const reportPages = new Set(["retrieval-project","learning-run", "training-failures", "initialization-width", "batch-noise", "chunked-cross-entropy", "generalization-analysis", "text-training-project", "reproducibility-lab", "training-step-profile"]);
const navigationPages = new Set(["training-engineer-guide", "deep-dive", "frontier", "transformer-lab"]);
const calculationPages = new Set([
  "representation-deep", "training-deep", "backpropagation", "numerical-computing", "tensor-layout",
  "attention-kernels", "position-cache", "distributed-execution", "activation-checkpointing", "decoding-algorithms",
  "packing-correctness", "lora-mechanics", "context-parallel", "rl-update", "quantized-linear", "paired-evaluation",
  "ann-index", "simulator", "parallel-correctness", "optimization-dynamics", "continual-learning",
]);

export function readingProfile(slug) {
  return {
    level: introductory.has(slug) ? "入门" : "进阶",
    format: reportPages.has(slug) ? "实测报告" : navigationPages.has(slug) ? "阅读导航" : calculationPages.has(slug) ? "计算演示" : "专题讲解",
  };
}

// Extensions have a primary learning stage and an optional next practical task.
export const learningAttachments = {
  "generalization-analysis": { stage: "evidence", next: "text-training-project", checkpoint: "能区分训练长度覆盖与真正的新长度外推。" },
  "text-training-project": { stage: "evidence", next: "reproducibility-lab", checkpoint: "能用简单基线、完整评估和实际生成分别判断学习结果。" },
  "reproducibility-lab": { stage: "reproduce", next: "training-step-profile", checkpoint: "能先对齐输入与状态，再定位首次可观察差异。" },
  "training-step-profile": { stage: "systems", next: "retrieval-project", checkpoint: "能在相同目标下分别解释整步耗时与保存引用。" },
  "retrieval-project": { stage: "systems", checkpoint: "能区分指定证据命中、近邻近似误差与候选集重排。" },
};
