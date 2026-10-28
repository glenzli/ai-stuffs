import { wikiGroups } from "./wiki.mjs";

export const site = {
  name: "模型工程 Wiki",
  description: "面向训练工程师的模型、Embedding 与工程 Wiki",
  url: "http://127.0.0.1:4173/",
  navigation: wikiGroups.map(group => ({ label: group.shortTitle, href: group.nav })),
};

export const pages = [
  {
    "slug": "architecture-variants",
    "file": "architecture-variants.html",
    "title": "架构变体与结论 · 模型工程 Wiki",
    "description": "Transformer 结构变体、注意力、位置编码、MoE、长上下文与选型结论。",
    "bodyPage": "variants",
    "styles": [
      "architecture"
    ],
    "features": [
      "architecture-variants"
    ],
    "footer": [
      "架构变体与结论",
      "把结构变化翻译成资源与能力的交换，再看证据是否支持这个解释。"
    ]
  },
  {
    "slug": "data-tokenizer",
    "file": "data-tokenizer.html",
    "title": "数据与 Tokenizer · 模型工程 Wiki",
    "description": "模型训练的数据清理、去重、污染隔离、tokenizer 与数据混合。",
    "bodyPage": "data-tokenizer",
    "styles": [
      "data-tokenizer"
    ],
    "features": [
      "data-tokenizer"
    ],
    "footer": [
      "数据与 Tokenizer",
      "tokenizer 决定输入长度，数据分布决定学习信号，去污染决定评估能否被相信。"
    ]
  },
  {
    "slug": "deep-dive",
    "file": "deep-dive.html",
    "title": "六步学习路线 · 模型工程 Wiki",
    "description": "从输入、模型、损失到真实学习、复现和系统扩展的六步路线，附前置、自检与选读。",
    "bodyPage": "deep",
    "styles": [
      "deep-dive"
    ],
    "features": [],
    "footer": [
      "六步学习路线",
      "连起一次学习，再按问题深入机制、系统与任务。"
    ]
  },
  {
    "slug": "engineering",
    "file": "engineering.html",
    "title": "训练工程全流程 · 模型工程 Wiki",
    "description": "从数据、batch、显存、并行到 checkpoint 的模型训练工程全流程。",
    "bodyPage": "engineering",
    "styles": [
      "engineering"
    ],
    "features": [
      "engineering"
    ],
    "footer": [
      "训练工程全流程",
      "跟踪每步数据、梯度、资源和评估结果，形成可复现的训练记录。"
    ]
  },
  {
    "slug": "frontier",
    "file": "frontier.html",
    "title": "模型能力与应用 · 模型工程 Wiki",
    "description": "自回归、扩散、CoT、J-space、多模态、Teacher、Agent、量化与模型评估。",
    "bodyPage": "frontier",
    "styles": [
      "frontier"
    ],
    "features": [],
    "footer": [
      "模型能力与应用",
      "按结构、训练目标、运行成本和评估结果比较模型方案。"
    ]
  },
  {
    "slug": "index",
    "file": "index.html",
    "title": "模型工程 Wiki：一次学习的全貌",
    "description": "从输入、模型预测、监督目标到参数更新，连接实际学习与泛化的第一步。",
    "bodyPage": "home",
    "styles": [
      "home"
    ],
    "features": [
      "home"
    ],
    "footer": [
      "模型工程：一次学习的全貌",
      "沿输入、目标、损失与更新，理解模型如何学习。"
    ]
  },
  {
    "slug": "inference-engine",
    "file": "inference-engine.html",
    "title": "推理引擎与 KV Cache · 模型工程 Wiki",
    "description": "prefill、decode、KV Cache、GQA、连续批处理与推理瓶颈。",
    "bodyPage": "inference",
    "styles": [
      "inference"
    ],
    "features": [
      "inference-engine"
    ],
    "footer": [
      "推理引擎与 KV Cache",
      "prefill 处理输入前缀，decode 逐步生成；权重、KV 和执行成本分别计算。"
    ]
  },
  {
    "slug": "lab",
    "file": "lab.html",
    "title": "检索几何沙盘 · 模型工程 Wiki",
    "description": "用可解释的小向量理解 cosine、分块、top-k 与重排。",
    "bodyPage": "lab",
    "styles": [
      "retrieval"
    ],
    "features": [
      "retrieval-lab"
    ],
    "footer": [
      "04 / 检索几何沙盘",
      "先看候选如何被找出来，再讨论大模型为什么回答得好或坏。"
    ]
  },
  {
    "slug": "model-artifacts",
    "file": "model-artifacts.html",
    "title": "模型工件与 Checkpoint · 模型工程 Wiki",
    "description": "模型 config、tokenizer、权重分片、checkpoint、恢复与部署工件的真实对应关系。",
    "bodyPage": "artifacts",
    "styles": [
      "artifacts"
    ],
    "features": [
      "model-artifacts"
    ],
    "footer": [
      "模型工件与 Checkpoint",
      "部署要能解释权重形状；恢复要能解释训练轨迹；发布要能解释全部依赖版本。"
    ]
  },
  {
    "slug": "model-case-studies",
    "file": "model-case-studies.html",
    "title": "现代模型结构案例 · 模型工程 Wiki",
    "description": "BERT、LLaMA、Mistral、Mixtral、DeepSeek 与 Mamba 的结构案例。",
    "bodyPage": "model-cases",
    "styles": [
      "frontier"
    ],
    "features": [
      "model-cases"
    ],
    "footer": [
      "现代模型结构案例",
      "对照具体版本的结构、运行成本和评估条件。"
    ]
  },
  {
    "slug": "model-evaluation",
    "file": "model-evaluation.html",
    "title": "模型评估方法论 · 模型工程 Wiki",
    "description": "模型工程评估：指标、切片、污染、统计可靠性与发布门槛。",
    "bodyPage": "evaluation",
    "styles": [
      "evaluation"
    ],
    "features": [
      "evaluation"
    ],
    "footer": [
      "模型评估方法论",
      "指标必须绑定数据版本、切片、统计可靠性与发布决策。"
    ]
  },
  {
    "slug": "posttraining-deep",
    "file": "posttraining-deep.html",
    "title": "后训练目标解剖 · 模型工程 Wiki",
    "description": "深入对比表示后训练、SFT、DPO、RLHF 的目标、梯度路径和训练数据结构。",
    "bodyPage": "posttraining-deep",
    "styles": [
      "posttraining-deep"
    ],
    "features": [
      "posttraining-deep"
    ],
    "footer": [
      "深入 03 / 后训练目标",
      "结合参考分布、mask 和可训练参数，追踪后训练的梯度路径。"
    ]
  },
  {
    "slug": "posttraining",
    "file": "posttraining.html",
    "title": "后训练在改什么 · 模型工程 Wiki",
    "description": "区分表示后训练、SFT、偏好对齐和蒸馏：它们分别在改什么。",
    "bodyPage": "posttraining",
    "styles": [
      "posttraining"
    ],
    "features": [
      "posttraining"
    ],
    "footer": [
      "03 / 后训练",
      "预训练给能力；后训练重新选择哪些行为、距离和排序被优先优化。"
    ]
  },
  {
    "slug": "representation-deep",
    "file": "representation-deep.html",
    "title": "表示几何解剖 · 模型工程 Wiki",
    "description": "深入句向量池化、各向异性、Matryoshka 截断与向量接口。",
    "bodyPage": "representation-deep",
    "styles": [
      "representation-deep"
    ],
    "features": [
      "representation-deep"
    ],
    "footer": [
      "深入 01 / 表示几何",
      "句向量由 token 状态、聚合规则、投影与训练目标共同形成。"
    ]
  },
  {
    "slug": "representation",
    "file": "representation.html",
    "title": "表示演化 · 模型工程 Wiki",
    "description": "Embedding 表示方法从 one-hot 到上下文向量和检索专用模型的演化。",
    "bodyPage": "representation",
    "styles": [
      "representation"
    ],
    "features": [
      "representation"
    ],
    "footer": [
      "01 / 表示演化",
      "从训练目标和数据出发，分析向量表示的变化。"
    ]
  },
  {
    "slug": "rl-posttraining",
    "file": "rl-posttraining.html",
    "title": "RL 后训练机制 · 模型工程 Wiki",
    "description": "PPO、GRPO、DPO、RLVR、reward model 与 reward hacking。",
    "bodyPage": "rl-posttraining",
    "styles": [
      "frontier"
    ],
    "features": [
      "rl-posttraining"
    ],
    "footer": [
      "RL 后训练机制",
      "奖励定义目标，算法决定怎样利用，reference 与评估防止策略走偏。"
    ]
  },
  {
    "slug": "scaling-laws",
    "file": "scaling-laws.html",
    "title": "Scaling Laws 与训练预算 · 模型工程 Wiki",
    "description": "Scaling Laws、训练 compute 分配、数据重复与推理感知训练。",
    "bodyPage": "scaling",
    "styles": [
      "frontier"
    ],
    "features": [
      "scaling-laws"
    ],
    "footer": [
      "Scaling Laws 与训练预算",
      "N、D、推理量、唯一数据与目标能力共同决定“更大”是否值得。"
    ]
  },
  {
    "slug": "simulator",
    "file": "simulator.html",
    "title": "训练动态模拟器 · 模型工程 Wiki",
    "description": "真实执行 InfoNCE 更新的二维 Embedding 训练模拟器。",
    "bodyPage": "simulator",
    "styles": [
      "simulator"
    ],
    "features": [
      "simulator"
    ],
    "footer": [
      "训练动态模拟器",
      "记录二维向量的每次更新，并比较损失、命中率和采样变化。"
    ]
  },
  {
    "slug": "training-deep",
    "file": "training-deep.html",
    "title": "损失与梯度解剖 · 模型工程 Wiki",
    "description": "深入推导 Embedding 的 gather/scatter、交叉熵、InfoNCE、温度与梯度。",
    "bodyPage": "training-deep",
    "styles": [
      "training-deep"
    ],
    "features": [
      "training-deep"
    ],
    "footer": [
      "深入 02 / 损失与梯度",
      "正负样本定义监督目标，反向传播计算各参数对损失的导数。"
    ]
  },
  {
    "slug": "training-engineer-guide",
    "file": "training-engineer-guide.html",
    "title": "训练工程师指南 · 模型工程 Wiki",
    "description": "六步学习主线、问题雷达、七个工程契约与按需专题查阅。",
    "bodyPage": "guide",
    "styles": [
      "guide"
    ],
    "features": [
      "engineer-guide",
      "site-index"
    ],
    "footer": [
      "训练工程师指南",
      "识别契约，选择证据，定位层次，再动手调参。"
    ]
  },
  {
    "slug": "training-systems-deep",
    "file": "training-systems-deep.html",
    "title": "训练系统深入 · 模型工程 Wiki",
    "description": "训练系统：DP、ZeRO、TP、PP、EP、精度、activation 内存与集合通信。",
    "bodyPage": "training-systems",
    "styles": [
      "training-systems"
    ],
    "features": [
      "training-systems"
    ],
    "footer": [
      "训练系统深入",
      "分别测量资源占用、计算与通信，再用 profiler 定位等待。"
    ]
  },
  {
    "slug": "training",
    "file": "training.html",
    "title": "训练回路 · 模型工程 Wiki",
    "description": "拆开 Embedding 训练：输入张量、正负样本、损失、梯度与参数更新。",
    "bodyPage": "training",
    "styles": [
      "training"
    ],
    "features": [
      "training"
    ],
    "footer": [
      "02 / 训练回路",
      "输入参与前向计算，目标定义损失，优化器更新可训练参数。"
    ]
  },
  {
    "slug": "transformer-anatomy",
    "file": "transformer-anatomy.html",
    "title": "Transformer 权重解剖 · 模型工程 Wiki",
    "description": "拆解 Transformer block、权重形状、参数量、Dense 与 MoE 路由。",
    "bodyPage": "transformer-anatomy",
    "styles": [
      "architecture"
    ],
    "features": [
      "transformer-anatomy"
    ],
    "footer": [
      "Transformer 权重解剖",
      "总参数回答“要存多少”，激活参数回答“每 token 走多少”，KV Cache 回答“推理时还要额外存多少”。"
    ]
  },
  {
    "slug": "transformer-evolution",
    "file": "transformer-evolution.html",
    "title": "Transformer 演化 · 模型工程 Wiki",
    "description": "Transformer 从 2017 到现代 decoder 的结构、训练目标与系统优化演化。",
    "bodyPage": "transformer-evolution",
    "styles": [
      "architecture"
    ],
    "features": [
      "transformer-evolution"
    ],
    "footer": [
      "Transformer 演化",
      "原始骨架保留 attention、残差与 FFN；现代变化主要发生在稳定性、位置、容量和推理成本。"
    ]
  },
  {
    "slug": "transformer-lab",
    "file": "transformer-lab.html",
    "title": "Transformer 架构解剖 · 模型工程 Wiki",
    "description": "Transformer 结构演化、权重解剖与推理工程入口。",
    "bodyPage": "transformer",
    "styles": [
      "architecture"
    ],
    "features": [],
    "footer": [
      "Transformer 架构解剖",
      "结构看张量，规模看参数，性能看访问模式。"
    ]
  }
];

for (const [slug, title, description] of [
  ["generation-mechanisms", "自回归、扩散与潜在空间", "自回归分解、teacher forcing、扩散去噪、Flow Matching、潜在表示与 JEPA。"],
  ["reasoning-compute", "CoT、Thinking 与推理计算", "CoT、可见 thinking、推理监督、潜在推理、验证器与 test-time compute。"],
  ["multimodal", "多模态模型工程", "视觉、音频、视频如何进入模型，连接器、token 预算、训练阶段与评估。"],
  ["agents-context", "工具、Agent 与上下文", "工具调用闭环、上下文状态、RAG、轨迹训练与端到端评估。"],
  ["distillation-data", "Teacher、蒸馏与合成数据", "教师角色、软标签蒸馏、序列蒸馏、合成数据筛选与泛化。"],
  ["efficient-inference", "量化与投机解码", "PTQ、QAT、校准、权重与 KV 量化、投机采样与吞吐延迟。"],
]) {
  pages.push({ slug, file: `${slug}.html`, title: `${title} · 模型工程 Wiki`, description,
    bodyPage: slug, styles: ["modern"], features: [slug],
    footer: [title, "按目标、数据、结构、成本与证据理解模型。"] });
}

for (const slug of ["data-tokenizer", "model-evaluation", "scaling-laws"]) {
  pages.find(page => page.slug === slug).styles.push("modern");
}
pages.find(page => page.slug === "data-tokenizer").features.push("data-curation");

for (const [slug, title, description] of [
  ["optimization-dynamics", "优化与训练稳定性", "AdamW、动量、更新尺度、初始化、warmup、梯度裁剪与超参数迁移。"],
  ["parallel-correctness", "并行训练的数学正确性", "有效 token 归一化、DDP 平均、梯度累积、空标签副本与等价性验证。"],
  ["performance-analysis", "训练性能与瓶颈分析", "Roofline、算术强度、HBM 流量、关键路径、通信重叠、MFU 与测量。"],
  ["engineering-cases", "训练工程诊断案例", "从多卡、数据过滤、MoE 和奖励异常，逐步揭示证据与控制变量实验。"],
]) {
  pages.push({ slug, file: `${slug}.html`, title: `${title} · 模型工程 Wiki`, description,
    bodyPage: slug, styles: ["modern", "engineering-lessons"], features: [slug],
    footer: [title, "先明确目标与统计口径，再用可复现证据判断。"] });
}
for (const slug of ["transformer-anatomy", "architecture-variants", "engineering", "training-engineer-guide"]) {
  pages.find(page => page.slug === slug).styles.push("modern", "engineering-lessons");
}

pages.push({
  slug: "finetuning-recipe", file: "finetuning-recipe.html", title: "开源模型微调配方 · 模型工程 Wiki",
  description: "从数据条数、模板、监督 token、epoch、packing 与 batch 到 LoRA 参数和独立评估。",
  bodyPage: "finetuning-recipe", 
  styles: ["modern", "engineering-lessons", "frontier-workflows"], features: ["finetuning-recipe"],
  footer: ["微调配方拆解", "记录数据覆盖、训练预算、参数范围和独立评估。"],
});
for (const [slug, feature] of [
  ["rl-posttraining", "rl-dynamics"], ["distillation-data", "on-policy-distillation"],
  ["agents-context", "agent-memory"], ["model-evaluation", "evaluation-reliability"],
  ["multimodal", "multimodal-stream"], ["model-case-studies", null],
]) {
  const page = pages.find(item => item.slug === slug);
  if (!page.styles.includes("modern")) page.styles.push("modern");
  page.styles.push("frontier-workflows");
  if (feature) page.features.push(feature);
}

for (const [slug, title, description] of [
  ["uncertainty-calibration", "不确定性与校准", "回答覆盖、错误率、Brier/ECE、语义熵与稳定错误。"],
  ["continual-learning", "持续学习与任务干扰", "梯度冲突、有限步长、重放、教师约束、EWC 与任务矩阵评估。"],
  ["learning-mechanisms", "上下文学习与机制解释", "固定权重的模式复制、induction head、superposition、SAE 与因果干预。"],
  ["training-environments", "训练环境与验证器", "环境契约、任务生成、状态验证、奖励漏洞、数据隔离和环境安全。"],
]) {
  pages.push({slug, file: `${slug}.html`, title: `${title} · 模型工程 Wiki`, description,
    bodyPage: slug, styles: ["modern", "frontier-workflows", "research-mechanisms"], features: [slug],
    footer: [title, "机制、假设与可检验的证据一起阅读。"]});
}
for (const [slug, feature] of [["optimization-dynamics", "matrix-optimizer"], ["reasoning-compute", "reasoning-evidence"]]) {
  const page = pages.find(item => item.slug === slug);
  page.features.push(feature);
  page.styles.push("research-mechanisms");
}
pages.find(page => page.slug === "frontier").styles.push("modern", "research-mechanisms");

for (const [slug, title, description] of [
  ["backpropagation", "整层 Transformer 反向传播", "小型 RMSNorm、Attention、残差与 SwiGLU 计算图，逐元素梯度与有限差分。"],
  ["numerical-computing", "浮点数与数值稳定性", "FP16/BF16/FP32 舍入、累加顺序、稳定 softmax 与混合精度路径。"],
  ["attention-kernels", "分块 Attention 与在线归一化", "运行最大值、分母与加权和，逐块核对同一注意力输出。"],
  ["distributed-execution", "多卡张量与集合通信", "两卡 MLP 的列/行分片、前向与反向、AllReduce 和其他集合操作。"],
  ["activation-checkpointing", "激活生命周期与重计算", "保存、重算、反向和释放，六层计算链的逻辑内存与梯度核对。"],
  ["decoding-algorithms", "采样、搜索与约束解码", "温度、Top-k、Top-p、逆 CDF 采样、有限语法和 Beam search。"],
]) {
  pages.push({slug, file: `${slug}.html`, title: `${title} · 模型工程 Wiki`, description,
    bodyPage: slug, styles: ["modern", "frontier-workflows", "exact-mechanics"], features: [slug],
    footer: [title, "按声明的算法逐步计算，与参考结果核对。"]});
}
for (const slug of ["engineering", "training-engineer-guide"]) pages.find(page => page.slug === slug).styles.push("exact-mechanics");

for (const [slug, title, description] of [
  ["tensor-layout", "张量布局与存储视图", "Shape、stride、offset、转置、切片、展平、共享存储与复制。"],
  ["engineering-check", "训练工程综合检查", "按固定小配置核对样本流、监督数量、形状、通信、缓存与重算。"],
]) pages.push({slug, file:`${slug}.html`, title:`${title} · 模型工程 Wiki`, description,
  bodyPage:slug, nav:"engineering.html", styles:["modern","frontier-workflows","exact-mechanics","engineering-details"],features:[slug],
  footer:[title,"跟踪数据与状态，按同一计算口径核对结果。"]});

for (const [slug, features] of [
  ["finetuning-recipe", ["data-order"]], ["data-tokenizer", ["bpe-mechanics"]],
  ["distributed-execution", ["vocab-parallel"]], ["training-systems-deep", ["pipeline-schedule"]],
  ["inference-engine", ["serving-scheduler"]],
]) {
  const page=pages.find(p=>p.slug===slug);
  page.features.push(...features);
  for(const style of ["modern","frontier-workflows","exact-mechanics","engineering-details"]) if(!page.styles.includes(style))page.styles.push(style);
}

pages.push({ slug: "position-cache", file: "position-cache.html", title: "RoPE 与 KV Cache 一致性 · 模型工程 Wiki",
  description: "位置旋转、相对点积、padding、后缀 mask 与完整/增量前向的 logits 对照。",
  bodyPage: "position-cache", features: ["position-cache"],
  styles: ["modern", "frontier-workflows", "exact-mechanics", "computation-chains"],
  footer: ["位置与缓存一致性", "固定输入与计算规则，逐位置比较两条前向路径。"] });
for (const [slug, feature] of [
  ["numerical-computing", "mixed-precision-step"], ["model-artifacts", "checkpoint-replay"],
  ["training-systems-deep", "fsdp-lifecycle"], ["transformer-anatomy", "moe-dispatch"],
  ["efficient-inference", "speculative-probability"],
]) {
  const page = pages.find(p => p.slug === slug);
  page.features.push(feature);
  for (const style of ["modern", "frontier-workflows", "exact-mechanics", "computation-chains"]) if (!page.styles.includes(style)) page.styles.push(style);
}

for (const [slug, title, description] of [
  ["packing-correctness", "SFT 样本打包与计算正确性", "多轮模板、监督标签、位置与 Attention 隔离；分别处理、padding 和 packing 的输出及梯度对照。"],
  ["lora-mechanics", "LoRA 梯度、更新与合并", "冻结底座、A/B 初始化、输入梯度、SGD 更新、量化底座与合并等价性。"],
]) pages.push({ slug, file: `${slug}.html`, title: `${title} · 模型工程 Wiki`, description,
  bodyPage: slug, features: [slug],
  styles: ["modern", "frontier-workflows", "exact-mechanics", "computation-chains", "objective-chains"],
  footer: [title, "固定目标与输入，用独立前向和梯度验证计算。"] });
const contrastivePage = pages.find(page => page.slug === "parallel-correctness");
contrastivePage.features.push("contrastive-distributed");
contrastivePage.styles.push("frontier-workflows", "exact-mechanics", "objective-chains");
contrastivePage.description = "全局 token 分母、DDP 与梯度累积，以及全局负例、远端表示梯度和共享参数归约。";
pages.find(page => page.slug === "backpropagation").styles.push("computation-chains");

for (const [slug, title, description, nav] of [
  ["context-parallel", "上下文并行的前向与梯度", "全局位置、跨 rank K/V、在线归一化、不等长分片、梯度归还与 Gloo 对照。", "engineering.html"],
  ["rl-update", "RL 参数梯度与更新", "固定 rollout 的 log-prob、优势、裁剪、精确 KL、梯度边界与连续 SGD 更新。", "frontier.html"],
  ["quantized-linear", "量化矩阵乘与整层误差", "W8A8 零点修正、整数累加、bias 单位、输出重定标和 W4A16 恢复对照。", "frontier.html"],
  ["paired-evaluation", "配对评估与统计实验", "逐题结果导入、按组 bootstrap、精确置换、训练 seed 与重复选择的偏差。", "engineering.html"],
  ["ann-index", "ANN 索引与精确检索", "IVF 列表、nprobe、Flat 参考、近邻与任务召回、编码版本和真实 CPU 索引实验。", "lab.html"],
]) pages.push({ slug, file: `${slug}.html`, title: `${title} · 模型工程 Wiki`, description,
  bodyPage: slug, nav, features: [slug], styles: ["modern", "exact-mechanics", "decision-chains"],
  footer: [title, "固定输入和目标，比较正确路径、故障与独立参考。"] });

for (const [slug, feature, description] of [
  ["attention-kernels", "attention-backward", "在线归一化、LSE 保存、分块概率重建、dQ/dK/dV 累加与计算存储账本。"],
  ["activation-checkpointing", "gradient-cache", "激活重算与梯度缓存：全局负例、表示梯度、RNG 重放、保存量和实际测量。"],
  ["parallel-correctness", "sharded-clipping", "全局分母、跨卡对比学习、分片范数、复制参数去重、统一裁剪和协调跳步。"],
  ["performance-analysis", "memory-report", "Roofline 与 CPU 实测：误差、保存张量引用、未插桩耗时、profiler 与本地报告查看。"],
]) {
  const page = pages.find(item => item.slug === slug);
  page.features.push(feature); page.description = description;
  for (const style of ["frontier-workflows", "exact-mechanics", "computation-chains", "objective-chains", "memory-chains"]) if (!page.styles.includes(style)) page.styles.push(style);
}

if (!pages.find(p => p.slug === "lab").styles.includes("modern")) pages.find(p => p.slug === "lab").styles.push("modern");

for (const [slug, title, description, nav] of [
  ["learning-run", "小模型学习、生成与泛化", "固定符号任务的小型 decoder：独立拆分、真实学习曲线、自由生成、多个种子、恢复和缓存一致性。", "training.html"],
  ["training-failures", "训练故障与确定性回放", "从同一 checkpoint 和输入回放数值、学习率与监督故障，定位首个差异并核验修复。", "engineering.html"],
  ["initialization-width", "初始化、残差与宽度迁移", "分别测量深度与宽度的激活和梯度尺度，对照标准参数化、μP 参数组与保留宽度的学习率迁移。", "engineering.html"],
  ["batch-noise", "Batch、梯度噪声与学习预算", "固定参数下的逐序列梯度、总体方差和抽样修正，以及相同消费 token 预算的真实训练对照。", "engineering.html"],
  ["chunked-cross-entropy", "分块输出投影与交叉熵", "不保存完整 logits 的精确输出层前后向，核对 loss、hidden/权重/bias 梯度、尾块与实际保存量。", "engineering.html"],
]) pages.push({ slug, file: `${slug}.html`, title: `${title} · 模型工程 Wiki`, description,
  bodyPage: slug, nav, features: [slug], styles: ["modern", "exact-mechanics", "decision-chains"],
  footer: [title, "固定任务、输入与评价口径，保存实际运行的结果和验证依据。"] });

for (const slug of ["learning-run", "training-failures", "initialization-width", "batch-noise", "chunked-cross-entropy"]) {
  pages.find(page => page.slug === slug).styles.push(slug);
}
// Both dynamics pages use the chart, control and provenance rules in this sheet.
pages.find(page => page.slug === "batch-noise").styles.splice(3, 0, "initialization-width");

for (const [slug, title, description] of [
  ["generalization-analysis", "长度泛化失败诊断", "固定权重下的 EOS 与位置探针、扩展训练长度的三 seed 对照，以及逐题结果和预算边界。"],
  ["text-training-project", "自然语言小模型项目", "从本站中文语料、训练集 BPE 到小型语言模型训练、独立文档评估、生成和新进程恢复。"],
  ["reproducibility-lab", "复现差异定位实验", "同初始权重与输入、精度转换和数据顺序探针；逐边界记录首次差异并保留历史原因的不确定性。"],
  ["training-step-profile", "完整训练 step 实测", "在同一真实小模型上比较完整与精确分块输出层，核对更新结果后分阶段测量 CPU 时间与保存引用。"],
]) pages.push({ slug, file: `${slug}.html`, title: `${title} · 模型工程 Wiki`, description,
  bodyPage: slug, features: ["practice-reports"], styles: ["modern", "exact-mechanics", "practice"],
  footer: [title, "从保存的输入、权重和逐步记录，核验页面上的每项结论。"] });
pages.find(page => page.slug === "deep-dive").styles.push("practice");
pages.push({ slug: "retrieval-project", file: "retrieval-project.html", title: "真实中文检索实战 · 模型工程 Wiki",
  description: "877 段真实中文文档、固定 BGE 模型、BM25、Flat/IVF 与 RRF 重排；完整问题、指定证据、权重和索引可核验。",
  bodyPage: "retrieval-project", features: ["retrieval-project", "experiment-check"], styles: ["modern", "exact-mechanics", "practice"],
  footer: ["真实中文检索实战", "区分指定证据命中、向量近邻近似与候选内重排。"] });
for (const slug of ["generalization-analysis", "text-training-project", "reproducibility-lab", "training-step-profile"]) pages.find(page => page.slug === slug).features.push("experiment-check");
pages.find(page => page.slug === "text-training-project").features.push("language-baselines");
pages.find(page => page.slug === "generalization-analysis").features.push("matched-budget");

// Topic ownership determines both header navigation and the knowledge directory.
const pageTopics = new Map(wikiGroups.flatMap(group => group.pages.map(([slug]) => [slug, group])));
for (const page of pages) {
  const topic = pageTopics.get(page.slug);
  if (!topic) throw new Error(`Missing topic for ${page.slug}`);
  page.nav = topic.nav;
}
