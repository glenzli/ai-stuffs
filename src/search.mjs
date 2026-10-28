export const searchMetadata = {
  "retrieval-project.html": { summary: "真实中文段落、固定 BGE、BM25、Flat/IVF、候选重排与混合召回，附来源组配对区间和完整候选证据", terms: "真实 检索 实战 中文 embedding BGE BM25 RRF hybrid 融合 混合 并集 候选 覆盖 bootstrap 配对 证据 相关性 语义 召回 重排 Flat IVF 检索闭环 retrieval project", priority: 10 },
  "generalization-analysis.html": { summary: "同权重的 EOS 约束与位置扰动、长度支持扩展训练、三 seed 逐题预测与预算混杂", terms: "长度泛化 失败诊断 EOS 位置 外推 长序列 训练支持 generalization length position probe", priority: 10 },
  "text-training-project.html": { summary: "38 篇中文文档的 byte BPE、小模型训练、词频基线、逐文档误差、宏微平均、原始生成和新进程恢复", terms: "自然语言 中文 小模型 项目 从零训练 tokenizer BPE 数据 划分 损失 逐文档 宏平均 微平均 micro macro 基线 区间 生成 交付 text project language model resume", priority: 10 },
  "reproducibility-lab.html": { summary: "同 seed 同初值与输入的精度和顺序对照，定位首个可观察差异，保留历史未知原因", terms: "复现 差异 定位 seed 精度 浮点 数据顺序 reproducibility randomness divergence float32 float64", priority: 10 },
  "training-step-profile.html": { summary: "完整小模型 step 的前向、输出层、反向和更新耗时，精确分块 CE 的整步收益", terms: "完整训练 step 性能 瓶颈 实测 profiler CPU 吞吐 交叉熵 分块 full step profile", priority: 10 },
  "learning-run.html": { summary: "真实 decoder 学习曲线、teacher-forced loss、自由生成、独立组合与长度外推、恢复和缓存核验", terms: "learning run decoder reversal teacher forcing token accuracy exact match held out generalization seed checkpoint cache 学习 训练 反转 泛化 自由生成 未见组合 长度外推 拆分 恢复 实测 报告", priority: 10 },
  "training-failures.html": { summary: "同 checkpoint 与输入回放数值异常、过大学习率、有限但错位的监督，并核验受控修复", terms: "training failure replay anomaly NaN infinity checkpoint RNG labels alignment learning rate hooks activation RMS gradient 故障 回放 异常 首个差异 数值 监督 错位 修复 学习率", priority: 10 },
  "initialization-width.html": { summary: "深度、Norm 与残差的尺度实测，以及标准/μP 参数组、readout、坐标检查和保留宽度迁移", terms: "initialization width depth residual pre ln post ln layer norm RMS mup μP maximal update parameterization readout fan in coordinate check learning rate 初始化 宽度 深度 残差 尺度 参数组 方差 学习率迁移", priority: 10 },
  "batch-noise.html": { summary: "逐序列梯度、协方差迹、有无放回抽样和零均值边界，按消费 token 对齐的 batch 学习曲线", terms: "batch noise gradient covariance variance trace replacement finite population correction token budget scheduler sequence normalization 梯度噪声 批量 总体方差 抽样 有放回 无放回 相关 重复 零均值 学习效率 预算", priority: 10 },
  "chunked-cross-entropy.html": { summary: "分块 H×Wᵀ、稳定 LSE 与完整梯度重建：ignore_index、尾块、零监督、保存张量引用及 CPU 时间", terms: "chunked cross entropy output projection logits logsumexp LSE dH dW db ignore index tied weight saved tensors memory CPU 分块 交叉熵 输出投影 词表 梯度 重算 权重绑定 忽略标签 尾块 保存量", priority: 10 },
  "context-parallel.html": { summary: "序列分片、远端 KV、全局因果 mask、梯度归还与真实 Gloo 通信", terms: "CP SP context parallel ring attention sequence K V dK dV global softmax LSE gloo 上下文并行 序列分片 全局位置 梯度 归还 不等长", priority: 10 },
  "rl-update.html": { summary: "rollout 到参数梯度：行为策略、精确 KL、裁剪、mask、长度归一化和连续更新", terms: "PPO GRPO RL policy gradient old reference logprob logits clip KL detach reward rollout SGD 策略梯度 参数更新 裁剪 行为概率 长度", priority: 10 },
  "quantized-linear.html": { summary: "W8A8 整数矩阵乘、零点、scale、INT32 bias、饱和和整层输出误差", terms: "quantization W8A8 W4A16 int8 int32 int4 GEMM zero point bias requantization scale 量化 矩阵乘 零点 整数 累加 校准 重定标", priority: 10 },
  "paired-evaluation.html": { summary: "逐题配对 JSON、按组重采样、差值区间、精确置换和 checkpoint 选择偏差", terms: "paired bootstrap permutation seed JSON confidence interval p-value selection 评估 配对 统计 重采样 置信区间 置换 导入 选择偏差", priority: 10 },
  "ann-index.html": { summary: "IVF 候选列表、Flat 精确参考、编码版本与真实向量的来源过滤前后对照", terms: "ANN IVF Faiss Flat HNSW nprobe efSearch recall L2 index 索引 近似检索 候选 列表 质心 召回 版本 重建 metadata filter 过滤 后过滤 预过滤 来源 候选不足 空范围 min(10,M)", priority: 10 },
  "position-cache.html": { summary: "RoPE 旋转与完整 Transformer 的左右填充、next logits 读取列、有效缓存及固定后缀续算核验", terms: "rope rotary position cache position_ids cache_position KV mask padding prefill decode logits 位置 旋转 相对位置 缓存 增量 一致性 后缀 偏移 左填充 右填充 last-column length-index 正弦位置 物理列 逻辑位置 完整模型 实测", priority: 10 },
  "tensor-layout.html": {summary:"地址、stride、offset、视图共享、复制与无拷贝展平",terms:"tensor layout stride offset storage view reshape contiguous transpose slice 内存 布局 张量 步长 视图 转置 拷贝 连续 共享",priority:10},
  "engineering-check.html": {summary:"固定配置下核对数据、监督、形状、词表分片和 KV",terms:"quiz check exercise 练习 综合 理解检查 数据 监督 梯度 形状 缓存 通信 自测",priority:10},
  "backpropagation.html": { summary: "整层计算图、有限差分，以及独立 PyTorch 自动微分、更新与恢复对照", terms: "backprop VJP backward jacobian matrix gradient rmsnorm swiglu attention chain rule pytorch autograd framework cpu gloo 反向传播 梯度 计算图 雅可比 有限差分 整层 框架 对照 实验", priority: 10 },
  "packing-correctness.html": { summary: "多轮对话到 token、labels 与 mask；独立 packing、padding、跨样本泄漏及重复 shift 的数值对照", terms: "sft packing padding labels shift mask position ids chat template assistant eos bos 截断 零标签 多轮 模板 样本隔离 监督 边界", priority: 10 },
  "lora-mechanics.html": { summary: "LoRA 前向、A/B 与输入梯度、零初始化、更新、量化底座和合并输出核验", terms: "lora qlora adapter gradient merge frozen detach no_grad initialization int4 quantization 低秩 冻结 梯度 初始化 合并 底座 量化", priority: 10 },
  "numerical-computing.html": { summary: "浮点舍入、稳定 softmax，以及梯度缩放、累积、裁剪、溢出跳步的完整更新", terms: "float fp16 bf16 fp32 ULP numerical stable softmax logsumexp AMP scaler unscale clipping overflow accumulation 数值 浮点 舍入 下溢 上溢 累加 NaN 精度 混合精度 梯度缩放 跳步 更新", priority: 10 },
  "attention-kernels.html": { summary: "在线前向与分块反传：LSE 保存、概率重建、dQ/dK/dV 累加和成本账本", terms: "FlashAttention tile online softmax attention kernel block SRAM HBM backward LSE logsumexp dq dk dv 分块 在线 归一化 反向 重缩放 重算 空行 梯度累加", priority: 10 },
  "distributed-execution.html": { summary: "两卡列行分片的前后向张量和四种集合通信", terms: "TP tensor parallel column row allreduce allgather reducescatter alltoall nccl 张量并行 多卡 分片 行并行 列并行 通信 vocab parallel cross entropy 词表并行 目标token 归约", priority: 10 },
  "activation-checkpointing.html": { summary: "激活重算与全局负例梯度缓存：表示梯度、RNG 重放、参数 VJP 和保存量", terms: "activation checkpointing rematerialization recompute lifetime RNG GradCache gradient cache VJP 激活 重算 检查点 生命周期 峰值 内存 dropout 梯度缓存 全局负例 随机状态", priority: 10 },
  "decoding-algorithms.html": { summary: "温度、Top-k/Top-p、采样、Beam search 与有限语法 mask", terms: "decode decoding greedy beam search topk topp nucleus temperature grammar JSON logits mask 采样 解码 搜索 语法 约束 温度 停止", priority: 10 },
  "uncertainty-calibration.html": { summary: "回答覆盖、错误率、Brier/ECE、语义熵与稳定错误", terms: "uncertainty calibration semantic entropy brier ECE hallucination confidence selective risk coverage 置信 幻觉 不确定性 校准 语义熵 拒答 回答门槛", priority: 10 },
  "continual-learning.html": { summary: "新旧任务梯度冲突、重放、EWC 与阶段评估矩阵", terms: "continual forgetting retention gradient conflict pcgrad replay EWC Fisher 干扰 遗忘 梯度冲突 持续学习 重放 可塑性 稳定性", priority: 10 },
  "learning-mechanisms.html": { summary: "上下文学习、induction head、superposition、SAE 与因果干预", terms: "in-context learning ICL induction head circuit tracing superposition SAE probe patching 上下文学习 归纳头 机制解释 特征 叠加 消融 激活替换 稀疏自编码器", priority: 10 },
  "training-environments.html": { summary: "任务生成、环境快照、可执行验证器、奖励漏洞与安全", terms: "environment verifier SWE-Gym reward sandbox simulation 训练环境 可解性 任务生成 验证器 状态 副作用 课程 环境隔离", priority: 10 },
  "finetuning-recipe.html": { summary: "数据条数、模板与监督、epoch/packing/batch 账本、LoRA/QLoRA 参数与数据学习曲线", terms: "finetune sft lora qlora rank alpha target modules 微调 配方 一万条 几千条 数据量 样本数 轮数 epoch packing batch 监督 token 学习曲线 sampler drop_last worker rank 补样 丢尾 重复 数据分发 base instruct", priority: 10 },
  "optimization-dynamics.html": { summary: "AdamW 状态账本、初始化、warmup、梯度裁剪与 μP 迁移", terms: "optimizer adamw momentum sgd beta initialization warmup clipping mup μp 优化器 更新 动量 稳定性 学习率 初始化 spike Muon SVD Newton Schulz 正交化 奇异值 矩阵优化器", priority: 9 },
  "parallel-correctness.html": { summary: "完整 Transformer 的真实累积/DDP 更新、变长监督分母、尾窗口与空监督，以及跨卡负例、全局范数和协调跳步", terms: "ddp loss reduction weighted mean accumulation no_sync token contrastive infonce all-gather detach negative gradcache global norm clip gradient shards replicated 全局 梯度累积 实测 Transformer AdamW 尾窗口 多卡 分母 归一化 空标签 对比学习 全局负例 表示梯度 范数 裁剪 复制 跳步 溢出", priority: 10 },
  "performance-analysis.html": { summary: "Roofline 与真实 CPU 实验：未插桩耗时、保存张量引用、profiler、逻辑成本和报告读取", terms: "performance roofline gemm flops hbm bandwidth mfu hfu profiler trace saved tensors report 吞吐 瓶颈 带宽 性能 算术强度 重算 实测 测量 报告 内存 保存量", priority: 9 },
  "engineering-cases.html": { summary: "多卡方向错误、过滤偏差、MoE 等待、奖励与能力的诊断", terms: "case diagnose debugging ablation 多卡 过滤 MoE 奖励 案例 诊断 控制变量 排查 证据", priority: 9 },
  "generation-mechanisms.html": { summary: "自回归、teacher forcing、DDPM、Flow Matching、扩散语言模型与 JEPA", terms: "autoregressive 自回归 diffusion difussion 扩散 denoising latent JEPA J-Space 潜在空间", priority: 10 },
  "reasoning-compute.html": { summary: "CoT 与可见 thinking、J-space 内部表示、监督推理、验证器与推理预算", terms: "cot chain of thought thinking 思维链 思考 reasoning verifier test time compute latent coconut J-space JSpace Jacobian Anthropic 推理 RLVR pass@k CoT-Pass 能力边界 过程正确 采样效率", priority: 10 },
  "multimodal.html": { summary: "图像 patch、原生多模态联合训练、Qwen3-Omni 与流式时间线", terms: "multimodal 多模态 vision vlm image audio video connector projector 视觉 音频 视频 streaming Thinker Talker codec Qwen3 Omni 流式 首包 队列 原生", priority: 9 },
  "agents-context.html": { summary: "工具调用、上下文压缩、证据记忆、冲突更新与持续学习", terms: "agent tool context memory rag 工具 上下文 记忆 轨迹 环境 compaction continual test time training 持续学习 压缩 约束 遗忘 prompt injection 提示注入 授权 信任", priority: 9 },
  "distillation-data.html": { summary: "教师角色、软标签、学生轨迹蒸馏、前缀分布与能力保留", terms: "teacher student distillation synthetic data 蒸馏 教师 合成 数据 软标签 on-policy opd reverse kl forward 前缀 轨迹 恢复 遗忘", priority: 9 },
  "efficient-inference.html": { summary: "量化误差、PTQ/QAT、KV 精度，以及投机采样接受与修正的概率账本", terms: "quantization int4 int8 fp8 awq gptq ptq qat 量化 校准 speculative draft residual distribution rejection 投机 解码 采样 残差 概率分布 接受 拒绝", priority: 9 },
  "index.html": {
    "summary": "一次学习的输入、预测、监督、损失和参数更新，连接六步主线",
    "terms": "overview 表示 token id embedding pipeline",
    "priority": 10
  },
  "training-engineer-guide.html": {
    "summary": "从头学习或按问题查阅：六步主线、问题雷达、七个契约与专题选读",
    "terms": "guide 训练工程师 排查 contract 问题 方向 学习实验 故障回放 初始化 宽度 batch 梯度噪声 分块交叉熵",
    "priority": 10
  },
  "engineering.html": {
    "summary": "预训练、表示训练、SFT、DPO 的执行流程、资源与故障",
    "terms": "engineering dataloader batch tokens update zero 显存 oom 故障",
    "priority": 9
  },
  "training.html": {
    "summary": "输入、查表、正负样本、损失、梯度与 SGD",
    "terms": "gather scatter cross entropy sgns word2vec gradient 梯度",
    "priority": 9
  },
  "transformer-anatomy.html": {
    "summary": "QKV、残差、Norm、SwiGLU、Dense/MoE 参数，以及专家分派与合并计算",
    "terms": "qkv residual rmsnorm swiglu dense moe parameter weights dispatch combine permutation alltoall 参数 权重 专家 分派 逆排列 容量 门控 梯度",
    "priority": 9
  },
  "inference-engine.html": {
    "summary": "prefill、decode、GQA/MQA、KV 计算与 serving",
    "terms": "prefill decode kv cache ttft tpot gqa mqa paged attention serving 调度 连续批 请求队列 前缀缓存 引用计数",
    "priority": 9
  },
  "data-tokenizer.html": {
    "summary": "规范化、质量筛选、去重、污染隔离、Tokenizer 与混合预算",
    "terms": "data tokenizer bpe unigram utf8 dedup contamination mixture fineweb filter curation 数据 清洗 质量 过滤 去重 poisoning 投毒 后门 提示注入 数据安全 BPE merge 合并 词频",
    "priority": 8
  },
  "model-evaluation.html": {
    "summary": "指标、泛化、重复执行、判分器误差与发布判据",
    "terms": "eval evaluation metric recall mrr ndcg f1 contamination confidence generalization early stopping 泛化 训练好了 过拟合 停止 评估 judge grader pass^k pass@k 判分器 混淆矩阵 可靠性 harness",
    "priority": 8
  },
  "posttraining.html": {
    "summary": "表示后训练、SFT、偏好对齐和回归",
    "terms": "posttraining sft dpo preference 后训练",
    "priority": 8
  },
  "rl-posttraining.html": {
    "summary": "PPO/GRPO、组内优势、探索、DAPO 与异步 RL 策略陈旧度",
    "terms": "rlhf ppo grpo rlvr reward hacking verifier rlaif preference dapo areal entropy staleness off-policy old behavior 策略 异步 裁剪 熵 陈旧度 全对 全错",
    "priority": 8
  },
  "architecture-variants.html": {
    "summary": "MHA/MQA/GQA/MLA、稀疏 attention、MoE、SSM",
    "terms": "mha mqa gqa mla sparse window rope swiglu moe ssm hybrid 变体",
    "priority": 8
  },
  "scaling-laws.html": {
    "summary": "compute、参数、token、重复数据与推理感知训练",
    "terms": "scaling chinchilla compute nd tokens epochs inference budget 幂律 推导 最优分配 微分 预算",
    "priority": 8
  },
  "model-case-studies.html": {
    "summary": "结构样本与 DeepSeek-V4.1-Flash 的跨层 KV、CSA2 和 CED",
    "terms": "bert llama mistral mixtral deepseek mamba model cards 模型案例 V4.1 Flash CSA2 CED cross layer KV reuse FP4 Engram mHC 跨层复用 压缩 SWA replay",
    "priority": 8
  },
  "transformer-lab.html": {
    "summary": "结构、权重、推理与变体四条路径",
    "terms": "transformer architecture map qkv residuals 架构",
    "priority": 8
  },
  "transformer-evolution.html": {
    "summary": "2017 到现代 decoder 的结构与系统变化",
    "terms": "history evolution preln rmsnorm rope swiglu gqa 演化",
    "priority": 7
  },
  "training-systems-deep.html": {
    "summary": "并行、数值精度、流水线，以及 FSDP 聚合/释放/预取/归约的参数生命周期",
    "terms": "fsdp deepspeed tensor parallel pipeline expert context precision bf16 fp8 activation collective gpipe 1f1b allgather reducescatter reshard prefetch 空泡 流水线 调度 参数生命周期 预取 分片 显存峰值",
    "priority": 7
  },
  "model-artifacts.html": {
    "summary": "工件版本、checkpoint 完整与遗漏状态恢复对照、CPU 小模型实验",
    "terms": "checkpoint safetensors config optimizer scheduler rng manifest resume restart cpu lab 工件 恢复 断点 轨迹 随机状态 数据游标 小模型 实验",
    "priority": 7
  },
  "representation.html": {
    "summary": "one-hot、词向量、上下文表示、句向量与原则",
    "terms": "onehot lsa word2vec glove fasttext bert sbert simcse embedding history",
    "priority": 7
  },
  "lab.html": {
    "summary": "chunk、cosine、top-k、reranker 与 RAG",
    "terms": "rag retrieval chunk cosine topk rerank ann index 检索",
    "priority": 7
  },
  "representation-deep.html": {
    "summary": "pooling、各向异性、Matryoshka 与向量接口",
    "terms": "pooling cls mean anisotropy whitening matryoshka normalize 句向量",
    "priority": 7
  },
  "training-deep.html": {
    "summary": "gather/scatter、InfoNCE、SGNS、温度与权重绑定",
    "terms": "infonce sgns temperature hard negative weighted tied embedding gradient 梯度 交叉熵 推导 p-y softmax 导数",
    "priority": 7
  },
  "posttraining-deep.html": {
    "summary": "SFT mask、DPO、reference、LoRA 与可训练范围",
    "terms": "sft dpo reference lora adapter freeze mask 后训练",
    "priority": 7
  },
  "frontier.html": {
    "summary": "自回归、扩散、CoT、Teacher、多模态、Agent、量化与规模规律入口",
    "terms": "frontier modern ai scaling rl model cases 前沿 现代 人工智能 决策",
    "priority": 7
  },
  "deep-dive.html": {
    "summary": "六步学习路线，以及按症状查找十三组实测、来源和独立重跑条件",
    "terms": "learning map deep dive 学习路径 六步 主线 前置 自检 复现 泛化 实测 实验 来源 核验 重跑 padding ddp 检索过滤",
    "priority": 10
  },
  "simulator.html": {
    "summary": "可选的 InfoNCE 二维动态实验",
    "terms": "simulator simulation dynamic infonce training loss recall 模拟",
    "priority": 5
  }
};
