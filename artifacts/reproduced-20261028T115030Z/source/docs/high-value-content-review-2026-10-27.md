# 高价值内容增量评审 · 2026-10-27

后续实施状态：六项内容及 CPU 训练恢复实验已完成，见 `docs/computation-chains-2026-10-27.md`。以下保留最初的缺口评审与建议验收条件，反映实施前的覆盖状态。

核对现有 48 页的目录、专题分工和关键交互实现后，建议优先补齐位置与缓存、混合精度更新、训练恢复这三条计算链。第二批再补 FSDP 参数生命周期、MoE 数据搬运和投机采样的概率验证。优先级按训练工程用途、相对现有内容的增量和可验证程度判断，不代表实现工时估计。

本轮交付内容评审与建议验收条件。现有页面已经覆盖反传、浮点舍入、在线 Attention、TP、词表并行、激活重算、流水线、BPE、数据分发、张量布局和服务调度，新增内容应接在这些基础上。

| 优先级 | 补充方向 | 现有覆盖与增量 | 建议归属 |
| --- | --- | --- | --- |
| 第一批 1 | RoPE 与 KV Cache 的数值一致性 | 已有位置编码说明、缓存计数和服务调度；增加旋转计算、位置与 mask 对齐、完整前向和增量前向的 logits 对照。 | 一个完整专题；架构页和推理页链接进入。 |
| 第一批 2 | 混合精度的完整 optimizer step | 已有格式舍入和精度路径说明；增加缩放、累积、unscale、裁剪、溢出跳步和计数状态。 | 深化 numerical-computing，连接 optimization-dynamics 和 parallel-correctness。 |
| 第一批 3 | Checkpoint 恢复的轨迹对照 | 已有文件浏览器、恢复清单和基线比较要求；增加真实小循环及逐项遗漏状态的对照。 | 深化 model-artifacts，连接数据分发与优化器账本。 |
| 第二批 4 | FSDP 参数与梯度生命周期 | 已有静态显存估算和集合通信数值；增加分片、临时完整参数、预取和反向归约的逐事件账本。 | 深化 training-systems-deep。 |
| 第二批 5 | MoE dispatch / combine 的具体计算 | 已有路由图、门控梯度和容量解释；增加 token 复制、排列、跨卡交换、专家计算与逆排列。 | 深化 transformer-anatomy，连接 distributed-execution。 |
| 第二批 6 | 投机采样为何保持目标分布 | 已有接受/修正公式；当前交互由用户指定拒绝位置。增加实际概率质量计算与有限分布验证。 | 深化 efficient-inference。 |

**1. RoPE 与 KV Cache：从位置坐标到相同 logits。**

本地证据：`src/scripts/features/inference-engine.js` 的 `inferenceState` 计算缓存与处理位置数量；`serving-scheduler.js` 处理请求和缓存管理；`backpropagation.js` 的小型 block 有 Attention 数值，但没有位置旋转或增量缓存路径。`architecture-variants.html` 和 `transformer-evolution.html` 已有 RoPE 概念说明。

建议先固定小型 causal decoder、权重、输入序列、位置规则与推理模式，逐项展示旋转前后的 Q/K、允许访问的 key、缓存内容和最终 logits。分别推进单 token 和多 token 后缀，观察逻辑位置、张量下标和缓存写入位置的关系。实现时核对站内已引用的 RoFormer 与 Transformers 缓存文档 [S1–S2]。

验收应包含：同一前缀下完整重算与增量缓存的 logits 在声明的容差内一致；padding 和分块后缀各有对照；把当前 token 位置错误重置、重复旋转已缓存 K 或错配后缀 mask 后能定位首次偏差。单独核验旋转保持范数及相对位置点积关系。固定频率的数值一致性不用于证明长上下文能力。

**2. 混合精度：一步更新中的顺序与状态。**

本地证据：`src/content/numerical-computing.html#precision-path` 已说明 dtype、master copy 和处理顺序；`src/scripts/features/numerical-computing.js` 主要计算舍入、求和和稳定 softmax。当前没有 scaler 与 optimizer 的联合状态演示。

建议展示原始梯度、缩放梯度、累积结果、unscale 后范数、裁剪系数、参数更新、scale 变化及实际更新次数。先固定一种明确的 FP16 教学路径，另外解释 autocast、梯度缩放和参数存储精度的各自职责；实现细节按引用版本核对 [S3]。

可以用一个直接算清的反例：梯度 `[3,4]`、scale 为 8、L2 裁剪阈值为 1，理想算术下先 unscale 再裁剪得到 `[0.6,0.8]`；先裁剪再 unscale 得到 `[0.075,0.1]`。这一例子展示阈值应用在不同尺度上的影响，不模拟 GPU kernel。

验收应覆盖正确顺序与错误顺序、累积窗口内的统一 scale、非有限梯度导致的更新跳过，以及尝试窗口数和成功更新数的区别。若加入 LR scheduler，明确它按哪一种计数推进。显式 master 权重作为另一种存储方案展示，不暗示所有 autocast 训练都创建独立 master copy。

**3. Checkpoint：恢复之后能否继续同一条轨迹。**

本地证据：`src/content/model-artifacts.html#checkpoint-title` 已明确要求与未中断基线比较；`src/scripts/features/model-artifacts.js` 的交互用于浏览工件说明，尚未执行训练恢复。

建议运行一个确定的小训练循环，在 optimizer step 边界保存。并列比较连续运行、完整恢复以及分别遗漏 optimizer、scheduler、RNG、数据游标的运行。使用随机算子的例子才检验 RNG 的作用。模型与优化器状态、随机源控制分别对照 PyTorch 文档核验 [S4–S5]。

每步显示样本 ID、学习率、更新前 loss、梯度范数和更新后参数差异，明确各读数时点。验收要求完整恢复与基线在约定条件下相符，遗漏状态的对照能显示首次偏差；只恢复权重时，不能仅凭恢复瞬间的相同 loss 判定轨迹恢复成功。累积中途保存涉及梯度及进度，适合后续扩展。真实框架的跨硬件逐位复现另行验证。

**4. FSDP：静态分片之外，临时完整参数何时出现。**

本地证据：`engineering.js` 提供 DP/ZeRO 静态状态估算，`training-systems.js` 说明 gather / reshard，`distributed-execution.js` 演示 collective 输入输出。三者尚未组合成 FSDP 的执行时间线。

建议固定两 rank、两层模型和一种明确的 FSDP2 策略，展示本地参数片、临时完整参数、梯度及 optimizer 状态的拥有者。逐事件推进前向 all-gather、参数释放、反向所需重聚、reduce-scatter 和本地更新；再比较前向后保留完整参数与重新分片的差异 [S6]。

验收应核对每个事件的张量拥有关系、逻辑存活量及与未分片参考的梯度/更新。临时峰值、预取量和静态状态分别统计，注明是否包含通信缓冲区、activation 和物理副本。FSDP2 的选定行为与 ZeRO 总览分别说明，避免把一种时间线当成所有实现的固定顺序。

**5. MoE：一个 token 怎样送到专家，再回到原位置。**

本地证据：`src/content/transformer-anatomy.html#moe-training` 已说明固定 top-k 集合下的梯度、均衡目标与容量；路由交互尚未计算完整的 dispatch / combine 数值路径。

建议用四个 token、四个专家、top-2 和两张逻辑卡，列出 token–expert 分配、复制后的条目、排列索引、每 rank 收发内容、专家输出及加权还原结果。实现时对照 NVIDIA 的 token dispatcher 文档核验本地预处理、通信和还原阶段 [S7]。

验收应与相同路由和门控定义的单机参考比较输出及梯度，覆盖空专家、偏斜负载和重复 token 的正确合并。容量丢弃改变参与计算的分支，应与同样丢弃规则的参考对照；无丢弃和有丢弃结果不能要求相等。数值例子无需模拟真实通信时延。

**6. 投机采样：把接受与修正拆成概率质量。**

本地证据：`src/content/efficient-inference.html#speculative-decoding` 已有接受率与正差修正公式；`src/scripts/features/efficient-inference.js` 使用 `data-spec-reject` 指定首次拒绝位置，承担流程演示作用。

建议保留现有流程图，再补小词表上的 p、q、接受概率、接受质量、拒绝总质量和修正分布。按原论文的采样目标验证最终分布 [S8]，而非把一次随机输出的相同视为证明。

例如设目标 `p=[0.5,0.3,0.2]`，草稿 `q=[0.2,0.5,0.3]`。接受质量为 `[0.2,0.3,0.2]`，拒绝总质量为 `0.3`，修正分布为 `[1,0,0]`；最终输出质量恢复为 `[0.5,0.3,0.2]`。该账本已用独立 Node 计算核对。

验收应覆盖 p=q、某些概率为零、第一次拒绝以及全部接受。先完成有限分布的解析核对，再把重复抽样频率作为辅助观察；明确保持的是相同前缀和采样规则下的目标分布，不保证两次独立采样得到同一个字符串。

**贯穿专题的补充形式：一个可在 CPU 上运行的小训练实验。**

项目现有源文件中未发现独立的 Python 训练脚本或 notebook。建议采用与站内小型 block 对齐的配置，提供固定小数据、前向/反向、一次更新、保存/恢复和对应断言。无需下载大模型，依赖与精度固定；逐项对照站内数值，运行输出才作为实测记录。CUDA AMP 和真实多卡内容另有环境要求，可在后续实验中接入。

建议先完成前三项及这个共用实验；新增完整页面集中在位置与缓存，其余内容在原专题深化。量化矩阵乘、成对统计实验与 ANN 索引也有增量空间，但相对当前训练工程主线，可排在上述执行链之后。

**本轮检查。**

`node scripts/check-source.mjs` 通过：48 页、61 个脚本、31 份样式。额外核算了上述混合精度裁剪反例与投机采样概率账本。本轮未实施候选交互或运行模型训练；这些条目的验收条件是后续补充的完成标准。

**后续实现需核对的一手资料。**

本轮联网检索与页面打开未返回可核验的正文或摘要，因此没有完成外部资料的实时复核。S1、S2、S3、S4 来自项目现有引用；其余为后续实现的候选来源，地址及适用版本需要确认。本文的覆盖判断来自本地源码，数值例子来自独立计算；框架示例实施时须读取对应文档、固定版本并核验代码行为。

- [S1] RoFormer: Enhanced Transformer with Rotary Position Embedding，2021：`https://arxiv.org/abs/2104.09864`。
- [S2] Transformers 缓存机制：`https://huggingface.co/docs/transformers/cache_explanation`；涉及 `cache_position` 的具体 API 参考固定版本：`https://huggingface.co/docs/transformers/v4.57.3/en/cache_explanation`。
- [S3] PyTorch 2.14，Automatic Mixed Precision examples：`https://docs.pytorch.org/docs/2.14/notes/amp_examples.html`。
- [S4] PyTorch，Saving and Loading Models：`https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html`。
- [S5] PyTorch main，Reproducibility：`https://docs.pytorch.org/docs/main/notes/randomness.html`；生成器状态接口另参考 PyTorch 2.14：`https://docs.pytorch.org/docs/2.14/generated/torch.Generator.html`。
- [S6] PyTorch 2.14，FSDP2 / fully_shard：`https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html`。
- [S7] NVIDIA Megatron Core，token dispatcher：`https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.token_dispatcher.html`。
- [S8] Leviathan 等，Fast Inference from Transformers via Speculative Decoding，ICML 2023：`https://proceedings.mlr.press/v202/leviathan23a.html`。
