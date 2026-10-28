# 高价值内容增量评审 · 2026-10-28

后续实施状态：五项内容、独立 CPU 对照和真实 Faiss 索引实验已完成，见 `docs/decision-chains-2026-10-28.md`。下文保留实施前的覆盖评审与建议验收条件。

核对当前页面元数据、相关专题正文、交互模块与最近两轮验收记录后，站点共有 51 页。
监督与 LoRA、跨卡对比学习、分块 Attention 反传、梯度缓存、分片裁剪已有数值实验和
独立框架对照。下一轮建议把这些基础接到长序列分布式计算、后训练参数更新和量化层计算。
优先级按训练工程用途、相对现有内容的增量、现有实验的复用程度判断，不代表工时估计。

| 顺序 | 补充方向 | 已有基础 | 建议新增的结果 |
| --- | --- | --- | --- |
| 1 | 上下文并行的完整前向与反向 | Context Parallel 概览、分块 Attention、两 rank MLP、全局梯度 | 一条序列跨 rank 后，Attention 输出与梯度和完整参考一致。 |
| 2 | RL 从 rollout 到一次参数更新 | 策略梯度推导、组内优势、标量概率比与裁剪 | 从实际 logits 计算 log-prob、目标、参数梯度和更新后的概率。 |
| 3 | 量化矩阵乘与整层误差 | 单值量化、outlier、低位 LoRA 底座 | 展示整数点积、零点、scale、bias 与输出重定标，分别核对算术和近似误差。 |
| 4 | 模型比较的配对统计实验 | 配对重采样说明、重复成功率、判分器误差 | 读取逐题结果，计算差值区间，并演示相关样本与反复选参的影响。 |
| 5 | ANN 索引与精确检索对照 | 小向量全库 cosine 排序、分块与重排 | 固定向量后比较精确搜索和近似候选，定位索引损失与版本错配。 |

## 1. 上下文并行：跨卡后如何保留同一个 Attention

本地依据：`src/content/training-systems-deep.html` 有 Context Parallel 入口，
`src/scripts/features/training-systems.js` 解释序列切分与 K/V 交换。
`src/scripts/features/attention-backward.js` 已有分块反向，
`src/scripts/features/distributed-execution.js` 的完整数值计算目前围绕两 rank MLP。
可以复用现有 Attention 数据和独立 CPU 实验，补齐序列分片后的通信与归约。

建议用五个位置切成 2 + 3 两段，每个 rank 保留本地 Q，逐块接收其他位置的 K/V。
图中逐步展示全局位置、因果可见性、在线 softmax 状态、局部输出，以及 dK/dV 返回拥有者的路径。
先核对完整 Attention 与分片 Attention，再计数消息、计算量和同时存活的逻辑张量。
Megatron Core 的 CP 文档说明了序列维切分、K/V 交换和反向归约；其中对 SP 的比较限于
该文档所指的 Megatron sequence parallelism，不能把所有叫 SP 的方案都定义成同一种机制。[S1]

建议故障开关：按本地 key 单独归一化、用分片内位置做因果 mask、遗漏远端 dK/dV、
各 rank 按本地 query 数平均后错误地等权汇总。

验收条件：

- 完整参考与分片方案的 O、dQ、dK、dV 在固定容差内一致；有共享投影参数时也核对参数梯度。
- 覆盖不等长切分、尾块、全局因果边界和只有一个分片的退化情形。
- 独立 PyTorch autograd 核对参考梯度；两进程 Gloo 验证真实传递与归约。
- 通信字节数注明发送、接收、dtype 和协议；CPU 正确性实验不用于宣称 GPU 通信重叠收益。

建议归属：新增一个上下文并行专题，从 Attention、分布式执行和系统页链接进入。
完整通信推导集中在该专题，原页面保留各自负责的计算。

## 2. RL：奖励怎样变成参数梯度

本地依据：`src/scripts/features/rl-dynamics.js` 的 `groupAdvantages` 和
`clippedSurrogate` 已计算优势与标量目标值；`src/content/rl-posttraining.html` 已说明
old policy、reference、KL、长度归一化和策略梯度恒等式。下一步可以把它们接到小策略的真实参数。
PPO 原论文以交替采样和优化 surrogate objective 为基本流程。[S2]

建议固定一个小词表和短回答集合，由可训练 logits 产生条件概率。
保留回答 token、mask、生成时的 old log-prob、固定 reference 和奖励，
逐项计算当前 log-prob、概率比、优势、裁剪项与 KL，再展示一次 SGD 后的概率。
首先声明具体教学目标：采用哪一种 token/序列归一化、哪一方向的 KL，以及哪些量停止梯度。
可以先用完整小词表计算精确 KL，采样估计器另作扩展。

建议故障开关：对 old log-prob 保留当前计算图、给 prompt/padding 计入策略目标、
把同批 rollout 的行为策略版本随每次更新一起改写、误用长度分母。
其中长度目标可以是不同算法的合法选择；只有与声明的目标不一致时才算实现错误。

验收条件：

- 固定 rollout 后，手算梯度、有限差分和独立 autograd 一致；有限差分避开裁剪不可导点。
- 单独枚举一个有限动作空间，核对期望奖励梯度和策略梯度恒等式；与采样 surrogate 实验分开。
- 覆盖正负优势、零优势、不同长度、零有效 token、裁剪平台及同批数据重复更新。
- old/reference/reward/mask 的梯度边界可见；更新后的概率和参数可追溯到对应目标。
- 框架实验固定版本与简化条件，结果用于核验计算，不宣称小策略实验能证明真实 RL 能力提升。

建议归属：深化 RL 后训练页；若交互与推导过长，再拆成完整计算专题。

## 3. 量化：从一个刻度到一个线性层

本地依据：`src/scripts/features/efficient-inference.js` 的 `quantizeSymmetric`
计算单值量化与 MSE，LoRA 实验使用恢复后的低位权重计算。
当前还可以补上输入和权重共同量化之后的矩阵乘、累加与输出尺度。
整数量化论文的第 2.2—2.4 节给出矩阵乘、零点修正、32 位累加、bias 和重定标的计算基础。[S3]

建议从一个小型线性层开始，固定输入逐张量 scale、权重逐输出通道 scale，
保证每次点积内部的 scale 不随归约位置改变。先做 W8A8 教学路径：

```text
整数累加 C[i,j] = Σk (qX[i,k] - zX) × (qW[k,j] - zW[j])
恢复输出 Yhat[i,j] = sX × sW[j] × C[i,j] + bias[j]
```

分开显示原始浮点输出、恢复后矩阵乘、整数累加再恢复三条路径。
前两者的差是量化近似误差，后两者应满足同一量化定义下的算术关系。
再扩展输出量化的舍入与饱和；如果展示固定点乘数和移位，单独记录它们引入的误差。
W4A16 可作为权重压缩后恢复计算的对照，明确与 W8A8 的计算路径区别。

验收条件：

- 逐元素核对整数累加、零点修正、scale 轴、bias 单位与输出重定标。
- 覆盖全零、正负值、outlier、饱和和多输出通道；累加范围预先检查。
- 小矩阵与独立 NumPy/PyTorch 参考一致；全精度与量化结果允许明确的近似误差。
- 展示权重误差与实际输入上的输出误差；校准输入和评估输入分离。
- 参数位宽、scale 元数据与实际运行耗时分别记录；教学 JS 的数值数组不冒充低位 kernel 或真实压缩存储。

建议归属：深化量化与投机解码页；完整量化计算较长时再独立成页。

## 4. 评估：这次提升是否足以支持选择

本地依据：`src/content/model-evaluation.html` 已写明逐题配对、相关样本按组重采样和
差值区间；`src/scripts/features/evaluation-reliability.js` 当前交互计算重复成功率和
判分器混淆账本。可以增加一套直接读取旧模型/新模型逐题结果的统计实验。
Dror 等人的论文提供按任务、实验设计和评估指标选择显著性检验的讨论；本轮读取其摘要，
具体重采样方法实施前仍需核对相应方法正文。[S4]

建议输入包含题目 ID、来源组、旧分数、新分数及可选训练 seed。
同时展示逐题赢输平、平均差值、配对 bootstrap 区间和预先声明的最小实际收益。
为同文档多题设置组 ID，对照逐题重采样与整组重采样；明确估计的是逐题平均还是组平均。
另设多次选择 checkpoint 的模拟，保留独立测试集，观察选择过程如何影响结论。

验收条件：

- 配对重采样始终同步抽取同一题的新旧分数；组重采样保留组内关系。
- 固定随机种子可复现；空样本、单样本、重复 ID、缺失配对和退化区间有清楚处理。
- 用可枚举的小型例子检查配对置换计算，并声明交换性等统计前提。
- 固定模型上的样本重采样区间与训练 seed 波动分开报告；不把前者称为全部实验不确定性。
- 发布规则同时考虑预定实际收益与回归切片，不能把一个 p 值解释成模型更好的概率。

建议归属：深化模型评估页，并提供本地结果导入入口，供后续量化、数据配方和 RL 实验共用。

## 5. ANN：检索误差出在表示还是索引

本地依据：`src/scripts/features/retrieval-lab.js` 对小语料执行完整 cosine 计算和排序，
现有页面覆盖候选与重排。可以固定向量，补出近似搜索自身引入的候选损失。
Faiss 官方索引文档列出 Flat 精确检索、IVF 倒排列表和 HNSW，说明 IVF 的 nprobe
选择搜索列表，漏掉包含最近邻的列表会导致失败。[S5]

建议先做小型 IVF：显示向量、质心、列表、查询距离、选中的列表和最终候选。
用相同向量的 Flat 搜索作参考，改变 nprobe，记录访问向量数与相对精确 top-k 的召回。
实际 CPU 索引作为可选实验，HNSW 搜索预算可作为后续扩展。

验收条件：

- 同一距离度量、过滤条件和并列分数规则下，搜索全部 IVF 列表应恢复精确结果。
- 把“ANN 对精确近邻的召回率”和“对人工相关文档的任务 Recall@k”分别命名、计算。
- 展示 query/document 编码版本、归一化和索引版本；同维度也要检查完整接口是否一致。
- 逻辑候选数、索引大小、实测时延分别标注；真实索引实验固定实现版本与构建参数。

建议归属：从现有检索沙盘进入独立索引专题。按训练工程主线排在前三项之后；
如果下一阶段重点转向 Embedding 应用，可以提前。

## 实施组织

建议第一批做前三项，共用现有独立 Python 环境、固定输入与本地报告格式。
新增内容各自给出一个最小正确例子、可定位的故障、独立参考和明确完成标准。
配对统计作为第二批优先项，让之后的模型质量实验可以复用；ANN 随检索主线安排。

本轮交付为评审文档，未修改网站正文或新增训练实验。页面总数来自当前 `src/pages.mjs`；
既有实验状态以 `docs/objective-chains-2026-10-27.md` 和
`docs/memory-chains-2026-10-28.md` 的验收记录为背景，本轮未重跑这些历史实验。

## 一手资料核对

2026-10-28 通过系统 HTTPS 客户端直接读取以下公开资料。检索工具本轮未返回可引用正文，
不作为资料已核实的依据。未将文档的 latest 路径解释为已确认的最新发布版本。

- [S1] NVIDIA Megatron Core，Context Parallel Package。已读 CP/SP 范围、KV 通信和反向归约正文：
  `https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html`。
  另读 Ring Attention with Blockwise Transformers for Near-Infinite Context（2023）摘要，
  用于确认分块序列与跨设备 KV 交换的研究来源；未据摘要推断任意硬件上的重叠收益：
  `https://arxiv.org/abs/2310.01889`。
- [S2] Schulman 等，Proximal Policy Optimization Algorithms（2017）。已读论文摘要，
  本地已有 clipped surrogate 的实现；详细梯度实验实施时须继续核对正文目标：
  `https://arxiv.org/abs/1707.06347`。
- [S3] Jacob 等，Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference（2017）。
  已读 HTML 正文第 2.2—2.4 节相关段落；用于整数算术基础，不外推该论文视觉模型的精度或速度到 LLM：
  `https://arxiv.org/html/1712.05877v1`。
- [S4] Dror 等，The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing（ACL 2018）。
  已读 ACL Anthology 摘要与元数据：`https://aclanthology.org/P18-1128/`。
- [S5] Faiss 官方 Wiki，Faiss indexes。已读索引表、IVF 和 HNSW 说明：
  `https://github.com/facebookresearch/faiss/wiki/Faiss-indexes`。
