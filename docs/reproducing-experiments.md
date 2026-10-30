# 实验复现指南

所有命令从仓库根目录执行，相对路径也以仓库根目录为准。
阅读与本地启动见 [README](../README.md)，页面编辑与浏览器检查见 [开发指南](development.md)。

## 先选择要核验的范围

| 目标 | 所需环境 | 输出与边界 |
| --- | --- | --- |
| 阅读网页、构建与 Node 计算测试 | README 中的 Node 环境；本地预览另需 Python 3 | 使用随仓库保存的来源，不启动模型训练 |
| MTP、GDN/KDA、AttnRes、输出门等固定机制 | 对应归档中的 Node 或 Python 3.10+ 标准库检查器 | 复算声明的公式、梯度与正文读出，不代表完整模型效果 |
| 小模型、框架梯度与 Gloo 对照 | Python 3.12 与 `.venv-framework` 固定依赖 | CPU 训练/恢复与局部通信正确性；按实验选择入口 |
| 真实中文检索 | 单独的 `.venv-retrieval` 固定依赖 | 使用保存的 BGE 模型、语料、向量与索引 |

先按实验选择环境，不必为了阅读页面安装全部依赖。固定依赖用于复现既有结果，
不表示最新版本；完整适用范围以来源卡和对应报告为准。

## 来源与运行前检查

从 `deep-dive.html#verified-experiments` 可按症状查找当前十三组实测来源，
查看参考对象、结论边界、运行条件和完整下载。支持用 `?verify=padding` 等链接
保存搜索条件；关闭 JavaScript 后仍可展开阅读全部来源。
独立重跑命令输出到新目录，来源清单登记的复现示例另行保留原文。
导航与命令检查见 [实测导航核验](experiment-navigation-2026-10-29.md)。

重跑前可先执行下面的检查，按来源组核对固定依赖与模块导入、已登记报告和原始文件、
输出目录冲突。检查不会启动训练、下载模型或创建实验输出目录；通过预检仍需实际核验结果。

```sh
npm run check:experiment -- --only padding-inference --out-dir artifacts/padding-inference-verify-run
```

省略 `--only` 检查全部十三组；`node scripts/check-experiment.mjs --json` 提供机器可读结果。
环境准备、退出码与检查边界见 [复现前检查](experiment-preflight-2026-10-29.md)。

`decoding-algorithms.html#sampling-state` 提供固定分布的随机流账本：
比较全局流、批次槽位和请求 ID 的分配，并核对暂停后的游标恢复。
输入清单全部公开，输出由浏览器直接计算；它是计算演示，独立于十三组模型实测。
边界与核验见 [随机流与请求身份](sampling-state-2026-10-29.md)。

当前十三组实测来源登记在 `src/experiment-sources.json`。学习、故障回放、
初始化/宽度/Batch、分块交叉熵已重新运行，报告与原始工件保存在
`artifacts/reproduced-20261028T115030Z/`。页面提供完整报告、源码与原始工件包、
SHA-256 校验表；构建前核对这些文件，缺失或被修改时停止构建。
来源迁移、当前结果和验证步骤见 [实验来源核验记录](experiment-reproduction-2026-10-28.md)。
新增语言基线、同监督 token 预算训练与真实中文检索的证据保存在
`artifacts/extension-20261029T022617Z/`，见 [五项扩展与核验](evidence-extension-2026-10-29.md)。
候选并集、真实配对统计和文本模型逐文档复算见
[误差与评估深化](evidence-audit-2026-10-29.md)，原始实验来源保持不变。

```sh
npm run check
npm run test:provenance
npm run test:learning
```

`scripts/package-experiments.mjs` 只打包已完成的新实验，
`scripts/experiment-sources.mjs` 负责来源核对、构建复制及页面下载入口。

Git 仅跟踪源码、文档、依赖锁文件及当前登记的完整实验来源。`.gitignore` 排除本地
虚拟环境、`node_modules/`、`dist/`、缓存和其他临时工件；当前报告、checkpoint、
张量和下载包通过明确清单保留。`.gitattributes` 保持来源文件的原始字节，避免换行
转换破坏 SHA-256。以后切换默认实验来源时，应同时更新 manifest 和工件保留清单。

## 已归档的训练机制核验

截至 2026-10-30 的近期公开训练技术、固定版本的一手来源及核验范围见
[训练技术补充记录](training-techniques-2026-10-30.md)。页面入口为后训练概览的
“近期公开训练技术”；小规模公式重算与完整模型训练复现分别记录。
采样与训练分布校正、原生低位训练和 rubric 奖励的机制补充，以及三组固定输入的
离线核验见[训练机制核验记录](training-contracts-2026-10-30.md)。
MTP 标签与梯度、草稿头后训练、WSD/checkpoint 平均及来源版本差异见
[MTP 与 WSD 补充记录](mtp-wsd-2026-10-30.md)，含固定输入、更新轨迹和可离线执行的核验命令。
MoE bias 控制、分模态统计与 mHC/Sinkhorn/Single-Pass 的版本区别和固定算例见
[MoE 与 mHC 核验记录](moe-mhc-2026-10-30.md)。
Engram 查表训练、逐头矩阵方向、有符号 L2 均衡更新和 DualPipe 资源依赖的固定输入与版本边界见
[查表、更新与重叠核验记录](lookup-updates-overlap-2026-10-30.md)。
DSpark 条件接受率、验收长度成本与调度选择偏差的版本边界和离线核验见
[验收预算补充记录](dspark-horizon-2026-10-30.md)。
Gated DeltaNet/KDA 的状态更新、分块输出与跨块梯度，以及固定状态和混合缓存的边界见
[循环状态核验记录](delta-state-2026-10-30.md)。
AttnRes 的深度来源、Full/Block 结构差异、精确两阶段合并及局部梯度见
[深度注意力核验记录](attnres-2026-10-30.md)。
Gated Attention 的输出门位置、逐头/逐通道粒度、局部学习与 KV 反例见
[输出门核验记录](gated-attention-2026-10-30.md)。

例如，输出门的四张正文表、17 项局部导数与 KV 反例可以直接重算：

```bash
python3 artifacts/gated-attention-20261030T125431Z/check-mechanisms.py
```

这类检查器默认比较保存结果，不覆盖旧实验；具体 flags 以各轮说明为准。
历史验收记录中的 README、源码和全站构建哈希对应当时的提交；需要核对整个历史
快照时应检出对应提交。后续文档更新不会改写旧归档的哈希记录。

## CPU 训练与恢复实验

```bash
npm run lab:cpu
```

无需额外依赖或下载模型。运行本站两位置 Transformer 的八次 AdamW 更新，第三次更新后
保存 JSON checkpoint，再分别启动新 Node 进程完整恢复或遗漏状态恢复。
默认写入新的 `artifacts/cpu-lab-<timestamp>/`；可以用 `-- --out-dir <新目录>` 指定位置。
记录包含运行环境、每步输入、loss、梯度、参数、随机状态与数据游标；完整恢复必须与连续基线一致。
命令行实验和页面复用计算模块。它验证本地数值及恢复语义，不执行 CUDA、框架 AMP 或多卡训练。
验收说明见 `docs/computation-chains-2026-10-27.md`。

## 独立 PyTorch 对照实验

```bash
python3.12 -m venv .venv-framework
.venv-framework/bin/python -m pip install -r scripts/framework-requirements.txt
npm run lab:framework
```

框架实验单独安装 PyTorch 2.9.0 和 NumPy 2.3.4；静态网站与 Node 测试保持原依赖。
默认使用 `.venv-framework/bin/python`，可设置 `FRAMEWORK_PYTHON` 为其他兼容环境的
Python 绝对路径。`-- --out-dir <新目录>` 指定输出位置，已存在目录会被拒绝，避免覆盖记录。

Node 导出三个新增实验和整层小模型的输入、权重与参考值，独立 Python 实现使用
CPU float64 自动微分计算实际结果。覆盖 100 种 packing 配置、48 种 LoRA 配置、
9 种对比学习配置、三个类别的整层梯度及八步 AdamW；第三步保存后启动新进程恢复。
另为三种负例/梯度方案各启动两个 CPU Gloo 进程，运行可微 gather、DDP 与 SGD。
Gloo 只使用本机 loopback 通信，需要运行环境允许本地进程绑定端口。

默认输出到新的 `artifacts/framework-lab-<timestamp>/`，包含 `reference.json`、
`checkpoint.pt`、连续/恢复轨迹、各 rank 结果、`environment.json` 和 `report.json`。
版本固定为本项目的可复现参考，不代表最新版本。实验不下载模型、不运行 CUDA/NCCL，
不评测真实模型质量或吞吐。验收说明见 `docs/objective-chains-2026-10-27.md`。

## 保存量、梯度与 CPU 性能实验

沿用上面的独立 Python 环境，执行：

```bash
npm run lab:memory
```

默认创建新的 `artifacts/memory-lab-<timestamp>/`；也可用 `-- --out-dir <新目录>`。
验证 144 种分块 Attention 配置、24 种梯度缓存配置、60 种裁剪配置；包括完整 autograd
对照、真实 PyTorch dropout 重放和两个 CPU Gloo 进程的独立拥有者归约与跳步协议。

在不同的执行中分别采集：3 次预热后 12 次调用耗时、`saved_tensors_hooks` 保存对象
所引用的唯一 storage 峰值，以及 CPU profiler 形状/时间/内存事件。前者不含 profiler
和 hooks；存储引用量包含保存的输入和参数，但不是分配器峰值或进程 RSS。
裁剪耗时只测本地归约算术，Gloo 通信正确性另验，不报告网络速度。

目录包含 `report.json`、`reference.json`、两份 rank 结果、`environment.json` 和五份
`*-trace.json`。性能页的“读取本地报告”可选择 `report.json`，在浏览器显示实际耗时、
存储引用、逐方案误差、逻辑账本和 profiler 热点；文件不会上传。
完整说明见 `docs/memory-chains-2026-10-28.md`。

## 上下文、策略、量化、统计与索引的独立实验

沿用 `.venv-framework` 的 PyTorch 2.9.0 / NumPy 2.3.4 环境：

```bash
npm run lab:decisions
```

默认创建新的 `artifacts/decision-lab-<timestamp>/`；可用 `-- --out-dir <新目录>`。
Node 导出与页面一致的固定输入，独立 Python 实现核对 300 种上下文配置、240 组四步策略更新、
20 种量化配置、30 种统计配置、5 种选择实验与 432 种 IVF 查询配置。
上下文用完整/分块目标 autograd 核对，RL 用实际 tensor 计算图求导，量化与统计用独立 NumPy。
另启动两个 CPU Gloo 进程，每 rank 验证 36 种配置，逐块交换 K/V、反向重放并归还梯度。

目录保存 `reference.json`、`rank-0.json`、`rank-1.json`、`environment.json` 和 `report.json`；
包含每个通信块、数值比较和参数更新。需要本机 loopback 通信权限；不下载模型或执行 CUDA。
这些输出检验声明的计算，不作为语言模型质量或 GPU 性能的证据。

真实 Faiss CPU 索引实验使用额外固定依赖：

```bash
.venv-framework/bin/python -m pip install -r scripts/index-requirements.txt
npm run lab:ann
```

如果已有 Python 环境由 uv 创建且没有 pip，可改用
`uv pip install --python .venv-framework/bin/python -r scripts/index-requirements.txt`。
固定 Faiss 1.12.0 / NumPy 2.3.4；这是复现实验版本，不代表最新版本。

默认创建新的 `artifacts/ann-lab-<timestamp>/`，同样支持 `-- --out-dir <新目录>`。
先用页面的 12 向量、固定质心和三条查询核对真实 IVFFlat，再用 8,192 个 32 维合成向量、
128 条查询、k=10、64 个列表比较 Flat 与不同 nprobe 的 IVF。
单线程 CPU、float32，3 次预热后记录 12 次批量查询时间；建索引和磁盘 I/O 不计入查询时间。
保存向量、查询、真实索引文件和 `report.json`，并重新加载索引验证结果。
索引文件大小区别于进程内存占用，合成向量的近邻召回区别于模型的任务相关性。

五个新专题可从训练工程师指南的“从计算正确走到实验判断”进入；逐题评估支持本地 JSON，
示例可在页面下载，文件不上传。实现和最终验证范围见 `docs/decision-chains-2026-10-28.md`。

## 学习动态与输出层实验

训练工程师指南的“从小模型的学习过程检验训练决策”连接本轮五个专题。
学习、故障回放与 batch/深度实验共用 `scripts/learning_core.py`；稳定接口和标签、
mask、缓存、评价约定见 `docs/learning-core-api.md`。宽度实验单独使用固定深度的 MLP，
并通过 `mup==1.0.0` 作者实现处理 μP；这个声明不用于小型 decoder。

沿用上面的 PyTorch 2.9.0 / NumPy 2.3.4 独立环境。尺度实验额外依赖固定版本的
μP 和 PyYAML，安装范围以 `scripts/dynamics-requirements.txt` 与实施说明为准。

先完成基础框架环境，再补齐这些固定依赖；`--no-deps` 保持已安装的共享依赖版本：

```bash
.venv-framework/bin/python -m pip install --no-deps -r scripts/dynamics-requirements.txt
```

```bash
npm run lab:learning
npm run lab:failures
npm run lab:dynamics
npm run lab:output
```

每个 runner 默认创建新的 `artifacts/` 子目录，可用 `-- --out-dir <新目录>` 指定，
已有目录会被拒绝。学习报告分开 teacher-forced loss、自由生成 token accuracy 与
完整序列 exact match，保存多种子曲线、逐题预测和恢复/缓存对照。故障报告保存同一
checkpoint 和输入的实际事件及修复比较。尺度与 batch 报告分开固定参数的梯度统计、
初始与更新后尺度，以及同消费 token 预算的真实训练。输出层报告用完整 CE 自动微分
核对 loss、dH/dW/db，分别记录逻辑元素、保存张量引用和未插桩 CPU 时间。

耗时测量应在重训练完成后单独安排，并记录线程、环境及并发干扰限制；普通正确性
测试可以同时运行。默认页面数据由实际报告提取，报告来源与哈希在页面内显示。
本地 JSON 只在浏览器读取；读取失败清空旧结果，外部文字转义且最后选择的文件生效。

浏览器入口为 `npm run test:learning-ui`，沿用现有 4173 网站与 9222 CDP 服务，
在 390px/768px 浅色与 1440px 暗色中实际操作控制、导入报告及边界状态。
所有新验收记录放入独立 `artifacts/learning-dynamics-qa/`，保留已有轮次；最终结果与
已覆盖范围见 `docs/learning-dynamics-acceptance-2026-10-28.md`。

## 五项可运行实战

沿六步学习路线进入以下实测专题，页面均链接到同一份完整来源包：

- `generalization-analysis.html`：冻结反转权重，检查 EOS、位置扰动与长度支持扩展；保留三个 seed 和预算混杂。
- `text-training-project.html`：38 篇本站中文文档、训练集专用 byte BPE、约 30 万参数 decoder；训练、完整评估、原始生成、保存与新进程恢复。
- `reproducibility-lab.html`：对齐初值和输入，记录精度及数据顺序变化后的首个可观察差异；历史 seed 47 变化仍未定位原因。
- `training-step-profile.html`：同一模型的完整 step 阶段计时、保存引用与精确分块 CE 对照，先核验数值再解释性能。
- `retrieval-project.html`：877 个真实段落、36 个预先标注的问题、固定 BGE 编码器、BM25、真实 Flat/IVF 索引与候选内 RRF 排名融合。

文本项目另提供同 tokenizer 的 unigram/bigram 基线；泛化页另提供三 seed、每次恰好
410,000 个监督 token 的长度支持对照。五页均有预测题、返回主线和可复制的实验选项链接。

```bash
npm run test:practice
npm run lab:practice -- --out-dir artifacts/practice-my-run
```

沿用固定的 `.venv-framework` 环境；重跑包含三 seed 的两组训练，需要数分钟。
输出目录必须尚不存在。runner 复用保存的语料和原始 HTML，避免 Wiki 后续文案变化改变训练输入；
`export-text-corpus.mjs` 可单独导出当前页面，但那是另一份数据，不能替代本轮冻结来源。
这四项原始实战的脚本不会下载外部语料、预训练模型或 tokenizer。再次运行只生成新报告、
页面数据候选和 `candidate.json`，不修改当前来源清单或页面默认值。检查后显式采纳：

```bash
npm run accept:experiment -- --candidate artifacts/practice-my-run/candidate.json
```

采纳会一起更新来源清单、页面投影和工件保留规则；写入失败会回滚，进程中断可用
`npm run accept:experiment -- --recover` 恢复。旧工件保留；采纳本地结果不执行 Git 提交。
复制到新项目时，需要一并保留原反转实验的完整来源。

加载已保存文本模型：

```bash
.venv-framework/bin/python scripts/text-inference.py \
  --checkpoint artifacts/practice-20261028T152417Z/practice/text/seed-11-final.pt \
  --tokenizer artifacts/practice-20261028T152417Z/practice/text/tokenizer.json \
  --prompt '模型训练'
```

CLI 按同一包的 `SHA256SUMS` 核对权重与 tokenizer 身份，再用固定 greedy 策略生成。
下载包解压后也可用对应的绝对路径；保留 `practice/SHA256SUMS` 与 `practice/text/` 的层级。
`test:practice` 从权重复算所有文本评估和样本、16 步恢复、逐题汇总、首次差异张量和完整模型梯度；
不会重新训练 800/1,200 步。报告的通过状态表示声明的链路核验通过，语言生成仍不连贯，
长度 9–10 的外推仍失败，CPU double 时间也不代表 GPU 性能。
实现和验证范围见 [四项实战说明](practice-expansion-2026-10-28.md)。

### 语言基线、等预算训练与真实检索

语言基线和等预算训练沿用 `.venv-framework`；中文检索使用单独的可选环境：

```bash
python3.12 -m venv .venv-retrieval
.venv-retrieval/bin/python -m pip install -r scripts/retrieval-requirements.txt
npm run test:evidence
npm run lab:evidence -- --out-dir artifacts/extension-my-run
```

固定版本是本轮复现参考。检索默认使用已保存来源包中的 BGE 权重与 tokenizer，离线运行；
需要另取原始模型时，可用 `scripts/download-retrieval-model.py --out-dir <新模型目录>`，
它只下载固定提交并校验发布者提供的权重哈希。`RETRIEVAL_PYTHON` 可指定兼容 Python，
runner 的 `--model-directory` 可指定另一份相同版本的本地模型。

编码和 Faiss 索引由 runner 分成独立进程执行，以保留已核对的运行边界。

`lab:evidence` 支持 `--only language-baselines`、`--only matched-budget` 或 `--only retrieval`。
每个子目录分别生成候选；核对后可把多个 `candidate.json` 一起传给 `accept:experiment`。
重跑不覆盖已有目录，也不自动采纳。`test:evidence` 从完整报告和保存的概率计数、权重、
向量、模型、索引独立复算，不重新训练全部模型；它会抽样重新编码真实输入。
详细命令、结论边界和失败记录见 [五项扩展与核验](evidence-extension-2026-10-29.md)。

### 从总体分数到逐项证据

检索页在原始排名上追加两路候选并集与截断 RRF，分别展示候选覆盖、最终命中、
MRR 和来源组重采样区间。可直接跳转配对评估页的真实数据预设，在浏览器复算同一批
24 道测试题、12 个来源组；模拟示例与真实来源分别标注。

文本项目从三个最终 checkpoint 重新计算全部逐位置损失，逐文档展示 bigram 对照、
token 权重、宏微平均及整文档重采样。本轮没有重新训练或按新切片重新选择模型。

```bash
npm run lab:audit -- --out-dir artifacts/audit-my-run
npm run test:evidence-audit
```

沿用 `.venv-framework`；只读已有来源，不下载新模型。runner 仅准备候选，支持
`--only retrieval-comparison` 或 `--only text-evaluation-audit`，仍由 `accept:experiment`
显式采纳。新结论使用已查看的数据集，标为探索性比较；区间不消除标注与数据复用偏差。
完整输入、逐位置分数、权重、实际候选和重采样分布见 [误差与评估深化](evidence-audit-2026-10-29.md)。

### 完整模型的梯度累积与双进程更新

并行正确性页增加同一份 298,176 参数模型的真实更新对照：不等量监督、均匀对照、
不完整尾窗口、全局空监督及一个副本空监督。比较完整 batch、累积、两个 CPU Gloo
进程与默认 DDP 平均；SGD 和恢复第 800 步状态的 AdamW 分别与自己的参考比较。

```bash
npm run lab:accumulation -- --out-dir artifacts/accumulation-my-run
npm run test:accumulation
```

沿用固定 `.venv-framework`；需要本机 loopback 通信，不下载模型。runner 只准备候选，
核对后用 `npm run accept:experiment -- --candidate artifacts/accumulation-my-run/candidate.json`
显式采纳。工件保留完整梯度、独立副本哈希及一次更新状态；下载包中的源码和输入还通过了
离开仓库目录的独立重放。结果和边界见 [完整模型更新核验](accumulation-equivalence-2026-10-29.md)。

## 真实向量的来源过滤与候选数量

沿用原检索项目的 877 段、36 问和保存向量，对照先限定范围的精确前十与全库前 L 条后过滤。
五种范围固定用于全部问题，覆盖范围不足十条、空范围，以及指定证据本来已在范围外的情形。

```bash
npm run lab:retrieval-filtering -- --out-dir artifacts/filtering-my-run
npm run test:retrieval-filtering
npm run accept:experiment -- --candidate artifacts/filtering-my-run/candidate.json
```

只需固定 NumPy 2.3.4，不运行编码器或下载模型。候选包包含输入向量、原文、问题、原始报告、
全部内积/排序和执行源码，可离开仓库独立重跑。机制与分母边界见
[来源过滤核验](retrieval-filtering-2026-10-29.md)。

## 完整模型的填充、读取列与缓存续算

复用文本项目的固定权重和三个原窗口，核验左右填充、逻辑位置、next logits 读取列和
四次固定原始后缀的缓存续算；等长无填充与额外填充均保留。完整模型使用固定正弦位置，
与位置页前半部分的 RoPE 教学配置分别声明。

```bash
npm run lab:padding-inference -- --out-dir artifacts/padding-my-run
npm run test:padding-inference
npm run accept:experiment -- --candidate artifacts/padding-my-run/candidate.json
```

候选包包含原始权重/窗口、全部执行源码、36 路实际张量、540 项逐阶段观测和缓存
前缀哈希，不重新训练或下载模型。详见[完整模型填充核验](padding-inference-2026-10-29.md)。
