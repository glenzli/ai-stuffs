# 模型工程 Wiki

当前五个实测页面的完整来源登记在 `src/experiment-sources.json`。学习、故障回放、
初始化/宽度/Batch、分块交叉熵已重新运行，报告与原始工件保存在
`artifacts/reproduced-20261028T115030Z/`。页面提供完整报告、源码与原始工件包、
SHA-256 校验表；构建前核对这些文件，缺失或被修改时停止构建。
来源迁移、当前结果和验证步骤见 [实验来源核验记录](docs/experiment-reproduction-2026-10-28.md)。

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

一个面向模型训练工程师的中文交互式 Wiki。使用轻量静态生成器，
构建期用 KaTeX 排版数学、parse5 解析 HTML。页面正文、导航、布局、样式和交互
分别维护，输出到 `dist/`；浏览器无需外部 CDN，也不运行公式排版 JavaScript。

## 快速开始

```bash
npm ci
npm run build
npm run serve
```

打开 `http://127.0.0.1:4173/`。

## 项目结构

```text
src/
├── content/             # 页面正文片段，不包含页头、导航和页脚
├── layouts/
│   └── base.html        # 全站共享 HTML 布局
├── scripts/
│   ├── core.js          # 主题、导航等全站基础行为
│   └── features/        # 页面交互模块，按页面按需加载
├── styles/
│   ├── tokens.css       # 颜色、字体等设计变量
│   ├── layout.css       # 页头、导航、主内容、页脚
│   ├── shared.css       # 跨页面组件
│   ├── overrides.css    # 全局响应式与顺序覆盖
│   └── *.css            # 页面或领域样式
├── assets/
│   └── favicon.svg
├── pages.mjs            # 页面元数据、导航、样式和能力映射
├── wiki.mjs             # 主题归属、顶部导航与图标
├── learning.mjs         # 六步学习主线、选读与阅读标签
├── references.mjs       # 每页适用范围与一手资料
└── search.mjs           # 全站概念搜索索引
```

`dist/` 是构建产物，不要直接编辑。页面 CSS/JS 文件名带内容哈希，
每次构建都会更新，避免浏览器继续使用旧资源。
全站搜索数据同样使用内容哈希。左侧知识目录由 `wiki.mjs` 生成；
移动端目录与搜索使用原生 dialog，页内目录从带 ID 的 H2 自动生成。

## 主题与配色

展示名称统一为“模型工程 Wiki”，目录和 npm 包名为 `ai-stuffs`。
阅读底色、文字、标题、导航和操作使用黑白灰；中性强调色由 `tokens.css` 的
`--accent` 系列维护。蓝、紫用于区分比较曲线和任务，橙用于差异或未达到阈值，
琥珀用于警告，红用于错误和非有限状态。浅深主题均不使用绿色。
图表同时保留文字标签、数值或线型，颜色只辅助表达信息。

## 学习路线与主题目录

指南提供“从头学习”和“按问题查阅”两种入口。默认顺序为输入与监督、模型前向、
损失与更新、学习与泛化、稳定与复现、系统与扩展；前四步形成最小学习闭环。
完整路线为每步提供前置、自检与按需深入，之后连接后训练、生成检索和能力应用。

`src/learning.mjs` 是学习顺序、选读和阅读标签的唯一配置；指南、完整路线和主线
页面的前后篇由 `scripts/render-learning.mjs` 生成。`src/wiki.mjs` 按主题维护每页
唯一归属，同时提供顶部导航，难度与结果形式在页面单独标注。
构建前的源码检查和 `test:reading` 会核对页面覆盖、主线、锚点和同题实测展示。
修改路线应同时检查正文、搜索摘要与适用范围；各计算示例的假设和数值保持独立。

## 编辑页面

1. 在 `src/content/<slug>.html` 修改正文。
2. 页面标题、描述、样式和需要加载的能力在 `src/pages.mjs` 配置；主题归属在 `src/wiki.mjs` 维护，顶部导航随主题生成。
3. 页面样式放入 `src/styles`，交互放入 `src/scripts/features`。
4. 执行 `npm run build`。

`src/references.mjs` 集中维护每页的适用范围与一手资料，构建时统一生成页尾。
新增公式应区分原始分数、logit 和求导变量；交互中的计算应与文案使用同一口径。
模拟数据、资源估算和真实模型配置需显式标注，不能把教学结果当成模型实测结论。
相同概念的完整推导只放在负责的专题页，概览与指南用链接连接。

共享页头、导航和页脚只存在于 `src/layouts/base.html`，不要在正文里复制。

## 页面地图

目前共 61 页，页面正文、目录、搜索与适用范围逐页对应。

- `training-engineer-guide.html`：建议的总入口、七个工程契约、生命周期和问题雷达
- `index.html`：Embedding 与训练全局路径
- `representation*.html`、`training*.html`、`posttraining*.html`：核心学习路径
- `engineering.html`、`data-tokenizer.html`、`model-evaluation.html`、
  `model-artifacts.html`、`training-systems-deep.html`：训练工程链
- `transformer-*.html`、`architecture-variants.html`、`inference-engine.html`：结构与推理
- `frontier.html`、`scaling-laws.html`、`rl-posttraining.html`、
  `model-case-studies.html`：前沿决策
- `lab.html`、`simulator.html`：可选的检索与训练互动实验
- `generation-mechanisms.html`、`reasoning-compute.html`：自回归、扩散、CoT、Thinking 与 J-space
- `multimodal.html`、`agents-context.html`：多模态接口、工具与上下文状态
- `distillation-data.html`、`efficient-inference.html`：Teacher、合成数据、量化与投机解码
- `optimization-dynamics.html`、`parallel-correctness.html`：优化器状态、初始化、全局 loss 与多卡正确性
- `performance-analysis.html`、`engineering-cases.html`：Roofline、性能关键路径与逐步诊断案例
- `finetuning-recipe.html`：数据条数、监督位置、packing、更新次数、LoRA 与数据学习曲线
- `packing-correctness.html`：多轮模板、token/label、样本隔离、padding 与 packing 的梯度一致性
- `lora-mechanics.html`：A/B 初始化、冻结与输入梯度、SGD、量化底座和合并
- `uncertainty-calibration.html`：选择性回答、校准、语义熵与查证边界
- `continual-learning.html`：新旧任务干扰、保留能力与阶段评估
- `learning-mechanisms.html`：上下文学习、归纳头、特征与因果解释
- `training-environments.html`：训练任务、状态、验证器和隔离
- `backpropagation.html`、`numerical-computing.html`：整层反传、数值梯度与浮点误差
- `attention-kernels.html`、`distributed-execution.html`：在线归一化与两卡分片计算
- `activation-checkpointing.html`、`decoding-algorithms.html`：激活生命周期和输出算法
- `tensor-layout.html`：stride、offset、视图共享与复制
- `position-cache.html`：RoPE 旋转、相对点积、padding 与增量缓存的 logits 一致性
- `engineering-check.html`：样本流、监督、形状、通信和缓存的综合检查
- `context-parallel.html`：序列分片、全局 mask、K/V 块通信和梯度归还
- `rl-update.html`：固定 rollout 的 log-prob、裁剪、精确 KL、参数梯度与连续更新
- `quantized-linear.html`：W8A8 整数矩阵乘、零点、scale、bias 与输出重定标
- `paired-evaluation.html`：逐题 JSON 导入、来源组重采样、精确置换与重复选择
- `ann-index.html`：IVF 列表、Flat 参考、近邻与任务召回、编码版本和真实索引
- `learning-run.html`：固定符号任务的真实学习、自由生成、独立组合、长度外推、恢复与缓存
- `training-failures.html`：同 checkpoint/输入的数值、学习率与监督故障回放和受控修复
- `initialization-width.html`：深度、Norm 与残差尺度，以及标准/μP 的参数组和保留宽度迁移
- `batch-noise.html`：逐序列梯度、总体协方差、有无放回抽样和同消费 token 预算训练
- `chunked-cross-entropy.html`：不保存完整 logits 的输出投影、精确 loss/梯度、保存量与 CPU 时间

计算链条补充沿原专题深化：数值页包含完整混合精度更新；工件页包含真实小模型的
checkpoint 恢复对照；系统页包含 FSDP 风格参数生命周期；权重页包含 MoE 搬运与合并；
量化与投机页包含接受/修正概率质量的精确验证。

监督与梯度补充增加两个计算专题，并在并行正确性页展示跨卡负例、远端表示梯度和
DDP 参数归约。反传页提供独立 PyTorch CPU 对照入口，使用同一输入和权重检查
自动微分、参数更新、新进程恢复和真实双进程 Gloo 通信。

保存量与梯度补充沿原专题扩展：Attention 页包含 LSE 重建和分块反向，激活页包含
全局负例的梯度缓存与 RNG 重放，并行正确性页包含独立参数范数、裁剪和统一跳步。
性能页可读取 `lab:memory` 的本地报告，比较梯度误差、逻辑成本、实际耗时和 profiler。

现代专题的扩展沿现有页面组织：RL 页负责组内信号与策略陈旧度；蒸馏页负责学生前缀；
Agent 页负责记忆保留与持续学习边界；评估页负责重复可靠性与判分器误差；
多模态页负责联合训练和流式处理；模型案例页按明确版本补充跨层 KV 与输入计算路径。

新增页面要同时配置 `pages.mjs`、`wiki.mjs`、`search.mjs`、`references.mjs`。
新专题共用 `modern.css`；全站阅读外观在 `wiki.css` 维护，不要在正文重复写导航。
工程专题的少量专属样式在 `engineering-lessons.css`；各领域响应式规则跟随领域样式，
架构页面不再依赖推理页样式。禁止用整页横向裁切隐藏溢出，宽表应有自己的滚动容器。
图标来自 Lucide，许可随构建输出于 `assets/lucide-LICENSE.txt`。

## 检查

2026-10-28 的当前副本缺少部分历史 `artifacts/` 原始报告、checkpoint 与张量文件。
页面构建使用随源码提供的内置快照，可以独立运行；涉及原始报告哈希、完整逐题结果
和已保存张量的来源核验仍需对应工件。缺失清单与本次结果见
[整体检查与主题调整记录](docs/model-engineering-review-2026-10-28.md)。

源代码检查与构建：

```bash
npm run check
npm run test:core
npm run test:reading
npm run test:content
npm run test:math
npm run test:workflows
npm run test:research
npm run test:exact
npm run test:details
npm run test:chains
npm run test:objectives
npm run test:memory
npm run test:decisions
```

浏览器回归需要先启动可调试的 Chrome：

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/ai-stuffs-check \
  about:blank
```

然后分别运行：

```bash
npm run test:interactions
npm run test:audit
npm run test:decision-ui
```

`test:interactions` 覆盖主要交互、控制台异常与移动端整页横向溢出；
`test:audit` 等待所有页面模块初始化完成，检查资源/HTTP/控制台错误、全站链接和锚点、
目录、重复 ID、表单标签、SVG 名称及正文溢出。可设置 `VIEWPORT_WIDTH`、
`VIEWPORT_HEIGHT` 和 `AUDIT_THEME=dark`，例如：

```bash
VIEWPORT_WIDTH=768 npm run test:audit
VIEWPORT_WIDTH=1440 AUDIT_THEME=dark npm run test:audit
EXPAND_MATH=1 VIEWPORT_WIDTH=390 npm run test:audit
```

`test:content` 用有限差分核对 InfoNCE/SGNS/DPO 的导数，并检查缓存计数、
模拟器负例语义、优化器位移、全局梯度分母、Roofline 单位和全站资料元数据；不启动模型训练。
`test:workflows` 检查微调记录与 packing 账本、LoRA 参数、组内优势、裁剪、前缀 KL、
重复成功率、判分器误差、记忆状态与流式队列，并覆盖空输入和边界状态。
`test:research` 检查选择性回答、语义熵、路径概率、任务冲突与精确二次损失、
矩阵方向、归纳电路消融和环境验证器；模拟与真实模型实测分别标注。
`test:exact` 核验完整小型 block 的解析梯度、在线 Attention 等价性、二进制舍入、
两卡张量重建、激活重算梯度、采样筛选和有限语言约束。
`test:details` 核验样本分发、BPE、存储地址、词表 CE、流水线依赖、请求缓存和综合题答案。
`test:chains` 核验位置旋转/缓存、缩放与裁剪顺序、恢复轨迹、分片生命周期、专家梯度和投机概率。
`test:objectives` 核验多轮监督、packing 等价和样本隔离、完整参数有限差分、LoRA 梯度及
合并、跨卡对比学习的梯度贡献与页面状态；不依赖 Python。
`test:memory` 核验分块反向有限差分、空行/尾块、梯度缓存随机重放、全局范数和
参数副本计数、非有限跳步及本地报告读取；不依赖 Python。
`test:decisions` 核验上下文分片与 Q/K/V 有限差分、RL 全参数梯度、量化算术、
配对统计、种子隔离、精确置换、选择偏差与 IVF 版本一致性；不依赖 Python。
`test:decision-ui` 在 390px、768px 和 1440px 暗色中操作五个新专题，覆盖故障切换、
连续更新、文件导入、转义、错误清理、异步读文件竞态和全部展开的布局；保存截图和 JSON 记录。
`check-source` 同时检查缺失或孤儿页面、搜索/引用/目录映射、重复页面与资源引用。

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

## 内容质量约定

### 数学排版

公式写在正文里，构建期生成可见 HTML 与辅助阅读 MathML，原始 TeX 保留在 annotation。
不要把解释段落放进公式；代码示例仍使用 `pre/code`，不会被公式渲染器扫描。

```html
<figure class="math-block">
  <div class="math-display" data-math>p_\theta(r,a\mid x)=p_\theta(r\mid x)\,p_\theta(a\mid x,r)</div>
  <figcaption>x 是问题，r 是推理序列，a 是答案，θ 是参数。</figcaption>
</figure>
```

行内形式使用 `<span class="math-inline" data-math="inline">...</span>`。
`data-math` 内只放 TeX 纯文本；HTML 中的 `<`、`&` 需分别写为 `&lt;`、`&amp;`，
例如 `\begin{aligned}a&amp;=b\\c&amp;=d\end{aligned}`。
长推导使用 `details.math-derivation`，主公式不隐藏；长式优先拆行，过宽时可键盘滚动。
交互式公式可预渲染多个状态后切换 `hidden`，不要用 textContent 覆盖已排版节点。
TeX 语法或结构错误会使构建失败，并保留此前输出（公式校验先于删除 dist）。
`test:math` 核验渲染、MathML、转义、代码隔离与新推导的数值关系；
`EXPAND_MATH=1` 会在全站浏览器审核中展开推导并验证字体加载。

文案先直接说明概念和机制，再写适用条件。删除无信息的连续否定、预设误解、口号式总结和重复提醒；保留有具体技术意义的区别、反例和限制。修改正文时同步检查动态说明、搜索摘要、页脚和资料说明。

每个完整专题应说明前提、机制、工程口径、失效方式和下一步证据。概览只连接专题，
不要复制完整推导。论文的特定配方、假设模型与真实实测必须分开；案例和模拟器不能
用来宣称真实模型的性能或泛化。模型版本、引用、计算与可见文案应同步修改。

主验收记录见 `docs/engineering-wiki-acceptance-2026-10-26.md`。

学习动态验证运行 `npm run test:learning`（四份 Node 与四份 Python 测试）；最终验收见 [learning-dynamics-acceptance-2026-10-28.md](docs/learning-dynamics-acceptance-2026-10-28.md)。

学习动态五项补充已完成；npm run test:learning 运行新增Node/Python检查，最终验收见 docs/learning-dynamics-acceptance-2026-10-28.md。
