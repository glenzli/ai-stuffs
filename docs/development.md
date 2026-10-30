# 开发与内容维护

本指南说明源码组织、编辑约定与按范围选择的检查。所有命令从仓库根目录执行；
代码中的相对路径也以仓库根目录为准。阅读与启动见 [README](../README.md)，
模型实验的依赖、重跑和结果采纳见 [实验复现指南](reproducing-experiments.md)。

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

阅读入口、实验预算口径及共享指标卡的校对见
[编辑与布局验收](editorial-alignment-2026-10-29.md)。
`npm run test:editorial` 检查 Tokenizer 三种模式与静态回退内容是否一致。

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

目前共 66 页，页面正文、目录、搜索与适用范围逐页对应。下面按责任列出主要入口；
五项实战的完整导航见 [实验复现指南](reproducing-experiments.md)。

- `training-engineer-guide.html`：总入口、六步学习主线、问题雷达与七个工程契约
- `index.html`：用 Embedding 小任务串起输入、预测、损失与更新
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

知识表述、公式条件与交互计数的最新复核见 [知识正确性复核](knowledge-review-2026-10-29.md)。

学习动态验证运行 `npm run test:learning`；最终验收见 [learning-dynamics-acceptance-2026-10-28.md](learning-dynamics-acceptance-2026-10-28.md)。

## 检查

2026-10-28 已重新生成学习、故障、尺度与 batch、输出层的完整来源，并追加四项实战。
构建会核对 `src/experiment-sources.json` 指定的报告、工件包和校验表；缺失或内容不符时
停止构建并保留原输出。仓库包含这些固定证据；其他历史实验是否重跑，以各轮验收说明为准。
来源恢复记录见 [实验重跑与核验](experiment-reproduction-2026-10-28.md)。

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

浏览器回归需要先运行 `npm run serve`，再在另一个终端启动可调试的 Chrome。
下面是 macOS 的可执行文件路径；其他平台应替换为实际 Chrome 路径：

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
