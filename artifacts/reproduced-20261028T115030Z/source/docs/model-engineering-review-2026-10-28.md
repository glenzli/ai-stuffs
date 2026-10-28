# 模型工程 Wiki：整体检查与主题调整 · 2026-10-28

本次在 `/Users/g4i/lab/ai-stuffs` 的文件副本上检查。目录没有 Git 元数据；
验证针对实际文件内容，原始源码备份保存在
`/private/tmp/ai-stuffs-before-review-20261028.tar.gz`。

## 结果与修复

共用入口原先直接读取、写入 `localStorage`。浏览器拒绝存储访问时，读取异常会
中断入口，后续搜索、导航和页面实验无法初始化；写入异常会中断主题标签同步。
新增用例在修复前复现这两种失败。现在存储不可用时仍能初始化并切换主题；
存储可用时继续保存偏好，且已保存偏好优先于系统主题。三个回归用例通过。

展示名、页面标题、搜索标题、README 和图标统一为“模型工程 Wiki”，npm 包名为
`ai-stuffs`。共用布局采用黑白灰，数据比较保留蓝、紫、橙；警告使用琥珀，错误使用红。
中性颜色使用 `--accent`，有比较意义的原绿色图表系列改用紫色。未修改实验快照或数值算法。

## 本次验证

| 范围 | 结果 | 记录 |
| --- | --- | --- |
| 源码与构建 | 61 页、91 个脚本、40 份样式、141 处公式通过 | `npm run check` |
| 全部 Node 测试入口 | 361 项通过，4 处因缺失历史工件失败；无跳过 | `artifacts/model-engineering-review-20261028/node-final.log` |
| Python 学习核心 | 12 项通过 | `test-learning-core.py.log` |
| Python 分块 CE | 13 项通过 | `test-chunked-ce.py.log` |
| Python 动态实验 | 7 项通过，1 项原始报告核验缺文件 | `test-training-dynamics.py.log` |
| Python 故障回放 | 缺报告及 checkpoint，初始化阶段中断 | `test-failure-replay.py.log` |
| 390px 浅色 | 全部 61 页通过 | `browser-audit.json` |
| 1440px 深色 | 全部 61 页通过 | `browser-audit.json` |
| 768px 浅色 | 总览、指南、五个学习动态专题、配对评估共 8 页通过 | `browser-audit.json` |

上表中缩写的日志均位于 `artifacts/model-engineering-review-20261028/`。
Node 基线为 358 项通过、相同 4 处工件缺失；修复后新增三个通过用例。
缺文件导致三个测试文件未完成加载，因此当前测试总数不能直接与旧验收记录的
完整用例数比较。未将这些失败隐藏为通过或跳过。

浏览器共检查 130 个页面/主题/宽度组合：初始化标记、单一 H1、页面标题、目录、
重复 ID、控件标签、SVG 名称、公式与 MathML 数量、页面宽度、图片和可见元素颜色。
另外核对全站本地链接及锚点，未发现断链、横向溢出、绿色残留或控制台异常。
这是渲染与结构检查，未穷举所有输入组合。

实际操作了首页训练步骤与表示尺度、全站 LoRA 搜索、Escape 关闭搜索、移动目录
打开与关闭，以及学习报告的 seed 47 / 未见长度切换。该条件仍显示训练 EM 93.33%、
独立测试 EM 91.37%、长度外推 EM 0.00%；图表保留蓝紫区分和实线/虚线。
浅深切换后恢复浅色及默认窗口尺寸，预览停在训练工程师指南。

源码重点复查了投机采样的接受/修正质量、量化零点与 bias 单位、梯度 Gram 与
有无放回的方差口径、任务生成与监督、缓存和恢复边界。已执行的数值验证未发现
新的公式计算失败；这些检查不构成对全部文字断言或任意真实模型配置的完整证明。

抽查版本化一手资料入口，包括
[DeepSeek-V4.1-Flash v1](https://arxiv.org/html/2609.19969v1)、
[J-space 研究](https://www.anthropic.com/research/global-workspace)、
[PyTorch AMP 示例](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html) 和
[投机解码论文](https://arxiv.org/abs/2211.17192)。
本地实验保持 PyTorch 2.9.0 / NumPy 2.3.4，文献中的版本与本地参考环境分别记录。

## 仍缺少的来源工件

| 类别 | 当前代码所引用的原始位置 |
| --- | --- |
| 学习与重新核验 | `artifacts/learning-revalidated-20261028T035637.650035Z/report.json`、`artifacts/learning-lab-20261028T034636.240515Z/report.json` |
| 故障与回放 | `artifacts/failure-lab-2026-10-28T03-56-27-850Z/` 中的报告、`pre-step.pt` 和逐次张量文件 |
| 初始化、宽度与 batch | `artifacts/dynamics-reanalysis-20261028-02/` 与 `artifacts/dynamics-lab-20261028-01/` 中的原始报告及重分析记录 |
| 输出层与计时 | `artifacts/output-lab-2026-10-28-timing-01/report.json` |

源码内置快照可用于展示和结构核验，无法替代这些原始文件的哈希及张量来源核验。
重跑实验会产生新的环境信息与工件身份，应写入新目录；不能补造历史文件或沿用旧哈希。

## 预览

- 本地服务：`http://127.0.0.1:4173/`
- 指南截图：`artifacts/model-engineering-review-20261028/guide-neutral.jpg`
- 曲线浅色截图：`artifacts/model-engineering-review-20261028/learning-light.jpg`
- 曲线深色截图：`artifacts/model-engineering-review-20261028/learning-dark.jpg`
