# 近期公开训练技术：来源与核验记录

核对日期：2026-10-30。基线：`918d1df86b03b9a3e30c48a6ee0e45d8f6bfaa07`。
本轮整合现有页面，不增加独立专题，也不改变既有模型实验的默认来源。修改保持未提交、未 push。

## 阅读入口与归属

后训练概览新增“近期公开训练技术”，以六个案例连接现有专题：

| 归属 | 案例与核对版本 | 核对重点 |
| --- | --- | --- |
| RL 后训练 | DeepSeek V4.1-Flash-1017，2026-10-17 发布说明 | 只确认 self-feedback RL 的采用；奖励、反馈生成器与优化配方不补写。 |
| RL 后训练 | Qwen GSPO v2 / SAPO v2（2025-12-01） | 长度归一化序列比率、裁剪方向、软门控与局部导数。GSPO 作为前置机制。 |
| RL 后训练 | QwenGyre v2，2026-10-20 | 弹性调度、轨迹树、评分状态、token 来源与共享目标去重。 |
| 训练环境 | Kimi K2.5 / PARL，2026-01 公开机制 | 训练编排者、冻结子 Agent、辅助奖励退火和关键步数。 |
| 训练环境 | Agent Lightning v1.0 论文 v1 / 官方 v1.0.0 | 真实 harness、重新分词、rollout 统计单位与同池异步调度。 |
| 蒸馏与合成数据 | Nemotron 3 Ultra，PDF 标注 2026-06-09 | 领域教师、学生轨迹、异步 MOPD 估计器与缺失中间权重。 |

评估页新增“公开训练技术的核验顺序”；六步路线、模型案例及搜索元数据增加入口。
DeepSeek 09-17 架构报告与 10-17 后训练说明分开引用。

## 一手来源与固定版本

逐项声明、依据位置和边界保存在
`artifacts/training-research-20261030T033831Z/primary-sources.json`。
其中记录 9 份一手材料和 3 个固定源码文档入口。摘要是人工对照原文后的归纳，机器检查不承担事实真实性判定。

- [DeepSeek 官方发布说明](https://api-docs.deepseek.com/news/news261017/)：完整项目开发段。单项采用声明不扩写为算法细节。
- [独立 RLSF 论文 v2](https://arxiv.org/abs/2507.21931v2)：2025 年首次发表，2026-09-15 修订。其置信度偏好机制不归到 DeepSeek 名下。
- [GSPO v2](https://arxiv.org/html/2507.18071v2)、[SAPO v2](https://arxiv.org/html/2511.20347v2)：SAPO 第 3 节式 5–8。
- [QwenGyre v2](https://arxiv.org/html/2609.33848v2)：第 4–5 节与附录 B。本轮只读论文机制，不执行分布式系统。
- [Kimi 官方 K2.5 技术博客](https://www.kimi.com/en/blog/kimi-k2-5)：第 2 节。未把社区 PARL 实现视为官方代码，也不推定后续 K3 系列机制。
- [Agent Lightning 论文 v1](https://arxiv.org/html/2608.17528v1)、[微软 10-07 技术说明](https://www.microsoft.com/en-us/research/blog/agent-lightning-v1-0-a-3500-line-lightweight-agentic-rl-framework-for-training-agents-with-real-harnesses/)：分别检查论文和后续博客日期。官方 v1.0.0 发布于 2026-08-17，annotated tag 对应提交 `8f8b8f95fd4ed664822734811b61bb3341f4ddfe`，不把 10-07 写成首次开源日。
- [NVIDIA Ultra 报告](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)：第 3.2–3.3 节、图 10。官方配方文档固定于 `99a7e45302f486c7e104647c128c2962ae7695d6`；明确缺少领域教师与第一轮 MOPD checkpoint，仅展示代表性单轮流程。没有将源码入口存在写成完整训练已可复现。

公司报告中的 benchmark、加速、训练规模和效率收益没有复制为本站成绩。新旧产品的发布名、机制论文和代码版本分别记录。

## 本地重算的范围

```sh
node artifacts/training-research-20261030T033831Z/mechanism-check.mjs
```

默认只核验和打印；`--write` 仅在结果文件不存在时创建证据，不能覆盖已有结果。
`mechanism-results.json` 保存固定输入及输出：

1. 检查现有 PPO 单项在正负优势下的有限差分，核对页面三行硬裁剪表。
2. 以 3 个温度、5 个比率，对 SAPO 的局部导数做 15 组有限差分比较；三个展示值四舍五入到三位小数。
3. 比较 GSPO 长度归一化比率与原始序列比率；它们不是同一统计量。
4. 按明确并行/屏障假设重算 PARL 手工输入：关键步数 12，总工作步数 17。没有测量真实时延。
5. 用固定 token loss 检查拆分记录的平均数与按 rollout 归约的不同含义。这是算术例子，不复现 Agent Lightning 训练。

这些核验没有运行模型、训练 rollout、调用 API、下载权重或使用 GPU。页面表格与对应 HTML 数据标记会和重算结果交叉检查。

## 验收方式

选用一次 `npm run check` 及现有内容、阅读、策略流程和公式测试，检查引用登记、全站本地链接/锚点、公式排版及原有交互的计算契约。
浏览器走查新增入口、全部六个案例、搜索别名、手机表格滚动及明暗主题；结果放入本轮 artifacts 目录。
最终验收通过：66 页、114 个前端脚本、44 份样式及 146 个公式构建成功；48 项聚焦测试通过。
7 个修改页面在 390、768、1440 像素和明暗主题下共检查 42 个布局，未出现整页横向溢出或公式渲染错误。
六个案例入口和六组搜索词均可到达对应专题。截图与坐标记录、源码摘要及实验校验写入 `verification-summary.json` 和同目录证据文件。

公开实验 manifest 的 SHA-256 仍为 `44afd76af506999af7ab58441de78ea5dab9001b5d0570c4ae8f061aee3948cd`；57 个登记文件逐字节校验通过。

保留原有 13 组模型实验及其报告、权重和投影；本轮新增的是来源账本与固定输入公式检查，不登记为新的模型训练实测。
