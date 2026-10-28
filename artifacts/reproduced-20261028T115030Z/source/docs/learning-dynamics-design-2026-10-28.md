# 学习动态扩展设计与审核契约 · 2026-10-28

本轮实施 `learning-dynamics-content-review-2026-10-28.md` 的全部五项。主 agent 负责设计、数学与证据审核；执行、编码、实验和整合由用户指定的 `gpt-6.1-sol` / `max` 子 agent 完成。

## 共用实验

1. 小型 decoder 使用固定符号词表和可复现任务生成器。建议选择反转序列，词表包括四种内容符号与 BOS/SEP/EOS/PAD。输入与输出反转组成同一拆分组，隔离 train/validation/test；另设长度外推集。规则基线、训练前模型、训练后模型分别记录。
2. 报告必须分开 teacher-forced loss、自由生成 token accuracy、完整序列 exact match。自由生成的提前结束、缺失 token 和多余输出均有明确计分；外推失败是有效结果，不为得到好看的分数调整独立测试。
3. 参数、优化器、数据顺序、RNG、计数与配置一并保存。确定性恢复对照、完整/增量 logits 对照均为验收项。
4. 训练和故障诊断使用同一 core；batch 实验也导入这一 core。初始化中的深度/Norm/残差试验使用 core；μP 宽度试验可单独用作者方法明确的 MLP，避免把未经验证的 Transformer 缩放称为 μP。
5. 沿用 PyTorch 2.9.0、NumPy 2.3.4 独立环境；CPU、固定线程数、种子和数据。可选 μP 依赖须查询并固定版本，不替换现有 torch/numpy。禁止下载预训练模型。

## Core 接口（由学习实验子 agent 拥有）

模块为 `scripts/learning_core.py`。以下命名供协作，具体字段如需调整先通知其他 worker，保持稳定兼容函数：

- `make_task_splits(data_seed=20261028)`：返回 train/validation/test/length 等记录列表。记录至少含稳定 `id`、`group_id`、内容序列、目标序列、长度。
- `collate_records(records)`：返回 `input_ids`、`labels`（忽略位置为 −100）、`attention_mask`（True 为有效）、记录 ID 与监督数量。
- `make_model(seed=11, **config)`：可配置 width/depth/heads/dropout/norm/residual_scale；返回 CPU 模型。参数配置与词表常量可导出。
- `forward_loss(model, batch, reduction="token")`：返回 scalar Tensor；另支持明确的逐序列归一化。全无标签时跳步，不伪造有效零 loss。
- `evaluate_records(model, records)`：teacher-forced 与自由生成指标及逐题结果；保证恢复调用前的 train/eval 状态。
- 模型 `forward(input_ids, attention_mask=...)` 返回 logits；缓存前向、监控 hooks 和快照辅助接口由 owner 文档化。

学习 worker 先交付可导入的 core 和简短接口说明，其他 worker 可以先编写独立数学、正文、图和报告读取，再接入 core。

## 文件所有权

| Worker | 页面和脚本 | 专有后端与测试 |
| --- | --- | --- |
| 学习实验 | `learning-run.html`、`learning-run.js`、可选 `learning-run-data.js` / `learning-run.css` | `learning_core.py`、`learning-run.py`、`run-learning-lab.mjs`、`test-learning-core.py`、`test-learning-run.mjs` |
| 故障回放 | `training-failures.html`、`training-failures.js`、可选同名前缀数据/样式 | `failure_replay.py`、`run-failure-lab.mjs`、同前缀测试 |
| 尺度与 batch | `initialization-width.html`、`batch-noise.html`、对应 feature/data/CSS | `training_dynamics.py`、`run-dynamics-lab.mjs`、`dynamics-requirements.txt`、同前缀测试 |
| 分块 CE | `chunked-cross-entropy.html`、`chunked-cross-entropy.js`、对应 data/CSS | `chunked_ce_reference.py`、`export-chunked-ce-fixtures.mjs`、`run-output-lab.mjs`、同前缀测试 |
| 站点整合（后续） | 所有共享目录、元数据、搜索、引用、导航和原页面链接 | `package.json`、README、build 映射、浏览器验收与最终记录 |

子 agent 只编辑自己的文件；不重置/提交/清理现有工作区，不修改其他 worker 的实现。所有报告使用独立新目录并拒绝覆盖。整合前各 worker 直接运行自身脚本，不并发重建 `dist/`。

## 页面与报告

- 沿用现有 `modern`、`exact-mechanics`、`decision-chains` 组件。必要时增加本专题样式，表格在自己的容器内滚动；不能裁切整页掩盖溢出。
- 图、表和事件必须同时提供可读数值与单位；数学使用现有 `data-math`。报告本地导入需有结构/范围检查、安全文字输出和异步竞态保护。
- 默认页面可包含从本轮真实报告提取的小型快照，明确运行日期、环境和来源。禁止手工编造训练曲线或把公式预测标为测量。
- 全部真实报告保存在 `artifacts/`；可供页面默认显示的精简 data 模块由各 worker 生成并记录报告 provenance。
- 每个 worker 提交实施摘要，列出新增入口、导出 API、styles、验证命令、报告路径、引用原文核对范围及任何未完成要求。

## 审核重点

- 学习：拆分无泄漏、因果与标签对齐、单一训练目标、自由生成计分、多个 seed、恢复与缓存一致性。
- 故障：数值异常、过大学习率和有限但错误的标签分别定位；同 checkpoint/输入回放；修复必须恢复约定目标，日志只报告实际观察。
- 尺度：初始与更新后尺度分开；深度和宽度单独改变；μP 使用可靠作者定义和参数组规则；验证宽度在 LR 选择中保留。
- Batch：固定参数的样本梯度与随训练变化的噪声分开；总体方差分母、有/无放回修正、零均值边界准确；训练比较固定消费 token 并明确 scheduler 计量。
- CE：精确完整梯度先行，loss、dH/dW/db、忽略标签与尾块独立核对；分块反传确实不保留完整 logits；保存张量引用、逻辑元素与时间分别测量。
- 最终：完整源码/构建检查、恰当数值测试、实际实验报告和 390/768/1440 暗色浏览器验证；主 agent 审核发现的问题交回实现者修正。
