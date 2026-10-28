# 模型工程 Wiki 验收 · 2026-10-26

## 本轮目标与范围

本轮在原有 31 页基础上，完成训练工程判断专题、工程案例与现有专题深化，
形成 35 页的中文交互式 Wiki。完成意味着下表中的明确要求已实现并验证，
不意味着覆盖全部 AI 研究或保证任何现实训练任务成功。

内容、信息结构、计算模型、视觉设计及最终审核由主 agent 负责。
GPT-6.1 Sol（max）仅完成 `scripts/check-source.mjs` 的机械引用检查增强；
其代码由主 agent 读取审核并运行。断电前未产出改动的子任务已结束。

## 逐项验收

| 要求 | 实现位置 | 验收证据 |
| --- | --- | --- |
| 优化器与稳定性 | `optimization-dynamics.html` | AdamW/momentum/SGD 状态与位移账本；初始尺度、warmup、裁剪、恢复及 μP；测试核对零梯度、动量、衰减与逐步位移 |
| 并行数学正确性 | `parallel-correctness.html` | 全局有效 token 分母、DDP 平均补偿、累积等价条件；0..16 的全部两副本数量组合与有限差分一致；空目标显式未定义 |
| 性能瓶颈分析 | `performance-analysis.html` | GEMM FLOPs/bytes/Roofline 下界、关键路径、通信重叠、MFU/HFU 与测量；测试验证单位和算力/带宽分支 |
| 工程诊断案例 | `engineering-cases.html` | 多卡方向错误、数据覆盖退化、MoE 等待与奖励偏差四例；选择观测后逐步揭示三条证据，结论和后续专题均可到达 |
| MoE 训练深化 | `transformer-anatomy.html#moe-training` | 区分离散 top-k、完整 softmax/选中重归一化的梯度、top-1 常数门控、均衡目标、容量与调度 |
| 长上下文深化 | `architecture-variants.html#long-context-training` | 位置、数据、训练、并行和长度×位置×任务评估；区分标称窗口与有效能力 |
| 数据配方深化 | `data-tokenizer.html#mixture-optimization` | token 权重、唯一量/重复预算、课程、DoReMi proxy/reference、迁移与独立消融 |
| 信息结构 | `pages.mjs` / `wiki.mjs` / `search.mjs` | 每页目录恰好一次，搜索/资料映射无孤儿；指南、工程总览与深入路径连接新专题，保留原公开 URL |
| 技术边界 | `references.mjs` 和各交互说明 | 每页有适用范围与一手来源；人为梯度、预设案例、理想硬件参数与真实模型测量分开 |
| 响应式与可访问性 | 领域 CSS / `wiki-shell.js` | 移除整页横向裁切；修复四个旧页面手机网格；宽表独立滚动且可聚焦；搜索 Esc 返回焦点；active-state 截图人工复查 |
| 可维护性 | `scripts/check-source.mjs` / `scripts/audit-site.mjs` | 校验目录、元数据、资源；等待明确初始化标记；CDP/网络超时；检查资源/HTTP/console 错误与全站链接/锚点 |
| 构建与服务 | `dist/` / 本地 4173 | 35 页构建成功；原服务地址恢复并返回 HTTP 200 |

## 验证结果

- `npm run check`：通过，35 页、36 个脚本、26 个样式文件。
- `npm run test:content`：16 组测试通过，包含已有梯度/推理/量化检查及新账本。
- `npm run test:interactions`：全部通过，含旧交互、新案例、空状态、焦点与宽表滚动。
- `npm run test:audit`：35/35 页，390px 浅色，无 issues。
- `VIEWPORT_WIDTH=768 npm run test:audit`：35/35 页，无 issues。
- `VIEWPORT_WIDTH=1100 npm run test:audit`：35/35 页，无 issues。
- `VIEWPORT_WIDTH=1440 AUDIT_THEME=dark npm run test:audit`：35/35 页，无 issues。
- 手工检查桌面/手机、浅色/暗色截图：优化器完整账本、Roofline 两种分支、
  案例展开结论、架构图、工件浏览器、数据流水线、评估切片与阅读路径。

## 本轮额外修复

- 纠正 InfoNCE “不能只抬正例”的绝对化表述，并同步初始梯度解释。
- 明确可截断向量不保证逐维按可解释的重要性排序。
- 前缀缓存匹配实际 token/模态输入、权重/adapter、位置和 attention 配置。
- 空评估切片显示“无数据”，全零混合权重显示无法分配预算。
- 激活参数账本注明 router 与 embedding 的计数口径。
- 架构响应式规则移回 `architecture.css`，架构页不再依赖 `inference.css`。
- 修复数据、评估、工件、深入路径四页被整页裁切掩盖的窄屏内容溢出。
- 增加页面模块初始化完成标记，浏览器审核不再仅依赖固定等待时间。

## 维护与限制

修改一个机制时，同时检查正文、动态文案、公式、数值实现、测试、搜索词和引用。
概览负责导航，专题负责完整推导，案例负责证据使用；避免在多页复制完整解释。
所有新计算的假设都显示在交互旁。没有运行真实 GPU 训练，也不以教学计算
推断现实模型的收敛、泛化、吞吐或优劣。外部论文与模型配置会继续发展，
具名研究按文中标注版本理解；站内检查验证引用结构，不保证外部链接永久可用。

推荐入口：`http://127.0.0.1:4173/training-engineer-guide.html`。
本轮新增内容集中入口：`http://127.0.0.1:4173/engineering.html#engineering-judgment`。
