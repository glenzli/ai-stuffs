# 内容审核记录 · 2026-10-26

范围：25 页正文、对应交互模块、站内探索链接。审核围绕机制正确性、
公式与数值一致性、模拟边界和公开模型版本；不是模型性能实测或实时排行榜更新。

## 已修正

- 完整词表输出梯度误写成只覆盖 label；补齐绑定权重、PAD、冻结、动量与衰减的边界。
- InfoNCE 混用 raw score 与 scaled logit；补回温度链式因子，区分导数和下降方向。
- 低温必然增大梯度的误导；固定其他负例分数并开放“负例高于正例”的情形。
- SGNS 被描述成完整 softmax 的简单近似；MLM 输入查表与 80/10/10 替换规则不清。
- causal 标签示例的 shift 和张量形状不一致；工程示例存在沿 batch 轴切片的错误。
- DPO 导数遗漏符号及 β；纠正 β 的局部效应与全局 KL 结论混淆。
- SFT prompt mask、packing、可训练读出、LoRA 范围与 reference 内存的绝对化表述。
- prefill 首 token 与缓存长度错一位；KV 模式判断改为固定 Hq 下显式比较 Hkv。
- 模拟器零噪声下仍采到同类假负例；loss 数值截断与梯度不一致；先平均再逐向量裁剪。
- 模拟器训练点指标误称独立评估、Hit@1 误称多相关文档 Recall@1，更新前后时点不清。
- CLS 示例直接取首个正文词；改为单独的特殊 token 状态。
- SwiGLU 公式把 SiLU 写成 SwiGLU；FlashAttention 的 HBM/SRAM 与计算复杂度表述。
- 参数账本公式漏写 final norm；KV heads 隐式截断后与控件不一致。
- T5、Llama 2/3、Mistral 的版本口径；MLA 与 GQA 不能视为同一机制。
- Scaling 中 D 混同信息量，服务成本公式混同 FLOPs；全局 batch 使用 DP 副本数。
- 补齐 pass@k、BPB、配对统计、checkpoint 轨迹复现的条件。
- 移动端梯度表的隐藏列与表头未同步。

## 内容维护

`src/references.mjs` 统一维护每页的适用范围与一手资料。
概览负责关系，深入页负责推导，工程页负责资源与接口。
新增机制必须说明假设、求导变量、张量形状和模拟范围；避免重复维护完整解释。

## 验证

- `npm run check`：语法与 25 页构建。
- `npm run test:content`：有限差分核对 InfoNCE/SGNS/DPO、温度极端行为、缓存计数、负例池及资料配置。
- `npm run test:interactions`：页面交互和控制台回归。
- `npm run test:audit`：页面结构、移动端溢出、站内链接/锚点及资料入口。
- 桌面/手机截图复查梯度表、损失面板、DPO、推理与资料区。

模型案例保留指定公开版本；资料链接不意味着已核验每篇论文的所有结论。
训练点模拟不能证明真实模型泛化，资源估算不能替代实际 profiler。
