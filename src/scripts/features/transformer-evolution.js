import { $, $$ } from "../core.js";

export function initTransformerEvolution() {
    const buttons = $$("[data-evo]");
    if (!buttons.length) return;
    const eras = [
      {
        era: "2017 · ORIGINAL TRANSFORMER",
        title: "Encoder-decoder + post-norm",
        summary: "用 self-attention 替代循环结构，让序列内所有位置并行交互。原始模型仍包含 encoder 与 decoder，使用正弦位置编码、ReLU FFN 和 post-layer-norm。",
        objective: "机器翻译：encoder 编码源语言，decoder 因果生成目标语言。",
        attention: "MHA；每个 query 位置看完整 source 或已生成 target。",
        norm: "每个子层输出后做 LayerNorm；残差帮助深层训练。",
        position: "正弦/余弦位置编码，注入顺序信息。",
        mlp: "两层 MLP + ReLU；通常 4d 宽，参数量约 2·d·d_ff。",
        inference: "decoder 自回归生成；MHA 的 KV 与 head 数同规模。",
        solved: "长距离依赖可并行建模，训练比 RNN 更充分利用硬件。",
        cost: "注意力成本随上下文长度二次增长；自回归 decode 每步仍需逐 token。",
      },
      {
        era: "2018–19 · ENCODER / DECODER SPLIT",
        title: "BERT、GPT 与 Pre-LN 稳定化",
        summary: "BERT 采用双向 encoder + masked LM，GPT 采用 causal decoder + next-token prediction。模型开始从单一翻译结构分化成不同预训练范式；pre-LN 也被用来改善深层训练稳定性。",
        objective: "Masked token prediction、causal LM、下一句预测等。",
        attention: "BERT 双向可见；GPT causal mask，只允许看左侧。",
        norm: "原版中使用 LayerNorm；Pre-LN 将归一化移到子层前，残差主干更稳定。",
        position: "仍常见 learned / absolute position embedding。",
        mlp: "GELU / ReLU 的 position-wise FFN 继续作为每个 block 的主要参数量。",
        inference: "encoder 可整段编码；decoder 逐 token 生成，KV Cache 价值上升。",
        solved: "预训练 + 下游迁移；decoder-only 成为生成式 LLM 基础。",
        cost: "目标与 attention mask 不同，不能把 BERT 的输出直接当 GPT 的生成接口。",
      },
      {
        era: "2020 · SCALE & SPARSE ATTENTION",
        title: "参数、数据与注意力路径一起扩张",
        summary: "GPT-3 展示规模化语言模型的通用能力；T5 使用 encoder-decoder 与相对位置；稀疏/局部 attention 被用于降低长序列成本。结构与训练规模的边界被重新认识。",
        objective: "自监督语言建模、span denoising；训练 token 预算成为独立变量。",
        attention: "full attention 仍是基准，同时出现 block/local/sparse 变体。",
        norm: "Pre-LN 在现代深层 decoder 中逐渐常见；训练稳定性与 warmup 成为关键。",
        position: "learned absolute、relative bias、T5 relative position 等并存。",
        mlp: "FFN 仍是稀疏化的主要目标之一；参数增长先于推理优化。",
        inference: "更大的 KV Cache 与多卡权重带来新的服务成本。",
        solved: "通用 few-shot、迁移与规模效应的工程可行性。",
        cost: "计算、显存与数据质量同时成为瓶颈；效率需结合有效计算、带宽和吞吐评价。",
      },
      {
        era: "2021–22 · EFFICIENT MODERN DECODER",
        title: "RoPE、SwiGLU、RMSNorm 与 FlashAttention",
        summary: "这一阶段逐渐组合 RMSNorm、SwiGLU、RoPE；这些部件提出时间不同。FlashAttention 减少 HBM 数据搬运。部分变化是结构，部分是执行优化。",
        objective: "大规模 causal LM 继续主导；训练目标未必变化，结构效率在变化。",
        attention: "GQA/MQA 开始用于降低 KV；FlashAttention 重排 attention 的内部读写。",
        norm: "Pre-RMSNorm 常见：少一次去均值、权重更少，也更依赖整体训练设置。",
        position: "RoPE 通过旋转 Q/K 表示相对位置，逐步替代绝对位置加法。",
        mlp: "SwiGLU 门控 FFN：gate ⊙ up 再 down；参数更多，表达更强。",
        inference: "KV head 共享与更高效 kernel 直接改善长上下文服务。",
        solved: "在相近算力下兼顾深层稳定性、长序列与吞吐。",
        cost: "SwiGLU 多矩阵、RoPE 长上下文外推与 FlashAttention kernel 支持仍需工程配套。",
      },
      {
        era: "2023 · STANDARDIZED DECODER & MoE",
        title: "GQA、MoE 与“参数激活比”",
        summary: "开源 decoder 逐渐收敛到 pre-RMSNorm、RoPE、SwiGLU 组合；GQA 让 KV heads 少于 query heads；MoE 用 router 选 experts，提高总容量而不让每个 token 走全部参数。",
        objective: "causal LM 仍是主干，SFT/偏好后训练成为产品模型流程的一部分。",
        attention: "GQA 成为常见推理折中；MHA/MQA/MoE 组合并存。",
        norm: "RMSNorm + pre-norm 常见；部分架构增加 QK-Norm 或额外稳定化。",
        position: "RoPE 及其频率/外推策略成为长上下文设计点。",
        mlp: "MoE 把 MLP 扩成 router + experts，top-k 决定每 token 激活路径。",
        inference: "KV 更小但 MoE 增加 all-to-all、专家并行与负载均衡问题。",
        solved: "低成本推理的 KV 压力、可扩展的总参数容量。",
        cost: "总参数必须驻留；router 不均衡、分布式通信与专家利用率成新瓶颈。",
      },
      {
        era: "2024+ · LONG CONTEXT & SYSTEM CO-DESIGN",
        title: "长上下文、混合架构与推理协同",
        summary: "结构选择越来越和 kernel、并行、缓存与服务一起设计：长上下文 attention、KV 压缩、混合状态空间/Transformer、paged/prefix KV 与 speculative decode 同时影响可用性。",
        objective: "预训练、SFT、偏好与蒸馏形成分层训练栈；目标也影响架构和推理策略。",
        attention: "稀疏/滑窗、GQA/MQA、KV 压缩与高效 attention kernel 共同演化。",
        norm: "现代 decoder 常见 pre-RMSNorm；细节扩展包括 QK-Norm 等。",
        position: "RoPE 仍是主流选择之一，长上下文外推与频率缩放是独立工程课题。",
        mlp: "细粒度 MoE、共享专家与不同 top-k 策略继续探索；仍需实测负载与质量。",
        inference: "prefill/decode 分相、continuous batching、paged KV、prefix cache、量化与猜测解码。",
        solved: "更长的上下文、分布式专家和更细的服务成本控制。",
        cost: "配置组合爆炸；benchmark 上的单一指标无法代表目标领域的质量、延迟和稳定性。",
      },
    ];

    const fields = {
      era: $("[data-evo-era]"),
      title: $("[data-evo-title]"),
      summary: $("[data-evo-summary]"),
      objective: $("[data-evo-objective]"),
      attention: $("[data-evo-attention]"),
      norm: $("[data-evo-norm]"),
      position: $("[data-evo-position]"),
      mlp: $("[data-evo-mlp]"),
      inference: $("[data-evo-inference]"),
      solved: $("[data-evo-solved]"),
      cost: $("[data-evo-cost]"),
    };
    const render = (index) => {
      const data = eras[index];
      buttons.forEach((button, buttonIndex) => {
        button.setAttribute("aria-pressed", String(buttonIndex === index));
      });
      Object.entries(fields).forEach(([key, element]) => {
        element.textContent = data[key];
      });
    };
    buttons.forEach((button) => {
      button.addEventListener("click", () => render(Number(button.dataset.evo)));
    });
    render(0);
}
