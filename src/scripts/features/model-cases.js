import { $, $$ } from "../core.js";

export function initModelCases() {
    const buttons = $$("[data-case]");
    if (!buttons.length) return;
    const cases = {
      bert: {
        kicker: "ENCODER-ONLY / MASKED LM",
        title: "BERT-base-uncased：双向 encoder + post-LN",
        summary: "用 masked LM 学习双向上下文。它输出每个 token 的上下文状态，采用双向上下文接口。",
        params: "110M",
        type: "Encoder-only",
        shape: "768 / 12",
        heads: "12 / 12 · MHA",
        ffn: "3072 / 30,522",
        position: "Learned absolute",
        norm: "Post-LN",
        activation: "GELU",
        context: "512 tokens",
        why: "双向 attention 让每个位置同时看左右文本，适合理解与 masked 目标；其可见上下文与 GPT 式因果生成不同。",
        caveat: "用于句向量时，需要按任务选择和训练 pooling 或输出头。",
      },
      llama2: {
        kicker: "DENSE DECODER / MHA",
        title: "LLaMA 2 7B：现代 decoder 配方的典型样本",
        summary: "pre-RMSNorm、RoPE、SwiGLU 与 causal MHA 组合，结构清晰，是理解后续 LLaMA 系变体的好基线。",
        params: "≈6.7B",
        type: "Decoder-only",
        shape: "4096 / 32",
        heads: "32 / 32 · MHA",
        ffn: "11008 / 32,000",
        position: "RoPE",
        norm: "Pre-RMSNorm",
        activation: "SwiGLU",
        context: "4K",
        why: "残差主干与 pre-norm 支持较深层；RoPE 把相对位置写进 Q/K；SwiGLU 提供门控 FFN。",
        caveat: "MHA 的 KV 与 query heads 一样多，长上下文 decode 的缓存与读取压力大。",
      },
      llama3: {
        kicker: "DENSE DECODER / GQA",
        title: "Llama 3 8B（2024-04）：更大词表与 GQA",
        summary: "保留现代 decoder 骨架，用 8 个 KV heads 降低缓存，并把词表扩大到约 128k。更大的词表也意味着 embedding/head 更大。",
        params: "≈8B",
        type: "Decoder-only",
        shape: "4096 / 32",
        heads: "32 / 8 · GQA",
        ffn: "14336 / 128,256",
        position: "RoPE",
        norm: "Pre-RMSNorm",
        activation: "SwiGLU",
        context: "8,192；非 Llama 3.1 配置",
        why: "GQA 显著降低 KV；更大词表可让多语言/代码片段更短，但 embedding 与输出头参数增加。",
        caveat: "不能把 GQA 的收益和词表变化混成一个因素；配置与训练数据都在变化。",
      },
      mistral: {
        kicker: "DENSE DECODER / SLIDING WINDOW",
        title: "Mistral 7B v0.1：局部窗口 + GQA",
        summary: "标准 decoder 配方上加入 sliding-window attention，限制每个 token 的局部访问，目标降低长序列 attention 成本。",
        params: "≈7.2B",
        type: "Decoder-only",
        shape: "4096 / 32",
        heads: "32 / 8 · GQA",
        ffn: "14336 / 32,000",
        position: "RoPE",
        norm: "Pre-RMSNorm",
        activation: "SwiGLU",
        context: "窗口 4096 / 按层配置",
        why: "局部窗口把 full attention 的 S² 交互限制在 W 附近，同时通过 GQA 控制 KV。",
        caveat: "跨窗口信息传播依赖层堆叠、训练与任务结构，可用长程任务评估有效范围。",
      },
      mixtral: {
        kicker: "MOE DECODER",
        title: "Mixtral 8×7B：总参数与每 token 激活分离",
        summary: "每个 MoE 层有 8 个 expert，router 每 token 选 top-2。总参数约 46.7B，但每 token 只走约 12.9B 级激活路径。",
        params: "≈46.7B",
        type: "MoE Decoder",
        shape: "4096 / 32",
        heads: "32 / 8 · GQA",
        ffn: "8 experts · top-2",
        position: "RoPE",
        norm: "Pre-RMSNorm",
        activation: "SwiGLU experts",
        context: "32K",
        why: "稀疏专家提升总容量/激活计算比；GQA 控制 attention KV；配置仍保留现代 decoder 基础。",
        caveat: "所有 experts 的权重仍需加载；服务成本还取决于路由热点、容量与 all-to-all 通信。",
      },
      deepseek: {
        kicker: "MLA + FINE-GRAINED MOE",
        title: "DeepSeek-V2 / V3：两条压缩/稀疏路线",
        summary: "MLA 压缩 K/V 的缓存表示，MoE 用细粒度 experts 与共享/路由结构扩大容量。V2 约 236B total / 21B active；V3 约 671B / 37B active。",
        params: "V2 236B / V3 671B",
        type: "MoE Decoder",
        shape: "依具体版本",
        heads: "MLA：低秩联合 KV latent，非 GQA 分组共享",
        ffn: "细粒度 experts",
        position: "RoPE 相关",
        norm: "RMSNorm 类",
        activation: "MoE",
        context: "128K 级（版本相关）",
        why: "MLA 把缓存从完整 K/V 推向潜在表示，MoE 则把容量与每 token MLP 计算解耦；两者正交，但都增加 kernel/路由复杂度。",
        caveat: "按具体版本的报告与 config 核对 MLA、expert 数量和训练目标。",
      },
      mamba: {
        kicker: "SELECTIVE STATE SPACE",
        title: "Mamba：用状态递推替代标准 attention",
        summary: "Mamba 属于 selective state-space model，通过随输入变化的状态递推处理序列；推理时保存递推状态，长上下文效果取决于状态容量与任务。",
        params: "依具体规模",
        type: "SSM / 非 Transformer",
        shape: "依具体版本",
        heads: "无标准 attention heads",
        ffn: "门控 MLP / scan blocks",
        position: "状态递推内含顺序",
        norm: "架构相关",
        activation: "门控 / SiLU 类",
        context: "依训练与版本",
        why: "递推状态可随序列长度避免完整 KV 增长，长序列吞吐有价值；选择性参数让状态随 token 内容变化。",
        caveat: "状态压缩也限制精确检索；kernel、并行训练、工具生态和具体任务质量都要比较。",
      },
    };
    const render = (key) => {
      const data = cases[key];
      buttons.forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.case === key));
      });
      $("[data-case-kicker]").textContent = data.kicker;
      $("[data-case-title]").textContent = data.title;
      $("[data-case-summary]").textContent = data.summary;
      $("[data-case-params]").textContent = data.params;
      $("[data-case-type]").textContent = data.type;
      $("[data-case-shape]").textContent = data.shape;
      $("[data-case-heads]").textContent = data.heads;
      $("[data-case-ffn]").textContent = data.ffn;
      $("[data-case-position]").textContent = data.position;
      $("[data-case-norm]").textContent = data.norm;
      $("[data-case-activation]").textContent = data.activation;
      $("[data-case-context]").textContent = data.context;
      $("[data-case-why]").textContent = data.why;
      $("[data-case-caveat]").textContent = data.caveat;
    };
    buttons.forEach((button) => {
      button.addEventListener("click", () => render(button.dataset.case));
    });
    render("bert");
}
