import { $, $$ } from "../core.js";

export function initArchitectureVariants() {
    const buttons = $$("[data-variant]");
    if (!buttons.length) return;
    const variants = {
      mha: {
        kicker: "STANDARD ATTENTION",
        title: "每个 query head 都有独立 K/V",
        copy: "最直接的 MHA。每个 query head 拥有自己的 K/V projection；表达灵活，但训练和推理的 KV 状态也最大。",
        structure: "Hq = Hkv; cache ∝ Hkv · D",
        gain: "更强 head 多样性；实现最直接。",
        cost: "KV Cache 大，decode 读取带宽高。",
        misread: "head 数量的收益需要结合每头维度、训练配方和任务结果评估。",
      },
      mqa: {
        kicker: "MULTI-QUERY ATTENTION",
        title: "所有 query heads 共享一组 K/V",
        copy: "query 仍可有多个 head，但只保留一套 K/V。缓存和 decode 读取显著下降，是最激进的共享方案。",
        structure: "Hq = H; Hkv = 1",
        gain: "最小标准 attention KV；decode 更省带宽。",
        cost: "共享过强可能限制 head 表达；需要训练验证。",
        misread: "共享 KV 会改变各 head 可独立表达的信息，需比较任务效果。",
      },
      gqa: {
        kicker: "GROUPED-QUERY ATTENTION",
        title: "多个 query heads 分组共享 KV",
        copy: "把 Hq 分成若干组，每组共享一个 K/V head。它在 MHA 表达与 MQA 缓存之间提供可控折中。",
        structure: "每组 query heads = Hq / Hkv; cache ∝ Hkv · D",
        gain: "显著降低 KV 与 decode 带宽，通常比 MQA 更温和。",
        cost: "仍损失一部分独立 KV 自由度；训练需匹配。",
        misread: "GQA 改变 projection 和 KV head 数量；还可与数值量化组合。",
      },
      mla: {
        kicker: "MULTI-HEAD LATENT ATTENTION",
        title: "用压缩隐变量保存 KV 信息",
        copy: "MLA 通过低秩投影把 K/V 信息压到潜在向量，在 attention 中重建所需表示；具体实现会仔细处理 RoPE 与投影。",
        structure: "KV → compressed latent c → up-project K/V",
        gain: "降低长上下文的缓存与读取压力，同时保留较强表达。",
        cost: "结构/kernel 更复杂；压缩与投影也有计算成本，需查看具体版本。",
        misread: "MLA 通过潜在缓存和上投影组织注意力，需按具体配置计算成本。",
      },
      window: {
        kicker: "SPARSE ATTENTION PATTERN",
        title: "只访问局部窗口或选定 block",
        copy: "每个 token 按设定的模式访问局部窗口、全局 token 或学习出的稀疏位置。不同层可混合 local/global pattern。",
        structure: "full S² → window W / block sparsity",
        gain: "降低长序列 attention 计算与部分 KV 访问。",
        cost: "可能看不到远处证据；窗口外信息依赖其他层或分层结构。",
        misread: "单层访问窗口和模型有效使用的上下文长度应分别测量。",
      },
      rope: {
        kicker: "ROTARY POSITION",
        title: "用旋转把相对位移写进 Q/K 点积",
        copy: "RoPE 在不同频率维度旋转 Q/K，使相对位置自然影响注意力分数。长上下文外推还涉及频率缩放与训练长度。",
        structure: "q' = R(pos) q; k' = R(pos) k",
        gain: "不额外增加位置 embedding 表；相对位置信号自然。",
        cost: "外推、频率和 attention 模式需要共同调，并验证目标长度下的效果。",
        misread: "支持长度要结合频率配置、训练长度与长程任务评估判断。",
      },
      norm: {
        kicker: "NORMALIZATION & DEPTH",
        title: "先归一化，再进子层；残差保留主干",
        copy: "Pre-RMSNorm 常见于现代 decoder；RMSNorm 只按均方根缩放。QK-Norm 还会对 query/key 维度再做稳定化。",
        structure: "x + Attn(RMSNorm(x)); x + MLP(RMSNorm(x))",
        gain: "深层训练更稳，计算比 LayerNorm 少一次均值。",
        cost: "具体收益与初始化/学习率耦合；必须和完整配方一起评估。",
        misread: "比较 Norm 时固定其他配方，单独验证稳定性、质量和计算成本。",
      },
      swiglu: {
        kicker: "GATED MLP",
        title: "用门控乘法提升 FFN 表达",
        copy: "SwiGLU 使用 gate 与 up 两个分支逐元素相乘，再通过 down 投影回 d 维。参数比普通两层 MLP 多一组 d×d_ff。",
        structure: "W_down(SiLU(W_gate x) ⊙ W_up x)",
        gain: "更强非线性与门控表达，常与更小 FFN width 搭配。",
        cost: "多一组矩阵；参数、activation 与 matmul 增加。",
        misread: "在相同参数或计算预算下比较，并记录中间宽度。",
      },
      moe: {
        kicker: "SPARSE CAPACITY",
        title: "每 token 只走 top-k experts",
        copy: "router 给每个 token 选专家并加权输出。总参数可以很大，但每 token 的 MLP 计算只覆盖少数 experts。",
        structure: "router: d → E; active MLP = topK × expert",
        gain: "提高总容量/激活计算比。",
        cost: "全部 experts 占内存；all-to-all、负载不均、容量与训练稳定。",
        misread: "分别计算总参数、激活参数与路由开销。",
      },
      hybrid: {
        kicker: "ALTERNATIVE MIXER",
        title: "用状态递推/卷积替换部分 attention",
        copy: "Mamba/SSM、线性 attention 或卷积层可在某些层替代 attention，保留状态而不缓存完整 KV；也常与 attention 混合。",
        structure: "state hₜ = f(hₜ₋₁, xₜ) / attention hybrid",
        gain: "长序列状态规模和吞吐可能有优势。",
        cost: "状态容量、训练并行、kernel 与长程检索能力不同；生态成熟度也重要。",
        misread: "在相同任务和硬件下比较状态容量、质量与吞吐。",
      },
    };

    const render = (key) => {
      const data = variants[key];
      buttons.forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.variant === key));
      });
      $("[data-variant-kicker]").textContent = data.kicker;
      $("[data-variant-title]").textContent = data.title;
      $("[data-variant-copy]").textContent = data.copy;
      $("[data-variant-structure]").textContent = data.structure;
      $("[data-variant-gain]").textContent = data.gain;
      $("[data-variant-cost]").textContent = data.cost;
      $("[data-variant-misread]").textContent = data.misread;
    };
    buttons.forEach((button) => {
      button.addEventListener("click", () => render(button.dataset.variant));
    });
    render("mha");

    const goalButtons = $$("[data-arch-goal]");
    if (!goalButtons.length) return;
    const goals = {
      kv: {
        kicker: "INFERENCE MEMORY",
        title: "先减少 KV heads，再考虑缓存精度与压缩",
        copy: "GQA 是常见折中：多个 query heads 共享 KV heads；MQA 更激进；MLA 则学习压缩潜在缓存。还要配合 paged KV、prefix cache 和 batch 调度。",
        stack: ["GQA / MQA", "MLA 或 KV compression", "Paged KV + prefix cache", "BF16/FP8 KV（需回归）"],
        warning: "KV 与权重占用应分别计算；长上下文吞吐还受 attention 计算和调度影响。",
      },
      capacity: {
        kicker: "CAPACITY PER TOKEN",
        title: "用 MoE 扩总容量，但要先解决路由与通信",
        copy: "增加 experts 让总参数变大，top-k 控制每 token 激活路径。更细粒度 experts 可以提高组合能力，但会让 route 与 all-to-all 更复杂。",
        stack: ["Dense backbone + MoE FFN", "top-k experts", "load-balance / router loss", "expert parallel + all-to-all"],
        warning: "计算总权重的驻留成本，并测量路由、通信和每 token 延迟。",
      },
      long: {
        kicker: "LONG CONTEXT",
        title: "位置、attention 模式、缓存与数据必须一起设计",
        copy: "RoPE 缩放需要配合训练长度、滑窗/全局模式、context parallel、KV 容量和长文任务评估。",
        stack: ["RoPE scaling", "local + global attention", "Context Parallel", "Paged/prefix KV", "long-context eval"],
        warning: "通过不同证据位置和长程任务，评估远端 token 对答案的影响。",
      },
      stable: {
        kicker: "DEEP TRAINING",
        title: "残差、Pre-Norm 与数值细节共同稳定深层训练",
        copy: "现代 decoder 常见 pre-RMSNorm、残差、适当初始化和 warmup。若出现不稳定，可检查 QK-Norm、logit softcap、梯度裁剪与混合精度累加。",
        stack: ["Pre-RMSNorm", "residual stream", "QK-Norm / careful init", "BF16 compute + FP32 accumulation", "gradient clipping"],
        warning: "调整 Norm 后，观察 grad norm、loss spike 和 activation 分布是否改善。",
      },
      simple: {
        kicker: "ENGINEERING SIMPLICITY",
        title: "先选成熟的 dense decoder 配方",
        copy: "需要成熟实现作为基线时，pre-RMSNorm + RoPE + SwiGLU + GQA 的 dense decoder 生态成熟，kernel 与训练经验都更多。",
        stack: ["Pre-RMSNorm", "RoPE", "SwiGLU", "GQA", "Dense FFN"],
        warning: "再根据目标工作负载，评估采用 MoE 或稀疏 attention 的收益与成本。",
      },
    };
    const renderGoal = (key) => {
      const data = goals[key];
      goalButtons.forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.archGoal === key));
      });
      $("[data-goal-kicker]").textContent = data.kicker;
      $("[data-goal-title]").textContent = data.title;
      $("[data-goal-copy]").textContent = data.copy;
      $("[data-goal-warning]").textContent = data.warning;
      $("[data-goal-stack]").innerHTML = data.stack.map((item, index) =>
        `<div><span>${String(index + 1).padStart(2, "0")}</span><strong>${item}</strong></div>`,
      ).join("");
    };
    goalButtons.forEach((button) => {
      button.addEventListener("click", () => renderGoal(button.dataset.archGoal));
    });
    renderGoal("kv");
}
