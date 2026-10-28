import { $, $$ } from "../core.js";

export function initEngineerGuide() {
    const list = $("[data-problem-list]");
    if (!list) return;
    const problems = [
      {
        id: "no-loss",
        category: "quality",
        label: "loss 完全不下降",
        title: "loss 完全不下降",
        observe: "先固定一个小 batch 尝试过拟合；检查真实 token 数、label shift、mask 与梯度。",
        cause: "labels 错位、mask 全空、预测/目标不对齐、优化器没更新、学习率错误、初始权重异常。",
        next: "用一条真实样本打印 token、label、logits、loss 与 grad_norm；再跑 20 步小 batch。",
        avoid: "先确认数据与标签对齐，随后再评估模型规模和学习率。",
      },
      {
        id: "loss-spike",
        category: "quality",
        label: "loss 突然尖峰 / NaN",
        title: "loss 突然尖峰或出现 NaN",
        observe: "定位 step、数据样本、精度溢出、grad_norm、被跳过的 optimizer step 与异常长度。",
        cause: "异常数据、数值溢出、mask/长度异常、学习率过高、精度累加不足、梯度爆炸。",
        next: "回放 spike 前后 batch，检查 per-token loss 最大值；对比 BF16/FP32 累加并临时开启裁剪。",
        avoid: "裁剪可限制梯度幅度，触发尖峰的样本仍需回放检查。",
      },
      {
        id: "train-eval-gap",
        category: "quality",
        label: "训练 loss 降，评估不升",
        title: "训练 loss 降低但 held-out 指标不提升",
        observe: "训练/验证分布、污染、有效 token、数据重复、指标与最终任务是否一致。",
        cause: "过拟合、模板泄漏、验证污染、tokenizer 不一致、训练目标与任务错配。",
        next: "按来源/长度/领域拆 loss；检查训练/评估重叠；换独立 prompt 集做 paired eval。",
        avoid: "checkpoint 选择应结合独立验证结果和任务切片。",
      },
      {
        id: "low-util",
        category: "throughput",
        label: "GPU 利用率周期性掉零",
        title: "GPU 利用率低或周期性掉到零",
        observe: "dataloader wait、CPU tokenization、worker 数、小文件读取、同步与 checkpoint 写入。",
        cause: "数据饥饿、collate 太慢、存储吞吐不足、collective 等待、日志/保存阻塞。",
        next: "做 timeline profile，区分 input wait 和 compute；离线预处理 shard 并启用预取。",
        avoid: "先确定等待来自输入、计算还是通信，再选择 kernel 或硬件调整。",
      },
      {
        id: "oom",
        category: "memory",
        label: "OOM 或 peak memory 异常",
        title: "显存不足，峰值无法解释",
        observe: "参数、梯度、optimizer、activation、T×T attention、workspace 与碎片分别测。",
        cause: "长序列、microbatch 太大、attention 未融合、activation 保留、optimizer state 复制。",
        next: "固定模型记录 peak；减小 T/microbatch、开 activation checkpointing、查 ZeRO/FSDP 与 attention kernel。",
        avoid: "accumulation 主要改变有效 batch；静态参数占用需通过分片、精度或模型规模处理。",
      },
      {
        id: "multi-slow",
        category: "distributed",
        label: "多卡反而更慢",
        title: "扩大 GPU 数后吞吐不升甚至下降",
        observe: "collective 时间、跨节点带宽、bucket、TP/PP 比例、全局 batch 与每卡工作量。",
        cause: "通信压过计算、过多小 collective、pipeline bubble、数据不均、straggler。",
        next: "按 DP/TP/PP 分组 profile；先小规模测单卡与 2/4/8 卡的 scaling efficiency。",
        avoid: "先测通信、计算和等待时间，再调整并行配置。",
      },
      {
        id: "resume",
        category: "artifact",
        label: "恢复后 loss 轨迹断裂",
        title: "checkpoint 恢复后轨迹与中断前不连续",
        observe: "optimizer/scheduler/RNG/data cursor、accumulation 边界、global step 与 sharding 映射。",
        cause: "只加载模型不加载优化器、数据重复/跳过、LR 位置错、随机数未恢复。",
        next: "从同一 checkpoint fork 两次各跑 5 步，逐 step 比较 loss、grad_norm、LR 与样本 ID。",
        avoid: "在相同后续样本上比较恢复结果，并声明数值容差。",
      },
      {
        id: "sft",
        category: "serving",
        label: "SFT 后生成退化",
        title: "SFT token loss 下降，生成效果反而变差",
        observe: "chat template、response mask、EOS、长度、packing 串样本、采样参数与 held-out 生成。",
        cause: "模板不一致、只学格式、过拟合短模板、停止 token 错、训练数据质量差。",
        next: "固定 decoding 参数做新旧模型 paired generation；按任务/长度/格式切片人工抽检。",
        avoid: "结合生成样本检查监督目标，再决定是否继续训练。",
      },
      {
        id: "retrieval",
        category: "serving",
        label: "后训练后检索退化",
        title: "模型更新后检索效果下降",
        observe: "pooling/projection/normalize、query/doc instruction、维度、索引版本与 chunk 版本。",
        cause: "模型接口变化、索引未重建、训练目标与检索不一致、false negatives、领域漂移。",
        next: "同一批 query 同时测旧/新模型完整索引；拆 query 端与 document 端变化。",
        avoid: "查询向量与文档索引的模型、维度和编码协议应保持兼容。",
      },
      {
        id: "long-context",
        category: "memory",
        label: "长上下文训练显存/通信爆炸",
        title: "长上下文运行成本急剧上升",
        observe: "T² attention、activation、KV、context parallel 通信、序列长度分布与 packing。",
        cause: "长样本比例、未融合 attention、global batch 过大、CP/TP 通信、KV 重复存储。",
        next: "按 T 分桶测峰值与 tokens/s；测试 FlashAttention、activation checkpointing、context parallel 与 microbatch。",
        avoid: "同时查看长度分布尾部及其对整批运行时间的影响。",
      },
    ];
    let selectedCategory = "all";
    let selectedProblem = problems[0];

    const renderList = () => {
      const filtered = selectedCategory === "all"
        ? problems
        : problems.filter((problem) => problem.category === selectedCategory);
      if (!filtered.includes(selectedProblem)) selectedProblem = filtered[0] || problems[0];
      $("[data-problem-list]").innerHTML = filtered.map((problem) => `
        <button type="button" data-problem-id="${problem.id}" aria-pressed="${problem === selectedProblem}">
          ${problem.label}
        </button>
      `).join("");
      $$("[data-problem-id]").forEach((button) => {
        button.addEventListener("click", () => {
          selectedProblem = problems.find((problem) => problem.id === button.dataset.problemId);
          renderList();
          renderDetail();
        });
      });
    };
    const renderDetail = () => {
      $("[data-problem-category-label]").textContent = selectedProblem.category.toUpperCase();
      $("[data-problem-title]").textContent = selectedProblem.title;
      $("[data-problem-observe]").textContent = selectedProblem.observe;
      $("[data-problem-cause]").textContent = selectedProblem.cause;
      $("[data-problem-next]").textContent = selectedProblem.next;
      $("[data-problem-avoid]").textContent = selectedProblem.avoid;
    };
    $$("[data-problem-category]").forEach((button) => {
      button.addEventListener("click", () => {
        selectedCategory = button.dataset.problemCategory;
        $$("[data-problem-category]").forEach((item) => {
          item.setAttribute("aria-pressed", String(item === button));
        });
        renderList();
        renderDetail();
      });
    });
    renderList();
    renderDetail();
}
