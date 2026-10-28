import { $, $$ } from "../core.js";

export function initPosttraining() {
    const stageData = {
      pretrain: {
        label: "GENERAL PRETRAINING",
        title: "先从海量文本学出通用预测能力",
        copy: "语言模型用下一词预测、掩码预测等自监督目标学习语言结构与世界统计。句向量检索质量和回答的有用性需要相应任务训练与评估。",
        objective: "L = -Σ log p(tokenₜ₊₁ | token≤ₜ)",
        data: "无标注文本；token ID 与错位目标",
        supervision: "文本自身",
        update: "通常全部参数",
        eval: "语言建模 / 下游任务",
        risk: "检索匹配和回答偏好尚需任务评估",
        impact: { geometry: "indirect", distribution: "direct", ranking: "none" },
      },
      representation: {
        label: "REPRESENTATION POST-TRAINING",
        title: "直接重塑“什么与什么靠近”",
        copy: "用 query、正样本和难负样本继续训练编码器。这里的 loss 就是在评价向量检索排序，梯度会直接改变句向量几何。",
        objective: "L = CE(q · dᵀ / τ, positive_index)",
        data: "(query, positive, hard negatives) 成组样本",
        supervision: "人工 / 规则 / 强模型排序",
        update: "编码器、投影层或 LoRA / Adapter",
        eval: "Recall@k、MRR、nDCG、领域相关性",
        risk: "错误难负例、目标过窄、其他领域表现退化",
        impact: { geometry: "direct", distribution: "indirect", ranking: "none" },
      },
      sft: {
        label: "SUPERVISED FINE-TUNING",
        title: "用示范回答改变 token 概率分布",
        copy: "SFT 常仍是 next-token cross entropy。区别在于 prompt 与 response 的角色被显式标出，通常只对 answer 部分计算 loss，让模型模仿示范回答。",
        objective: "L = -Σ log p(responseₜ | prompt, response<t)",
        data: "(prompt, high-quality response) 指令数据",
        supervision: "人类 / 强模型示范",
        update: "全部参数或参数高效微调",
        eval: "指令遵循、任务准确率、人工偏好",
        risk: "模仿风格、过拟合示范、泛化与拒答能力下降",
        impact: { geometry: "indirect", distribution: "direct", ranking: "indirect" },
      },
      preference: {
        label: "PREFERENCE OPTIMIZATION",
        title: "比较两个回答，学习偏好",
        copy: "RLHF 常先学 reward model，再用策略优化；DPO 则直接比较 chosen 与 rejected 的相对概率。它处理开放式回答的主观取舍与安全边界。",
        objective: "提高 chosen 相对 rejected 的策略/参考模型对数概率优势",
        data: "(prompt, chosen response, rejected response)",
        supervision: "人类或 AI 偏好排序",
        update: "策略模型；RLHF 还涉及 reward model",
        eval: "成对偏好胜率、安全评测、人工审阅",
        risk: "奖励投机、偏好偏差、过度拒答、分布漂移",
        impact: { geometry: "none", distribution: "indirect", ranking: "direct" },
      },
    };

    const buttons = $$("[data-post-stage]");
    if (!buttons.length) return;
    const fields = {
      label: $("[data-post-label]"),
      title: $("[data-post-title]"),
      copy: $("[data-post-copy]"),
      objective: $("[data-post-objective]"),
      data: $("[data-post-data]"),
      supervision: $("[data-post-supervision]"),
      update: $("[data-post-update]"),
      eval: $("[data-post-eval]"),
      risk: $("[data-post-risk]"),
    };

    const render = (key) => {
      const data = stageData[key];
      buttons.forEach((button) => {
        button.setAttribute("aria-selected", String(button.dataset.postStage === key));
      });
      fields.label.textContent = data.label;
      fields.title.textContent = data.title;
      fields.copy.textContent = data.copy;
      fields.objective.textContent = data.objective;
      fields.data.textContent = data.data;
      fields.supervision.textContent = data.supervision;
      fields.update.textContent = data.update;
      fields.eval.textContent = data.eval;
      fields.risk.textContent = data.risk;
      $$("[data-impact]").forEach((lane) => {
        lane.classList.remove("is-direct", "is-indirect");
        const state = data.impact[lane.dataset.impact];
        if (state === "direct") lane.classList.add("is-direct");
        if (state === "indirect") lane.classList.add("is-indirect");
      });
    };

    buttons.forEach((button) => {
      button.addEventListener("click", () => render(button.dataset.postStage));
    });
    render("pretrain");
}
