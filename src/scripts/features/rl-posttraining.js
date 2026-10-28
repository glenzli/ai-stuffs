import { $, $$ } from "../core.js";

export function initRlPosttraining() {
    const buttons = $$("[data-reward-source]");
    if (!buttons.length) return;
    const sources = {
      preference: {
        kicker: "PREFERENCE PAIR",
        title: "同一 prompt 下比较 chosen 与 rejected",
        copy: "人工或 AI 给出相对偏好。可以用 Bradley-Terry 训练 reward model，也可以直接做 DPO；偏好数据覆盖有限，评分者一致性会直接进入模型。",
        data: "(prompt, chosen, rejected) 成对记录",
        algos: "RLHF + reward model / DPO",
        risk: "偏好偏差、长度偏置、judge 与训练同源、对开放式任务覆盖不足",
      },
      verifier: {
        kicker: "VERIFIABLE REWARD",
        title: "用测试、答案检查或约束满足自动判分",
        copy: "数学、代码和结构化任务可让 verifier 输出确定信号。这类信号有明确判定规则，覆盖范围受验证器限制，测试本身也可能存在漏洞。",
        data: "可执行任务、隐藏测试、答案校验器",
        algos: "RLVR + PPO/GRPO 等策略优化",
        risk: "测试泄漏、硬编码、格式漏洞、不可验证能力无法覆盖",
      },
      rubric: {
        kicker: "RUBRIC / AI JUDGE",
        title: "按维度给开放式回答评分",
        copy: "Rubric 可分解正确性、引用、格式、语气与安全。judge 的提示、模型版本和评分尺度都是训练协议的一部分。",
        data: "任务、rubric、judge 评分或成对判断",
        algos: "RLAIF / RLHF / DPO / 拒绝采样",
        risk: "judge 偏差、自我强化、可读格式胜过真实内容",
      },
      process: {
        kicker: "PROCESS REWARD",
        title: "为中间步骤提供过程奖励",
        copy: "过程奖励希望减少“蒙对答案”的奖励漏洞，但步骤标注昂贵，且推理路径并不唯一。需要与最终结果验证器结合。",
        data: "步骤级标注、工具轨迹、过程模型",
        algos: "PRM + search / policy optimization",
        risk: "步骤标注成本、路径偏置、模型迎合评审格式",
      },
    };
    const render = (key) => {
      const data = sources[key];
      buttons.forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.rewardSource === key));
      });
      $("[data-reward-kicker]").textContent = data.kicker;
      $("[data-reward-title]").textContent = data.title;
      $("[data-reward-copy]").textContent = data.copy;
      $("[data-reward-data]").textContent = data.data;
      $("[data-reward-algos]").textContent = data.algos;
      $("[data-reward-risk]").textContent = data.risk;
    };
    buttons.forEach((button) => {
      button.addEventListener("click", () => render(button.dataset.rewardSource));
    });
    render("preference");
}
