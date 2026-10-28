import { $, $$ } from "../core.js";

export function dpoReadout(beta, delta) {
  const margin = beta * delta;
  return {
    margin,
    loss: Math.max(-margin, 0) + Math.log1p(Math.exp(-Math.abs(margin))),
    gradient: -beta / (1 + Math.exp(margin)),
  };
}

export function initPosttrainingDeep() {
    const scopeButtons = $$("[data-scope]");
    if (scopeButtons.length) {
      const scopeData = {
        full: {
          kicker: "FULL FINE-TUNE",
          title: "所有层都能改变",
          copy: "可训练自由度最大，优化器状态通常也最多。小数据下可能过拟合或遗忘，与 LoRA 的差异需要在同预算、同数据下比较，reference 取决于训练目标。",
          geometry: "强",
          cost: "最高",
          risk: "高",
          encoder: "trainable",
          pooler: "trainable",
          projection: "trainable",
          head: "trainable",
        },
        lora: {
          kicker: "LOW-RANK UPDATE",
          title: "只让低秩增量参与训练",
          copy: "LoRA 固定原权重，在选定矩阵旁学习低秩增量；Adapter 通常插入小模块，两者各有相应的参数配置。本示意额外开放读出与输出头，实际是否训练这些模块由配置决定。LoRA 省参数梯度和优化器状态，但仍需要反传及激活。",
          geometry: "中到强",
          cost: "中等",
          risk: "中",
          encoder: "partial",
          pooler: "trainable",
          projection: "trainable",
          head: "trainable",
        },
        output: {
          kicker: "PROJECTION / HEAD ONLY",
          title: "冻结大部分表示，只改最后映射",
          copy: "backbone 与 pooling 固定，投影和任务头可训练。对于 embedding API，这仍然能改变最终坐标，编码器内部的 token 表示保持固定。",
          geometry: "有限",
          cost: "低",
          risk: "较低",
          encoder: "frozen",
          pooler: "frozen",
          projection: "trainable",
          head: "trainable",
        },
        frozen: {
          kicker: "BACKBONE FROZEN",
          title: "固定编码器，训练读出与输出",
          copy: "编码器固定，训练有参数的读出、投影和任务头。最终能力受冻结表示保留的信息与读出容量限制；只有线性读出才直接受线性可分性限制。mean pooling 或直接读 EOS 本身无可训练参数。",
          geometry: "受限于读出层",
          cost: "较低",
          risk: "低到中",
          encoder: "frozen",
          pooler: "trainable",
          projection: "trainable",
          head: "trainable",
        },
      };

      const renderScope = (scope) => {
        const data = scopeData[scope];
        scopeButtons.forEach((button) => {
          button.setAttribute("aria-pressed", String(button.dataset.scope === scope));
        });
        $("[data-scope-kicker]").textContent = data.kicker;
        $("[data-scope-title]").textContent = data.title;
        $("[data-scope-copy]").textContent = data.copy;
        $("[data-scope-geometry]").textContent = data.geometry;
        $("[data-scope-cost]").textContent = data.cost;
        $("[data-scope-risk]").textContent = data.risk;
        ["encoder", "pooler", "projection", "head"].forEach((block) => {
          const element = $(`[data-scope-block="${block}"]`);
          element.classList.remove("is-trainable", "is-frozen", "is-partial");
          const state = data[block];
          element.classList.add(state === "trainable" ? "is-trainable" : state === "partial" ? "is-partial" : "is-frozen");
        });
      };
      scopeButtons.forEach((button) => {
        button.addEventListener("click", () => renderScope(button.dataset.scope));
      });
      renderScope("full");
    }

    const betaInput = $("[data-dpo-beta-input]");
    if (!betaInput) return;
    const marginInput = $("[data-dpo-margin-input]");
    const renderDpo = () => {
      const beta = Number(betaInput.value) / 100;
      const logprobMargin = Number(marginInput.value) / 100;
      const { margin: rewardMargin, loss, gradient } = dpoReadout(beta, logprobMargin);
      $("[data-dpo-beta]").textContent = beta.toFixed(2);
      $("[data-dpo-margin]").textContent = logprobMargin.toFixed(2);
      $("[data-dpo-reward]").textContent = rewardMargin.toFixed(3);
      $("[data-dpo-loss]").textContent = loss.toFixed(3);
      $("[data-dpo-gradient]").textContent = gradient.toFixed(3);

      let explanation;
      if (rewardMargin < 0) {
        explanation = "Δ 为负：策略的 chosen/rejected 对数概率差低于 reference 的同一差值，不代表 chosen 的绝对概率一定更低。∂L/∂Δ 为负，梯度下降会增加这个相对差；幅度也取决于 β。";
      } else if (rewardMargin < 2) {
        explanation = "Δ 为正：相对 reference，策略增加了 chosen 对 rejected 的优势。面板显示的是对 Δ 的导数 −βσ(−βΔ)，对 m 的导数则为 −σ(−m)；固定 Δ 改 β，会同时改变缩放系数和 sigmoid 饱和程度。";
      } else {
        explanation = "正 margin 较大，sigmoid 开始饱和。当前样本梯度可能减弱；训练后的 KL 还取决于整体优化轨迹。固定奖励的 KL 正则目标里，β 越大约束越强；有限数据的 DPO 动态还取决于学习率和训练过程。";
      }
      $("[data-dpo-explanation]").textContent = explanation;
    };
    betaInput.addEventListener("input", renderDpo);
    marginInput.addEventListener("input", renderDpo);
    renderDpo();
}
