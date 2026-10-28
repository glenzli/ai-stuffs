import { $, $$ } from "../core.js";

export function lossReadout(scores, tau, mode = "infonce") {
  const softplus = (x) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
  if (mode === "sgns") {
    const probabilities = scores.map((s) => 1 / (1 + Math.exp(-s)));
    return {
      probabilities,
      loss: softplus(-scores[0]) + scores.slice(1).reduce((sum, s) => sum + softplus(s), 0),
      gradients: probabilities.map((p, i) => p - Number(i === 0)),
    };
  }
  const logits = scores.map((s) => s / tau);
  const max = Math.max(...logits);
  const exps = logits.map((z) => Math.exp(z - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  const probabilities = exps.map((value) => value / sum);
  return {
    probabilities,
    loss: max - logits[0] + Math.log(sum),
    gradients: probabilities.map((p, i) => (p - Number(i === 0)) / tau),
  };
}

export function initTrainingDeep() {
    const caseButtons = $$("[data-row-case]");
    const rowTable = $(".token-row-table");
    if (!caseButtons.length || !rowTable) return;

    const tieInput = $("[data-row-tie]");
    const rowFields = {
      kicker: $("[data-row-kicker]"),
      title: $("[data-row-title]"),
      formula: $("[data-row-formula]"),
      copy: $("[data-row-copy]"),
    };
    let activeCase = "repeat";

    const renderRows = () => {
      const tied = tieInput.checked;
      const states = {
        repeat: {
          kicker: "REPEATED GATHER",
          title: "两次 gather，梯度累加到同一行",
          formula: tied ? "dE[17] = dx[0] + dx[4] + dE_output[17]" : "dE_lookup[17] = dx[0] + dx[4]",
          copy: "ID 17 出现两次，lookup 分支把两个位置的梯度相加。完整词表 softmax 的输出分支通常对所有词表行都有梯度；绑定权重时，两条路径相加。表格显示当前梯度的来源；实际参数位移还取决于优化器状态。",
        },
        target: {
          kicker: "INPUT VS OUTPUT ROLE",
          title: tied ? "权重绑定让未输入的词表行也收到输出梯度" : "未输入的 token：无 lookup 梯度，有输出梯度",
          formula: tied ? "dE_output[77] = Σ_t (p[t,77] - y[t,77]) h[t]" : "77 ∉ ids → dE_lookup[77] = 0",
          copy: tied
            ? "token 77 没有被输入查表，但参与了输出 softmax。绑定后输出梯度回到 E[77]；这不仅适用于目标 token，未命中的候选行通常也有输出梯度。"
            : "token 77 没有 lookup 梯度。W_out 采用 [V,d] 存储时，第 77 行及其他候选行通常都有输出梯度：目标类系数为 p−1，非目标类为 p。",
        },
        padding: {
          kicker: "MASKED OUT",
          title: tied ? "PAD 的 lookup 被屏蔽，绑定输出仍能给它梯度" : "PAD 标签位置与输出类别分别处理",
          formula: tied ? "dE_lookup[PAD] = 0; dE_output[PAD] 通常 ≠ 0" : "ignore_index 屏蔽位置；softmax 仍含 PAD 类别",
          copy: "假设 PAD 输入不影响有效位置，PAD 标签也不计 loss。ignore_index 排除的是训练位置，并不从其他有效位置的 softmax 分母删除 PAD 类别。padding_idx 也只保护 lookup 分支；绑定输出、动量与权重衰减仍需单独检查。",
        },
        frozen: {
          kicker: "STOP GRADIENT",
          title: "冻结 E 后，输入查表行不再更新",
          formula: tied ? "E 冻结 → 输入与输出共享参数都不更新" : "E 不求参数梯度；独立 W_out 仍可训练",
          copy: tied
            ? "如果输入 embedding 冻结且输出投影又绑定到同一矩阵，两边都无法通过常规优化更新。"
            : "E 不参与参数优化；独立 W_out 的各候选行仍接收完整词表交叉熵梯度。这里假设冻结参数已从优化器更新中排除；Transformer 仍可学习。",
        },
      };

      caseButtons.forEach((button) => {
        const selected = button.dataset.rowCase === activeCase;
        button.setAttribute("aria-pressed", String(selected));
      });
      Object.entries(states[activeCase]).forEach(([key, value]) => {
        rowFields[key].textContent = value;
      });

      const setState = (token, inputUpdate, outputUpdate, inputLabel, outputLabel) => {
        const row = $(`[data-row="${token}"]`, rowTable);
        if (!row) return;
        const input = $(".row-state:nth-child(4)", row);
        const output = $(".row-state:nth-child(5)", row);
        input.textContent = inputLabel;
        output.textContent = outputLabel;
        input.className = `row-state ${inputUpdate ? "is-update" : inputLabel === "冻结" ? "is-frozen" : "is-zero"}`;
        output.className = `row-state ${outputUpdate ? "is-update" : outputLabel === "绑定" ? "is-frozen" : "is-zero"}`;
      };

      for (const token of ["17", "4821", "77", "PAD"]) {
        const frozen = activeCase === "frozen";
        const lookup = token === "17" || token === "4821";
        const inputLabel = frozen ? "冻结" : tied ? (lookup ? "两路梯度" : "输出梯度") : lookup ? "查表梯度" : "零梯度";
        const outputLabel = frozen && tied ? "绑定冻结" : tied ? "共享 E" : "输出梯度";
        setState(token, !frozen && (lookup || tied), !(frozen && tied), inputLabel, outputLabel);
      }
    };

    caseButtons.forEach((button) => {
      button.addEventListener("click", () => {
        activeCase = button.dataset.rowCase;
        renderRows();
      });
    });
    tieInput.addEventListener("change", renderRows);
    renderRows();

    const lossModeButtons = $$("[data-loss-mode]");
    if (!lossModeButtons.length) return;
    const tauInput = $("[data-math-tau-input]");
    const hardInput = $("[data-math-hard-input]");
    let lossMode = "infonce";

    const renderLossMath = () => {
      const tau = Number(tauInput.value) / 100;
      const hard = Number(hardInput.value) / 100;
      const rawScores = [2, hard, 1.05, 0.6, 0.2];
      $("[data-math-tau]").textContent = tau.toFixed(2);
      $("[data-math-hard]").textContent = hard.toFixed(2);
      lossModeButtons.forEach((button) => {
        button.setAttribute("aria-selected", String(button.dataset.lossMode === lossMode));
      });

      const { probabilities, loss, gradients } = lossReadout(rawScores, tau, lossMode);
      tauInput.disabled = lossMode === "sgns";
      $$("[data-loss-equation]").forEach(equation => { equation.hidden = equation.dataset.lossEquation !== lossMode; });
      let explanation;
      if (lossMode === "infonce") {
        $("[data-math-p]").previousElementSibling.textContent = "真实类概率";
        explanation = probabilities[1] > probabilities[0]
          ? "难负 A 分数超过正例。它对原始分数的导数为正，梯度下降会压低它；正例导数为负，更新会抬高它。温度变小会强化这次错误竞争。"
          : "正例分数最高。降低温度会让分布更尖，但当正例已经明显胜出，p₊ 接近 1，梯度反而可能趋近零；需要同时看概率差与 1/τ 的作用。";
        $("[data-loss-key]").textContent = "InfoNCE 对所有候选分数一起归一化。改变一个负例的分数，也会改变其他候选的概率。";
      } else {
        $("[data-math-p]").previousElementSibling.textContent = "正配对概率";
        explanation = "SGNS 把每对样本当独立二分类。正配对概率不因负样本变多而除以新的分母；但每增加一个负例，总 loss 就会多一项。";
        $("[data-loss-key]").textContent = "SGNS 的负例各自独立做 logistic 分类，各候选概率分别计算。";
      }

      $("[data-math-p]").textContent = probabilities[0].toFixed(3);
      $("[data-math-loss]").textContent = loss.toFixed(3);
      $("[data-math-grad]").textContent = gradients[0].toFixed(3);
      $("[data-math-negative-grad]").textContent = Math.max(...gradients.slice(1)).toFixed(3);
      $("[data-math-explanation]").textContent = explanation;
      $("[data-candidate-chart]").innerHTML = probabilities.map((probability, index) => `
        <div class="candidate-column ${index === 0 ? "is-positive" : ""}">
          <span>${probability.toFixed(3)}</span>
          <i style="height:${Math.max(3, probability * 100)}%"></i>
        </div>
      `).join("");
    };

    lossModeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        lossMode = button.dataset.lossMode;
        renderLossMath();
      });
    });
    tauInput.addEventListener("input", renderLossMath);
    hardInput.addEventListener("input", renderLossMath);
    renderLossMath();
}
