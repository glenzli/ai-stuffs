import { $ } from "../core.js";

function validateProbability(value, name) {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(`${name} must be a finite probability between 0 and 1`);
  }
}

export function repeatedSuccess(p = 0.8, k = 5) {
  validateProbability(p, "p");
  if (!Number.isInteger(k) || k < 1) {
    throw new RangeError("k must be a positive integer");
  }
  return { any: 1 - (1 - p) ** k, all: p ** k };
}

export function judgeLedger({ truth = 0.6, sensitivity = 0.9, specificity = 0.9 } = {}) {
  validateProbability(truth, "truth");
  validateProbability(sensitivity, "sensitivity");
  validateProbability(specificity, "specificity");
  const tp = truth * sensitivity;
  const fn = truth * (1 - sensitivity);
  const fp = (1 - truth) * (1 - specificity);
  const tn = (1 - truth) * specificity;
  const observed = tp + fp;
  return { tp, fn, fp, tn, observed, bias: observed - truth };
}

export function initEvaluationReliability() {
  const pInput = $("[data-er-p]");
  if (!pInput) return;
  const kInput = $("[data-er-k]");
  const truthInput = $("[data-er-truth]");
  const sensitivityInput = $("[data-er-sensitivity]");
  const specificityInput = $("[data-er-specificity]");
  const pct = value => `${(value * 100).toFixed(2)}%`;
  const setText = (selector, value) => {
    const output = $(selector);
    if (output) output.textContent = value;
  };
  const render = () => {
    for (const [name, input] of [["p", pInput], ["k", kInput], ["truth", truthInput], ["sensitivity", sensitivityInput], ["specificity", specificityInput]]) {
      if (input) setText(`[data-er-${name}-value]`, `${input.value}${name === "k" ? "" : "%"}`);
    }
    if (kInput) {
      const result = repeatedSuccess(Number(pInput.value) / 100, Number(kInput.value));
      setText("[data-er-any]", pct(result.any));
      setText("[data-er-all]", pct(result.all));
    }
    if (truthInput && sensitivityInput && specificityInput) {
      const ledger = judgeLedger({
        truth: Number(truthInput.value) / 100,
        sensitivity: Number(sensitivityInput.value) / 100,
        specificity: Number(specificityInput.value) / 100,
      });
      setText("[data-er-observed]", pct(ledger.observed));
      const points = Number((ledger.bias * 100).toFixed(1));
      setText("[data-er-bias]", `${points > 0 ? "+" : ""}${points.toFixed(1)} pp`);
      const rows = $("[data-er-rows]");
      if (rows) rows.innerHTML = [
        ["正确结果判通过 (TP)", ledger.tp],
        ["正确结果判失败 (FN)", ledger.fn],
        ["错误结果判通过 (FP)", ledger.fp],
        ["错误结果判失败 (TN)", ledger.tn],
      ].map(([label, fraction]) => `<tr><th scope="row">${label}</th><td>${pct(fraction)}</td></tr>`).join("");
    }
    setText("[data-er-note]", `每次成功率 ${pInput.value}%，重复 ${kInput?.value || 1} 次。至少一次成功反映尝试机会，全部成功反映重复可靠性；两者在独立同分布假设下计算。`);
  };
  [pInput, kInput, truthInput, sensitivityInput, specificityInput].forEach(input => input?.addEventListener("input", render));
  render();
}
