import { $ } from "../core.js";

export function selectionProbabilities(p, n, q) {
  const allCorrect = p ** n;
  const coverage = 1 - (1 - p) ** n;
  return { coverage, selected: allCorrect + q * Math.max(0, coverage - allCorrect) };
}

export function initReasoningCompute() {
  const pInput = $("[data-reason-p]");
  if (!pInput) return;
  const nInput = $("[data-reason-n]");
  const qInput = $("[data-reason-q]");
  const pct = x => `${(100 * x).toFixed(1)}%`;
  const render = () => {
    const p = Number(pInput.value) / 100, n = Number(nInput.value), q = Number(qInput.value) / 100;
    const { coverage, selected } = selectionProbabilities(p, n, q);
    $("[data-reason-p-value]").textContent = pct(p);
    $("[data-reason-n-value]").textContent = n;
    $("[data-reason-q-value]").textContent = pct(q);
    $("[data-reason-coverage]").textContent = pct(coverage);
    $("[data-reason-selected]").textContent = pct(selected);
    $("[data-reason-cost]").textContent = `${n}×`;
    $("[data-reason-bars]").innerHTML = [["单条候选", p], ["候选覆盖", coverage], ["实际选中", selected]].map(([label, value]) => `<div class="bar-row"><span>${label}</span><div class="bar-track"><i style="width:${value * 100}%"></i></div><output>${pct(value)}</output></div>`).join("");
    $("[data-reason-note]").textContent = n === 1 ? "只有一个候选时无选择问题，结果就是 p；q 不影响结果。" : `候选覆盖与最终选中的差距是 ${pct(coverage - selected)}。增加候选后，最终收益还取决于选择器，应一起评估。`;
  };
  [pInput, nInput, qInput].forEach(input => input.addEventListener("input", render));
  render();
}
