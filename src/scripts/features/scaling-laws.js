import { $, $$ } from "../core.js";

export function initScalingLaws() {
    const compute = $("[data-scaling-compute]");
    if (!compute) return;
    const model = $("[data-scaling-model]");
    const unique = $("[data-scaling-unique]");

    const formatTokens = (value) => {
      if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
      if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
      if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
      return Math.round(value).toLocaleString();
    };
    const modelFromSlider = (index) => 0.1 * Math.pow(10, (index / 20) * Math.log10(700));

    const render = () => {
      const totalCompute = Number(compute.value);
      const params = modelFromSlider(Number(model.value)) * 1e9;
      const tokens = totalCompute / (6 * params);
      const uniqueTokens = Number(unique.value);
      const ratio = tokens / params;
      const epochs = tokens / uniqueTokens;
      $("[data-scaling-model-value]").textContent = `${(params / 1e9).toFixed(params >= 10e9 ? 0 : 1)}B`;
      $("[data-scaling-tokens]").textContent = formatTokens(tokens);
      $("[data-scaling-ratio]").textContent = ratio.toFixed(ratio >= 10 ? 0 : 1);
      $("[data-scaling-epochs]").textContent = `${epochs.toFixed(2)}×`;
      let verdict;
      if (ratio < 10) {
        verdict = "D/N 偏低：在固定 compute 下，这个模型可能偏向“容量大但训练 token 少”。是否值得取决于数据质量与推理需求。";
      } else if (ratio <= 40) {
        verdict = "本图用 10–40 作教学分档，围绕常被引用的约 20 tokens/param 基准；分档只用于观察预算分配，最优范围应通过目标设置下的实验估计。";
      } else if (ratio <= 100) {
        verdict = "这个分配更偏数据密集：同样 compute 下模型较小、训练 token 更多。推理成本可能更友好，但容量上限与训练时长需单独分析。";
      } else {
        verdict = "D/N 很高，训练更偏“小模型、多 token”。如果唯一数据不足，会引入大量重复；重复不能等同全新信息。";
      }
      if (epochs > 1) verdict += ` 当前唯一数据约需 ${epochs.toFixed(1)} 个 epoch，需单独评估重复数据的收益与过拟合。`;
      $("[data-scaling-verdict]").textContent = verdict;

      const width = 520;
      const height = 260;
      const pad = 32;
      const minLogN = Math.log10(0.1e9);
      const maxLogN = Math.log10(70e9);
      const minLogD = Math.log10(totalCompute / (6 * 70e9));
      const maxLogD = Math.log10(totalCompute / (6 * 0.1e9));
      const point = (n) => {
        const d = totalCompute / (6 * n);
        return [
          pad + ((Math.log10(n) - minLogN) / (maxLogN - minLogN)) * (width - pad * 2),
          height - pad - ((Math.log10(d) - minLogD) / (maxLogD - minLogD)) * (height - pad * 2),
        ];
      };
      const values = Array.from({ length: 31 }, (_, index) => {
        const n = 0.1e9 * Math.pow(700, index / 30);
        return point(n);
      });
      const path = values.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
      const [px, py] = point(params);
      $("[data-scaling-chart]").innerHTML = `
        <title>固定 compute 下参数与 token 的权衡</title>
        <rect x="0.5" y="0.5" width="519" height="259" rx="7" fill="none" stroke="var(--line)" />
        <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="var(--line)" />
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="var(--line)" />
        <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2" />
        <circle cx="${px}" cy="${py}" r="5" fill="var(--coral)" />
        <text x="${pad}" y="16" fill="var(--ink-faint)" font-size="8">D tokens ↑</text>
        <text x="${width - pad - 35}" y="${height - 8}" fill="var(--ink-faint)" font-size="8">N params →</text>
      `;
    };
    [compute, model, unique].forEach((input) => {
      input.addEventListener("input", render);
      input.addEventListener("change", render);
    });
    render();
}
