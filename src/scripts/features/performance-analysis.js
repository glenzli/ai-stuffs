import { $ } from "../core.js";

export function gemmRoofline({ m, k = 4096, n = 4096, bytes = 2, peak = 100e12, bandwidth = 1e12 }) {
  const flops = 2 * m * k * n, traffic = bytes * (m * k + k * n + m * n);
  const intensity = flops / traffic, computeTime = flops / peak, memoryTime = traffic / bandwidth;
  return { flops, traffic, intensity, computeTime, memoryTime, time: Math.max(computeTime, memoryTime), bound: computeTime >= memoryTime ? "算力" : "带宽", attainable: Math.min(peak, bandwidth * intensity), ridge: peak / bandwidth };
}

export function initPerformanceAnalysis() {
  const mInput = $("[data-perf-m]"); if (!mInput) return;
  const bytes = $("[data-perf-bytes]"), compute = $("[data-perf-compute]"), bandwidth = $("[data-perf-bandwidth]");
  const render = () => {
    const config = { m: Number(mInput.value), bytes: Number(bytes.value), peak: Number(compute.value) * 1e12, bandwidth: Number(bandwidth.value) * 1e12 };
    const r = gemmRoofline(config);
    $("[data-perf-intensity]").textContent = r.intensity.toFixed(1);
    $("[data-perf-time]").textContent = (r.time * 1000).toFixed(4);
    $("[data-perf-bound]").textContent = r.bound;
    $("[data-perf-rows]").innerHTML = `<tr><td>${(r.flops / 1e9).toFixed(3)} GFLOPs</td><td>${(r.traffic / 1e6).toFixed(2)} MB</td><td>${(r.computeTime * 1000).toFixed(4)} ms</td><td>${(r.memoryTime * 1000).toFixed(4)} ms</td></tr>`;
    $("[data-perf-note]").textContent = `当前 I=${r.intensity.toFixed(1)}，屋顶拐点 P/BW=${r.ridge.toFixed(1)} FLOPs/B，理想限制来自${r.bound}。增加 M 会增加工作总量，也能摊薄权重读取；这里的 ${ (r.attainable / 1e12).toFixed(2) } TFLOP/s 表示理论上界；设备实测速率还受 kernel 和调度影响。`;
    const x = value => 58 + (Math.log10(value) + 1) / 5 * 530;
    const y = value => 220 - (Math.log10(value / 1e12) + 2) / 5 * 180;
    const points = Array.from({ length: 70 }, (_, i) => { const intensity = 10 ** (-1 + 5 * i / 69); return [x(intensity), y(Math.min(config.peak, config.bandwidth * intensity))]; });
    const ticks = [0.1, 1, 10, 100, 1000, 10000].map(value => `<text x="${x(value)}" y="241" text-anchor="middle">${value}</text>`).join("");
    const yticks = [0.01, 0.1, 1, 10, 100, 1000].map(value => `<text x="49" y="${y(value * 1e12) + 4}" text-anchor="end">${value}</text>`).join("");
    $("[data-perf-chart]").innerHTML = `<title>Roofline 理想吞吐上限，横纵均为对数轴</title><g fill="var(--ink-soft)" font-size="11" font-family="var(--mono)">${ticks}${yticks}<text x="58" y="18">TFLOP/s</text><text x="420" y="265">FLOPs / byte →</text></g><path d="M58,30 V220 H590" stroke="var(--line-strong)" fill="none"/><path d="${points.map(([px,py],i) => `${i ? "L" : "M"}${px},${py}`).join(" ")}" fill="none" stroke="var(--accent)" stroke-width="3"/><line x1="${x(r.intensity)}" y1="220" x2="${x(r.intensity)}" y2="${y(r.attainable)}" stroke="var(--blue)" stroke-dasharray="4 4"/><circle cx="${x(r.intensity)}" cy="${y(r.attainable)}" r="6" fill="var(--coral)"/>`;
  };
  [mInput, bytes, compute, bandwidth].forEach(input => input.addEventListener("input", render)); render();
}
