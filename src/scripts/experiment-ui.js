// Pure presentation shared by measured reports; contains no experiment data.
export const esc = value => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
export const pct = x => `${(100 * x).toFixed(2)}%`;
export const num = x => x === 0 ? "0" : Math.abs(x) < .001 ? x.toExponential(3) : x.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");

export function table(headers, rows, label) {
  return `<div class="compare-wrap practice-scroll" tabindex="0" role="region" aria-label="${esc(label)}"><table class="decision-table"><thead><tr>${headers.map(h => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map((value, i) => `<${i ? "td" : "th scope=\"row\""}>${value}</${i ? "td" : "th"}>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
export function lineChart(series, xLabel, yLabel) {
  const points = series.flatMap(row => row.points), xmax = Math.max(1, ...points.map(p => p.x)), ymax = Math.max(.000001, ...points.map(p => p.y));
  const x = value => 65 + value / xmax * 475, y = value => 225 - value / ymax * 145;
  return `<svg class="practice-chart" viewBox="0 0 620 285" role="img" aria-label="${esc(xLabel + '；' + yLabel)}"><line x1="65" y1="225" x2="540" y2="225"/><line x1="65" y1="80" x2="65" y2="225"/><text x="8" y="85">${esc(num(ymax))}</text><text x="25" y="225">0</text><text x="65" y="249">0</text><text x="540" y="249" text-anchor="end">${esc(num(xmax))}</text><text x="540" y="275" text-anchor="end">${esc(xLabel)}</text><text x="8" y="25">${esc(yLabel)}</text>${series.map((row, i) => `<g class="practice-series-${i % 2}"><text x="${65 + i * 250}" y="52">${esc(row.name)}</text><polyline points="${row.points.map(p => `${x(p.x)},${y(p.y)}`).join(' ')}" fill="none"/>${row.points.map(p => `<circle cx="${x(p.x)}" cy="${y(p.y)}" r="3"><title>${esc(`${row.name}：${p.x} / ${num(p.y)}`)}</title></circle>`).join('')}</g>`).join('')}</svg>`;
}
export function metrics(items) { return `<dl class="practice-metrics">${items.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>`; }
