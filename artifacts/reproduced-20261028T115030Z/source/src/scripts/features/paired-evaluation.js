import { $ } from "../core.js";
import { escapeText, formatValue } from "./objective-math.js";

export function randomStream(seed = 73) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw new RangeError("随机种子必须是 uint32。");
  let state = seed >>> 0;
  return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
}
const average = a => a.reduce((s, x) => s + x, 0) / a.length;
export function linearQuantile(sorted, probability) {
  const index = (sorted.length - 1) * probability, lower = Math.floor(index), fraction = index - lower;
  return sorted[lower] * (1 - fraction) + sorted[Math.min(lower + 1, sorted.length - 1)] * fraction;
}
export function pairedExample(kind = "correlated") {
  if (!["correlated", "identical", "multi-seed"].includes(kind)) throw new RangeError("未知的评估示例。");
  const rows = [], sizes = [2, 6, 3, 5], shifts = [.18, .16, -.09, .06];
  for (const seed of kind === "multi-seed" ? ["11", "29", "47"] : ["11"]) {
    for (let group = 0; group < sizes.length; group++) for (let i = 0; i < sizes[group]; i++) {
      const old = .42 + .025 * ((i + group) % 4), delta = shifts[group] + (seed === "29" ? -.04 : seed === "47" ? .03 : 0);
      rows.push({ id: `doc${group + 1}-q${i + 1}`, group: `doc${group + 1}`, seed, old, new: kind === "identical" ? old : old + delta });
    }
  }
  return { schema: 1, rows };
}
export function validatePairedData(value) {
  if (!value || value.schema !== 1 || !Array.isArray(value.rows) || !value.rows.length || value.rows.length > 2000) throw new RangeError("需要 schema=1 和 1…2000 条 rows。");
  const seen = new Set();
  const rows = value.rows.map((row, index) => {
    if (!row || typeof row !== "object") throw new TypeError(`第 ${index + 1} 行不是对象。`);
    for (const name of ["id", "group"]) if (typeof row[name] !== "string" || !row[name].trim() || row[name].length > 100) throw new TypeError(`第 ${index + 1} 行需要非空 ${name}（最多 100 字符）。`);
    const seed = row.seed === undefined ? "default" : row.seed;
    if (typeof seed !== "string" || !seed.trim() || seed.length > 100) throw new TypeError("seed 必须是非空字符串。");
    if (![row.old, row.new].every(x => Number.isFinite(x) && x >= 0 && x <= 1)) throw new RangeError(`第 ${index + 1} 行需要成对的 old/new 分数，范围 0…1。`);
    const key = JSON.stringify([seed, row.id]); if (seen.has(key)) throw new RangeError(`同一 seed 内题目 ID 重复：${row.id}`); seen.add(key);
    return { id: row.id, group: row.group, seed, old: row.old, new: row.new };
  });
  return { schema: 1, rows };
}
function unitsFor(rows, mode) {
  if (!["row", "group"].includes(mode)) throw new RangeError("重采样单位必须是题目或来源组。");
  const groups = new Map();
  for (const row of rows) { const key = mode === "group" ? row.group : row.id; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(row); }
  return [...groups.values()].map(group => ({ total: group.reduce((s, row) => s + row.new - row.old, 0), count: group.length }));
}
export function exactPairedPermutation(rows, mode = "row") {
  const data = validatePairedData({ schema: 1, rows });
  if (new Set(data.rows.map(row => row.seed)).size !== 1) throw new RangeError("一次检验只接受一个训练 seed。");
  const units = unitsFor(data.rows, mode), observed = Math.abs(units.reduce((s, u) => s + u.total, 0));
  if (units.length > 16) return { p: null, permutations: null, units: units.length, reason: "独立单位超过 16，省略指数规模的精确枚举。" };
  let sums = [0]; for (const unit of units) sums = sums.flatMap(sum => [sum + unit.total, sum - unit.total]);
  const extreme = sums.filter(sum => Math.abs(sum) >= observed - 1e-12).length;
  return { p: extreme / sums.length, permutations: sums.length, units: units.length, extreme };
}
export function pairedStatistics({ data = pairedExample(), seed, mode = "group", randomSeed = 73, repetitions = 2000, threshold = .02 } = {}) {
  const validated = validatePairedData(data), seeds = [...new Set(validated.rows.map(row => row.seed))], chosenSeed = seed ?? seeds[0];
  const rows = validated.rows.filter(row => row.seed === chosenSeed);
  if (!rows.length) throw new RangeError("找不到指定的训练 seed。");
  if (!Number.isInteger(repetitions) || repetitions < 100 || repetitions > 5000 || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new RangeError("统计参数越界。");
  const units = unitsFor(rows, mode), random = randomStream(randomSeed), distribution = [];
  for (let b = 0; b < repetitions; b++) {
    let total = 0, count = 0;
    for (let i = 0; i < units.length; i++) { const unit = units[Math.floor(random() * units.length)]; total += unit.total; count += unit.count; }
    distribution.push(total / count);
  }
  distribution.sort((a, b) => a - b);
  const delta = average(rows.map(row => row.new - row.old)), interval = [linearQuantile(distribution, .025), linearQuantile(distribution, .975)];
  const groups = new Map(); for (const row of rows) { if (!groups.has(row.group)) groups.set(row.group, []); groups.get(row.group).push(row.new - row.old); }
  const wins = rows.filter(row => row.new > row.old + 1e-12).length, losses = rows.filter(row => row.new < row.old - 1e-12).length;
  const perSeed = seeds.map(value => { const selected = validated.rows.filter(row => row.seed === value); return { seed: value, count: selected.length, delta: average(selected.map(row => row.new - row.old)) }; });
  return { rows, seed: chosenSeed, mode, randomSeed, repetitions, count: rows.length, units: units.length, delta, interval, distribution, wins, losses, ties: rows.length - wins - losses,
    perSeed, slices: [...groups].map(([group, differences]) => ({ group, count: differences.length, delta: average(differences) })), threshold,
    exceedsThreshold: units.length > 1 && interval[0] > threshold, degenerate: interval[0] === interval[1], permutation: exactPairedPermutation(rows, mode) };
}

export function checkpointSelection({ candidates = 10, trials = 300, randomSeed = 91 } = {}) {
  if (!Number.isInteger(candidates) || candidates < 1 || candidates > 50 || !Number.isInteger(trials) || trials < 1 || trials > 1000) throw new RangeError("选择实验参数越界。");
  const validationRandom = randomStream(randomSeed), testRandom = randomStream((randomSeed + 1307) >>> 0);
  const score = random => { let wins = 0; for (let i = 0; i < 40; i++) wins += Number(random() < .6); return wins / 40; };
  let selected = 0, independent = 0;
  for (let t = 0; t < trials; t++) {
    let best = -Infinity, heldout = 0;
    for (let c = 0; c < candidates; c++) { const validation = score(validationRandom), test = score(testRandom); if (validation > best) { best = validation; heldout = test; } }
    selected += best; independent += heldout;
  }
  return { candidates, trials, truth: .6, selected: selected / trials, independent: independent / trials };
}

export function initPairedEvaluation() {
  const mode = $("[data-pe-mode]"); if (!mode) return;
  const preset = $("[data-pe-preset]"), seed = $("[data-pe-seed]"), threshold = $("[data-pe-threshold]"), input = $("[data-pe-file]"), candidates = $("[data-pe-candidates]");
  let data = pairedExample(), request = 0;
  const set = (name, value) => { $(`[data-pe-${name}]`).textContent = value; };
  const selectSeeds = () => { seed.replaceChildren(...[...new Set(data.rows.map(row => row.seed))].map(value => { const option = document.createElement("option"); option.value = value; option.textContent = value; return option; })); };
  const render = () => {
    if (!data) return;
    const r = pairedStatistics({ data, seed: seed.value, mode: mode.value, threshold: +threshold.value });
    $("[data-pe-results]").hidden = false;
    set("delta", `${(r.delta * 100).toFixed(2)} pp`); set("interval", r.interval.map(x => `${(100 * x).toFixed(2)}`).join(" … ") + " pp"); set("counts", `${r.count} 题 / ${r.units} 重采样单位`);
    set("permutation", r.permutation.p === null ? r.permutation.reason : `双侧精确置换 p=${formatValue(r.permutation.p)}；${r.permutation.permutations} 种组内共同交换或逐题交换。`);
    set("decision", r.units < 2 ? "只有一个独立单位，重采样不能提供有用的泛化不确定性。" : `${r.exceedsThreshold ? "本次区间下界高于" : "本次区间下界未高于"}预设收益 ${(100 * r.threshold).toFixed(1)} pp；仍需检查各切片、训练 seed 与独立测试。${r.degenerate ? " 区间退化，只反映观察样本没有差异波动。" : ""}`);
    set("wins", `${r.wins} 赢 / ${r.losses} 输 / ${r.ties} 平`);
    $("[data-pe-rows]").innerHTML = r.rows.map(row => `<tr><td>${escapeText(row.id)}</td><td>${escapeText(row.group)}</td><td>${formatValue(row.old)}</td><td>${formatValue(row.new)}</td><td>${((row.new - row.old) * 100).toFixed(2)}</td></tr>`).join("");
    $("[data-pe-seeds]").innerHTML = r.perSeed.map(item => `<tr><td>${escapeText(item.seed)}</td><td>${item.count}</td><td>${(100 * item.delta).toFixed(2)} pp</td></tr>`).join("");
    $("[data-pe-slices]").innerHTML = r.slices.map(item => `<tr><td>${escapeText(item.group)}</td><td>${item.count}</td><td>${(100 * item.delta).toFixed(2)} pp</td></tr>`).join("");
    const low = Math.min(0, r.distribution[0]), high = Math.max(.01, r.distribution.at(-1)), bins = Array(20).fill(0);
    r.distribution.forEach(x => bins[Math.min(19, Math.floor((x - low) / (high - low) * 20))]++);
    const peak = Math.max(...bins), x = value => 40 + (value - low) / (high - low) * 540;
    $("[data-pe-chart]").innerHTML = `<line x1="40" y1="185" x2="580" y2="185" class="decision-axis"/>${bins.map((b, i) => `<rect x="${40 + i * 27}" y="${185 - b / peak * 145}" width="24" height="${b / peak * 145}" class="decision-bar"/>`).join("")}<line x1="${x(0)}" y1="20" x2="${x(0)}" y2="185" class="decision-zero"/><text x="40" y="211">${(low * 100).toFixed(1)} pp</text><text x="520" y="211">${(high * 100).toFixed(1)} pp</text><text x="40" y="18">差值的重采样分布 · 2,000 次</text>`;
  };
  const renderSelection = () => { const r = checkpointSelection({ candidates: +candidates.value }); set("selection", `所有候选真实成功率都为 60%；${r.trials} 次模拟中，选中者验证均值 ${(r.selected * 100).toFixed(2)}%，独立测试均值 ${(r.independent * 100).toFixed(2)}%。每个数据集 40 次 Bernoulli 试验，候选与试验在本模拟中独立。`); };
  preset.addEventListener("change", () => { request++; input.value = ""; data = pairedExample(preset.value); selectSeeds(); set("status", "已载入教学数据。"); render(); });
  [mode, seed, threshold].forEach(el => el.addEventListener("change", render)); candidates.addEventListener("change", renderSelection);
  input.addEventListener("change", async () => {
    const ticket = ++request; data = null; $("[data-pe-results]").hidden = true; seed.replaceChildren(); set("status", "正在读取本地文件…");
    try {
      const file = input.files?.[0]; if (!file) { set("status", "未选择文件，选择示例可恢复。"); return; }
      if (file.size > 1048576) throw new RangeError("文件上限 1 MiB。");
      const parsed = validatePairedData(JSON.parse(await file.text())); if (ticket !== request) return;
      data = parsed; selectSeeds(); set("status", `已读取 ${file.name}，共 ${data.rows.length} 行；文件仅在浏览器处理。`); render();
    } catch (error) { if (ticket === request) set("status", `读取失败：${error.message}`); }
  });
  $("[data-pe-download]").addEventListener("click", () => { const url = URL.createObjectURL(new Blob([JSON.stringify(pairedExample("multi-seed"), null, 2)], { type: "application/json" })); const link = document.createElement("a"); link.href = url; link.download = "paired-evaluation-example.json"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); });
  selectSeeds(); render(); renderSelection();
}
