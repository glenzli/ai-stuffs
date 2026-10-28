import { $ } from "../core.js";

const tokens = ["{\"ok\":", "true", "false", "}", "<eos>", "junk"];
const logits = [2, 1.4, 0.8, 0.4, 0, -0.5];
const validStrings = ["{\"ok\":true}", "{\"ok\":false}"];

function completePrefix(prefix) {
  if (!validStrings.includes(prefix)) return false;
  try {
    const parsed = JSON.parse(prefix);
    return Object.keys(parsed).length === 1 && typeof parsed.ok === "boolean";
  } catch {
    return false;
  }
}

export function decodingDistribution({ temperature = 1, topk = 6, topp = 1, grammar = false, prefix = "" } = {}) {
  if (!Number.isFinite(temperature) || temperature < 0.1 || temperature > 3) {
    throw new RangeError("温度须为 0.1 到 3 的有限数。");
  }
  if (!Number.isInteger(topk) || topk < 1 || topk > tokens.length) {
    throw new RangeError("top-k 须为 1 到 6 的整数。");
  }
  if (!Number.isFinite(topp) || topp <= 0 || topp > 1) {
    throw new RangeError("top-p 须为大于 0 且不超过 1 的有限数。");
  }
  if (typeof grammar !== "boolean") {
    throw new RangeError("语法开关须为布尔值。");
  }
  if (grammar && (typeof prefix !== "string" || !validStrings.some(value => value.startsWith(prefix)))) {
    throw new RangeError("前缀须为两个固定合法字符串之一的字符前缀。");
  }

  const complete = grammar && completePrefix(prefix);
  const rows = tokens.map((token, index) => ({
    token,
    logit: logits[index],
    allowed: !grammar || (token === "<eos>" ? complete : validStrings.some(value => value.startsWith(prefix + token))),
    probBeforeTopk: 0,
    keptK: false,
    keptP: false,
    probability: 0,
  }));
  const allowed = rows.map((row, index) => index).filter(index => rows[index].allowed);
  // A known character prefix can stop inside a token with no available continuation.
  if (!allowed.length) return { rows, selected: null, candidates: [], mass: 0 };

  const maxLogit = Math.max(...allowed.map(index => rows[index].logit));
  const weights = allowed.map(index => Math.exp((rows[index].logit - maxLogit) / temperature));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  allowed.forEach((index, position) => { rows[index].probBeforeTopk = weights[position] / total; });

  const ranked = allowed.sort((a, b) => rows[b].probBeforeTopk - rows[a].probBeforeTopk || a - b);
  const keptK = ranked.slice(0, topk);
  const massK = keptK.reduce((sum, index) => sum + rows[index].probBeforeTopk, 0);
  keptK.forEach(index => { rows[index].keptK = true; });

  let cumulative = 0;
  for (const index of keptK) {
    rows[index].keptP = true;
    cumulative += rows[index].probBeforeTopk / massK;
    if (topp < 1 && cumulative >= topp) break;
  }
  const massP = rows.reduce((sum, row) => sum + (row.keptP ? row.probBeforeTopk / massK : 0), 0);
  for (const row of rows) {
    if (row.keptP) row.probability = (row.probBeforeTopk / massK) / massP;
  }

  const candidates = rows.filter(row => row.probability > 0);
  const selected = candidates.reduce((best, row) => row.probability > best.probability ? row : best, candidates[0]);
  return {
    rows,
    selected: selected.token,
    candidates: candidates.map(row => row.token),
    mass: rows.reduce((sum, row) => sum + row.probability, 0),
  };
}

export function sampleFrom(distribution, u) {
  if (!Number.isFinite(u) || u < 0 || u >= 1) {
    throw new RangeError("采样位置须为 0 到 1 之间的有限数，不包含 1。");
  }
  const rows = Array.isArray(distribution) ? distribution : distribution?.rows;
  if (!Array.isArray(rows) || !rows.length) {
    throw new RangeError("采样分布须含候选行。");
  }
  for (const row of rows) {
    if (!row || typeof row.token !== "string" || !Number.isFinite(row.probability) || row.probability < 0 || row.probability > 1) {
      throw new RangeError("采样行须含字符串 token 与 0 到 1 的有限概率。");
    }
  }
  const positive = rows.filter(row => row.probability > 0);
  const mass = positive.reduce((sum, row) => sum + row.probability, 0);
  if (!positive.length || Math.abs(mass - 1) > 1e-12) {
    throw new RangeError("采样分布须有正概率候选且概率总和为 1。");
  }
  let cumulative = 0;
  for (const row of positive) {
    cumulative += row.probability;
    if (u < cumulative) return row.token;
  }
  // Cover only the rounding gap at the end of a normalized CDF.
  return positive[positive.length - 1].token;
}

const beamTree = {
  "": [["A", 0.6], ["B", 0.4]],
  A: [["x", 0.51], ["y", 0.49]],
  B: [["x", 0.99], ["y", 0.01]],
};

export function beamExample(width = 2) {
  if (width !== 1 && width !== 2) throw new RangeError("束宽须为 1 或 2。");
  const nodes = [];
  let beam = [{ path: "", probability: 1, logProbability: 0 }];
  for (let step = 1; step <= 2; step++) {
    const expanded = beam.flatMap(parent => beamTree[parent.path].map(([token, probability]) => ({
      step,
      path: parent.path + token,
      probability: parent.probability * probability,
      logProbability: parent.logProbability + Math.log(probability),
      kept: false,
    })));
    const ranked = expanded.map((node, index) => index).sort((a, b) => expanded[b].logProbability - expanded[a].logProbability || a - b);
    beam = ranked.slice(0, width).map(index => {
      expanded[index].kept = true;
      return expanded[index];
    });
    nodes.push(...expanded);
  }
  const leaves = beamTree[""].flatMap(([first, firstProbability]) => beamTree[first].map(([last, lastProbability]) => ({
    path: first + last,
    probability: firstProbability * lastProbability,
    logProbability: Math.log(firstProbability) + Math.log(lastProbability),
  })));
  const optimum = leaves.reduce((best, leaf) => leaf.logProbability > best.logProbability ? leaf : best, leaves[0]);
  const winner = beam[0];
  return { nodes, selected: winner.path, probability: winner.probability, logProbability: winner.logProbability, optimum };
}

export function initDecodingAlgorithms() {
  const temperatureInput = $("[data-dec-temperature]");
  const beamInput = $("[data-dec-beam]");
  if (!temperatureInput && !beamInput) return;
  const setText = (selector, value) => {
    const output = $(selector);
    if (output) output.textContent = value;
  };
  const percent = value => `${(value * 100).toFixed(2)}%`;

  if (temperatureInput) {
    const kInput = $("[data-dec-k]");
    const pInput = $("[data-dec-p]");
    const grammarInput = $("[data-dec-grammar]");
    const prefixInput = $("[data-dec-prefix]");
    const uInput = $("[data-dec-u]");
    const render = () => {
      const temperature = Number(temperatureInput.value) / 100;
      const topp = Number(pInput?.value ?? 100) / 100;
      const u = Number(uInput?.value ?? 500) / 1000;
      const grammar = grammarInput?.checked ?? false;
      const result = decodingDistribution({ temperature, topk: Number(kInput?.value ?? 6), topp, grammar, prefix: prefixInput?.value ?? "" });
      setText("[data-dec-temp-value]", temperature.toFixed(2));
      setText("[data-dec-p-value]", topp.toFixed(2));
      setText("[data-dec-u-value]", u.toFixed(3));
      setText("[data-dec-sampled]", result.candidates.length ? sampleFrom(result, u) : "无候选");
      setText("[data-dec-greedy]", result.selected ?? "无候选");
      setText("[data-dec-mass]", percent(result.mass));
      const rows = $("[data-dec-rows]");
      if (rows) rows.innerHTML = result.rows.map(row => {
        const token = row.token.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
        return `<tr><th scope="row"><code>${token}</code></th><td>${row.logit.toFixed(1)}</td><td>${row.allowed ? "允许" : "屏蔽"}</td><td>${percent(row.probBeforeTopk)}</td><td>top-k：${row.keptK ? "保留" : "剔除"}；top-p：${row.keptP ? "保留" : "剔除"}</td><td>${percent(row.probability)}</td></tr>`;
      }).join("");
      const grammarNote = grammar
        ? '语法约束只检查 {"ok":true} 与 {"ok":false} 的字符前缀；完整字符串还须通过 JSON.parse，才允许 <eos>。'
        : "语法约束未启用，前缀不参与分布计算。";
      setText("[data-dec-note]", `固定 logits；顺序为语法屏蔽、温度 softmax、top-k、重归一化、top-p、重归一化，最后取贪心结果或按原 token 顺序做逆累积分布采样。top-p 保留累计概率达到阈值的最小前缀；排序同分时保留原索引较小者。${grammarNote}${result.candidates.length ? "" : "当前前缀没有可接续的 token。"}`);
    };
    temperatureInput.addEventListener("input", render);
    pInput?.addEventListener("input", render);
    uInput?.addEventListener("input", render);
    kInput?.addEventListener("change", render);
    grammarInput?.addEventListener("change", render);
    prefixInput?.addEventListener("change", render);
    render();
  }

  if (beamInput) {
    const renderBeam = () => {
      const width = Number(beamInput.value);
      const result = beamExample(width);
      const greedy = beamExample(1);
      setText("[data-dec-beam-result]", `束宽 ${width}：${result.selected}，联合概率 ${result.probability.toFixed(3)}。贪心 ${greedy.selected} 为 ${greedy.probability.toFixed(3)}；本树最优 ${result.optimum.path} 为 ${result.optimum.probability.toFixed(3)}。完整路径均为两步，按累计对数概率比较，无 EOS 或长度归一化。`);
      const rows = $("[data-dec-beam-rows]");
      if (rows) rows.innerHTML = result.nodes.map(node =>
        `<tr><td>${node.step}</td><th scope="row"><code>${node.path}</code></th><td>${node.probability.toFixed(3)}</td><td>${node.kept ? "保留" : "剔除"}</td></tr>`
      ).join("");
    };
    beamInput.addEventListener("change", renderBeam);
    renderBeam();
  }
}
