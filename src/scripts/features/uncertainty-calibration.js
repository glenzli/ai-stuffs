import { $ } from "../core.js";

const records = [
  { score: 0.99, correct: true },
  { score: 0.95, correct: true },
  { score: 0.90, correct: false },
  { score: 0.85, correct: true },
  { score: 0.80, correct: true },
  { score: 0.75, correct: false },
  { score: 0.70, correct: true },
  { score: 0.65, correct: false },
  { score: 0.55, correct: true },
  { score: 0.40, correct: false },
];

export function selectiveReadout(threshold = 0.7, stableError = false) {
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new RangeError("阈值须为 0 到 1 的有限数。");
  }
  const rows = records.map((record, index) => ({
    id: index + 1,
    score: record.score,
    correct: stableError === true && index === 0 ? false : record.correct,
    accepted: record.score >= threshold,
  }));
  const accepted = rows.filter(row => row.accepted);
  const answered = accepted.length;
  const errors = accepted.filter(row => !row.correct).length;
  const brier = rows.reduce((sum, row) => sum + (row.score - Number(row.correct)) ** 2, 0) / rows.length;
  const ranges = [
    { label: "[0, 0.5)", lower: 0, upper: 0.5 },
    { label: "[0.5, 0.75)", lower: 0.5, upper: 0.75 },
    { label: "[0.75, 1]", lower: 0.75, upper: 1 },
  ];
  const bins = ranges.map(({ label, lower, upper }) => {
    const members = rows.filter(row => row.score >= lower && (upper === 1 ? row.score <= upper : row.score < upper));
    const n = members.length;
    return {
      label,
      n,
      confidence: n ? members.reduce((sum, row) => sum + row.score, 0) / n : null,
      accuracy: n ? members.filter(row => row.correct).length / n : null,
    };
  });
  const ece = bins.reduce((sum, bin) => sum + (bin.n ? bin.n / rows.length * Math.abs(bin.confidence - bin.accuracy) : 0), 0);
  return { rows, coverage: answered / rows.length, risk: answered ? errors / answered : null, answered, errors, brier, bins, ece };
}

const semanticSamples = {
  paraphrase: [
    { text: "北京是中国的首都。", cluster: "A", count: 3, correct: true },
    { text: "中国的首都是北京。", cluster: "A", count: 2, correct: true },
    { text: "中国首都：北京。", cluster: "A", count: 1, correct: true },
  ],
  conflict: [
    { text: "中国的首都是北京。", cluster: "A", count: 3, correct: true },
    { text: "中国的首都是上海。", cluster: "B", count: 2, correct: false },
    { text: "中国没有首都。", cluster: "C", count: 1, correct: false },
  ],
  stablewrong: [
    { text: "中国的首都是上海。", cluster: "B", count: 6, correct: false },
  ],
};

export function semanticReadout(mode) {
  if (typeof mode !== "string" || !Object.hasOwn(semanticSamples, mode)) {
    throw new RangeError("语义模式无效。");
  }
  const samples = semanticSamples[mode].map(sample => ({ ...sample }));
  const total = samples.reduce((sum, sample) => sum + sample.count, 0);
  const counts = new Map();
  for (const { cluster, count } of samples) counts.set(cluster, (counts.get(cluster) || 0) + count);
  const clusters = Array.from(counts, ([id, count]) => ({ id, count, probability: count / total }));
  const entropy = clusters.reduce((sum, cluster) => sum - cluster.probability * Math.log(cluster.probability), 0);
  const accuracy = samples.reduce((sum, sample) => sum + (sample.correct ? sample.count : 0), 0) / total;
  return { samples, clusters, entropy, accuracy };
}

export function initUncertaintyCalibration() {
  const thresholdInput = $("[data-uc-threshold]");
  if (!thresholdInput) return;
  const stableInput = $("[data-uc-stable]");
  const semanticInput = $("[data-uc-semantic]");
  const percent = value => value === null ? "无样本" : `${(value * 100).toFixed(2)}%`;
  const setText = (selector, value) => {
    const output = $(selector);
    if (output) output.textContent = value;
  };
  const render = () => {
    const result = selectiveReadout(Number(thresholdInput.value) / 100, stableInput?.checked ?? false);
    setText("[data-uc-threshold-value]", `${thresholdInput.value}%`);
    setText("[data-uc-coverage]", percent(result.coverage));
    setText("[data-uc-risk]", result.risk === null ? "无作答" : percent(result.risk));
    setText("[data-uc-answered]", String(result.answered));
    setText("[data-uc-brier]", result.brier.toFixed(4));
    setText("[data-uc-ece]", result.ece.toFixed(4));
    const rows = $("[data-uc-rows]");
    if (rows) rows.innerHTML = result.rows.map(row =>
      `<tr><th scope="row">${row.id}</th><td>${row.score.toFixed(2)}</td><td>${row.correct ? "正确" : "错误"}</td><td>${row.accepted ? "作答" : "弃答"}</td></tr>`
    ).join("");
    const bins = $("[data-uc-bins]");
    if (bins) bins.innerHTML = result.bins.map(bin =>
      `<tr><th scope="row">${bin.label}</th><td>${bin.n}</td><td>${percent(bin.confidence)}</td><td>${percent(bin.accuracy)}</td></tr>`
    ).join("");
    const semantic = semanticReadout(semanticInput?.value || "paraphrase");
    setText("[data-uc-entropy]", semantic.entropy.toFixed(4));
    setText("[data-uc-semantic-accuracy]", percent(semantic.accuracy));
    const meanings = $("[data-uc-meanings]");
    if (meanings) meanings.innerHTML = semantic.samples.map(sample =>
      `<tr><th scope="row">${sample.text}</th><td>簇 ${sample.cluster}</td><td>${sample.count}</td></tr>`
    ).join("");
    setText("[data-uc-note]", "固定教学样本；语义簇与真值均为人工标注，概率来自频数。零熵不保证正确，高分也可能错答。");
  };
  thresholdInput.addEventListener("input", render);
  stableInput?.addEventListener("change", render);
  semanticInput?.addEventListener("change", render);
  render();
}
