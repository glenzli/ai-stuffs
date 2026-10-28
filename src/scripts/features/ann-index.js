import { $ } from "../core.js";
import { formatValue, formatVector } from "./objective-math.js";

export function annInputs() {
  return { centroids: [[-1.5, 0], [0, 1.5], [1.5, 0]],
    documents: [[-1.4, -.2], [-1.1, .4], [-.5, .9], [0, .8], [.1, 1.6], [.4, 1], [.8, .2], [1.6, .1], [1.1, -.4], [-.8, -.7], [.2, 1.9], [1.8, -.5]].map((vector, i) => ({ id: `d${String(i + 1).padStart(2, "0")}`, vector, group: i % 2 ? "B" : "A", relevant: [3, 5, 8].includes(i) })),
    queries: { center: [.35, .25], left: [-.8, .1], upper: [.1, 1.2] } };
}
const distance = (a, b) => a.reduce((sum, x, i) => sum + (x - b[i]) ** 2, 0);
const rotate = ([x, y]) => [-y, x];
const ranking = (documents, query) => documents.map(doc => ({ ...doc, distance: distance(doc.vector, query) })).sort((a, b) => a.distance - b.distance || a.id.localeCompare(b.id, "en"));
export function annSearch({ nprobe = 1, k = 3, query = "center", filter = "all", version = "aligned", inputs = annInputs() } = {}) {
  if (!Number.isInteger(nprobe) || nprobe < 1 || nprobe > inputs.centroids.length || !Number.isInteger(k) || k < 1 || k > 12 || !Object.hasOwn(inputs.queries, query) || !["all", "A", "B", "empty"].includes(filter) || !["aligned", "stale", "rebuilt"].includes(version)) throw new RangeError("索引参数无效。");
  const ids = new Set(); for (const doc of inputs.documents) { if (ids.has(doc.id) || !Array.isArray(doc.vector) || doc.vector.length !== 2 || !doc.vector.every(Number.isFinite)) throw new RangeError("文档 ID 或二维向量无效。"); ids.add(doc.id); }
  if (!inputs.centroids.every(v => v.length === 2 && v.every(Number.isFinite)) || !inputs.queries[query].every(Number.isFinite)) throw new RangeError("质心或查询无效。");
  const centroids = inputs.centroids.map(v => version === "rebuilt" ? rotate(v) : [...v]);
  const q = version === "aligned" ? [...inputs.queries[query]] : rotate(inputs.queries[query]);
  const docs = inputs.documents.map(doc => ({ ...doc, vector: version === "rebuilt" ? rotate(doc.vector) : [...doc.vector] }));
  const lists = centroids.map(() => []);
  for (const doc of docs) {
    let best = 0; for (let i = 1; i < centroids.length; i++) if (distance(doc.vector, centroids[i]) < distance(doc.vector, centroids[best])) best = i;
    doc.list = best; lists[best].push(doc);
  }
  const centroidsRanked = centroids.map((vector, list) => ({ list, vector, distance: distance(q, vector) })).sort((a, b) => a.distance - b.distance || a.list - b.list);
  const probed = centroidsRanked.slice(0, nprobe).map(c => c.list), eligible = doc => filter === "all" || doc.group === filter;
  const visited = probed.flatMap(list => lists[list]), candidates = visited.filter(eligible);
  const exact = ranking(docs.filter(eligible), q).slice(0, k), approximate = ranking(candidates, q).slice(0, k);
  const oracle = ranking(inputs.documents.filter(eligible), inputs.queries[query]).slice(0, k);
  const overlap = (actual, expected) => expected.length ? actual.filter(row => expected.some(x => x.id === row.id)).length / expected.length : null;
  // One pedagogical query has fixed relevance labels. Other queries intentionally omit relevance judgments.
  const relevant = query === "center" ? docs.filter(doc => eligible(doc) && doc.relevant) : [];
  const taskRecall = relevant.length ? approximate.filter(doc => doc.relevant).length / relevant.length : null;
  const contract = { query: { encoder: version === "aligned" ? "v1" : "v2-rotation", dimensions: 2, normalization: "none", metric: "squared-L2", chunks: "chunks-1" },
    index: { encoder: version === "rebuilt" ? "v2-rotation" : "v1", dimensions: 2, normalization: "none", metric: "squared-L2", chunks: "chunks-1" } };
  return { inputs, nprobe, k, query, q, filter, version, docs, centroids, centroidsRanked, lists, probed, candidates, exact, approximate, oracle, contract,
    compatible: Object.keys(contract.query).every(key => contract.query[key] === contract.index[key]),
    annRecall: overlap(approximate, exact), oracleRecall: overlap(approximate, oracle), taskRecall, relevant: relevant.map(doc => doc.id),
    visitedCount: visited.length, distanceCount: candidates.length, fullDistanceCount: docs.filter(eligible).length,
    storage: { vectors: docs.length * 2 * 4, ids: docs.length * 8, centroids: centroids.length * 2 * 4 } };
}

export function initAnnIndex() {
  const nprobe = $("[data-ai-nprobe]"); if (!nprobe) return;
  const k = $("[data-ai-k]"), query = $("[data-ai-query]"), filter = $("[data-ai-filter]"), version = $("[data-ai-version]");
  const set = (name, value) => { $(`[data-ai-${name}]`).textContent = value; }, pct = value => value === null ? "未定义" : `${(100 * value).toFixed(1)}%`;
  const render = () => {
    const r = annSearch({ nprobe: +nprobe.value, k: +k.value, query: query.value, filter: filter.value, version: version.value });
    set("recall", pct(r.annRecall)); set("oracle", pct(r.oracleRecall)); set("task", pct(r.taskRecall)); set("visits", `${r.distanceCount}/${r.fullDistanceCount}`);
    set("contract", r.compatible ? "查询与索引版本一致；二维向量、无归一化、平方 L2、chunks-1。" : "版本校验失败：query=v2-rotation，index=v1。下方继续计算仅为故障演示，真实服务应拒绝该组合或切换到匹配的索引。");
    set("work", `先计算 ${r.centroids.length} 个质心距离，再读取 ${r.visitedCount} 个列表条目；过滤后计算 ${r.distanceCount} 个文档距离。Flat 对相同过滤集合计算 ${r.fullDistanceCount} 次。`);
    set("storage", `假设 FP32 二维向量与 int64 ID：向量 ${r.storage.vectors} B，ID ${r.storage.ids} B，质心 ${r.storage.centroids} B；未含列表头、分配器与框架对象。浏览器实际使用 JS 对象。`);
    $("[data-ai-lists]").innerHTML = r.centroidsRanked.map(item => `<tr><th scope="row">${item.list}</th><td>${formatVector(item.vector)}</td><td>${formatValue(item.distance)}</td><td>${r.probed.includes(item.list) ? "搜索" : "跳过"}</td><td>${r.lists[item.list].map(doc => doc.id).join(", ") || "空"}</td></tr>`).join("");
    $("[data-ai-results]").innerHTML = Array.from({ length: Math.max(1, r.k) }, (_, i) => `<tr><td>${i + 1}</td><td>${r.exact[i] ? `${r.exact[i].id} / ${formatValue(r.exact[i].distance)}` : "—"}</td><td>${r.approximate[i] ? `${r.approximate[i].id} / ${formatValue(r.approximate[i].distance)}` : "—"}</td><td>${r.oracle[i]?.id || "—"}</td></tr>`).join("");
    $("[data-ai-documents]").innerHTML = r.docs.map(doc => `<tr><td>${doc.id}</td><td>${formatVector(doc.vector)}</td><td>${doc.group}</td><td>${doc.list}</td><td>${query.value === "center" ? doc.relevant ? "相关" : "不相关" : "未判定"}</td><td>${r.candidates.some(x => x.id === doc.id) ? "进入候选" : "未进入"}</td></tr>`).join("");
    set("note", !r.exact.length ? "过滤后没有文档，结果为空，Recall 分母为零，显示未定义。" : r.compatible ? "搜索全部列表后应恢复同一向量与过滤条件下的 Flat 结果；并列距离按 ID 排序。标注相关文档与几何近邻分别计分。" : "即使搜索全部列表、ANN 相对当前 Flat 的召回率达到 100%，两者也可能共同使用错误的 query/index 接口。查看匹配版本参考列定位这种偏差。");
    const sx = x => 310 + 115 * x, sy = y => 235 - 88 * y;
    $("[data-ai-chart]").innerHTML = `<line x1="40" y1="235" x2="580" y2="235" class="decision-axis"/><line x1="310" y1="20" x2="310" y2="370" class="decision-axis"/>${r.docs.map(doc => `<circle cx="${sx(doc.vector[0])}" cy="${sy(doc.vector[1])}" r="${r.approximate.some(x => x.id === doc.id) ? 8 : 5}" class="ann-point ann-list-${doc.list}${r.candidates.some(x => x.id === doc.id) ? " ann-visited" : ""}"/><text x="${sx(doc.vector[0]) + 9}" y="${sy(doc.vector[1]) - 8}">${doc.id}</text>`).join("")}${r.centroids.map((c, i) => `<rect x="${sx(c[0]) - 6}" y="${sy(c[1]) - 6}" width="12" height="12" class="ann-centroid ann-list-${i}"/><text x="${sx(c[0]) - 23}" y="${sy(c[1]) - 12}">C${i}</text>`).join("")}<path d="M${sx(r.q[0]) - 8},${sy(r.q[1])}h16 M${sx(r.q[0])},${sy(r.q[1]) - 8}v16" class="ann-query"/><text x="${sx(r.q[0]) + 10}" y="${sy(r.q[1]) + 17}">query</text>`;
  };
  [nprobe, k, query, filter, version].forEach(el => el.addEventListener("change", render)); render();
}
