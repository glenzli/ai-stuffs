import { $ } from "../core.js";

function requireInteger(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer in ${min}..${max}.`);
  }
}

function requireBoolean(value, name) {
  if (typeof value !== "boolean") throw new TypeError(`${name} must be boolean.`);
}

function teachingOrder(count, epoch, shuffle) {
  const order = Array.from({ length: count }, (_, index) => index);
  if (!shuffle) return order;
  // Pinned LCG32: s0 = (0x6d2b79f5 + epoch) mod 2^32,
  // s = (1664525 * s + 1013904223) mod 2^32; u = s / 2^32.
  let state = (0x6d2b79f5 + (epoch >>> 0)) >>> 0;
  for (let index = order.length - 1; index > 0; index--) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    const other = Math.floor((state / 0x100000000) * (index + 1));
    [order[index], order[other]] = [order[other], order[index]];
  }
  return order;
}

export function distributedSamples({
  count = 10, ranks = 3, batch = 2, samplerDrop = false,
  loaderDrop = false, epoch = 0, shuffle = false,
} = {}) {
  requireInteger(count, 0, 30, "count");
  requireInteger(ranks, 1, 8, "ranks");
  requireInteger(batch, 1, 8, "batch");
  requireInteger(epoch, 0, Number.MAX_SAFE_INTEGER, "epoch");
  requireBoolean(samplerDrop, "samplerDrop");
  requireBoolean(loaderDrop, "loaderDrop");
  requireBoolean(shuffle, "shuffle");

  const order = teachingOrder(count, epoch, shuffle);
  const localSize = samplerDrop ? Math.floor(count / ranks) : Math.ceil(count / ranks);
  const total = localSize * ranks;
  const expanded = samplerDrop
    ? order.slice(0, total)
    : Array.from({ length: total }, (_, index) => order[index % count]);
  const perRank = Array.from({ length: ranks }, (_, rank) => {
    const assigned = [];
    for (let index = rank; index < total; index += ranks) assigned.push(expanded[index]);
    const kept = loaderDrop ? Math.floor(assigned.length / batch) * batch : assigned.length;
    const batches = [];
    for (let start = 0; start < kept; start += batch) {
      batches.push(assigned.slice(start, Math.min(start + batch, kept)));
    }
    return { rank, assigned, batches, dropped: assigned.slice(kept) };
  });
  // Rank-major occurrences retain padding repeats; metrics describe delivered batches.
  const delivered = perRank.flatMap(({ batches }) => batches.flat());
  const occurrences = Array(count).fill(0);
  for (const index of delivered) occurrences[index]++;
  const duplicates = occurrences.map((count, index) => ({ index, count }))
    .filter(({ count }) => count > 1);
  const missing = occurrences.flatMap((count, index) => count === 0 ? [index] : []);

  return {
    order, expanded, perRank, delivered, duplicates, missing,
    readCount: delivered.length,
    uniqueCount: occurrences.filter(count => count > 0).length,
    steps: perRank[0].batches.length,
    samplerRemoved: samplerDrop ? order.slice(total) : [],
    paddingCount: samplerDrop ? 0 : total - count,
  };
}

export function iterableWorkers(count = 10, ranks = 2, workers = 2, shard = true) {
  requireInteger(count, 0, 30, "count");
  requireInteger(ranks, 1, 8, "ranks");
  requireInteger(workers, 1, 8, "workers");
  requireBoolean(shard, "shard");
  const indices = Array.from({ length: count }, (_, index) => index);
  const totalWorkers = ranks * workers;
  return Array.from({ length: totalWorkers }, (_, globalWorker) => ({
    rank: Math.floor(globalWorker / workers),
    worker: globalWorker % workers,
    globalWorker,
    ids: shard ? indices.filter(index => index % totalWorkers === globalWorker) : indices.slice(),
  }));
}

function readInteger(input, min, max, fallback) {
  if (!input) return fallback;
  const raw = String(input.value ?? "").trim();
  const value = raw === "" ? NaN : Number(raw);
  return Number.isSafeInteger(value) && value >= min && value <= max ? value : null;
}

export function initDataOrder() {
  if (typeof document === "undefined") return;
  const countInput = $("[data-do-count]");
  if (!countInput) return;
  const ranksInput = $("[data-do-ranks]");
  const batchInput = $("[data-do-batch]");
  const samplerInput = $("[data-do-sampler-drop]");
  const loaderInput = $("[data-do-loader-drop]");
  const shuffleInput = $("[data-do-shuffle]");
  const epochInput = $("[data-do-epoch]");
  const shardInput = $("[data-do-shard]");
  const setText = (key, value) => {
    const node = $(`[data-do-${key}]`);
    if (node) node.textContent = String(value);
  };
  const setRows = (key, html) => {
    const node = $(`[data-do-${key}]`);
    if (node) node.innerHTML = html;
  };

  const render = () => {
    const count = readInteger(countInput, 0, 30, 10);
    const ranks = readInteger(ranksInput, 1, 4, 3);
    const batch = readInteger(batchInput, 1, 4, 2);
    const epoch = readInteger(epochInput, 0, 2, 0);
    if ([count, ranks, batch, epoch].includes(null)) {
      for (const key of ["order", "expanded", "reads", "unique", "missing"]) setText(key, "-");
      setRows("rows", "");
      setRows("worker-rows", "");
      setText("note", "样本数需为 0 到 30 的整数；进程数和批大小为 1 到 4；epoch 为 0 到 2。");
      return;
    }

    const shuffle = shuffleInput?.checked ?? false;
    const result = distributedSamples({
      count, ranks, batch, epoch, shuffle,
      samplerDrop: samplerInput?.checked ?? false,
      loaderDrop: loaderInput?.checked ?? false,
    });
    setText("order", JSON.stringify(result.order));
    setText("expanded", JSON.stringify(result.expanded));
    setText("reads", result.readCount);
    setText("unique", result.uniqueCount);
    setText("missing", JSON.stringify(result.missing));
    setRows("rows", result.perRank.map(({ rank, assigned, batches, dropped }) =>
      `<tr><td>${rank}</td><td>${JSON.stringify(assigned)}</td><td>${JSON.stringify(batches)}</td><td>${JSON.stringify(dropped)}</td></tr>`
    ).join(""));
    const repeats = result.duplicates.map(({ index, count }) => `${index}:${count}`).join(", ") || "无";
    const loaderRemoved = result.perRank.reduce((sum, row) => sum + row.dropped.length, 0);
    setText("note", `${shuffle ? "固定种子 + epoch 的教学置换，不复现 PyTorch RNG。" : "按索引顺序分配。"}采样器补齐 ${result.paddingCount} 次，移除 ${result.samplerRemoved.length} 项；加载器丢弃 ${loaderRemoved} 次。每进程 ${result.steps} 批；重复读入次数：${repeats}。`);

    setRows("worker-rows", iterableWorkers(count, 2, 2, shardInput?.checked ?? true)
      .map(({ rank, worker, globalWorker, ids }) =>
        `<tr><td>${rank}</td><td>${worker}</td><td>${globalWorker}</td><td>${JSON.stringify(ids)}</td></tr>`
      ).join(""));
  };

  for (const input of [countInput, ranksInput, batchInput, samplerInput, loaderInput, shuffleInput, epochInput, shardInput]) {
    for (const event of ["input", "change"]) input?.addEventListener(event, render);
  }
  render();
}
