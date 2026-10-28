import { $ } from "../core.js";

function checkedInteger(value, name, minimum = 0) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`${name}必须是${minimum ? "正" : "非负"}安全整数。`);
  }
  return value;
}

export function recipeLedger({
  records = 10000, prompt = 600, answer = 400, limit = 2048,
  epochs = 3, microbatch = 2, dp = 4, accum = 8,
  packing = false, supervise = "answer",
} = {}) {
  for (const [value, name, minimum] of [
    [records, "记录数", 0], [prompt, "提示长度", 0], [answer, "回答长度", 0],
    [limit, "序列上限", 1], [epochs, "轮数", 1], [microbatch, "每卡 microbatch", 1],
    [dp, "数据并行数", 1], [accum, "梯度累积步数", 1],
  ]) checkedInteger(value, name, minimum);
  if (!checkedInteger(prompt + answer, "原始记录长度")) {
    throw new RangeError("提示与回答的总长度必须大于 0。");
  }
  if (typeof packing !== "boolean") throw new RangeError("packing 必须是布尔值。");
  if (supervise !== "answer" && supervise !== "all") {
    throw new RangeError("监督范围必须是 answer 或 all。");
  }

  const usedPrompt = Math.min(prompt, limit);
  const usedAnswer = Math.min(answer, Math.max(0, limit - prompt));
  const inputPerRecord = usedPrompt + usedAnswer;
  // Lengths already include template/BOS/EOS; independent masks apply the causal shift per record.
  const supervisedPerRecord = supervise === "all"
    ? Math.max(0, inputPerRecord - 1)
    : usedPrompt > 0 ? usedAnswer : Math.max(0, usedAnswer - 1);
  const recordsPerSequence = packing ? Math.max(1, Math.floor(limit / inputPerRecord)) : 1;
  const sequencesPerEpoch = Math.ceil(records / recordsPerSequence);
  const sequencesPerMicrostep = checkedInteger(microbatch * dp, "microstep 序列容量");
  const microstepsPerEpoch = Math.ceil(sequencesPerEpoch / sequencesPerMicrostep);
  const updatesPerEpoch = Math.ceil(microstepsPerEpoch / accum);
  const globalSequences = checkedInteger(sequencesPerMicrostep * accum, "完整全局序列 batch");
  const slotsPerEpoch = checkedInteger(microstepsPerEpoch * sequencesPerMicrostep * limit, "每轮 token 槽位");
  const totalSlots = checkedInteger(slotsPerEpoch * epochs, "总 token 槽位");
  const readOccurrences = checkedInteger(records * epochs, "读取次数");
  const processedTokens = checkedInteger(readOccurrences * inputPerRecord, "处理 token 总数");
  const supervisedTokens = checkedInteger(readOccurrences * supervisedPerRecord, "监督标签总数");
  const updates = checkedInteger(updatesPerEpoch * epochs, "计划更新窗口总数");
  // A tail is an incomplete final window, including a short final microstep.
  const tailSequences = sequencesPerEpoch % globalSequences;
  const tailMicrosteps = Math.ceil(tailSequences / sequencesPerMicrostep);

  return {
    records, prompt, answer, limit, epochs, microbatch, dp, accum, packing, supervise,
    usedPrompt, usedAnswer, inputPerRecord, supervisedPerRecord, recordsPerSequence,
    readOccurrences, sequencesPerEpoch, sequencesPerMicrostep, microstepsPerEpoch,
    updatesPerEpoch, globalSequences, slotsPerEpoch, totalSlots, processedTokens,
    supervisedTokens, utilization: totalSlots ? processedTokens / totalSlots : 0,
    updates, tailSequences, tailMicrosteps,
  };
}

export function loraLedger(rank = 16, target = "qv") {
  checkedInteger(rank, "LoRA rank", 1);
  if (!["qv", "attention", "all"].includes(target)) {
    throw new RangeError("LoRA 目标必须是 qv、attention 或 all。");
  }
  const layers = 32, d = 4096, kv = 1024, f = 11008, base = 7e9;
  const projections = [["q", d, d], ["v", d, kv]];
  if (target !== "qv") projections.push(["k", d, kv], ["o", d, d]);
  if (target === "all") projections.push(["gate", d, f], ["up", d, f], ["down", f, d]);
  const rows = projections.map(([name, input, output]) => {
    const perLayer = checkedInteger(rank * (input + output), `${name} 每层参数量`);
    const total = checkedInteger(perLayer * layers, `${name} 总参数量`);
    return { name, input, output, perLayer, total };
  });
  const params = checkedInteger(rows.reduce((sum, row) => sum + row.total, 0), "LoRA 总参数量");
  return { rank, target, layers, base, rows, params, ratio: params / base };
}

const formatNumber = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 }).format;
const formatPercent = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const selector = name => `[data-ft-${name}]`;

function writeText(name, value) {
  const node = $(selector(name));
  if (node) node.textContent = value;
}

function writeRows(name, rows) {
  const node = $(selector(name));
  if (node) node.innerHTML = rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join("")}</tr>`).join("");
}

function writeBar(name, fraction) {
  const node = $(selector(`${name}-bar`));
  if (node) node.style.width = `${fraction * 100}%`;
}

function recipeNote(result) {
  const notes = [];
  if (!result.records) notes.push("记录数为 0，计划更新窗口为 0。");
  if (result.usedPrompt < result.prompt) {
    notes.push(`提示被截断，保留 ${formatNumber(result.usedPrompt)}/${formatNumber(result.prompt)} 个 token。`);
  }
  if (result.usedAnswer < result.answer) {
    notes.push(`回答被截断，保留 ${formatNumber(result.usedAnswer)}/${formatNumber(result.answer)} 个 token。`);
  }
  if (!result.supervisedTokens) {
    notes.push("有效监督标签为 0，没有有效训练目标；优化器必须跳过更新。");
  }
  if (result.packing) {
    notes.push(`每个序列最多放 ${formatNumber(result.recordsPerSequence)} 条完整记录，各条独立 mask。`);
  }
  if (result.tailSequences) {
    notes.push(`每轮尾窗口有 ${formatNumber(result.tailSequences)} 个序列、${formatNumber(result.tailMicrosteps)} 个 microstep，轮末结清。`);
  } else if (result.records) notes.push("每轮计划窗口均满，无尾批。");
  return notes.join("");
}

function maskText(result) {
  if (!result.supervisedPerRecord) return "每条记录的有效监督标签为 0，没有有效训练目标。";
  const count = formatNumber(result.supervisedPerRecord);
  if (result.supervise === "all") {
    return `每条记录首个 token 和 padding 不参与 loss，其余位置共 ${count} 个有效标签。`;
  }
  return result.usedPrompt > 0
    ? `提示和 padding 不参与 loss；每条回答有 ${count} 个有效标签。`
    : `首个 token 和 padding 不参与 loss；每条回答有 ${count} 个有效标签。`;
}

export function initFinetuningRecipe() {
  if (typeof document === "undefined") return;
  const recordsInput = $(selector("records"));
  if (!recordsInput) return;
  const inputs = {
    records: recordsInput, prompt: $(selector("prompt")), answer: $(selector("answer")),
    limit: $(selector("limit")), epochs: $(selector("epochs")), microbatch: $(selector("micro")),
    dp: $(selector("dp")), accum: $(selector("accum")),
  };
  const packingInput = $(selector("packing")), superviseInput = $(selector("supervise"));
  const rankInput = $(selector("rank")), targetInput = $(selector("target"));
  const readNumber = input => input ? (input.value.trim() ? Number(input.value) : NaN) : undefined;
  const recipeOutputs = ["processed", "supervised", "updates", "sequences", "perseq", "fill"];
  const renderRecipe = () => {
    try {
      const options = Object.fromEntries(Object.entries(inputs).map(([name, input]) => [name, readNumber(input)]));
      const result = recipeLedger({ ...options, packing: packingInput?.checked, supervise: superviseInput?.value });
      for (const [name, value] of [
        ["processed", result.processedTokens], ["supervised", result.supervisedTokens],
        ["updates", result.updates], ["sequences", result.sequencesPerEpoch], ["perseq", result.recordsPerSequence],
      ]) writeText(name, formatNumber(value));
      writeText("fill", formatPercent(result.utilization));
      writeRows("ledger", [
        ["原始记录", formatNumber(result.records)],
        ["读取次数", formatNumber(result.readOccurrences)],
        ["序列 / 轮", formatNumber(result.sequencesPerEpoch)],
        ["microstep / 轮", formatNumber(result.microstepsPerEpoch)],
        ["计划更新窗口 / 轮", formatNumber(result.updatesPerEpoch)],
        ["完整全局序列 batch", formatNumber(result.globalSequences)],
        ["尾窗口序列 / 轮", formatNumber(result.tailSequences)],
      ]);
      writeBar("prompt", result.usedPrompt / result.limit);
      writeBar("answer", result.usedAnswer / result.limit);
      writeBar("pad", (result.limit - result.inputPerRecord) / result.limit);
      writeText("mask", maskText(result));
      writeText("note", recipeNote(result));
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      recipeOutputs.forEach(name => writeText(name, "-"));
      writeRows("ledger", []);
      ["prompt", "answer", "pad"].forEach(name => writeBar(name, 0));
      writeText("mask", "输入无效，无法计算监督标签。");
      writeText("note", error.message);
    }
  };
  const renderLora = () => {
    try {
      const result = loraLedger(readNumber(rankInput), targetInput?.value);
      writeText("rank-value", formatNumber(result.rank));
      writeText("params", formatNumber(result.params));
      writeText("ratio", formatPercent(result.ratio, 3));
      writeRows("lora-rows", result.rows.map(row => [
        row.name, `${formatNumber(row.input)} → ${formatNumber(row.output)}`, formatNumber(row.perLayer), formatNumber(row.total),
      ]));
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      ["rank-value", "params", "ratio"].forEach(name => writeText(name, "-"));
      writeRows("lora-rows", []);
    }
  };
  const bind = (controls, render) => controls.filter(Boolean).forEach(input => {
    input.addEventListener("input", render);
    input.addEventListener("change", render);
  });
  bind([...Object.values(inputs), packingInput, superviseInput], renderRecipe);
  bind([rankInput, targetInput], renderLora);
  renderRecipe();
  renderLora();
}
