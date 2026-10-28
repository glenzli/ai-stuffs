import { $ } from "../core.js";

const records = {
  goal: { id: "goal", label: "目标", text: "准备一天的活动方案。" },
  evidence: {
    id: "evidence",
    label: "证据",
    text: '“周六场次在 10:00 开始。”（来源：example:schedule-1）',
    quote: "周六场次在 10:00 开始。",
    sourceId: "example:schedule-1",
  },
  constraint: { id: "constraint", label: "约束", text: "总预算不超过 500；只提供方案，不进行预订。" },
  stale: { id: "stale", label: "旧日期", text: "原定周五，已被后来的用户要求替代。", date: "Fri" },
  update: { id: "update", label: "新要求", text: "用户将日期改为周六。", date: "Sat" },
};

export const memoryStrategies = Object.freeze({
  full: Object.freeze({
    label: "完整记录",
    items: Object.freeze(["goal", "evidence", "constraint", "stale", "update"]),
    note: "保留来源与预算、预订约束，也留下了旧日期。后来的周六要求覆盖原定周五。",
  }),
  compact: Object.freeze({
    label: "简短摘要",
    items: Object.freeze(["goal", "update"]),
    note: "只保留目标与周六日期，预算、禁止预订的约束和证据来源已丢失。",
  }),
  structured: Object.freeze({
    label: "结构化记录",
    items: Object.freeze(["goal", "evidence", "constraint", "update"]),
    note: "保留来源、预算、预订约束与周六日期，移除已被替代的旧日期。",
  }),
});

export function memoryReadout(mode, verify = false) {
  if (typeof mode !== "string" || !Object.hasOwn(memoryStrategies, mode)) {
    throw new RangeError("mode must be full, compact, or structured");
  }
  if (typeof verify !== "boolean") throw new TypeError("verify must be a boolean");
  const strategy = memoryStrategies[mode];
  const items = strategy.items.map(id => ({ ...records[id] }));
  const evidenceAvailable = items.some(item => item.id === "evidence");
  const constraint = items.find(item => item.id === "constraint");
  const latestDate = items.findLast(item => item.date);
  const verificationNote = verify
    ? evidenceAvailable
      ? "本次模拟回读可以定位保留的来源。真实任务仍需请求来源并检查当前结果。"
      : "摘要中没有来源，无法据此回读；需要重新检索。"
    : "来源尚未回读，历史证据的时效仍待确认。";
  return {
    items,
    constraints: constraint ? constraint.text : "未保留",
    date: latestDate.date,
    evidenceAvailable,
    note: `${strategy.note} ${verificationNote}`,
  };
}

export function initAgentMemory() {
  const modeInput = $("[data-memory-mode]");
  if (!modeInput) return;
  const verifyInput = $("[data-memory-verify]");
  const setText = (selector, value) => {
    const output = $(selector);
    if (output) output.textContent = value;
  };
  const render = () => {
    const verify = Boolean(verifyInput?.checked);
    const readout = memoryReadout(modeInput.value, verify);
    const list = $("[data-memory-items]");
    if (list) list.innerHTML = readout.items.map(item => `<li><strong>${item.label}:</strong> ${item.text}</li>`).join("");
    setText("[data-memory-constraints]", readout.items.some(item => item.id === "constraint") ? "已保留" : "缺失");
    setText("[data-memory-date]", readout.date === "Sat" ? "周六" : "周五");
    setText("[data-memory-grounding]", verify && readout.evidenceAvailable ? "模拟已回读" : "未回读");
    setText("[data-memory-note]", readout.note);
  };
  modeInput.addEventListener("change", render);
  verifyInput?.addEventListener("change", render);
  render();
}
