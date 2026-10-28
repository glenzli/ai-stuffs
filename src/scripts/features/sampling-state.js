import { sampleFrom } from "./decoding-algorithms.js";
import { esc, table } from "../experiment-ui.js";

// Deliberately specified uniforms, not outputs attributed to a framework seed.
const ids = ["A", "B", "C"], tokens = ["x", "y", "z"];
const probabilities = { A: [.2, .3, .5], B: [.5, .25, .25], C: [.1, .2, .7] };
const globalTape = [.12, .87, .34, .66, .49, .91, .23, .72, .05];
const requestTapes = { A: [.12, .66, .23], B: [.87, .49, .72], C: [.34, .91, .05] };
export const samplingOrders = Object.freeze(["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]);
const emptyCursors = () => ({ global: 0, requests: { A: 0, B: 0, C: 0 }, slots: [0, 0, 0] });

export function samplingSchedule(order, bSteps) {
  if (!samplingOrders.includes(order) || ![1, 3].includes(bSteps)) throw new RangeError("选择固定的请求排列与 B 的输出预算。");
  const events = [];
  for (let step = 1; step <= 3; step++) {
    const active = [...order].filter(id => id !== "B" || step <= bSteps);
    for (const [slot, request] of active.entries()) events.push({ event: events.length, step, request, slot });
  }
  return events;
}

function consume(events, allocation, cursors) {
  return events.map(event => {
    let stream, index, tape;
    if (allocation === "global") { stream = "global"; index = cursors.global++; tape = globalTape; }
    else if (allocation === "request") { stream = `request:${event.request}`; index = cursors.requests[event.request]++; tape = requestTapes[event.request]; }
    else { stream = `slot:${event.slot}`; index = cursors.slots[event.slot]++; tape = requestTapes[ids[event.slot]]; }
    const u = tape[index];
    const distribution = probabilities[event.request].map((probability, i) => ({ token: tokens[i], probability }));
    return { ...event, stream, index, u, token: sampleFrom(distribution, u) };
  });
}

export function samplingStateComparison({ order = "CBA", allocation = "global", bSteps = 3, pauseAfter = 4, recovery = "state" } = {}) {
  const schedule = samplingSchedule(order, bSteps);
  if (!["global", "slot", "request"].includes(allocation)) throw new RangeError("未知随机流分配策略。");
  if (!["state", "reset"].includes(recovery)) throw new RangeError("未知恢复策略。");
  if (!Number.isInteger(pauseAfter) || pauseAfter < 0 || pauseAfter > schedule.length) throw new RangeError("暂停位置须在本次取数范围内。");
  const continuous = consume(schedule, allocation, emptyCursors());
  const reference = consume(samplingSchedule("ABC", 3), "request", emptyCursors());
  const byRequestStep = new Map(reference.map(row => [`${row.request}:${row.step}`, row]));
  const allocationRows = continuous.map(row => {
    const expected = byRequestStep.get(`${row.request}:${row.step}`);
    return { ...row, referenceU: expected.u, referenceToken: expected.token, sameU: row.u === expected.u, sameToken: row.token === expected.token };
  });
  const cursors = emptyCursors();
  const prefix = consume(schedule.slice(0, pauseAfter), allocation, cursors);
  // Serialize the progress and every active stream cursor at the same boundary.
  const snapshot = JSON.parse(JSON.stringify({ schema: "sampling-tape-state-v1", order, allocation, bSteps, nextEvent: pauseAfter, cursors, prefix }));
  const restoredCursors = recovery === "state" ? structuredClone(snapshot.cursors) : emptyCursors();
  const suffix = consume(schedule.slice(snapshot.nextEvent), allocation, restoredCursors);
  const resumed = snapshot.prefix.concat(suffix);
  const recoveryRows = suffix.map(row => {
    const expected = continuous[row.event];
    return { ...row, referenceU: expected.u, referenceToken: expected.token, sameU: row.u === expected.u, sameToken: row.token === expected.token };
  });
  return { order, allocation, bSteps, pauseAfter, recovery, schedule, continuous, resumed, snapshot,
    allocationRows, recoveryRows, allocationDifferences: allocationRows.filter(row => !row.sameU).length,
    tokenDifferences: allocationRows.filter(row => !row.sameToken).length,
    firstRecoveryDifference: recoveryRows.find(row => !row.sameU) || null };
}

export function samplingLedgerMarkup(result) {
  const ledger = (rows, label) => table(["轮次 / 请求", "当前槽位", "随机流 / 下标", "当前 u", "参考 u", "当前 / 参考 token"], rows.map(row => [
    `${row.step} / ${esc(row.request)}`, row.slot, `<code>${esc(row.stream)} [${row.index}]</code>`,
    `<span class="${row.sameU ? "sampling-match" : "sampling-change"}">${row.u.toFixed(2)}</span>`, row.referenceU.toFixed(2),
    `<span class="${row.sameToken ? "sampling-match" : "sampling-change"}">${esc(row.token)} / ${esc(row.referenceToken)}</span>`,
  ]), label);
  const first = result.firstRecoveryDifference;
  return `<div class="sampling-metrics"><p><span>按请求与步数比较，取数不同</span><strong>${result.allocationDifferences} / ${result.allocationRows.length}</strong></p><p><span>同一步输出 token 不同</span><strong>${result.tokenDifferences} / ${result.allocationRows.length}</strong></p><p><span>恢复后首个取数差异</span><strong>${first ? `第 ${first.step} 轮 / 请求 ${esc(first.request)}` : result.recoveryRows.length ? "本例一致" : "无后续取数"}</strong></p></div>
    <h4>一、调度后，随机数分给了谁？</h4><p>参考固定为 ABC 顺序、按请求 ID 分配、每请求三步。只比较共同的请求步；B 少输出的步骤不记作差异。</p>
    ${ledger(result.allocationRows, "请求调度与随机流分配账本")}
    <h4>二、暂停之后，从哪一位继续？</h4><p>已消费 ${result.pauseAfter} 次取数，参考为当前同一调度的连续运行。下表只列暂停后的取数，计算期间的分布保持不变。</p>
    <p class="sampling-cursors">保存的游标：<code>${esc(JSON.stringify(result.snapshot.cursors))}</code></p>
    ${result.recoveryRows.length ? ledger(result.recoveryRows, "暂停后的随机游标恢复账本") : "<p>已到本次计算末尾，没有后续随机数可比较。</p>"}
    <p class="sampling-legend">蓝色表示与该表参考一致，琥珀色表示发生变化。取数不同可以仍抽中同一 token；样本路径变化不直接证明采样分布有偏。</p>`;
}

export function initSamplingState() {
  if (typeof document === "undefined") return;
  const root = document.querySelector("[data-sampling-state]");
  if (!root || root.dataset.initialized) return;
  const control = name => root.querySelector(`[data-sampling-${name}]`);
  const render = () => {
    const bSteps = Number(control("finish").value), total = bSteps + 6;
    const result = samplingStateComparison({ order: control("order").value, allocation: control("allocation").value,
      bSteps, pauseAfter: control("pause").value === "end" ? total : Number(control("pause").value), recovery: control("recovery").value });
    control("view").innerHTML = samplingLedgerMarkup(result);
  };
  root.querySelectorAll("select").forEach(input => input.addEventListener("change", render));
  control("reset").addEventListener("click", () => {
    for (const [name, value] of [["order", "CBA"], ["allocation", "global"], ["finish", "3"], ["pause", "4"], ["recovery", "state"]]) control(name).value = value;
    render();
  });
  render(); control("controls").disabled = false; root.dataset.initialized = "true";
  if (location.hash === "#sampling-state") root.open = true;
}
