import { $ } from "../core.js";

export function streamSchedule({ chunks = 4, interval = 200, encode = 80, decode = 100 } = {}) {
  if (!Number.isSafeInteger(chunks) || chunks < 1) {
    throw new RangeError("chunks must be a positive safe integer");
  }
  for (const [name, value] of Object.entries({ interval, encode, decode })) {
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError(`${name} must be a finite nonnegative time`);
    }
  }
  const service = encode + decode;
  const offlineFirst = chunks * interval + service;
  const offlineEnd = chunks * interval + chunks * service;
  if (!Number.isFinite(offlineEnd)) throw new RangeError("schedule duration must be finite");
  const rows = [];
  let previousFinish = 0;
  for (let index = 0; index < chunks; index++) {
    const arrival = (index + 1) * interval;
    const start = Math.max(arrival, previousFinish);
    const finish = start + service;
    rows.push({ index, arrival, start, finish, wait: start - arrival });
    previousFinish = finish;
  }
  return {
    rows,
    first: rows[0].finish,
    end: previousFinish,
    offlineFirst,
    offlineEnd,
    backlog: rows[rows.length - 1].wait,
  };
}

export function initMultimodalStream() {
  const intervalInput = $("[data-stream-interval]");
  if (!intervalInput) return;
  const serviceInput = $("[data-stream-service]");
  const chunksInput = $("[data-stream-chunks]");
  if (!serviceInput || !chunksInput) return;
  const ms = value => `${Number.isInteger(value) ? value : value.toFixed(1)} ms`;
  const setText = (selector, value) => {
    const output = $(selector);
    if (output) output.textContent = value;
  };
  const render = () => {
    setText("[data-stream-interval-value]", ms(Number(intervalInput.value)));
    setText("[data-stream-service-value]", ms(Number(serviceInput.value)));
    setText("[data-stream-chunks-value]", chunksInput.value);
    const schedule = streamSchedule({
      chunks: Number(chunksInput.value),
      interval: Number(intervalInput.value),
      encode: 0,
      decode: Number(serviceInput.value),
    });
    setText("[data-stream-first]", ms(schedule.first));
    setText("[data-stream-end]", ms(schedule.end));
    setText("[data-stream-offline]", ms(schedule.offlineEnd));
    setText("[data-stream-backlog]", ms(schedule.backlog));
    const rows = $("[data-stream-rows]");
    if (rows) rows.innerHTML = schedule.rows.map(row => `<tr><th scope="row">${row.index + 1}</th><td>${ms(row.arrival)}</td><td>${ms(row.start)}</td><td>${ms(row.finish)}</td><td>${ms(row.wait)}</td></tr>`).join("");
    const timeline = $("[data-stream-timeline]");
    const horizon = Math.max(schedule.end, 1);
    if (timeline) timeline.innerHTML = schedule.rows.map(row => {
      const label = `第 ${row.index + 1} 块：${ms(row.arrival)} 到达，${ms(row.start)} 开始，${ms(row.finish)} 完成，等待 ${ms(row.wait)}`;
      return `<div class="stream-row"><span>第 ${row.index + 1} 块</span><div class="stream-track"><i role="img" aria-label="${label}" title="${label}" style="left:${row.start / horizon * 100}%;width:${(row.finish - row.start) / horizon * 100}%"></i></div></div>`;
    }).join("");
    setText("[data-stream-note]", `输入每 ${ms(Number(intervalInput.value))} 到达一块，每块处理 ${ms(Number(serviceInput.value))}。整段接收后处理的首块在 ${ms(schedule.offlineFirst)} 完成。${schedule.backlog > 0 ? "处理速度落后于输入到达速度，后续块出现排队。" : "处理能跟上到达速度，当前没有排队积压。"}时间线色条显示处理区间，等待时间列在表中。`);
  };
  [intervalInput, serviceInput, chunksInput].forEach(input => input.addEventListener("input", render));
  render();
}
