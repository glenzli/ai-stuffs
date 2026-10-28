import assert from "node:assert/strict";
import { test } from "node:test";
import { parseFragment } from "parse5";
import { repeatedSuccess, judgeLedger, initEvaluationReliability } from "../src/scripts/features/evaluation-reliability.js";
import { memoryStrategies, memoryReadout, initAgentMemory } from "../src/scripts/features/agent-memory.js";
import { streamSchedule, initMultimodalStream } from "../src/scripts/features/multimodal-stream.js";

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);
const ids = readout => readout.items.map(item => item.id);
const invalidProbabilities = [-0.01, 1.01, NaN, Infinity, -Infinity, "0.8", null, {}, []];
const descendants = (node, tag) => (node.tagName === tag ? [node] : []).concat((node.childNodes || []).flatMap(child => descendants(child, tag)));
const textContent = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(textContent).join("");
const attr = (node, name) => node.attrs?.find(item => item.name === name)?.value;

function mount(t, values = {}) {
  const nodes = new Map(Object.entries(values).map(([key, initial]) => {
    const listeners = new Map();
    const node = {
      value: String(initial),
      checked: initial === true,
      textContent: "",
      innerHTML: "",
      addEventListener(type, listener) {
        if (!listeners.has(type)) listeners.set(type, []);
        listeners.get(type).push(listener);
      },
      emit(type) {
        for (const listener of listeners.get(type) || []) listener();
      },
    };
    return [`[data-${key}]`, node];
  }));
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: selector => nodes.get(selector) || null } });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, "document", previous);
    else delete globalThis.document;
  });
  return key => nodes.get(`[data-${key}]`);
}

test("repetition defaults distinguish any success from all successes", () => {
  const result = repeatedSuccess();
  close(result.any, 0.99968);
  close(result.all, 0.32768);
});

test("repetition handles certain, impossible, and single-trial outcomes", () => {
  for (const k of [1, 5, 100]) {
    assert.deepEqual(repeatedSuccess(0, k), { any: 0, all: 0 });
    assert.deepEqual(repeatedSuccess(1, k), { any: 1, all: 1 });
  }
  for (const p of [0, 0.2, 0.8, 1]) {
    close(repeatedSuccess(p, 1).any, p);
    close(repeatedSuccess(p, 1).all, p);
  }
});

test("more repetitions increase coverage and decrease all-trial reliability", () => {
  for (const p of [0, 0.1, 0.5, 0.8, 1]) {
    let previous = repeatedSuccess(p, 1);
    for (let k = 2; k <= 25; k++) {
      const result = repeatedSuccess(p, k);
      assert.ok(result.any >= previous.any && result.all <= previous.all);
      assert.ok(result.any >= result.all && result.any <= 1 && result.all >= 0);
      previous = result;
    }
  }
});

test("repetition rejects invalid probabilities and trial counts", () => {
  for (const p of invalidProbabilities) assert.throws(() => repeatedSuccess(p), RangeError);
  for (const k of [0, -1, 1.5, NaN, Infinity, -Infinity, "5", null, {}]) {
    assert.throws(() => repeatedSuccess(0.8, k), RangeError);
  }
});

test("judge ledger defaults account for all four population fractions", () => {
  const ledger = judgeLedger();
  for (const [key, expected] of Object.entries({ tp: 0.54, fn: 0.06, fp: 0.04, tn: 0.36, observed: 0.58, bias: -0.02 })) {
    close(ledger[key], expected);
  }
  assert.deepEqual(judgeLedger({}), ledger);
});

test("confusion-matrix endpoints conserve the population and true rate", () => {
  for (const truth of [0, 0.3, 1]) {
    for (const sensitivity of [0, 1]) {
      for (const specificity of [0, 1]) {
        const ledger = judgeLedger({ truth, sensitivity, specificity });
        close(ledger.tp + ledger.fn, truth);
        close(ledger.fp + ledger.tn, 1 - truth);
        close(ledger.tp + ledger.fn + ledger.fp + ledger.tn, 1);
        close(ledger.observed, ledger.tp + ledger.fp);
        close(ledger.bias, ledger.observed - truth);
        for (const cell of [ledger.tp, ledger.fn, ledger.fp, ledger.tn]) assert.ok(cell >= 0 && cell <= 1);
      }
    }
  }
});

test("perfect and reversed judges expose zero and signed bias", () => {
  for (const truth of [0, 0.2, 0.6, 1]) {
    const perfect = judgeLedger({ truth, sensitivity: 1, specificity: 1 });
    close(perfect.observed, truth);
    close(perfect.bias, 0);
    const reversed = judgeLedger({ truth, sensitivity: 0, specificity: 0 });
    close(reversed.observed, 1 - truth);
    close(reversed.bias, 1 - 2 * truth);
  }
  assert.ok(judgeLedger({ sensitivity: 1, specificity: 0.5 }).bias > 0);
  assert.ok(judgeLedger({ sensitivity: 0.5, specificity: 1 }).bias < 0);
});

test("judge validates every probability independently", () => {
  for (const key of ["truth", "sensitivity", "specificity"]) {
    for (const value of invalidProbabilities) assert.throws(() => judgeLedger({ [key]: value }), RangeError);
  }
});

test("stream defaults process each chunk as it arrives with the same offline work", () => {
  const schedule = streamSchedule();
  assert.deepEqual(schedule.rows, [
    { index: 0, arrival: 200, start: 200, finish: 380, wait: 0 },
    { index: 1, arrival: 400, start: 400, finish: 580, wait: 0 },
    { index: 2, arrival: 600, start: 600, finish: 780, wait: 0 },
    { index: 3, arrival: 800, start: 800, finish: 980, wait: 0 },
  ]);
  assert.equal(schedule.first, 380);
  assert.equal(schedule.end, 980);
  assert.equal(schedule.offlineFirst, 980);
  assert.equal(schedule.offlineEnd, 1520);
  assert.equal(schedule.backlog, 0);
});

test("a slower shared worker accumulates queue wait rather than parallel service", () => {
  const schedule = streamSchedule({ chunks: 4, interval: 100, encode: 80, decode: 100 });
  assert.deepEqual(schedule.rows.map(row => row.start), [100, 280, 460, 640]);
  assert.deepEqual(schedule.rows.map(row => row.finish), [280, 460, 640, 820]);
  assert.deepEqual(schedule.rows.map(row => row.wait), [0, 80, 160, 240]);
  assert.equal(schedule.first, 280);
  assert.equal(schedule.end, 820);
  assert.equal(schedule.offlineFirst, 580);
  assert.equal(schedule.offlineEnd, 1120);
  assert.equal(schedule.backlog, 240);
});

test("schedule boundaries include one chunk, zero work, simultaneous arrivals, and fractional times", () => {
  const single = streamSchedule({ chunks: 1, interval: 100, encode: 0, decode: 50 });
  assert.equal(single.first, 150);
  assert.equal(single.end, single.offlineEnd);
  assert.equal(single.first, single.offlineFirst);
  const zero = streamSchedule({ interval: 0, encode: 0, decode: 0 });
  assert.equal(zero.first, 0);
  assert.equal(zero.end, 0);
  assert.equal(zero.offlineEnd, 0);
  assert.ok(zero.rows.every(row => row.wait === 0 && row.start === 0 && row.finish === 0));
  const simultaneous = streamSchedule({ interval: 0, encode: 0, decode: 50 });
  assert.deepEqual(simultaneous.rows.map(row => row.wait), [0, 50, 100, 150]);
  assert.equal(simultaneous.end, simultaneous.offlineEnd);
  const fractional = streamSchedule({ chunks: 2, interval: 0.5, encode: 0.25, decode: 0.5 });
  assert.equal(fractional.end, 2);
  assert.equal(fractional.backlog, 0.25);
  assert.equal(streamSchedule({ interval: 180 }).backlog, 0);
});

test("all slider boundaries produce ordered schedules and fair offline comparisons", () => {
  for (const chunks of [2, 4, 10]) {
    for (const interval of [100, 200, 500]) {
      for (const service of [50, 180, 600]) {
        const schedule = streamSchedule({ chunks, interval, encode: 0, decode: service });
        let previousFinish = 0;
        for (const row of schedule.rows) {
          assert.equal(row.arrival, (row.index + 1) * interval);
          assert.ok(row.start >= row.arrival && row.start >= previousFinish);
          assert.equal(row.finish - row.start, service);
          assert.equal(row.wait, row.start - row.arrival);
          previousFinish = row.finish;
        }
        assert.equal(schedule.end, interval + (chunks - 1) * Math.max(interval, service) + service);
        assert.equal(schedule.backlog, (chunks - 1) * Math.max(service - interval, 0));
        assert.equal(schedule.offlineFirst, chunks * interval + service);
        assert.equal(schedule.offlineEnd, chunks * interval + chunks * service);
        assert.ok(schedule.first <= schedule.offlineFirst && schedule.end <= schedule.offlineEnd);
      }
    }
  }
});

test("longer service, later arrivals, and more chunks keep finish times monotonic", () => {
  for (const interval of [100, 200, 500]) {
    let previous;
    for (const service of [50, 100, 200, 600]) {
      const schedule = streamSchedule({ interval, encode: 0, decode: service });
      if (previous) assert.ok(schedule.first >= previous.first && schedule.end >= previous.end && schedule.backlog >= previous.backlog);
      previous = schedule;
    }
  }
  for (const service of [50, 600]) {
    let previous;
    for (const interval of [100, 200, 500]) {
      const schedule = streamSchedule({ interval, encode: 0, decode: service });
      if (previous) assert.ok(schedule.first >= previous.first && schedule.end >= previous.end);
      previous = schedule;
    }
    previous = undefined;
    for (const chunks of [1, 2, 4, 10]) {
      const schedule = streamSchedule({ chunks, encode: 0, decode: service });
      if (previous) {
        assert.equal(schedule.first, previous.first);
        assert.ok(schedule.end >= previous.end && schedule.offlineEnd >= previous.offlineEnd && schedule.backlog >= previous.backlog);
      }
      previous = schedule;
    }
  }
});

test("schedule rejects invalid counts, negative or nonfinite times, and overflow", () => {
  for (const chunks of [0, -1, 1.5, NaN, Infinity, -Infinity, "4", null, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => streamSchedule({ chunks }), RangeError);
  }
  for (const key of ["interval", "encode", "decode"]) {
    for (const value of [-1, NaN, Infinity, -Infinity, "100", null, {}]) {
      assert.throws(() => streamSchedule({ [key]: value }), RangeError);
    }
  }
  assert.throws(() => streamSchedule({ encode: Number.MAX_VALUE, decode: Number.MAX_VALUE }), RangeError);
  assert.throws(() => streamSchedule({ interval: Number.MAX_VALUE }), RangeError);
});

test("memory strategies specify static record ids with no generated summaries", () => {
  assert.deepEqual(Object.keys(memoryStrategies), ["full", "compact", "structured"]);
  for (const strategy of Object.values(memoryStrategies)) {
    assert.equal(typeof strategy.label, "string");
    assert.equal(typeof strategy.note, "string");
    assert.ok(strategy.items.every(id => typeof id === "string"));
  }
});

test("full memory retains evidence and constraints while the latest Sat update wins over Fri", () => {
  const readout = memoryReadout("full");
  assert.deepEqual(ids(readout), ["goal", "evidence", "constraint", "stale", "update"]);
  assert.equal(readout.items.find(item => item.id === "stale").date, "Fri");
  assert.equal(readout.date, "Sat");
  assert.match(readout.constraints, /500/);
  assert.match(readout.constraints, /不进行预订/);
  assert.equal(readout.evidenceAvailable, true);
  const evidence = readout.items.find(item => item.id === "evidence");
  assert.equal(evidence.sourceId, "example:schedule-1");
  assert.ok(evidence.text.includes(evidence.quote) && evidence.text.includes(evidence.sourceId));
});

test("compact memory loses the source and constraints without recovering them from the goal", () => {
  const readout = memoryReadout("compact");
  assert.deepEqual(ids(readout), ["goal", "update"]);
  assert.equal(readout.constraints, "未保留");
  assert.equal(readout.evidenceAvailable, false);
  assert.equal(readout.date, "Sat");
  assert.ok(readout.items.every(item => !item.sourceId && !item.quote && !item.text.includes("500") && !item.text.includes("预订")));
});

test("structured memory retains source and constraints while omitting the stale date", () => {
  const readout = memoryReadout("structured");
  assert.deepEqual(ids(readout), ["goal", "evidence", "constraint", "update"]);
  assert.equal(readout.date, "Sat");
  assert.equal(readout.evidenceAvailable, true);
  assert.equal(readout.constraints, memoryReadout("full").constraints);
});

test("verification requests expose availability without claiming live retrieval or restoring lost records", () => {
  for (const mode of Object.keys(memoryStrategies)) {
    const stored = memoryReadout(mode);
    const requested = memoryReadout(mode, true);
    assert.deepEqual(requested.items, stored.items);
    assert.equal(requested.constraints, stored.constraints);
    assert.equal(requested.date, stored.date);
    assert.equal(requested.evidenceAvailable, stored.evidenceAvailable);
    assert.ok(!Object.hasOwn(requested, "verified") && !Object.hasOwn(requested, "grounding"));
    assert.match(requested.note, requested.evidenceAvailable ? /模拟回读.*真实任务/ : /没有来源.*重新检索/);
  }
});

test("memory rejects unsupported modes and nonboolean verification flags", () => {
  for (const mode of [undefined, null, "", "Full", "unknown", "__proto__", "constructor", ["full"]]) {
    assert.throws(() => memoryReadout(mode), RangeError);
  }
  for (const verify of [null, 0, 1, "true", {}]) assert.throws(() => memoryReadout("full", verify), TypeError);
});

test("readout callers cannot change later static records", () => {
  const readout = memoryReadout("full");
  readout.items[0].text = "Changed goal";
  readout.items.find(item => item.id === "evidence").sourceId = "changed";
  readout.items.pop();
  const fresh = memoryReadout("full");
  assert.equal(fresh.items[0].text, "准备一天的活动方案。");
  assert.equal(fresh.items.find(item => item.id === "evidence").sourceId, "example:schedule-1");
  assert.equal(fresh.date, "Sat");
});

test("initializers return harmlessly when their page markers are absent", t => {
  mount(t);
  assert.doesNotThrow(initEvaluationReliability);
  assert.doesNotThrow(initAgentMemory);
  assert.doesNotThrow(initMultimodalStream);
});

test("evaluation renders percent inputs, signed percentage-point bias, and semantic ledger rows", t => {
  const get = mount(t, {
    "er-p": 80, "er-k": 5, "er-truth": 60, "er-sensitivity": 90, "er-specificity": 90,
    "er-any": "", "er-all": "", "er-observed": "", "er-bias": "", "er-rows": "", "er-note": "",
  });
  initEvaluationReliability();
  assert.equal(get("er-any").textContent, "99.97%");
  assert.equal(get("er-all").textContent, "32.77%");
  assert.equal(get("er-observed").textContent, "58.00%");
  assert.equal(get("er-bias").textContent, "-2.0 pp");
  const tree = parseFragment(`<table><tbody>${get("er-rows").innerHTML}</tbody></table>`);
  assert.deepEqual(descendants(tree, "td").map(textContent), ["54.00%", "6.00%", "4.00%", "36.00%"]);
  assert.ok(descendants(tree, "th").every(node => attr(node, "scope") === "row"));
  get("er-p").value = "0";
  get("er-p").emit("input");
  assert.equal(get("er-any").textContent, "0.00%");
  assert.equal(get("er-all").textContent, "0.00%");
  get("er-p").value = "50";
  get("er-k").value = "1";
  get("er-k").emit("input");
  assert.equal(get("er-any").textContent, "50.00%");
  assert.equal(get("er-all").textContent, "50.00%");
  get("er-sensitivity").value = "100";
  get("er-specificity").value = "100";
  get("er-specificity").emit("input");
  assert.equal(get("er-bias").textContent, "0.0 pp");
  get("er-truth").value = "25";
  get("er-truth").emit("input");
  assert.equal(get("er-observed").textContent, "25.00%");
  get("er-specificity").value = "50";
  get("er-sensitivity").emit("input");
  assert.equal(get("er-bias").textContent, "+37.5 pp");
  assert.match(get("er-note").textContent, /独立同分布/);
});

test("memory mode and verification controls render semantic items and simulated grounding", t => {
  const get = mount(t, {
    "memory-mode": "full", "memory-verify": false, "memory-items": "", "memory-constraints": "",
    "memory-date": "", "memory-grounding": "", "memory-note": "",
  });
  initAgentMemory();
  const tree = parseFragment(`<ul>${get("memory-items").innerHTML}</ul>`);
  assert.equal(descendants(tree, "li").length, 5);
  assert.equal(descendants(tree, "strong").length, 5);
  assert.match(textContent(tree), /来源：example:schedule-1/);
  assert.equal(get("memory-date").textContent, "周六");
  assert.equal(get("memory-grounding").textContent, "未回读");
  get("memory-verify").checked = true;
  get("memory-verify").emit("change");
  assert.equal(get("memory-grounding").textContent, "模拟已回读");
  assert.match(get("memory-note").textContent, /模拟回读/);
  get("memory-mode").value = "compact";
  get("memory-mode").emit("change");
  assert.equal(descendants(parseFragment(get("memory-items").innerHTML), "li").length, 2);
  assert.equal(get("memory-constraints").textContent, "缺失");
  assert.equal(get("memory-grounding").textContent, "未回读");
  get("memory-mode").value = "structured";
  get("memory-mode").emit("change");
  assert.equal(descendants(parseFragment(get("memory-items").innerHTML), "li").length, 4);
  assert.equal(get("memory-grounding").textContent, "模拟已回读");
  assert.equal(get("memory-date").textContent, "周六");
  get("memory-verify").checked = false;
  get("memory-verify").emit("change");
  assert.equal(get("memory-grounding").textContent, "未回读");
});

test("stream controls render queue-aware HTML bars and the offline end time", t => {
  const get = mount(t, {
    "stream-interval": 100, "stream-service": 180, "stream-chunks": 4,
    "stream-first": "", "stream-end": "", "stream-offline": "", "stream-backlog": "",
    "stream-rows": "", "stream-timeline": "", "stream-note": "",
  });
  initMultimodalStream();
  assert.equal(get("stream-first").textContent, "280 ms");
  assert.equal(get("stream-end").textContent, "820 ms");
  assert.equal(get("stream-offline").textContent, "1120 ms");
  assert.equal(get("stream-backlog").textContent, "240 ms");
  // Input ranges belong to the authored HTML; the initializer preserves them.
  const table = parseFragment(`<table><tbody>${get("stream-rows").innerHTML}</tbody></table>`);
  assert.equal(descendants(table, "tr").length, 4);
  assert.deepEqual(descendants(table, "tr").map(node => descendants(node, "td").map(textContent)), [
    ["100 ms", "100 ms", "280 ms", "0 ms"],
    ["200 ms", "280 ms", "460 ms", "80 ms"],
    ["300 ms", "460 ms", "640 ms", "160 ms"],
    ["400 ms", "640 ms", "820 ms", "240 ms"],
  ]);
  const tree = parseFragment(get("stream-timeline").innerHTML);
  assert.equal(descendants(tree, "svg").length, 0);
  assert.equal(descendants(tree, "div").filter(node => attr(node, "class") === "stream-row").length, 4);
  assert.equal(descendants(tree, "div").filter(node => attr(node, "class") === "stream-track").length, 4);
  assert.deepEqual(descendants(tree, "span").map(textContent), ["第 1 块", "第 2 块", "第 3 块", "第 4 块"]);
  const bars = descendants(tree, "i");
  assert.equal(bars.length, 4);
  for (const [index, bar] of bars.entries()) {
    assert.equal(attr(bar, "role"), "img");
    assert.match(attr(bar, "aria-label"), /到达.*开始.*完成.*等待/);
    const style = Object.fromEntries(attr(bar, "style").split(";").map(part => part.split(":")));
    const left = parseFloat(style.left), width = parseFloat(style.width);
    close(left, (100 + index * 180) / 820 * 100);
    close(width, 180 / 820 * 100);
    assert.ok(left >= 0 && width >= 0 && left + width <= 100 + 1e-10);
  }
  get("stream-interval").value = "500";
  get("stream-interval").emit("input");
  assert.equal(get("stream-backlog").textContent, "0 ms");
  get("stream-service").value = "600";
  get("stream-service").emit("input");
  assert.equal(get("stream-backlog").textContent, "300 ms");
  get("stream-chunks").value = "2";
  get("stream-chunks").emit("input");
  assert.equal(descendants(parseFragment(get("stream-timeline").innerHTML), "i").length, 2);
  assert.equal(get("stream-backlog").textContent, "100 ms");
  assert.match(get("stream-note").textContent, /处理速度.*排队/);
});
