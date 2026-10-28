import assert from "node:assert/strict";
import { test } from "node:test";
import { recipeLedger, loraLedger, initFinetuningRecipe } from "../src/scripts/features/finetuning-recipe.js";

test("default ledger counts processed tokens, causal response labels and scheduled windows", () => {
  const result = recipeLedger();
  assert.equal(result.usedPrompt, 600);
  assert.equal(result.usedAnswer, 400);
  assert.equal(result.inputPerRecord, 1000);
  assert.equal(result.supervisedPerRecord, 400);
  assert.equal(result.recordsPerSequence, 1);
  assert.equal(result.readOccurrences, 30000);
  assert.equal(result.sequencesPerEpoch, 10000);
  assert.equal(result.sequencesPerMicrostep, 8);
  assert.equal(result.microstepsPerEpoch, 1250);
  assert.equal(result.updatesPerEpoch, 157);
  assert.equal(result.updates, 471);
  assert.equal(result.globalSequences, 64);
  assert.equal(result.slotsPerEpoch, 20480000);
  assert.equal(result.totalSlots, 61440000);
  assert.equal(result.processedTokens, 30000000);
  assert.equal(result.supervisedTokens, 12000000);
  assert.equal(result.utilization, 0.48828125);
  assert.equal(result.tailSequences, 16);
  assert.equal(result.tailMicrosteps, 2);
});

test("packing whole records changes capacity but preserves token and label accounting", () => {
  const result = recipeLedger({ packing: true });
  assert.equal(result.recordsPerSequence, 2);
  assert.equal(result.sequencesPerEpoch, 5000);
  assert.equal(result.microstepsPerEpoch, 625);
  assert.equal(result.updatesPerEpoch, 79);
  assert.equal(result.updates, 237);
  assert.equal(result.slotsPerEpoch, 10240000);
  assert.equal(result.totalSlots, 30720000);
  assert.equal(result.processedTokens, 30000000);
  assert.equal(result.supervisedTokens, 12000000);
  assert.equal(result.utilization, 0.9765625);
  assert.equal(result.tailSequences, 8);
  assert.equal(result.tailMicrosteps, 1);
  assert.equal(recipeLedger({ supervise: "all", packing: true }).supervisedTokens, 29970000);
});

test("right truncation keeps prompt first and never adds BOS outside the record", () => {
  for (const [prompt, answer, limit, keptPrompt, keptAnswer, answerLabels, allLabels] of [
    [6, 4, 10, 6, 4, 4, 9],
    [6, 4, 8, 6, 2, 2, 7],
    [6, 4, 6, 6, 0, 0, 5],
    [6, 4, 3, 3, 0, 0, 2],
    [0, 4, 3, 0, 3, 2, 2],
    [4, 0, 8, 4, 0, 0, 3],
  ]) {
    const options = { records: 1, prompt, answer, limit, epochs: 1, microbatch: 1, dp: 1, accum: 1 };
    const result = recipeLedger(options);
    assert.equal(result.usedPrompt, keptPrompt);
    assert.equal(result.usedAnswer, keptAnswer);
    assert.equal(result.inputPerRecord, keptPrompt + keptAnswer);
    assert.equal(result.processedTokens, keptPrompt + keptAnswer);
    assert.equal(result.supervisedPerRecord, answerLabels);
    assert.equal(result.supervisedTokens, answerLabels);
    assert.equal(recipeLedger({ ...options, supervise: "all" }).supervisedPerRecord, allLabels);
  }
});

test("prompt-free answers lose their first causal label with or without packing", () => {
  for (const packing of [false, true]) {
    for (const supervise of ["answer", "all"]) {
      const result = recipeLedger({ records: 5, prompt: 0, answer: 3, limit: 8, epochs: 2, packing, supervise });
      assert.equal(result.supervisedPerRecord, 2);
      assert.equal(result.supervisedTokens, 20);
      assert.equal(result.processedTokens, 30);
      assert.equal(result.recordsPerSequence, packing ? 2 : 1);
      assert.equal(result.sequencesPerEpoch, packing ? 3 : 5);
    }
  }
});

test("single-token records have zero labels even when many records share one packed sequence", () => {
  for (const [prompt, answer] of [[0, 1], [1, 0], [5, 3]]) {
    for (const supervise of ["answer", "all"]) {
      const result = recipeLedger({ records: 8, prompt, answer, limit: 1, epochs: 2, packing: true, supervise });
      assert.equal(result.inputPerRecord, 1);
      assert.equal(result.supervisedPerRecord, 0);
      assert.equal(result.supervisedTokens, 0);
      assert.equal(result.processedTokens, 16);
      assert.equal(result.updates, 2);
    }
  }
  const packed = recipeLedger({ records: 8, prompt: 0, answer: 1, limit: 8, packing: true });
  assert.equal(packed.recordsPerSequence, 8);
  assert.equal(packed.sequencesPerEpoch, 1);
  assert.equal(packed.supervisedTokens, 0);
});

test("greedy packing never splits records or carries an unfinished sequence across epochs", () => {
  const unsplittable = recipeLedger({ records: 5, prompt: 3, answer: 3, limit: 10, packing: true, epochs: 1 });
  assert.equal(unsplittable.recordsPerSequence, 1);
  assert.equal(unsplittable.sequencesPerEpoch, 5);
  const tail = recipeLedger({ records: 5, prompt: 1, answer: 2, limit: 8, packing: true, epochs: 3, microbatch: 1, dp: 1, accum: 1 });
  assert.equal(tail.recordsPerSequence, 2);
  assert.equal(tail.sequencesPerEpoch, 3);
  assert.equal(tail.slotsPerEpoch, 24);
  assert.equal(tail.totalSlots, 72);
  assert.equal(tail.readOccurrences, 15);
  assert.equal(tail.processedTokens, 45);
  assert.equal(tail.utilization, 0.625);
  assert.equal(tail.updates, 9);
  const exact = recipeLedger({ records: 6, prompt: 1, answer: 3, limit: 8, packing: true, microbatch: 1, dp: 1, accum: 1 });
  assert.equal(exact.recordsPerSequence, 2);
  assert.equal(exact.sequencesPerEpoch, 3);
  assert.equal(exact.utilization, 1);
});

test("each epoch flushes accumulation and accounts all final microstep slots without dropping records", () => {
  const result = recipeLedger({ records: 9, prompt: 1, answer: 1, limit: 2, epochs: 3, microbatch: 1, dp: 1, accum: 8 });
  assert.equal(result.microstepsPerEpoch, 9);
  assert.equal(result.updatesPerEpoch, 2);
  assert.equal(result.updates, 6);
  assert.equal(result.processedTokens, 54);
  assert.equal(result.tailSequences, 1);
  assert.equal(result.tailMicrosteps, 1);
  for (const [records, steps, updates, tailSequences, tailMicrosteps, utilization] of [
    [1, 1, 1, 1, 1, 0.125], [63, 8, 1, 63, 8, 0.984375],
    [64, 8, 1, 0, 0, 1], [65, 9, 2, 1, 1, 65 / 72],
  ]) {
    const ledger = recipeLedger({ records, prompt: 1, answer: 1, limit: 2, epochs: 1 });
    assert.equal(ledger.microstepsPerEpoch, steps);
    assert.equal(ledger.updatesPerEpoch, updates);
    assert.equal(ledger.tailSequences, tailSequences);
    assert.equal(ledger.tailMicrosteps, tailMicrosteps);
    assert.equal(ledger.slotsPerEpoch, steps * 16);
    assert.equal(ledger.processedTokens, records * 2);
    assert.equal(ledger.utilization, utilization);
  }
});

test("zero records leave all dataset and schedule accounting zero while retaining configured capacities", () => {
  for (const packing of [false, true]) {
    for (const supervise of ["answer", "all"]) {
      const result = recipeLedger({ records: 0, packing, supervise });
      for (const key of [
        "readOccurrences", "sequencesPerEpoch", "microstepsPerEpoch", "updatesPerEpoch", "updates",
        "slotsPerEpoch", "totalSlots", "processedTokens", "supervisedTokens", "utilization", "tailSequences", "tailMicrosteps",
      ]) assert.equal(result[key], 0, key);
      assert.equal(result.inputPerRecord, 1000);
      assert.equal(result.recordsPerSequence, packing ? 2 : 1);
      assert.equal(result.globalSequences, 64);
    }
  }
});

test("recipe rejects invalid numeric inputs and enum values with RangeError", () => {
  const numericKeys = ["records", "prompt", "answer", "limit", "epochs", "microbatch", "dp", "accum"];
  for (const key of numericKeys) {
    for (const value of [-1, 0.5, NaN, Infinity, -Infinity, Number.MAX_SAFE_INTEGER + 1, "2", null, true, 2n]) {
      assert.throws(() => recipeLedger({ [key]: value }), RangeError, `${key}: ${String(value)}`);
    }
  }
  for (const key of ["limit", "epochs", "microbatch", "dp", "accum"]) {
    assert.throws(() => recipeLedger({ [key]: 0 }), RangeError, key);
  }
  assert.throws(() => recipeLedger({ prompt: 0, answer: 0 }), RangeError);
  assert.throws(() => recipeLedger({ records: 0, prompt: 0, answer: 0 }), RangeError);
  for (const supervise of ["", "response", null, false]) assert.throws(() => recipeLedger({ supervise }), RangeError);
  for (const packing of [0, 1, "false", null]) assert.throws(() => recipeLedger({ packing }), RangeError);
});

test("unsafe derived counts throw instead of silently rounding or overflowing", () => {
  const max = Number.MAX_SAFE_INTEGER;
  for (const options of [
    { prompt: max, answer: 1 }, { microbatch: max, dp: 2 }, { microbatch: max, dp: 1, accum: 2 },
    { records: max, prompt: 0, answer: 1, limit: 1, epochs: 2, microbatch: 1, dp: 1, accum: 1 },
    { records: max, prompt: 1, answer: 1, limit: 2, epochs: 1, microbatch: 1, dp: 1, accum: 1 },
    { records: 1, prompt: 0, answer: 1, limit: max, epochs: 1, microbatch: 2, dp: 1, accum: 1 },
    { records: 1, prompt: 0, answer: 1, limit: max, epochs: 2, microbatch: 1, dp: 1, accum: 1 },
  ]) assert.throws(() => recipeLedger(options), RangeError);
  const bounded = recipeLedger({ records: max, prompt: 0, answer: 1, limit: 1, epochs: 1, microbatch: 1, dp: 1, accum: 1 });
  assert.equal(bounded.processedTokens, max);
  assert.equal(bounded.slotsPerEpoch, max);
  assert.equal(bounded.updates, max);
  assert.equal(bounded.utilization, 1);
});

test("small exhaustive ledgers match a record-by-record packing and scheduling oracle", () => {
  for (const records of [0, 1, 2, 3, 7, 9]) {
    for (const [prompt, answer, limit] of [[0, 1, 4], [0, 3, 5], [2, 3, 4], [4, 2, 2], [1, 1, 4]]) {
      for (const packing of [false, true]) {
        for (const supervise of ["answer", "all"]) {
          for (const [microbatch, dp, accum, epochs] of [[1, 1, 1, 1], [2, 2, 3, 2], [1, 2, 2, 3]]) {
            const tokens = Array(prompt).fill("prompt").concat(Array(answer).fill("answer")).slice(0, limit);
            const labels = tokens.slice(1).filter(token => supervise === "all" || token === "answer").length;
            const sequences = [];
            for (let record = 0; record < records; record++) {
              if (!packing || !sequences.length || sequences.at(-1) + tokens.length > limit) sequences.push(0);
              sequences[sequences.length - 1] += tokens.length;
            }
            let steps = 0, windows = 0, accumulated = 0;
            for (let offset = 0; offset < sequences.length; offset += microbatch * dp) {
              steps++;
              if (++accumulated === accum) { windows++; accumulated = 0; }
            }
            if (accumulated) windows++;
            const result = recipeLedger({ records, prompt, answer, limit, packing, supervise, microbatch, dp, accum, epochs });
            assert.equal(result.sequencesPerEpoch, sequences.length);
            assert.equal(result.microstepsPerEpoch, steps);
            assert.equal(result.updatesPerEpoch, windows);
            assert.equal(result.updates, windows * epochs);
            assert.equal(result.processedTokens, records * tokens.length * epochs);
            assert.equal(result.supervisedTokens, records * labels * epochs);
            assert.ok(result.supervisedTokens <= result.processedTokens);
            assert.ok(result.processedTokens <= result.totalSlots);
            assert.ok(result.utilization >= 0 && result.utilization <= 1);
          }
        }
      }
    }
  }
});

test("LoRA q/v accounting respects asymmetric GQA dimensions", () => {
  const result = loraLedger();
  assert.equal(result.rank, 16);
  assert.equal(result.target, "qv");
  assert.equal(result.layers, 32);
  assert.equal(result.base, 7000000000);
  assert.deepEqual(result.rows, [
    { name: "q", input: 4096, output: 4096, perLayer: 131072, total: 4194304 },
    { name: "v", input: 4096, output: 1024, perLayer: 81920, total: 2621440 },
  ]);
  assert.equal(result.params, 6815744);
  assert.equal(result.ratio, 6815744 / 7000000000);
});

test("attention and all targets add the stated projections, and parameter count scales linearly with rank", () => {
  assert.deepEqual(loraLedger(16, "attention").rows, [
    ...loraLedger().rows,
    { name: "k", input: 4096, output: 1024, perLayer: 81920, total: 2621440 },
    { name: "o", input: 4096, output: 4096, perLayer: 131072, total: 4194304 },
  ]);
  assert.deepEqual(loraLedger(16, "all").rows.slice(4), [
    { name: "gate", input: 4096, output: 11008, perLayer: 241664, total: 7733248 },
    { name: "up", input: 4096, output: 11008, perLayer: 241664, total: 7733248 },
    { name: "down", input: 11008, output: 4096, perLayer: 241664, total: 7733248 },
  ]);
  for (const [target, count] of [["qv", 425984], ["attention", 851968], ["all", 2301952]]) {
    for (const rank of [1, 8, 16, 64, 128]) {
      const result = loraLedger(rank, target);
      assert.equal(result.params, count * rank);
      assert.equal(result.ratio, result.params / 7000000000);
      assert.equal(result.rows.reduce((sum, row) => sum + row.total, 0), result.params);
    }
  }
});

test("LoRA rejects invalid rank, unknown targets and unsafe parameter totals", () => {
  for (const rank of [0, -1, 1.5, NaN, Infinity, -Infinity, "16", null, true, 16n, Number.MAX_SAFE_INTEGER]) {
    assert.throws(() => loraLedger(rank), RangeError);
  }
  for (const target of ["", "mlp", null, 0]) assert.throws(() => loraLedger(16, target), RangeError);
  const largestQvRank = Math.floor(Number.MAX_SAFE_INTEGER / 425984);
  assert.equal(loraLedger(largestQvRank).params, largestQvRank * 425984);
  assert.throws(() => loraLedger(largestQvRank + 1), RangeError);
});

class Element extends EventTarget {
  constructor(value = "") {
    super();
    this.value = String(value);
    this.checked = false;
    this.textContent = "";
    this.innerHTML = "";
    this.style = {};
  }
}

function fixture(overrides = {}) {
  const elements = new Map();
  for (const [name, value] of Object.entries({
    records: 10000, prompt: 600, answer: 400, limit: 2048, epochs: 3, micro: 2,
    dp: 4, accum: 8, supervise: "answer", rank: 16, target: "qv", ...overrides,
  })) elements.set(`[data-ft-${name}]`, new Element(value));
  for (const name of [
    "packing", "processed", "supervised", "updates", "sequences", "perseq", "fill", "note", "ledger",
    "prompt-bar", "answer-bar", "pad-bar", "mask", "params", "ratio", "lora-rows", "rank-value",
  ]) elements.set(`[data-ft-${name}]`, new Element());
  return {
    document: { querySelector: selector => elements.get(selector) || null },
    get: name => elements.get(`[data-ft-${name}]`),
    change(name, value, event = "input") {
      const element = elements.get(`[data-ft-${name}]`);
      if (name === "packing") element.checked = value;
      else element.value = String(value);
      element.dispatchEvent(new Event(event));
    },
  };
}

function withDocument(document, callback) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { value: document, configurable: true });
  try { callback(); }
  finally {
    if (previous) Object.defineProperty(globalThis, "document", previous);
    else delete globalThis.document;
  }
}

test("initializer safely guards absent DOM or missing page marker and tolerates partial markup", () => {
  withDocument(undefined, () => assert.doesNotThrow(initFinetuningRecipe));
  withDocument({ querySelector: () => null }, () => assert.doesNotThrow(initFinetuningRecipe));
  withDocument({ querySelector: selector => selector === "[data-ft-records]" ? new Element(0) : null }, () => {
    assert.doesNotThrow(initFinetuningRecipe);
  });
});

test("DOM renders plain counts, one-record bars and both ledgers", () => {
  const view = fixture();
  withDocument(view.document, () => {
    initFinetuningRecipe();
    for (const [name, value] of [
      ["processed", "30,000,000"], ["supervised", "12,000,000"], ["updates", "471"],
      ["sequences", "10,000"], ["perseq", "1"], ["fill", "48.8%"],
      ["params", "6,815,744"], ["ratio", "0.097%"], ["rank-value", "16"],
    ]) assert.equal(view.get(name).textContent, value, name);
    assert.equal(view.get("prompt-bar").style.width, "29.296875%");
    assert.equal(view.get("answer-bar").style.width, "19.53125%");
    assert.equal(view.get("pad-bar").style.width, "51.171875%");
    assert.match(view.get("ledger").innerHTML, /<td>计划更新窗口 \/ 轮<\/td><td>157<\/td>/);
    assert.match(view.get("ledger").innerHTML, /<td>读取次数<\/td><td>30,000<\/td>/);
    assert.match(view.get("ledger").innerHTML, /<td>尾窗口序列 \/ 轮<\/td><td>16<\/td>/);
    assert.match(view.get("lora-rows").innerHTML, /<td>v<\/td><td>4,096 → 1,024<\/td><td>81,920<\/td><td>2,621,440<\/td>/);
    for (const row of view.get("lora-rows").innerHTML.matchAll(/<tr>(.*?)<\/tr>/g)) {
      assert.equal([...row[1].matchAll(/<td>/g)].length, 4);
    }
    assert.match(view.get("mask").textContent, /提示和 padding 不参与 loss/);
    assert.match(view.get("note").textContent, /轮末结清/);
  });
});

test("packing and supervision selects rerender labels without treating packed records as one causal stream", () => {
  const view = fixture();
  withDocument(view.document, () => {
    initFinetuningRecipe();
    view.change("packing", true, "change");
    assert.equal(view.get("perseq").textContent, "2");
    assert.equal(view.get("sequences").textContent, "5,000");
    assert.equal(view.get("updates").textContent, "237");
    assert.equal(view.get("fill").textContent, "97.7%");
    assert.equal(view.get("prompt-bar").style.width, "29.296875%");
    assert.match(view.get("note").textContent, /完整记录，各条独立 mask/);
    view.change("supervise", "all", "change");
    assert.equal(view.get("supervised").textContent, "29,970,000");
    assert.match(view.get("mask").textContent, /每条记录首个 token/);
    view.change("prompt", 0);
    assert.equal(view.get("supervised").textContent, "11,970,000");
    view.change("supervise", "answer", "change");
    assert.equal(view.get("supervised").textContent, "11,970,000");
    assert.match(view.get("mask").textContent, /首个 token/);
  });
});

test("every numeric recipe input updates its corresponding accounting", () => {
  const view = fixture();
  withDocument(view.document, () => {
    initFinetuningRecipe();
    view.change("records", 9);
    assert.equal(view.get("processed").textContent, "27,000");
    view.change("epochs", 1, "change");
    assert.equal(view.get("processed").textContent, "9,000");
    view.change("accum", 1);
    assert.equal(view.get("updates").textContent, "2");
    view.change("micro", 1);
    assert.equal(view.get("updates").textContent, "3");
    view.change("dp", 1, "change");
    assert.equal(view.get("updates").textContent, "9");
    view.change("answer", 300);
    assert.equal(view.get("supervised").textContent, "2,700");
    view.change("prompt", 700);
    assert.equal(view.get("processed").textContent, "9,000");
    view.change("limit", 800);
    assert.equal(view.get("processed").textContent, "7,200");
    assert.equal(view.get("supervised").textContent, "900");
    assert.match(view.get("note").textContent, /回答被截断，保留 100\/300/);
    assert.equal(view.get("prompt-bar").style.width, "87.5%");
    assert.equal(view.get("answer-bar").style.width, "12.5%");
    assert.equal(view.get("pad-bar").style.width, "0%");
  });
});

test("zero-label DOM states retain scheduled windows and require optimizer skips", () => {
  const view = fixture({ records: 4, prompt: 0, answer: 1, limit: 8, epochs: 2 });
  withDocument(view.document, () => {
    initFinetuningRecipe();
    view.change("packing", true, "change");
    assert.equal(view.get("processed").textContent, "8");
    assert.equal(view.get("supervised").textContent, "0");
    assert.equal(view.get("updates").textContent, "2");
    assert.match(view.get("note").textContent, /没有有效训练目标；优化器必须跳过更新/);
    assert.match(view.get("mask").textContent, /有效监督标签为 0/);
    view.change("records", 0);
    for (const name of ["processed", "supervised", "updates", "sequences"]) assert.equal(view.get(name).textContent, "0");
    assert.equal(view.get("fill").textContent, "0.0%");
    assert.match(view.get("note").textContent, /记录数为 0/);
    assert.match(view.get("note").textContent, /优化器必须跳过更新/);
    view.change("records", 4);
    view.change("prompt", 8);
    view.change("answer", 2);
    assert.match(view.get("note").textContent, /回答被截断，保留 0\/2/);
    assert.equal(view.get("prompt-bar").style.width, "100%");
    assert.equal(view.get("answer-bar").style.width, "0%");
    assert.equal(view.get("pad-bar").style.width, "0%");
    view.change("supervise", "all", "change");
    assert.equal(view.get("supervised").textContent, "56");
    assert.doesNotMatch(view.get("note").textContent, /优化器必须跳过/);
    view.change("limit", 1);
    assert.equal(view.get("supervised").textContent, "0");
    assert.match(view.get("note").textContent, /提示被截断/);
    assert.match(view.get("note").textContent, /没有有效训练目标/);
  });
});

test("invalid or blank recipe input clears stale accounting and recovers independently of LoRA", () => {
  const view = fixture();
  withDocument(view.document, () => {
    initFinetuningRecipe();
    for (const value of ["", " ", "invalid", "1.5", "-1", "Infinity", "9007199254740992"]) {
      view.change("records", value);
      for (const name of ["processed", "supervised", "updates", "sequences", "perseq", "fill"]) {
        assert.equal(view.get(name).textContent, "-");
      }
      assert.equal(view.get("ledger").innerHTML, "");
      assert.equal(view.get("prompt-bar").style.width, "0%");
      assert.equal(view.get("params").textContent, "6,815,744");
      assert.match(view.get("note").textContent, /安全整数/);
    }
    view.change("rank", 32);
    assert.equal(view.get("params").textContent, "13,631,488");
    view.change("records", 10000, "change");
    assert.equal(view.get("processed").textContent, "30,000,000");
    assert.equal(view.get("prompt-bar").style.width, "29.296875%");
    assert.doesNotMatch(view.get("note").textContent, /安全整数/);
  });
});

test("LoRA controls update rank and targets, clear invalid results and recover", () => {
  const view = fixture();
  withDocument(view.document, () => {
    initFinetuningRecipe();
    view.change("rank", 32);
    assert.equal(view.get("rank-value").textContent, "32");
    assert.equal(view.get("params").textContent, "13,631,488");
    view.change("target", "attention", "change");
    assert.equal(view.get("params").textContent, "27,262,976");
    view.change("target", "all", "change");
    assert.equal(view.get("params").textContent, "73,662,464");
    assert.equal(view.get("ratio").textContent, "1.052%");
    assert.match(view.get("lora-rows").innerHTML, /<td>down<\/td><td>11,008 → 4,096<\/td>/);
    view.change("rank", "", "change");
    for (const name of ["rank-value", "params", "ratio"]) assert.equal(view.get(name).textContent, "-");
    assert.equal(view.get("lora-rows").innerHTML, "");
    assert.equal(view.get("processed").textContent, "30,000,000");
    view.change("rank", 16);
    assert.equal(view.get("params").textContent, "36,831,232");
    view.change("target", "invalid", "change");
    assert.equal(view.get("params").textContent, "-");
    view.change("target", "qv", "change");
    assert.equal(view.get("params").textContent, "6,815,744");
  });
});
