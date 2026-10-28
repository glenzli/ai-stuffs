import assert from "node:assert/strict";
import { test } from "node:test";
import { parseFragment } from "parse5";
import { distributedSamples, iterableWorkers, initDataOrder } from "../src/scripts/features/data-order.js";
import { bpeTraining, encodeBpe, initBpeMechanics } from "../src/scripts/features/bpe-mechanics.js";

const corpus = [
  { word: "low", freq: 5 }, { word: "lower", freq: 2 },
  { word: "newest", freq: 6 }, { word: "widest", freq: 3 },
];
const initialVocabulary = () => corpus.map(({ word, freq }) => ({ word, freq, tokens: [...word, "</w>"] }));
const knownMerges = [
  ["e", "s"], ["es", "t"], ["est", "</w>"], ["l", "o"],
  ["lo", "w"], ["e", "w"], ["ew", "est</w>"], ["n", "ewest</w>"],
  ["low", "</w>"], ["d", "est</w>"], ["i", "dest</w>"], ["w", "idest</w>"],
];

function referenceOrder(count, shuffle, epoch) {
  const order = Array.from({ length: count }, (_, index) => index);
  if (shuffle) {
    let state = (0x6d2b79f5n + BigInt(epoch)) & 0xffffffffn;
    for (let remaining = count; remaining > 1; remaining--) {
      state = (1664525n * state + 1013904223n) & 0xffffffffn;
      const chosen = Number(state * BigInt(remaining) / 0x100000000n);
      const last = order[remaining - 1];
      order[remaining - 1] = order[chosen];
      order[chosen] = last;
    }
  }
  return order;
}

function referenceDistribution(order, ranks, batch, samplerDrop, loaderDrop) {
  const expanded = order.slice(), samplerRemoved = [];
  let paddingCount = 0;
  if (samplerDrop) {
    while (expanded.length % ranks) samplerRemoved.unshift(expanded.pop());
  } else if (expanded.length) {
    while (expanded.length % ranks) {
      expanded.push(order[paddingCount % order.length]);
      paddingCount++;
    }
  }
  const perRank = Array.from({ length: ranks }, (_, rank) => ({ rank, assigned: [], batches: [], dropped: [] }));
  expanded.forEach((id, position) => perRank[position % ranks].assigned.push(id));
  const delivered = [];
  for (const row of perRank) {
    const pending = row.assigned.slice();
    while (pending.length) {
      const group = pending.splice(0, batch);
      if (loaderDrop && group.length < batch) row.dropped.push(...group);
      else {
        row.batches.push(group);
        delivered.push(...group);
      }
    }
  }
  const duplicates = [], missing = [];
  for (let index = 0; index < order.length; index++) {
    const count = delivered.filter(id => id === index).length;
    if (count > 1) duplicates.push({ index, count });
    if (count === 0) missing.push(index);
  }
  return {
    order: order.slice(), expanded, perRank, delivered, duplicates, missing,
    readCount: delivered.length, uniqueCount: new Set(delivered).size,
    steps: perRank[0].batches.length, samplerRemoved, paddingCount,
  };
}

// Enumerate candidates, then count each sliding window independently of the module.
function brutePairs(vocabulary) {
  const candidates = new Set();
  for (const { tokens } of vocabulary) {
    for (let index = 1; index < tokens.length; index++) candidates.add(JSON.stringify(tokens.slice(index - 1, index + 1)));
  }
  const results = Array.from(candidates, key => {
    const pair = JSON.parse(key);
    let count = 0;
    for (const { tokens, freq } of vocabulary) {
      for (let index = 0; index < tokens.length - 1; index++) {
        if (tokens[index] === pair[0] && tokens[index + 1] === pair[1]) count += freq;
      }
    }
    return { pair, count };
  });
  results.sort((left, right) => {
    if (left.count !== right.count) return right.count - left.count;
    const a = JSON.stringify(left.pair), b = JSON.stringify(right.pair);
    return a < b ? -1 : a > b ? 1 : 0;
  });
  return results;
}

function referenceMerge(tokens, pair) {
  const result = tokens.slice();
  for (let index = 0; index + 1 < result.length; index++) {
    if (result[index] === pair[0] && result[index + 1] === pair[1]) {
      result.splice(index, 2, pair[0] + pair[1]);
    }
  }
  return result;
}

const descendants = (node, tag) => (node.tagName === tag ? [node] : [])
  .concat((node.childNodes || []).flatMap(child => descendants(child, tag)));
const content = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(content).join("");
const tableRows = html => descendants(parseFragment(`<table><tbody>${html}</tbody></table>`), "tr")
  .map(row => (row.childNodes || []).filter(node => ["td", "th"].includes(node.tagName)).map(content));

function mount(t, values = {}) {
  const nodes = new Map(Object.entries(values).map(([key, value]) => {
    const listeners = new Map();
    return [`[data-${key}]`, {
      value: typeof value === "boolean" ? "on" : String(value),
      checked: typeof value === "boolean" ? value : false,
      textContent: "", innerHTML: "",
      addEventListener(event, listener) {
        if (!listeners.has(event)) listeners.set(event, []);
        listeners.get(event).push(listener);
      },
      emit(event) { for (const listener of listeners.get(event) || []) listener(); },
    }];
  }));
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: selector => nodes.get(selector) || null } });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, "document", previous);
    else delete globalThis.document;
  });
  return key => nodes.get(`[data-${key}]`);
}

const dataValues = {
  "do-count": 10, "do-ranks": 3, "do-batch": 2,
  "do-sampler-drop": false, "do-loader-drop": false, "do-shuffle": false, "do-epoch": 0,
  "do-order": "", "do-expanded": "", "do-reads": "", "do-unique": "", "do-missing": "", "do-note": "", "do-rows": "",
  "do-shard": true, "do-worker-rows": "",
};
const bpeValues = {
  "bpe-steps": 4, "bpe-word": "lowest", "bpe-steps-value": "", "bpe-vocab": "",
  "bpe-encoded": "", "bpe-decoded": "", "bpe-rounds": "", "bpe-pairs": "", "bpe-note": "",
};

test("默认采样保留填充读入，并区分两个阶段的丢弃", () => {
  const result = distributedSamples();
  assert.deepEqual(result.expanded, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]);
  assert.deepEqual(result.perRank.map(row => row.assigned), [[0, 3, 6, 9], [1, 4, 7, 0], [2, 5, 8, 1]]);
  assert.deepEqual(result.delivered, [0, 3, 6, 9, 1, 4, 7, 0, 2, 5, 8, 1]);
  assert.deepEqual(result.duplicates, [{ index: 0, count: 2 }, { index: 1, count: 2 }]);
  assert.equal(result.readCount, 12);
  assert.equal(result.uniqueCount, 10);
  assert.equal(result.steps, 2);
  const both = distributedSamples({ samplerDrop: true, loaderDrop: true });
  assert.deepEqual(both.samplerRemoved, [9]);
  assert.deepEqual(both.perRank.map(row => row.dropped), [[6], [7], [8]]);
  assert.deepEqual(both.missing, [6, 7, 8, 9]);
  assert.equal(both.readCount, 6);
  assert.equal(both.steps, 1);
  assert.deepEqual(distributedSamples({ count: 1, ranks: 8 }).expanded, Array(8).fill(0));
  assert.deepEqual(distributedSamples({ count: 0, ranks: 8 }).expanded, []);
});

test("固定教学 LCG32 的 epoch 置换可重复", () => {
  const pinned = [
    [1, 9, 7, 4, 8, 3, 6, 2, 5, 0],
    [3, 5, 4, 8, 2, 1, 7, 9, 6, 0],
    [4, 6, 3, 2, 1, 8, 9, 5, 7, 0],
  ];
  pinned.forEach((order, epoch) => {
    assert.deepEqual(distributedSamples({ shuffle: true, epoch }).order, order);
    assert.deepEqual(distributedSamples({ shuffle: true, epoch }).order, order);
    assert.deepEqual(referenceOrder(10, true, epoch), order);
  });
  for (const epoch of [0xffffffff, 0x100000000, Number.MAX_SAFE_INTEGER]) {
    assert.deepEqual(distributedSamples({ count: 30, shuffle: true, epoch }).order, referenceOrder(30, true, epoch));
  }
  const changed = distributedSamples();
  changed.order[0] = -1;
  changed.perRank[0].assigned[0] = -2;
  changed.perRank[0].batches[0][0] = -3;
  assert.deepEqual(distributedSamples(), referenceDistribution(referenceOrder(10, false, 0), 3, 2, false, false));
});

test("47,616 种采样组合符合独立分配、批处理和读入计数", () => {
  let cases = 0;
  for (let count = 0; count <= 30; count++) {
    for (const shuffle of [false, true]) {
      for (const epoch of [0, 1, 2]) {
        const order = referenceOrder(count, shuffle, epoch);
        for (let ranks = 1; ranks <= 8; ranks++) {
          for (let batch = 1; batch <= 8; batch++) {
            for (const samplerDrop of [false, true]) {
              for (const loaderDrop of [false, true]) {
                const options = { count, ranks, batch, samplerDrop, loaderDrop, shuffle, epoch };
                const result = distributedSamples(options);
                assert.deepEqual(result, referenceDistribution(order, ranks, batch, samplerDrop, loaderDrop), JSON.stringify(options));
                assert.ok(result.perRank.every(row => row.batches.length === result.steps));
                assert.equal(result.expanded.length + result.samplerRemoved.length, count + result.paddingCount);
                assert.equal(result.readCount + result.perRank.reduce((sum, row) => sum + row.dropped.length, 0), result.expanded.length);
                assert.equal(result.uniqueCount + result.missing.length, count);
                cases++;
              }
            }
          }
        }
      }
    }
  }
  assert.equal(cases, 47616);
});

test("worker 分片覆盖每个索引一次；未分片每个 worker 读取全部索引", () => {
  assert.deepEqual(iterableWorkers(), [
    { rank: 0, worker: 0, globalWorker: 0, ids: [0, 4, 8] },
    { rank: 0, worker: 1, globalWorker: 1, ids: [1, 5, 9] },
    { rank: 1, worker: 0, globalWorker: 2, ids: [2, 6] },
    { rank: 1, worker: 1, globalWorker: 3, ids: [3, 7] },
  ]);
  for (let count = 0; count <= 30; count++) {
    const indices = Array.from({ length: count }, (_, index) => index);
    for (let ranks = 1; ranks <= 8; ranks++) {
      for (let workers = 1; workers <= 8; workers++) {
        const sharded = iterableWorkers(count, ranks, workers, true);
        const unsharded = iterableWorkers(count, ranks, workers, false);
        assert.equal(sharded.length, ranks * workers);
        assert.deepEqual(sharded.flatMap(row => row.ids).sort((a, b) => a - b), indices);
        assert.equal(new Set(sharded.flatMap(row => row.ids)).size, count);
        sharded.forEach((row, globalWorker) => {
          assert.equal(row.rank * workers + row.worker, globalWorker);
          assert.equal(row.globalWorker, globalWorker);
          assert.ok(row.ids.every(id => id % (ranks * workers) === globalWorker));
          assert.deepEqual(unsharded[globalWorker].ids, indices);
        });
        assert.equal(unsharded.flatMap(row => row.ids).length, count * ranks * workers);
      }
    }
  }
  const unsharded = iterableWorkers();
  unsharded[0].ids.push(99);
  assert.ok(!unsharded[1].ids.includes(99));
});

test("采样与 worker API 拒绝越界、非整数和非布尔参数", () => {
  const invalid = {
    count: [-1, 31, 0.5, NaN, Infinity, "10", null],
    ranks: [0, 9, 1.5, NaN, "2", null],
    batch: [0, 9, 1.5, NaN, "2", null],
    epoch: [-1, 0.5, NaN, Infinity, "0", null, Number.MAX_SAFE_INTEGER + 1],
  };
  for (const [key, values] of Object.entries(invalid)) {
    for (const value of values) assert.throws(() => distributedSamples({ [key]: value }), RangeError);
  }
  for (const key of ["samplerDrop", "loaderDrop", "shuffle"]) {
    for (const value of [0, 1, "false", null]) assert.throws(() => distributedSamples({ [key]: value }), TypeError);
  }
  for (const count of invalid.count) assert.throws(() => iterableWorkers(count), RangeError);
  for (const ranks of invalid.ranks) assert.throws(() => iterableWorkers(10, ranks), RangeError);
  for (const workers of invalid.ranks) assert.throws(() => iterableWorkers(10, 2, workers), RangeError);
  assert.throws(() => iterableWorkers(10, 2, 2, "true"), TypeError);
});

test("BPE 固定 12 轮合并、加权频次和最终切分", () => {
  const model = bpeTraining(12);
  assert.deepEqual(model.merges, knownMerges);
  assert.deepEqual(model.rounds.map(round => round.count), [9, 9, 9, 7, 7, 6, 6, 6, 5, 3, 3, 3]);
  assert.deepEqual(model.splits, [
    { word: "low", freq: 5, tokens: ["low</w>"] },
    { word: "lower", freq: 2, tokens: ["low", "e", "r", "</w>"] },
    { word: "newest", freq: 6, tokens: ["newest</w>"] },
    { word: "widest", freq: 3, tokens: ["widest</w>"] },
  ]);
  const initial = bpeTraining(0);
  assert.deepEqual(initial.splits, initialVocabulary());
  assert.equal(initial.vocabCount, 11);
  assert.equal(initial.learnedVocabCount, 11);
  for (let steps = 1; steps <= 12; steps++) assert.equal(bpeTraining(steps).learnedVocabCount, 11 + steps);
  assert.deepEqual(initial.rounds, []);
  assert.deepEqual(initial.merges, []);
  const countOf = pair => initial.pairs.find(candidate => JSON.stringify(candidate.pair) === JSON.stringify(pair)).count;
  assert.equal(countOf(["e", "s"]), 6 + 3);
  assert.equal(countOf(["s", "t"]), 6 + 3);
  assert.equal(countOf(["t", "</w>"]), 6 + 3);
  assert.equal(countOf(["l", "o"]), 5 + 2);
  assert.deepEqual(initial.pairs.slice(0, 3).map(candidate => candidate.pair), [["e", "s"], ["s", "t"], ["t", "</w>"]]);
  assert.deepEqual(bpeTraining(), bpeTraining(4));
});

test("每轮 BPE 候选与独立滑窗计数一致，快照保留合并后的完整语料", () => {
  for (let steps = 0; steps <= 12; steps++) {
    const model = bpeTraining(steps);
    let expected = initialVocabulary();
    assert.equal(model.rounds.length, steps);
    for (const [index, round] of model.rounds.entries()) {
      const pairs = brutePairs(expected);
      assert.equal(round.step, index + 1);
      assert.deepEqual(round.pairs, pairs);
      assert.deepEqual(round.pair, pairs[0].pair);
      assert.equal(round.count, pairs[0].count);
      expected = expected.map(({ word, freq, tokens }) => ({ word, freq, tokens: referenceMerge(tokens, pairs[0].pair) }));
      assert.deepEqual(round.vocabulary, expected);
      assert.deepEqual(model.merges[index], round.pair);
      round.vocabulary.forEach(({ word, freq, tokens }, wordIndex) => {
        assert.equal(word, corpus[wordIndex].word);
        assert.equal(freq, corpus[wordIndex].freq);
        assert.equal(tokens.join(""), word + "</w>");
      });
    }
    assert.deepEqual(model.splits, expected);
    assert.deepEqual(model.pairs, brutePairs(expected));
    assert.equal(model.vocabCount, new Set(expected.flatMap(row => row.tokens)).size);
  }
  const model = bpeTraining(4);
  model.rounds[0].vocabulary[0].tokens[0] = "changed";
  model.rounds[0].pair[0] = "changed";
  assert.deepEqual(model.rounds[1].vocabulary, bpeTraining(2).splits);
  assert.deepEqual(model.merges, knownMerges.slice(0, 4));
  assert.deepEqual(model.splits, bpeTraining(4).splits);
});

test("编码按学得顺序合并所有不重叠位置，保留未见码点和边界", () => {
  assert.deepEqual(encodeBpe("aaaaa", [["a", "a"]]), {
    tokens: ["aa", "aa", "a", "</w>"], decoded: "aaaaa", unknown: ["a"],
  });
  assert.deepEqual(encodeBpe("abc", [["a", "b"], ["ab", "c"]]).tokens, ["abc", "</w>"]);
  assert.deepEqual(encodeBpe("abc", [["ab", "c"], ["a", "b"]]).tokens, ["ab", "c", "</w>"]);
  const words = ["low", "lower", "lowest", "newer", "widest", "newest", "", "z🙂z", "汉字", "</w>"];
  for (let steps = 0; steps <= 12; steps++) {
    const { merges } = bpeTraining(steps);
    const original = structuredClone(merges);
    for (const word of words) {
      let tokens = [...Array.from(word), "</w>"];
      for (const pair of merges) tokens = referenceMerge(tokens, pair);
      const encoded = encodeBpe(word, merges);
      assert.deepEqual(encoded.tokens, tokens);
      assert.equal(encoded.decoded, word);
      assert.equal(encoded.tokens.join(""), word + "</w>");
      assert.deepEqual(encodeBpe(word, merges), encoded);
    }
    assert.deepEqual(merges, original);
    assert.deepEqual(encodeBpe("z🙂z", merges).unknown, ["z", "🙂"]);
    assert.deepEqual(encodeBpe("lowest", merges).unknown, []);
  }
  assert.deepEqual(encodeBpe("lowest", knownMerges).tokens, ["low", "est</w>"]);
  assert.deepEqual(encodeBpe("newer", knownMerges).tokens, ["n", "ew", "e", "r", "</w>"]);
});

test("BPE API 拒绝无效轮数和合并表", () => {
  for (const steps of [-1, 13, 1.5, NaN, Infinity, "4", null]) assert.throws(() => bpeTraining(steps), RangeError);
  for (const word of [null, undefined, 5, {}, []]) assert.throws(() => encodeBpe(word, []), TypeError);
  for (const merges of [undefined, null, {}, ["ab"], [["a"]], [["a", "b", "c"]], [["", "b"]], [[1, "b"]]]) {
    assert.throws(() => encodeBpe("low", merges), TypeError);
  }
});

test("无 DOM 或页面标记时两个初始化函数安全返回", t => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  try {
    delete globalThis.document;
    assert.doesNotThrow(initDataOrder);
    assert.doesNotThrow(initBpeMechanics);
  } finally {
    if (previous) Object.defineProperty(globalThis, "document", previous);
  }
  mount(t);
  assert.doesNotThrow(initDataOrder);
  assert.doesNotThrow(initBpeMechanics);
});

test("采样界面所有控件更新真实四列表格，worker 控件独立固定为 2 x 2", t => {
  const get = mount(t, dataValues);
  initDataOrder();
  let eventCount = 0;
  const check = () => {
    const result = distributedSamples({
      count: Number(get("do-count").value), ranks: Number(get("do-ranks").value), batch: Number(get("do-batch").value),
      epoch: Number(get("do-epoch").value), shuffle: get("do-shuffle").checked,
      samplerDrop: get("do-sampler-drop").checked, loaderDrop: get("do-loader-drop").checked,
    });
    assert.deepEqual(JSON.parse(get("do-order").textContent), result.order);
    assert.deepEqual(JSON.parse(get("do-expanded").textContent), result.expanded);
    assert.equal(Number(get("do-reads").textContent), result.readCount);
    assert.equal(Number(get("do-unique").textContent), result.uniqueCount);
    assert.deepEqual(JSON.parse(get("do-missing").textContent), result.missing);
    assert.deepEqual(tableRows(get("do-rows").innerHTML), result.perRank.map(({ rank, assigned, batches, dropped }) =>
      [String(rank), JSON.stringify(assigned), JSON.stringify(batches), JSON.stringify(dropped)]));
    const workers = iterableWorkers(Number(get("do-count").value), 2, 2, get("do-shard").checked);
    assert.deepEqual(tableRows(get("do-worker-rows").innerHTML), workers.map(({ rank, worker, globalWorker, ids }) =>
      [String(rank), String(worker), String(globalWorker), JSON.stringify(ids)]));
    assert.match(get("do-note").textContent, new RegExp(`每进程 ${result.steps} 批`));
    if (get("do-shuffle").checked) assert.match(get("do-note").textContent, /教学置换.*不复现 PyTorch RNG/);
  };
  check();
  for (const [key, value] of [
    ["do-shuffle", true], ["do-epoch", 1], ["do-epoch", 2],
    ["do-sampler-drop", true], ["do-loader-drop", true], ["do-shard", false],
    ["do-count", 1], ["do-ranks", 4], ["do-batch", 4],
    ["do-sampler-drop", false], ["do-loader-drop", false], ["do-count", 0],
    ["do-count", 30], ["do-shard", true],
  ]) {
    if (typeof value === "boolean") get(key).checked = value;
    else get(key).value = String(value);
    get(key).emit(eventCount++ % 2 ? "change" : "input");
    check();
  }
});

test("采样界面无效输入清空结果，恢复后没有 NaN 或旧表格", t => {
  const get = mount(t, dataValues);
  initDataOrder();
  for (const [key, values] of Object.entries({
    "do-count": ["", " ", "NaN", "Infinity", "-1", "31", "2.5"],
    "do-ranks": ["", "0", "5"], "do-batch": ["", "0", "5"], "do-epoch": ["", "-1", "3"],
  })) {
    const previous = get(key).value;
    for (const value of values) {
      get(key).value = value;
      get(key).emit("input");
      for (const output of ["do-order", "do-expanded", "do-reads", "do-unique", "do-missing"]) {
        assert.equal(get(output).textContent, "-");
        assert.doesNotMatch(get(output).textContent, /NaN|undefined/);
      }
      assert.equal(get("do-rows").innerHTML, "");
      assert.equal(get("do-worker-rows").innerHTML, "");
      assert.match(get("do-note").textContent, /整数/);
    }
    get(key).value = previous;
    get(key).emit("change");
    assert.equal(get("do-reads").textContent, "12");
    assert.equal(tableRows(get("do-rows").innerHTML).length, 3);
  }
});

test("BPE 界面显示四列合并快照和三列下一轮候选，并转义边界符", t => {
  const get = mount(t, bpeValues);
  initBpeMechanics();
  for (let steps = 0; steps <= 10; steps++) {
    get("bpe-steps").value = String(steps);
    get("bpe-steps").emit(steps % 2 ? "change" : "input");
    const training = bpeTraining(steps);
    assert.equal(Number(get("bpe-steps-value").textContent), steps);
    assert.equal(Number(get("bpe-vocab").textContent), training.vocabCount);
    const rounds = tableRows(get("bpe-rounds").innerHTML);
    assert.equal(rounds.length, steps);
    rounds.forEach((values, index) => {
      const round = training.rounds[index];
      assert.equal(values.length, 4);
      assert.equal(Number(values[0]), round.step);
      assert.deepEqual(JSON.parse(values[1]), round.pair);
      assert.equal(Number(values[2]), round.count);
      assert.equal(values[3], round.vocabulary.map(({ word, freq, tokens }) => `${word} (${freq}): ${tokens.join(" | ")}`).join("; "));
    });
    assert.deepEqual(tableRows(get("bpe-pairs").innerHTML), training.pairs.map(({ pair, count }, index) =>
      [JSON.stringify(pair), String(count), index === 0 ? "下一轮" : ""]));
    assert.doesNotMatch(get("bpe-rounds").innerHTML, /<\/w>/);
    assert.doesNotMatch(get("bpe-pairs").innerHTML, /<\/w>/);
    if (steps > 0) assert.match(get("bpe-rounds").innerHTML, /&lt;\/w&gt;/);
    for (const word of ["low", "lower", "lowest", "newer", "widest"]) {
      get("bpe-word").value = word;
      get("bpe-word").emit(word === "lower" ? "change" : "input");
      assert.deepEqual(JSON.parse(get("bpe-encoded").textContent), encodeBpe(word, training.merges).tokens);
      assert.equal(get("bpe-decoded").textContent, word);
    }
    assert.match(get("bpe-note").textContent, /训练.*编码.*当前不同符号.*无字节回退/);
  }
});

test("BPE 界面拒绝空值和越界，缺少可选输出仍可初始化", t => {
  const get = mount(t, bpeValues);
  initBpeMechanics();
  for (const steps of ["", " ", "NaN", "Infinity", "-1", "11", "1.5"]) {
    get("bpe-steps").value = steps;
    get("bpe-steps").emit("input");
    for (const key of ["bpe-steps-value", "bpe-vocab", "bpe-encoded", "bpe-decoded"]) assert.equal(get(key).textContent, "-");
    assert.equal(get("bpe-rounds").innerHTML, "");
    assert.equal(get("bpe-pairs").innerHTML, "");
    assert.doesNotMatch(get("bpe-note").textContent, /NaN|undefined/);
  }
  get("bpe-steps").value = "4";
  get("bpe-word").value = "<script>";
  get("bpe-word").emit("change");
  assert.equal(get("bpe-encoded").textContent, "-");
  get("bpe-word").value = "lowest";
  get("bpe-word").emit("change");
  assert.equal(get("bpe-decoded").textContent, "lowest");
  assert.equal(tableRows(get("bpe-rounds").innerHTML).length, 4);
});

test("只有主输入而无可选节点时仍可渲染和监听", t => {
  const get = mount(t, { "do-count": 10, "bpe-steps": 4 });
  assert.doesNotThrow(initDataOrder);
  assert.doesNotThrow(initBpeMechanics);
  get("do-count").value = "";
  get("bpe-steps").value = "";
  assert.doesNotThrow(() => get("do-count").emit("input"));
  assert.doesNotThrow(() => get("bpe-steps").emit("change"));
});
