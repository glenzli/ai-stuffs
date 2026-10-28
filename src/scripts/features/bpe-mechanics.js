import { $ } from "../core.js";

const BOUNDARY = "</w>";
const CORPUS = [
  { word: "low", freq: 5 }, { word: "lower", freq: 2 },
  { word: "newest", freq: 6 }, { word: "widest", freq: 3 },
];
const ALPHABET = new Set(CORPUS.flatMap(({ word }) => Array.from(word)));
const WORDS = new Set(["low", "lower", "lowest", "newer", "widest"]);
const snapshot = vocabulary => vocabulary.map(({ word, freq, tokens }) => ({ word, freq, tokens: tokens.slice() }));

function countPairs(vocabulary) {
  const counts = new Map();
  for (const { freq, tokens } of vocabulary) {
    for (let index = 0; index + 1 < tokens.length; index++) {
      const key = JSON.stringify([tokens[index], tokens[index + 1]]);
      counts.set(key, (counts.get(key) || 0) + freq);
    }
  }
  // Compare JSON keys by code unit, independent of the host's locale.
  return Array.from(counts, ([key, count]) => ({ key, count }))
    .sort((left, right) => right.count - left.count || (left.key < right.key ? -1 : left.key > right.key ? 1 : 0))
    .map(({ key, count }) => ({ pair: JSON.parse(key), count }));
}

function mergeTokens(tokens, pair) {
  const merged = [];
  for (let index = 0; index < tokens.length; index++) {
    if (tokens[index] === pair[0] && tokens[index + 1] === pair[1]) {
      merged.push(pair[0] + pair[1]);
      index++;
    } else {
      merged.push(tokens[index]);
    }
  }
  return merged;
}

export function bpeTraining(steps = 4) {
  if (!Number.isInteger(steps) || steps < 0 || steps > 12) {
    throw new RangeError("steps must be an integer in 0..12.");
  }
  let vocabulary = CORPUS.map(({ word, freq }) => ({ word, freq, tokens: [...Array.from(word), BOUNDARY] }));
  const rounds = [], merges = [];
  for (let step = 1; step <= steps; step++) {
    const pairs = countPairs(vocabulary);
    if (!pairs.length) break;
    const { pair, count } = pairs[0];
    merges.push(pair.slice());
    vocabulary = vocabulary.map(({ word, freq, tokens }) => ({ word, freq, tokens: mergeTokens(tokens, pair) }));
    // Candidates are counted before this merge; vocabulary is the resulting snapshot.
    rounds.push({ step, pair: pair.slice(), count, vocabulary: snapshot(vocabulary), pairs });
  }
  return {
    rounds, merges, splits: snapshot(vocabulary),
    vocabCount: new Set(vocabulary.flatMap(({ tokens }) => tokens)).size,
    learnedVocabCount: new Set([...ALPHABET, BOUNDARY, ...merges.map(pair => pair.join(""))]).size,
    pairs: countPairs(vocabulary),
  };
}

export function encodeBpe(word, merges) {
  if (typeof word !== "string") throw new TypeError("word must be a string.");
  if (!Array.isArray(merges) || merges.some(pair =>
    !Array.isArray(pair) || pair.length !== 2 || pair.some(token => typeof token !== "string" || !token.length)
  )) throw new TypeError("merges must contain pairs of nonempty strings.");
  const characters = Array.from(word);
  let tokens = [...characters, BOUNDARY];
  for (const pair of merges) tokens = mergeTokens(tokens, pair);
  return {
    tokens,
    decoded: tokens.join("").slice(0, -BOUNDARY.length),
    unknown: Array.from(new Set(characters.filter(character => !ALPHABET.has(character)))),
  };
}

function escapeHtml(value) {
  const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(value).replace(/[&<>"']/g, character => entities[character]);
}

export function initBpeMechanics() {
  if (typeof document === "undefined") return;
  const stepsInput = $("[data-bpe-steps]");
  if (!stepsInput) return;
  const wordInput = $("[data-bpe-word]");
  const setText = (key, value) => {
    const node = $(`[data-bpe-${key}]`);
    if (node) node.textContent = String(value);
  };
  const setRows = (key, html) => {
    const node = $(`[data-bpe-${key}]`);
    if (node) node.innerHTML = html;
  };
  const render = () => {
    const raw = String(stepsInput.value ?? "").trim();
    const steps = raw === "" ? NaN : Number(raw);
    const word = wordInput?.value ?? "low";
    if (!Number.isInteger(steps) || steps < 0 || steps > 10 || !WORDS.has(word)) {
      for (const key of ["steps-value", "vocab", "learned", "encoded", "decoded"]) setText(key, "-");
      setRows("rounds", "");
      setRows("pairs", "");
      setText("note", "合并轮数需为 0 到 10 的整数，并选择有效单词。");
      return;
    }

    const training = bpeTraining(steps);
    const encoded = encodeBpe(word, training.merges);
    setText("steps-value", steps);
    setText("vocab", training.vocabCount);
    setText("learned", training.learnedVocabCount);
    setText("encoded", JSON.stringify(encoded.tokens));
    setText("decoded", encoded.decoded);
    setRows("rounds", training.rounds.map(({ step, pair, count, vocabulary }) => {
      const splits = vocabulary.map(({ word, freq, tokens }) => `${word} (${freq}): ${tokens.join(" | ")}`).join("; ");
      return `<tr><td>${step}</td><td>${escapeHtml(JSON.stringify(pair))}</td><td>${count}</td><td>${escapeHtml(splits)}</td></tr>`;
    }).join(""));
    setRows("pairs", training.pairs.map(({ pair, count }, index) =>
      `<tr><td>${escapeHtml(JSON.stringify(pair))}</td><td>${count}</td><td>${index === 0 ? "下一轮" : ""}</td></tr>`
    ).join(""));
    setText("note", `训练按词频统计相邻对，每轮合并一对；编码按已学顺序应用合并。当前不同符号只统计本轮语料使用的符号；累计词表保留初始符号和全部合并产物。${encoded.unknown.length ? `未见字符：${encoded.unknown.join(", ")}；保留原字符，无字节回退。` : "本词没有未见字符，本例无字节回退。"}`);
  };
  for (const input of [stepsInput, wordInput]) {
    for (const event of ["input", "change"]) input?.addEventListener(event, render);
  }
  render();
}
