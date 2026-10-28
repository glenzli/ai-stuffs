import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { parseFragment } from "parse5";
import { initDataTokenizer } from "../src/scripts/features/data-tokenizer.js";

const walk = (node, predicate) => [...(predicate(node) ? [node] : []), ...(node.childNodes || []).flatMap(child => walk(child, predicate))];
const attrs = node => Object.fromEntries((node.attrs || []).map(({ name, value }) => [name, value]));
const text = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(text).join("");
const root = parseFragment(await readFile("src/content/data-tokenizer.html", "utf8"));
const marked = name => walk(root, node => name in attrs(node))[0];
const markers = ["count", "ratio", "capacity", "note", "chips", "seq", "vocab", "kv"];

function mount(t) {
  const nodes = new Map(markers.map(key => [key, { textContent: text(marked(`data-tokenizer-${key}`)), innerHTML: "" }]));
  const buttons = ["char", "bpe", "byte"].map(mode => ({
    dataset: { tokenMode: mode }, attributes: {},
    setAttribute(key, value) { this.attributes[key] = value; },
    addEventListener(event, fn) { assert.equal(event, "click"); this.click = fn; },
  }));
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: {
    querySelector: selector => nodes.get(selector.replace(/^\[data-tokenizer-|\]$/g, "")) || null,
    querySelectorAll: selector => selector === "[data-token-mode]" ? buttons : [],
  } });
  t.after(() => previous ? Object.defineProperty(globalThis, "document", previous) : delete globalThis.document);
  initDataTokenizer();
  return { nodes, buttons };
}

test("the saved BPE fallback agrees with the initialized UI, including its unit note", t => {
  const { nodes } = mount(t);
  for (const key of ["count", "ratio", "capacity", "note", "seq", "vocab", "kv"]) {
    assert.equal(nodes.get(key).textContent, text(marked(`data-tokenizer-${key}`)), key);
  }
  const note = text(walk(root, node => attrs(node).id === "tokenizer-budget-note")[0]);
  assert.match(note, /15.*Unicode/);
  assert.match(note, /4,096/);
  assert.match(note, /未计模板与特殊 token/);
});

test("all three modes update the same units, selection and actual token chips", t => {
  const { nodes, buttons } = mount(t);
  const expected = { char: [15, "1.00", "4,096"], bpe: [8, "0.53", "7,680"], byte: [45, "3.00", "1,365"] };
  const sentence = "负样本让模型学会区分错误候选。";
  assert.equal(Buffer.byteLength(sentence, "utf8"), expected.byte[0]);
  for (const mode of ["byte", "char", "bpe", "byte", "bpe"]) {
    buttons.find(button => button.dataset.tokenMode === mode).click();
    const [count, ratio, capacity] = expected[mode];
    assert.equal(nodes.get("count").textContent, String(count));
    assert.equal(nodes.get("ratio").textContent, ratio);
    assert.equal(nodes.get("capacity").textContent, capacity);
    assert.deepEqual(buttons.map(button => button.attributes["aria-pressed"]), buttons.map(button => String(button.dataset.tokenMode === mode)));
    const chips = walk(parseFragment(nodes.get("chips").innerHTML), node => node.tagName === "span").map(text);
    assert.equal(chips.length, count);
    if (mode === "byte") {
      assert.equal(Buffer.from(chips.map(value => parseInt(value, 16))).toString("utf8"), sentence);
      assert.match(nodes.get("vocab").textContent, /256/);
      assert.match(nodes.get("note").textContent, /byte fallback 不同/);
    } else assert.equal(chips.join(""), sentence);
  }
});
