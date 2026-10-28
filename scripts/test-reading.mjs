import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { parse } from "parse5";
import { pages, site } from "../src/pages.mjs";
import { wikiGroups } from "../src/wiki.mjs";
import { learningPath } from "../src/learning.mjs";
import { learningRunSnapshot } from "../src/scripts/features/learning-run-data.js";
import { learningExampleMarkup, validateLearningReport } from "../src/scripts/features/learning-run.js";

const attrs = node => Object.fromEntries((node.attrs || []).map(a => [a.name, a.value]));
const walk = (node, predicate) => [...(predicate(node) ? [node] : []), ...(node.childNodes || []).flatMap(child => walk(child, predicate))];
const text = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(text).join("");
const hasClass = (node, name) => attrs(node).class?.split(/\s+/).includes(name);
const byId = (root, id) => walk(root, node => attrs(node).id === id)[0];
const byMarker = (root, marker) => walk(root, node => marker in attrs(node))[0];
const hasAncestor = (node, predicate) => node?.parentNode ? predicate(node.parentNode) || hasAncestor(node.parentNode, predicate) : false;
const built = new Map(await Promise.all(pages.map(async page => [page.file, parse(await readFile(`dist/${page.file}`, "utf8"))])));

test("topic ownership covers every page once and agrees with header navigation", () => {
  const slugs = wikiGroups.flatMap(group => group.pages.map(([slug]) => slug));
  assert.equal(new Set(slugs).size, pages.length);
  assert.deepEqual(slugs.toSorted(), pages.map(page => page.slug).toSorted());
  assert.equal(new Set(site.navigation.map(item => item.href)).size, wikiGroups.length);
  for (const page of pages) {
    const group = wikiGroups.find(group => group.pages.some(([slug]) => slug === page.slug));
    assert.equal(page.nav, group.nav);
    const root = built.get(page.file), context = walk(root, node => hasClass(node, "reading-context"));
    assert.equal(context.length, 1, page.file);
    assert.equal(attrs(context[0])["data-reading-topic"], group.title);
    const nav = walk(root, node => hasClass(node, "site-nav"))[0];
    assert.deepEqual(walk(nav, node => attrs(node)["aria-current"] === "page").map(node => attrs(node).href), [page.nav]);
  }
});

test("built reading routes and all local authored links resolve to pages and anchors", async () => {
  const ids = new Map([...built].map(([file, root]) => [file, new Set(walk(root, node => attrs(node).id).map(node => attrs(node).id))]));
  for (const [file, root] of built) for (const link of walk(root, node => node.tagName === "a" && attrs(node).href)) {
    const url = new URL(attrs(link).href, `http://127.0.0.1:4173/${file}`);
    if (url.origin !== "http://127.0.0.1:4173") continue;
    const target = url.pathname.slice(1) || "index.html";
    if (!ids.has(target)) {
      assert.ok(target.startsWith("assets/experiments/") && "download" in attrs(link), `${file}: ${url.pathname}`);
      await readFile(`dist/${target}`);
      continue;
    }
    if (url.hash) assert.ok(ids.get(target).has(decodeURIComponent(url.hash.slice(1))), `${file}: ${url.pathname}${url.hash}`);
  }
});

test("guide starts with learning and problem lookup; advanced collections are optional", () => {
  const root = built.get("training-engineer-guide.html");
  const ordered = walk(root, node => attrs(node).id).map(node => attrs(node).id);
  assert.ok(ordered.indexOf("guide-learn") < ordered.indexOf("learning-mainline"));
  assert.ok(ordered.indexOf("learning-mainline") < ordered.indexOf("radar-title"));
  assert.ok(ordered.indexOf("radar-title") < ordered.indexOf("exact-mechanics-route"));
  for (const id of ["exact-mechanics-route", "computation-chains-route", "decision-chains-route", "learning-dynamics-route"]) {
    const section = byId(root, id).parentNode.parentNode;
    const collection = walk(section, node => hasClass(node, "exact-route-list"))[0];
    assert.ok(collection, id);
    assert.ok(hasAncestor(collection, node => node.tagName === "details" && hasClass(node, "guide-reference") && !("open" in attrs(node))), id);
  }
});

test("guide and detailed route share six steps with a real learning result in the first four", () => {
  assert.deepEqual(learningPath.slice(0, 4).map(step => step.href), ["index.html", "transformer-lab.html", "training.html", "learning-run.html"]);
  assert.equal(learningPath.length, 6);
  for (const file of ["training-engineer-guide.html", "deep-dive.html"]) {
    const root = built.get(file), path = walk(root, node => hasClass(node, "learning-path"))[0];
    const steps = path.childNodes.filter(node => node.tagName === "li");
    assert.equal(steps.length, 6, file);
    assert.deepEqual(steps.map(node => attrs(walk(node, n => n.tagName === "h3")[0].childNodes.find(n => n.tagName === "a")).href), learningPath.map(step => step.href));
  }
  for (let i = 0; i < learningPath.length; i++) {
    const root = built.get(learningPath[i].href), continuation = walk(root, node => attrs(node)["aria-label"] === "学习主线前后篇")[0];
    assert.ok(continuation, learningPath[i].href);
    const targets = walk(continuation, node => node.tagName === "a").map(node => attrs(node).href);
    if (i) assert.ok(targets.includes(learningPath[i - 1].href));
    assert.ok(targets.includes(i + 1 < learningPath.length ? learningPath[i + 1].href : "deep-dive.html#learning-branches"));
  }
});

test("before/after examples come from the same saved question across every seed and split", () => {
  assert.equal(validateLearningReport(learningRunSnapshot), learningRunSnapshot);
  for (const run of learningRunSnapshot.runs) for (const split of ["train", "validation", "test", "length"]) {
    const root = parse(learningExampleMarkup(learningRunSnapshot, { seed: run.seed, split }));
    const source = text(walk(root, node => hasClass(node, "learning-example-source"))[0]);
    const row = run.learned[split].predictions.find(row => source.includes(`${row.id} ·`));
    assert.ok(row, `${run.seed}/${split}`);
    const initial = run.initial[split].predictions.find(old => old.id === row.id && old.group_id === row.group_id);
    assert.deepEqual(initial.content, row.content); assert.deepEqual(initial.target, row.target);
    const tokens = values => values.map(id => learningRunSnapshot.task.vocab[id]).join(" ");
    assert.deepEqual(walk(root, node => node.tagName === "code").map(text), [tokens(row.content), tokens([...row.target, 3]), tokens(initial.predicted), tokens(row.predicted)]);
  }
});

test("partial or conflicting saved pairs never fabricate a before/after comparison", () => {
  const data = structuredClone(learningRunSnapshot);
  for (const row of data.runs[0].initial.test.predictions) row.group_id += "-different";
  validateLearningReport(data);
  const root = parse(learningExampleMarkup(data));
  assert.equal(walk(root, node => node.tagName === "code").length, 0);
  assert.ok(text(root).includes("没有同一道题"));
});

test("report question identifiers are escaped in the new example", () => {
  const data = structuredClone(learningRunSnapshot);
  for (const phase of ["initial", "learned"]) for (const row of data.runs[0][phase].test.predictions) row.id = '<img src=x onerror="unsafe()">' + row.id;
  validateLearningReport(data);
  const html = learningExampleMarkup(data);
  assert.ok(html.includes("&lt;img"));
  assert.equal(walk(parse(html), node => node.tagName === "img").length, 0);
});

test("report starts with evidence while failure outcomes and all existing controls remain available", () => {
  const root = built.get("learning-run.html"), ids = walk(root, node => attrs(node).id).map(node => attrs(node).id);
  assert.ok(ids.indexOf("lr-report") < ids.indexOf("lr-task"));
  const elements = walk(root, () => true);
  assert.ok(elements.indexOf(byMarker(root, "data-lr-preview")) < elements.indexOf(byMarker(root, "data-lr-chart")));
  assert.ok(!hasAncestor(byMarker(root, "data-lr-outcome"), node => node.tagName === "details"));
  for (const marker of ["data-lr-file", "data-lr-provenance", "data-lr-computation", "data-lr-fit", "data-lr-summary-rows", "data-lr-cache", "data-lr-model", "data-lr-paired"]) assert.ok(byMarker(root, marker), marker);
  const lossRoot = built.get("training-deep.html");
  const ce = walk(lossRoot, node => node.tagName === "details" && text(node).includes("为什么 softmax + 交叉熵"))[0];
  assert.ok(hasAncestor(ce, node => attrs(node)["aria-labelledby"] === "tie-title"));
});
