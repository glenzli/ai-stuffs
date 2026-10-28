import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { parse } from "parse5";
import { experimentNavigation } from "../src/experiment-navigation.mjs";
import { matchesExperiment } from "../src/scripts/features/experiment-navigator.js";
import { experimentStudies, renderExperimentNavigator } from "./render-experiment-navigator.mjs";
import { verifyExperimentSources, experimentReplayPlan, experimentRuntime, renderExperimentSource } from "./experiment-sources.mjs";

const experiments = await verifyExperimentSources();
const html = await readFile("dist/deep-dive.html", "utf8"), root = parse(html);
const attrs = node => Object.fromEntries((node.attrs || []).map(item => [item.name, item.value]));
const walk = (node, predicate) => [...(predicate(node) ? [node] : []), ...(node.childNodes || []).flatMap(child => walk(child, predicate))];
const has = key => node => Object.hasOwn(attrs(node), key);
const text = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(text).join("");
const panel = walk(root, has("data-experiment-navigator-panel"))[0];
const cards = walk(panel, has("data-experiment-study"));

test("navigation covers every accepted source exactly once and enforces ownership", () => {
  assert.equal(experimentStudies(experiments).length, 13);
  assert.deepEqual(experimentNavigation.map(row => row.id).sort(), experiments.map(row => row.id).sort());
  for (const navigation of [experimentNavigation.slice(1), [...experimentNavigation, experimentNavigation[0]], [...experimentNavigation, { id: "unknown" }]]) assert.throws(() => experimentStudies(experiments, navigation), /exactly once/);
  const bad = structuredClone(experimentNavigation); bad[0].href = "index.html";
  assert.throws(() => experimentStudies(experiments, bad), /outside its owner/);
});

test("all primary study links resolve to an owning page and an actual anchor", async () => {
  for (const study of experimentNavigation) {
    const [page, anchor] = study.href.split("#"), destination = parse(await readFile(`dist/${page}`, "utf8"));
    if (anchor) assert.equal(walk(destination, node => attrs(node).id === anchor).length, 1, study.href);
  }
  for (const page of ["training-engineer-guide", "engineering-check"]) assert.ok((await readFile(`dist/${page}.html`, "utf8")).includes('href="deep-dive.html#verified-experiments"'));
});

test("static fallback contains all thirteen studies and exactly thirty-nine pinned downloads", () => {
  assert.equal(cards.length, 13);
  assert.ok(!("open" in attrs(panel)), "optional collection starts collapsed");
  assert.equal(walk(panel, has("data-experiment-controls")).length, 1);
  assert.ok("hidden" in attrs(walk(panel, has("data-experiment-controls"))[0]));
  const downloads = walk(panel, has("download")).map(node => attrs(node).href);
  assert.equal(downloads.length, 39);
  assert.deepEqual(downloads.sort(), experiments.flatMap(item => Object.values(item.assets).map(asset => asset.href)).sort());
  assert.ok(!html.includes("{{experimentNavigator}}"));
  assert.equal(walk(root, node => attrs(node).id === "verified-experiments").length, 1);
  assert.ok(renderExperimentNavigator(experiments).includes('data-experiment-study="padding-inference"'));
});

test("environment labels omit absent packages while retaining recorded values", () => {
  assert.equal(experimentRuntime({ python: "3.12", numpy: "2.3", device: "cpu" }), "Python 3.12 / NumPy 2.3 / 设备 cpu");
  assert.equal(experimentRuntime({ torch: null, numpy: "", threads: 0 }), "线程 0");
  assert.equal(experimentRuntime(), "运行环境未记录");
  assert.equal(experimentRuntime(null), "运行环境未记录");
  const filtering = renderExperimentSource(experiments.find(row => row.id === "retrieval-filtering"));
  assert.ok(filtering.includes("NumPy 2.3.4"));
  assert.doesNotMatch(filtering, /undefined|null|PyTorch/);
  assert.ok(renderExperimentSource({ ...experiments[0], runtime: { python: '<3.12 "test">' } }).includes("&lt;3.12 &quot;test&quot;&gt;"));
});

test("replay recipes use separate output folders, preserve numeric settings and keep original provenance", () => {
  const outputs = [];
  for (const experiment of experiments) {
    const before = JSON.stringify(experiment), recipe = experimentReplayPlan(experiment);
    assert.ok(recipe.command, experiment.id);
    assert.equal(recipe.original, experiment.command);
    assert.equal(JSON.stringify(experiment), before);
    const output = recipe.command.match(/--out-dir (\S+)/)?.[1];
    assert.equal(output, `artifacts/${experiment.id}-verify-run`); outputs.push(output);
    assert.doesNotMatch(recipe.command, /--write-page-data|--snapshot-out|--reanalyze/);
    if (["language-baselines", "matched-budget", "retrieval", "retrieval-comparison", "text-evaluation-audit"].includes(experiment.id)) assert.ok(recipe.command.endsWith(`--only ${experiment.id}`));
    const markup = renderExperimentSource(experiment);
    assert.ok(markup.includes(experiment.command));
    assert.ok(markup.includes(recipe.command));
  }
  assert.equal(new Set(outputs).size, 13);
  assert.match(experimentReplayPlan(experiments[0]).command, /--steps 1200 --seeds 11,29,47 --batch-size 64 --lr 0.003 --threads 1$/);
  assert.ok(experimentReplayPlan(experiments.find(row => row.id === "dynamics")).note.includes("移除"));
});

test("unreviewed or ambiguous commands receive no automatic replay recipe", () => {
  const learning = experiments[0];
  for (const command of [learning.command + "; echo unsafe", learning.command + "\necho unsafe", learning.command.replace("1200", '"1200"'), learning.command + " --snapshot-out src/page.js", learning.command + " --out-dir artifacts/duplicate", learning.command.replace("run-learning-lab", "another-script"), "node scripts/run-learning-lab.mjs --steps 1200"]) {
    assert.equal(experimentReplayPlan({ ...learning, command }).command, null, command);
  }
  const retrieval = experiments.find(row => row.id === "retrieval");
  assert.equal(experimentReplayPlan({ ...retrieval, command: retrieval.command + " --only matched-budget" }).command, null);
});

test("symptom search normalizes width and case, requires all terms and treats punctuation literally", () => {
  const visible = query => cards.filter(card => matchesExperiment(query, attrs(card)["data-experiment-terms"])).map(card => attrs(card)["data-experiment-study"]);
  assert.equal(visible("").length, 13);
  assert.deepEqual(visible("ＰＡＤＤＩＮＧ"), ["padding-inference"]);
  assert.deepEqual(visible("ddp"), ["accumulation"]);
  assert.deepEqual(visible("检索　过滤"), ["retrieval-filtering"]);
  assert.deepEqual(visible("不存在的症状"), []);
  assert.deepEqual(visible(".*"), []);
  assert.equal(matchesExperiment("bGe \n cpu", "BGE / CPU"), true);
});

test("rendered source details distinguish independent recipes from immutable registered examples", async () => {
  for (const experiment of experiments) {
    const card = cards.find(node => attrs(node)["data-experiment-study"] === experiment.id);
    const current = walk(card, node => attrs(node)["aria-label"] === "独立重跑命令");
    const original = walk(card, node => attrs(node)["aria-label"] === "登记的复现命令");
    assert.equal(current.length, 1); assert.equal(original.length, 1);
    assert.equal(text(current[0]), experimentReplayPlan(experiment).command);
    assert.equal(text(original[0]), experiment.command);
  }
  assert.doesNotMatch(html, /PyTorch undefined|NumPy undefined|Python undefined/);
  const entry = walk(root, node => node.tagName === "script" && attrs(node).type === "module")[0];
  const script = await readFile(`dist/${attrs(entry).src}`, "utf8");
  assert.match(script, /experiment-navigator\.[a-f0-9]{10}\.js/);
  const alias = script.match(/initExperimentNavigator as (\w+)/)?.[1];
  assert.ok(alias && script.includes(`${alias}();`));
  const stylesheet = walk(root, node => node.tagName === "link" && attrs(node).rel === "stylesheet")[0];
  assert.ok((await readFile(`dist/${attrs(stylesheet).href}`, "utf8")).includes(".experiment-navigator-grid"));
});
