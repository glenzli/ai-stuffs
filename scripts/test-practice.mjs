import test from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, mkdir, copyFile, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { tmpdir } from "node:os";
import path from "node:path";
import { practiceSnapshot as data } from "../src/scripts/features/practice-data.js";
import { validatePracticeReport, generalizationRows, textProjectView, reproductionRows, profileRows } from "../src/scripts/features/practice-reports.js";
import { pages } from "../src/pages.mjs";
import { learningPath, readingProfile } from "../src/learning.mjs";

const read = async file => JSON.parse(await readFile(file, "utf8"));
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const manifest = await read("src/experiment-sources.json"), entry = manifest.experiments.practice;
const root = path.dirname(entry.report.path), full = await read(entry.report.path);
const cap = rows => [...new Set(rows.map(row => row.length))].flatMap(n => rows.filter(row => row.length === n).slice(0, 6));
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} != ${b}`);

test("practice snapshot preserves raw report identity and all unabridged studies", async () => {
  assert.equal(validatePracticeReport(data), data);
  assert.equal(sha(await readFile(entry.report.path)), data.provenance.report_sha256);
  assert.equal(await readFile(`${root}/page-data.js`, "utf8"), await readFile("src/scripts/features/practice-data.js", "utf8"));
  for (const key of ["text", "reproduction", "profile"]) assert.deepEqual(data.reports[key], full.reports[key]);
  assert.deepEqual(data.provenance.source_sha256, full.provenance.source_sha256);
});

test("saved prediction examples are the declared subset; statistics retain every record", () => {
  const g = full.reports.generalization;
  for (const run of g.baselines) {
    const view = data.reports.generalization.baselines.find(row => row.seed === run.seed);
    for (const key of ["standard", "eos_diagnostic", "shifted_positions"]) {
      assert.deepEqual(view[key], cap(run[key]));
      assert.deepEqual(view.summaries[key], run.summaries[key]);
    }
    for (const stat of view.position_summary) {
      const records = run.teacher_forced.filter(row => row.length === stat.length), content = records.flatMap(row => row.positions.slice(0, -1));
      assert.equal(stat.records, records.length);
      close(stat.teacher_content_accuracy, content.filter(p => p.target === p.top1).length / content.length);
      close(stat.mean_premature_eos_probability, content.reduce((s, p) => s + p.eos_probability, 0) / content.length);
      close(stat.mean_gold_eos_probability, records.reduce((s, r) => s + r.positions.at(-1).eos_probability, 0) / records.length);
    }
  }
  for (const run of g.extended) {
    const view = data.reports.generalization.extended.find(row => row.seed === run.seed);
    assert.deepEqual(view.predictions, cap(run.predictions)); assert.deepEqual(view.summary, run.summary);
  }
});

test("every runtime source and corpus source is the executed, hashed version", async () => {
  for (const [file, hash] of Object.entries(full.provenance.source_sha256)) {
    assert.equal(sha(await readFile(file)), hash, file);
    assert.equal(sha(await readFile(path.join(root, "../source", file))), hash, file);
  }
  const corpus = await read(`${root}/corpus.json`);
  for (const doc of corpus.documents) assert.equal(sha(await readFile(`${root}/corpus-sources/${doc.id}.html`)), doc.source_sha256);
});

test("frozen HTML regenerates the exact saved corpus even after live Wiki edits", async t => {
  const directory = await mkdtemp(path.join(tmpdir(), "ai-stuffs-corpus-")); t.after(() => rm(directory, { recursive: true, force: true }));
  await mkdir(`${directory}/src/content`, { recursive: true });
  const corpus = await read(`${root}/corpus.json`);
  for (const doc of corpus.documents) await copyFile(`${root}/corpus-sources/${doc.id}.html`, `${directory}/${doc.source_path}`);
  await promisify(execFile)(process.execPath, [path.resolve("scripts/export-text-corpus.mjs"), `${directory}/corpus.json`], { cwd: directory });
  assert.deepEqual(await read(`${directory}/corpus.json`), corpus);
});

test("all seeds and policies keep their actual outcomes, including failed interventions", () => {
  for (const seed of [11, 29, 47]) for (const policy of ["standard", "eos_diagnostic", "shifted_positions", "extended"]) {
    const view = generalizationRows(data, seed, policy);
    assert.equal(view.rows.length, 9);
    for (const row of view.predictions) assert.ok(view.original.standard.some(r => r.id === row.id));
    if (["standard", "eos_diagnostic"].includes(policy)) for (const row of view.rows.filter(r => r.length >= 6)) {
      assert.equal(row.exact_match, 0); if (policy === "eos_diagnostic") assert.equal(row.early_eos, 0);
    }
    if (policy === "extended") {
      assert.ok(view.extended.supervised_tokens > view.extended.baseline_supervised_tokens);
      assert.ok(view.rows.filter(r => r.length >= 9).every(r => r.exact_match === 0));
    }
  }
  assert.notEqual(generalizationRows(data, 11, "extended").rows[6].exact_match, generalizationRows(data, 29, "extended").rows[6].exact_match);
});

test("selectors reject absent, prototype or invalid conditions", () => {
  for (const value of ["toString", "__proto__", "lengths"]) assert.throws(() => generalizationRows(data, 11, value), RangeError);
  assert.throws(() => generalizationRows(data, 777), RangeError);
  assert.throws(() => textProjectView(data, 11, "validation+test"), RangeError);
  assert.throws(() => reproductionRows(data, "a made-up OS"), RangeError);
  assert.throws(() => profileRows(data, "GPU"), RangeError);
});

test("validator rejects corrupted metrics, leakage, fabricated history and inconsistent timing", () => {
  for (const mutate of [
    d => d.reports.generalization.baselines[0].summaries.standard[0].exact_match = 1.1,
    d => d.reports.text.tokenizer.training_documents.push("test-document"),
    d => d.reports.text.resume.errors.trace = 1e-6,
    d => d.reports.reproduction.historical_difference.status = "caused by OS",
    d => d.reports.profile.rows[0].stages_ms.total.median += 1,
    d => d.reports.profile.rows[0].stages_ms.total.samples[0] = -1,
  ]) { const changed = structuredClone(data); mutate(changed); assert.throws(() => validatePracticeReport(changed), RangeError); }
});

test("full-step rates, stage sums and stored sample summaries agree", async () => {
  const raw = await read(`${root}/profile/raw-step-times.json`);
  for (const row of data.reports.profile.rows) {
    const samples = raw[row.method]; assert.equal(samples.length, 12);
    for (const sample of samples) close(sample.times_ms.total, Object.entries(sample.times_ms).filter(([key]) => key !== "total").reduce((s, [, v]) => s + v, 0));
    for (const [key, stats] of Object.entries(row.stages_ms)) assert.deepEqual(stats.samples, samples.map(s => s.times_ms[key]));
    close(row.supervised_tokens_per_second, samples.reduce((s, r) => s + r.supervised_tokens, 0) / (samples.reduce((s, r) => s + r.times_ms.total, 0) / 1000));
  }
});

test("four report pages have unique topics, evidence downloads, search entries and route links", async () => {
  assert.equal(learningPath.length, 6);
  const route = await readFile("dist/deep-dive.html", "utf8");
  for (const slug of entry.pages) {
    const page = pages.find(p => p.slug === slug); assert.ok(page); assert.deepEqual(page.features, ["practice-reports"]);
    assert.equal(readingProfile(slug).format, "实测报告");
    const html = await readFile(`dist/${slug}.html`, "utf8");
    assert.ok(html.includes('data-practice-kind=')); assert.ok(html.includes('本页实测结果的完整来源'));
    assert.equal((html.match(/download>/g) || []).length, 3);
    assert.ok(route.includes(`href="${slug}.html"`));
  }
});

test("practice styles use defined theme tokens so SVG labels remain visible in dark mode", async () => {
  const tokens = await readFile("src/styles/tokens.css", "utf8"), css = await readFile("src/styles/practice.css", "utf8");
  const defined = new Set([...tokens.matchAll(/(--[\w-]+)\s*:/g)].map(match => match[1]));
  for (const [, name] of css.matchAll(/var\((--[\w-]+)\)/g)) assert.ok(defined.has(name), `Undefined theme token: ${name}`);
});
