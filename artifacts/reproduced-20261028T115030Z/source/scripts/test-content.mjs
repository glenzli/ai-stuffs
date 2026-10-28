import assert from "node:assert/strict";
import { test } from "node:test";
import { pages } from "../src/pages.mjs";
import { pageEvidence, sources } from "../src/references.mjs";
import { lossReadout } from "../src/scripts/features/training-deep.js";
import { dpoReadout } from "../src/scripts/features/posttraining-deep.js";
import { inferenceState } from "../src/scripts/features/inference-engine.js";
import { negativePool } from "../src/scripts/features/simulator.js";
import { selectionProbabilities } from "../src/scripts/features/reasoning-compute.js";
import { visualTokenBudget } from "../src/scripts/features/multimodal.js";
import { distillationReadout } from "../src/scripts/features/distillation-data.js";
import { quantizeSymmetric } from "../src/scripts/features/efficient-inference.js";
import { curateSamples } from "../src/scripts/features/data-curation.js";
import { wikiGroups } from "../src/wiki.mjs";
import { optimizerLedger, gradientSequences } from "../src/scripts/features/optimization-dynamics.js";
import { reductionExample } from "../src/scripts/features/parallel-correctness.js";
import { gemmRoofline } from "../src/scripts/features/performance-analysis.js";
import { diagnosticCases } from "../src/scripts/features/engineering-cases.js";

const near = (actual, expected, tolerance = 1e-6) => {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
};
const derivative = (fn, x) => (fn(x + 1e-5) - fn(x - 1e-5)) / 2e-5;

test("InfoNCE and SGNS raw-score gradients match finite differences", () => {
  for (const mode of ["infonce", "sgns"]) {
    for (const tau of [0.05, 0.2, 1]) {
      for (const hard of [0.1, 1.4, 2, 2.6]) {
        const scores = [2, hard, 1.05, 0.6, 0.2];
        const result = lossReadout(scores, tau, mode);
        for (let i = 0; i < scores.length; i++) {
          const numeric = derivative((value) => {
            const changed = [...scores];
            changed[i] = value;
            return lossReadout(changed, tau, mode).loss;
          }, scores[i]);
          near(result.gradients[i], numeric);
        }
        assert.ok(result.gradients[0] <= 0);
        assert.ok(result.gradients.slice(1).every((g) => g >= 0));
        if (mode === "infonce") near(result.probabilities.reduce((a, b) => a + b), 1);
      }
    }
  }
});

test("temperature can saturate correct predictions and amplify incorrect ones", () => {
  const correct = [2, 1.4, 1.05, 0.6, 0.2];
  assert.ok(Math.abs(lossReadout(correct, 0.05).gradients[0]) < Math.abs(lossReadout(correct, 0.2).gradients[0]));
  const wrong = [2, 2.6, 1.05, 0.6, 0.2];
  assert.ok(Math.abs(lossReadout(wrong, 0.05).gradients[0]) > Math.abs(lossReadout(wrong, 0.2).gradients[0]));
  near(lossReadout([0, 0, 0], 1).loss, Math.log(3));
  assert.ok(Number.isFinite(lossReadout([-1000, 1000], 0.05).loss));
});

test("DPO derivative is with respect to reference-corrected delta", () => {
  for (const beta of [0.01, 0.1, 0.5, 1]) {
    for (const delta of [-2, 0, 1.2, 3]) {
      near(dpoReadout(beta, delta).gradient, derivative((d) => dpoReadout(beta, d).loss, delta));
    }
    near(dpoReadout(beta, 0).loss, Math.log(2));
    near(dpoReadout(beta, 0).gradient, -beta / 2);
  }
});

test("sampled output is one position ahead of cached generated input", () => {
  assert.deepEqual(inferenceState(1024, 4, 0), { cacheTokens: 0, computedTokens: 0 });
  assert.deepEqual(inferenceState(1024, 4, 1), { cacheTokens: 1024, computedTokens: 4096 });
  assert.deepEqual(inferenceState(1024, 4, 11), { cacheTokens: 1034, computedTokens: 4 });
});

test("clean simulator negatives never include another valid same-class document", () => {
  const docs = [{ classIndex: 0 }, { classIndex: 0 }, { classIndex: 1 }, { classIndex: 2 }];
  assert.deepEqual(negativePool(docs, 0, 0).map((x) => x.index), [2, 3]);
  assert.deepEqual(negativePool(docs, 0, 2).map((x) => x.index), [3]);
});

test("every page has explicit scope and valid primary-source metadata", () => {
  assert.deepEqual(Object.keys(pageEvidence).sort(), pages.map((p) => p.slug).sort());
  for (const page of pages) {
    const evidence = pageEvidence[page.slug];
    assert.ok(evidence.scope && evidence.refs.length);
    for (const key of evidence.refs) {
      assert.ok(sources[key]?.title && sources[key]?.note, `${page.slug}: ${key}`);
      assert.equal(new URL(sources[key].url).protocol, "https:");
    }
  }
});

test("candidate coverage does not equal final selection; single candidate ignores selector", () => {
  for(const p of [.1,.3,.8]) {
    near(selectionProbabilities(p,1,0).selected,p);
    near(selectionProbabilities(p,1,1).selected,p);
    for(const n of [2,4,16]) {
      const result=selectionProbabilities(p,n,.7);
      assert.ok(result.selected<=result.coverage);
      near(selectionProbabilities(p,n,1).selected,result.coverage);
      near(selectionProbabilities(p,n,0).selected,p**n);
    }
  }
});
test("vision budget scales with area, merge and frames",()=>{
  assert.deepEqual(visualTokenBudget(448,14,4,1),{patches:1024,tokens:256});
  assert.equal(visualTokenBudget(896,14,4,1).tokens,1024);
  assert.equal(visualTokenBudget(448,14,4,8).tokens,2048);
});
test("soft labels remain normalized and higher temperature softens fixed logits",()=>{
  for(const t of [1,2,8]) {
    const result=distillationReadout(t);
    near(result.teacher.reduce((a,b)=>a+b),1);
    near(result.student.reduce((a,b)=>a+b),1);
    assert.ok(result.kl>=0);near(result.scaled,t*t*result.kl);
  }
  assert.ok(distillationReadout(8).teacher[0]<distillationReadout(1).teacher[0]);
});
test("symmetric quantization respects range, reconstructs extrema and handles zero",()=>{
  const values=[-.92,-.43,.08,.37,.84,3];
  const result=quantizeSymmetric(values,4);
  assert.ok(result.rows.every(row=>row.integer>=-7 && row.integer<=7));
  near(result.rows.at(-1).restored,3);
  assert.ok(quantizeSymmetric(values,8).mse<result.mse);
  near(quantizeSymmetric([0,0,0],4).mse,0);
});
test("curation records independent score, duplicate and contamination decisions",()=>{
  const filtered=curateSamples(.5,true,true);
  assert.equal(filtered.filter(x=>x.kept).length,4);
  assert.equal(filtered[1].reason,"重复簇");
  assert.equal(filtered[3].reason,"低于阈值");
  assert.equal(filtered[4].reason,"评测隔离");
  assert.ok(curateSamples(.5,true,false)[4].kept);
});
test("knowledge library contains each page exactly once",()=>{
  const slugs=wikiGroups.flatMap(group=>group.pages.map(([slug])=>slug));
  assert.equal(new Set(slugs).size,slugs.length);
  assert.deepEqual([...slugs].sort(),pages.map(page=>page.slug).sort());
});

test("optimizer ledger separates task update, historical state and decoupled decay", () => {
  const sgd = optimizerLedger([2, -1], { method: "sgd", lr: 0.1, decay: 0 });
  near(sgd[0].theta, 0.8); near(sgd[1].theta, 0.9);
  const momentum = optimizerLedger([1, 0], { method: "momentum", lr: 0.1, beta: 0.9, decay: 0 });
  near(momentum[1].m, 0.9); near(momentum[1].theta, 0.81);
  const first = optimizerLedger([2], { method: "adamw", lr: 0.1, beta: 0.9, decay: 0.2 })[0];
  near(first.m, 0.2); near(first.v, 0.04);
  near(first.gradientDelta, -0.1 * 2 / (2 + 1e-8));
  near(first.decayDelta, -0.02);
  for (const method of ["sgd", "momentum", "adamw"]) {
    const rows = optimizerLedger([0, 0, 0], { method, lr: 0.1, decay: 0.2 });
    near(rows.at(-1).theta, 0.98 ** 3);
    assert.ok(rows.every(row => row.gradientDelta === 0));
    const trace = optimizerLedger(gradientSequences.reverse, { method });
    for (const row of trace) near(row.theta - row.before, row.gradientDelta + row.decayDelta);
  }
});

test("global token gradient matches finite differences for every UI count including empty ranks", () => {
  for (let a = 0; a <= 16; a++) for (let b = 0; b <= 16; b++) {
    const result = reductionExample(a, b);
    if (!a && !b) { assert.equal(result.target, null); assert.equal(result.theta, null); continue; }
    const loss = theta => (a * (theta - 2) ** 2 + b * (theta + 1) ** 2) / (2 * (a + b));
    near(result.target, derivative(loss, 0));
    near(result.actual, result.target);
    near(reductionExample(a, b, "double").actual, result.target / 2);
    if (!a || !b) assert.equal(reductionExample(a, b, "mean").actual, null);
  }
  near(reductionExample(2, 8).target, 0.4);
  near(reductionExample(2, 8, "mean").actual, -0.5);
});

test("GEMM roofline obeys work/traffic units and compute/bandwidth bounds", () => {
  const small = gemmRoofline({ m: 1 });
  assert.equal(small.flops, 33554432);
  assert.equal(small.traffic, 33570816);
  assert.equal(small.bound, "带宽");
  const large = gemmRoofline({ m: 4096 });
  assert.equal(large.bound, "算力");
  assert.ok(large.intensity > small.intensity);
  near(large.flops / large.time, large.attainable, 1);
  near(gemmRoofline({ m: 1, bandwidth: 2e12 }).time, small.time / 2);
  near(gemmRoofline({ m: 4096, peak: 200e12 }).time, large.time / 2);
});

test("diagnostic cases provide distinct evidence and real follow-up topics", () => {
  const files = new Set(pages.map(page => page.file));
  assert.equal(Object.keys(diagnosticCases).length, 4);
  for (const item of Object.values(diagnosticCases)) {
    assert.equal(item.options.length, 3);
    assert.equal(item.evidence.length, 3);
    assert.ok(item.conclusion && files.has(item.link.split("#")[0]));
  }
});
