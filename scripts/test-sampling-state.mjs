import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { parse } from "parse5";
import { samplingSchedule, samplingStateComparison, samplingLedgerMarkup, samplingOrders, initSamplingState } from "../src/scripts/features/sampling-state.js";

const key = row => `${row.request}:${row.step}`;
const rows = result => result.continuous.map(row => [row.request, row.step, row.u, row.token]);

test("the CBA counterexample follows the published tape and independently hand-calculated CDF intervals", () => {
  const result = samplingStateComparison();
  assert.deepEqual(rows(result), [["C",1,.12,"y"],["B",1,.87,"z"],["A",1,.34,"y"],["C",2,.66,"z"],["B",2,.49,"x"],["A",2,.91,"z"],["C",3,.23,"y"],["B",3,.72,"y"],["A",3,.05,"x"]]);
  assert.equal(result.allocationDifferences, 6); assert.equal(result.tokenDifferences, 4);
  assert.equal(result.firstRecoveryDifference, null);
});

test("ABC without early completion deliberately aligns all three allocation policies", () => {
  const reference = samplingStateComparison({ order: "ABC", allocation: "request" });
  for (const allocation of ["global","slot","request"]) {
    const result = samplingStateComparison({ order: "ABC", allocation });
    assert.deepEqual(rows(result), rows(reference));
    assert.equal(result.allocationDifferences, 0);
  }
});

test("request-owned tapes preserve every common request step across all permutations and completion budgets", () => {
  for (const order of samplingOrders) for (const bSteps of [1,3]) {
    const result = samplingStateComparison({ order, bSteps, allocation: "request" });
    assert.equal(result.allocationDifferences, 0); assert.equal(result.tokenDifferences, 0);
    assert.equal(result.continuous.length, 6 + bSteps);
    assert.equal(new Set(result.continuous.map(key)).size, 6 + bSteps);
    assert.ok(result.continuous.filter(row => row.request === "B").every(row => row.step <= bSteps));
  }
});

test("slot compaction moves C onto the B slot while leaving A's path intact", () => {
  const result = samplingStateComparison({ order: "ABC", allocation: "slot", bSteps: 1 });
  const secondC = result.allocationRows.find(row => key(row) === "C:2"), thirdC = result.allocationRows.find(row => key(row) === "C:3");
  assert.equal(secondC.slot, 1); assert.equal(secondC.stream, "slot:1");
  assert.equal(secondC.u, .49); assert.equal(secondC.referenceU, .91);
  assert.equal(thirdC.u, .72); assert.equal(thirdC.referenceU, .05);
  assert.ok(result.allocationRows.filter(row => row.request === "A").every(row => row.sameU && row.sameToken));
  assert.equal(result.allocationDifferences, 2); assert.equal(result.tokenDifferences, 1);
});

test("all 288 declared combinations preserve prefixes and recover exact traces from serialized cursors", () => {
  let cases = 0;
  for (const order of samplingOrders) for (const allocation of ["global","slot","request"]) for (const bSteps of [1,3]) {
    const total = 6 + bSteps;
    for (const pauseAfter of [0,1,4,total]) for (const recovery of ["state","reset"]) {
      const result = samplingStateComparison({ order, allocation, bSteps, pauseAfter, recovery }); cases++;
      assert.deepEqual(result.resumed.slice(0,pauseAfter), result.continuous.slice(0,pauseAfter));
      assert.equal(result.snapshot.nextEvent, pauseAfter);
      assert.deepEqual(JSON.parse(JSON.stringify(result.snapshot)), result.snapshot);
      assert.ok(result.continuous.every(row => row.u >= 0 && row.u < 1 && Number.isInteger(row.index)));
      if (recovery === "state" || pauseAfter === 0 || pauseAfter === total) {
        assert.deepEqual(result.resumed, result.continuous);
        assert.equal(result.firstRecoveryDifference, null);
      }
    }
  }
  assert.equal(cases, 288);
});

test("a reset cursor can first change the uniform while leaving its sampled token unchanged", () => {
  const result = samplingStateComparison({ recovery: "reset" });
  assert.deepEqual(result.snapshot.cursors, { global: 4, requests: { A: 0, B: 0, C: 0 }, slots: [0,0,0] });
  const first = result.firstRecoveryDifference;
  assert.equal(key(first), "B:2"); assert.equal(first.u, .12); assert.equal(first.referenceU, .49);
  assert.equal(first.sameToken, true); assert.equal(first.sameU, false);
  assert.notDeepEqual(result.resumed, result.continuous);
});

test("returned snapshots and rows cannot mutate a later calculation", () => {
  const original = samplingStateComparison(), changed = samplingStateComparison();
  changed.snapshot.cursors.global = 99; changed.continuous[0].u = .99; changed.resumed[0].token = "tampered";
  assert.deepEqual(samplingStateComparison(), original);
});

test("invalid controls reject calculations and markup marks the no-future-draw boundary", () => {
  for (const spec of [{order:"AAA"},{order:"AB"},{allocation:"other"},{bSteps:2},{pauseAfter:-1},{pauseAfter:NaN},{pauseAfter:10},{pauseAfter:.5},{recovery:"guess"}]) assert.throws(() => samplingStateComparison(spec), RangeError);
  assert.throws(() => samplingSchedule("ABC",0), RangeError);
  const html = samplingLedgerMarkup(samplingStateComparison({pauseAfter:9,recovery:"reset"}));
  assert.ok(html.includes("无后续取数")); assert.ok(html.includes("没有后续随机数可比较"));
  assert.doesNotThrow(initSamplingState);
});

test("built page preserves static inputs, scopes the optional feature, and links the related lessons", async () => {
  const html = await readFile("dist/decoding-algorithms.html","utf8"), root = parse(html);
  const attrs = node => Object.fromEntries((node.attrs||[]).map(item=>[item.name,item.value]));
  const walk = (node, predicate) => [...(predicate(node)?[node]:[]),...(node.childNodes||[]).flatMap(child=>walk(child,predicate))];
  const panel = walk(root,node=>Object.hasOwn(attrs(node),"data-sampling-state"))[0];
  assert.ok(panel); assert.ok(!Object.hasOwn(attrs(panel),"open"));
  const controls = walk(panel,node=>Object.hasOwn(attrs(node),"data-sampling-controls"))[0];
  assert.ok(Object.hasOwn(attrs(controls),"disabled"));
  assert.equal(walk(panel,node=>node.tagName==="select").length,5);
  assert.equal(walk(root,node=>attrs(node).id==="sampling-state").length,1);
  assert.ok(html.includes("0.12, 0.87, 0.34, 0.66, 0.49, 0.91, 0.23, 0.72, 0.05"));
  assert.ok(html.includes("不是某个框架 seed 的实际输出"));
  const entry = attrs(walk(root,node=>node.tagName==="script"&&attrs(node).type==="module")[0]).src;
  const script = await readFile(`dist/${entry}`,"utf8");
  assert.match(script,/sampling-state\.[a-f0-9]{10}\.js/);
  const alias=script.match(/initSamplingState as (\w+)/)?.[1];assert.ok(alias&&script.includes(`${alias}();`));
  for(const page of ["reproducibility-lab","inference-engine"])assert.ok((await readFile(`dist/${page}.html`,"utf8")).includes('href="decoding-algorithms.html#sampling-state"'));
});
