import assert from "node:assert/strict";
import { test } from "node:test";
import { roundFormat, sumTrace, formatReadout, stabilityExample } from "../src/scripts/features/numerical-computing.js";
import { onlineAttention } from "../src/scripts/features/attention-kernels.js";
const near = (a, b, eps = 1e-12) => assert.ok(Math.abs(a - b) <= eps, `${a} versus ${b}`);

test("nearest-even handles half ties, sign and signed zero", () => {
  assert.equal(roundFormat(1 + 2 ** -11), 1);
  assert.equal(roundFormat(1 + 3 * 2 ** -11), 1 + 2 ** -9);
  assert.equal(roundFormat(-(1 + 2 ** -11)), -1);
  assert.ok(Object.is(roundFormat(-0), -0));
  assert.ok(Object.is(roundFormat(-(2 ** -26)), -0));
});
test("fp16 gradual underflow and overflow boundary", () => {
  assert.equal(roundFormat(2 ** -24), 2 ** -24);
  assert.equal(roundFormat(2 ** -25), 0);
  assert.equal(roundFormat(3 * 2 ** -25), 2 ** -23);
  assert.equal(roundFormat(2 ** -14), 2 ** -14);
  assert.equal(roundFormat(65504), 65504);
  assert.equal(roundFormat(65519), 65504);
  assert.equal(roundFormat(65520), Infinity);
  assert.equal(roundFormat(-65520), -Infinity);
});
test("bf16 matches expected representable spacing", () => {
  assert.equal(roundFormat(256 + 1, "bf16"), 256);
  assert.equal(roundFormat(256 + 3, "bf16"), 260);
  assert.equal(roundFormat(10000, "bf16"), 9984);
  assert.equal(formatReadout(1, "bf16").ulp, 2 ** -7);
});
test("fp32 agrees with native fround across signs, boundaries, mantissas and exponents", () => {
  const values = [0, -0, Number.MIN_VALUE, 1e-45, 2 ** -126, 2 ** -149, 3.4028234663852886e38, 3.4028235677973366e38, 1e40, 2 - Number.EPSILON];
  for(let exp=-150;exp<130;exp+=7) for(const significand of [1,1.00000006,1.23456789,1.9999999]) values.push(significand*2**exp);
  for(const x of values) for(const sign of [1,-1]) assert.ok(Object.is(roundFormat(sign*x,"fp32"),Math.fround(sign*x)), `${sign*x}`);
});
test("format conversion is idempotent and finite errors stay within half a quantum", () => {
  for(const format of ["fp16","bf16","fp32"]) for(const x of [.000001, .123456789,1.0001,17.125,10000,-7.34567]) {
    const r=formatReadout(x,format); assert.equal(roundFormat(r.rounded,format),r.rounded);
    assert.ok(Math.abs(r.error)<=r.ulp/2);
  }
});
test("order determines whether small contribution is lost", () => {
  for(const format of ["fp16","bf16"]) {
    assert.equal(sumTrace([10000,1,-10000],format).result,0);
    assert.equal(sumTrace([10000,-10000,1],format).result,1);
  }
  assert.equal(sumTrace([10000,1,-10000],"fp32").result,1);
});
test("stable softmax is shift invariant when direct exponentials fail", () => {
  const baseline=stabilityExample(0);
  for(const offset of [-1000,0,1000]) {
    const result=stabilityExample(offset); result.stable.forEach((x,i)=>near(x,baseline.stable[i]));
    near(result.stable.reduce((a,b)=>a+b),1); near(result.lse-offset,baseline.lse,1e-10);
    if(offset!==0) assert.ok(result.naive.every(Number.isNaN));
  }
});
test("numeric input contracts reject invalid format or finite readout", () => {
  assert.throws(()=>roundFormat(1,"bad")); assert.throws(()=>roundFormat("1"));
  assert.ok(Number.isNaN(roundFormat(NaN))); assert.equal(roundFormat(Infinity),Infinity);
  assert.throws(()=>formatReadout(NaN)); assert.throws(()=>sumTrace([Infinity])); assert.throws(()=>stabilityExample(1001));
});
test("all block sizes and offsets match full stable attention", () => {
  for(let block=1;block<=6;block++) for(const offset of [-1000,0,1000]) {
    const r=onlineAttention({block,offset}); assert.ok(r.error<1e-12); assert.equal(r.rows.length,Math.ceil(6/block));
    for(const row of r.rows) {
      const prefix=r.scores.slice(0,row.end),m=Math.max(...prefix),e=prefix.map(x=>Math.exp(x-m)),l=e.reduce((a,b)=>a+b);
      near(row.m,m);near(row.l,l);
      for(let j=0;j<2;j++) { const a=e.reduce((s,v,i)=>s+v*r.values[i][j],0);near(row.a[j],a);near(row.output[j],a/l); }
    }
  }
});
test("new maxima rescale previous state and reads are nonoverlapping", () => {
  const r=onlineAttention(); assert.equal(r.rows[0].scale,0);near(r.rows[1].scale,Math.exp(-1));assert.equal(r.rows[2].scale,1);
  assert.deepEqual(r.rows.flatMap(row=>Array.from({length:row.end-row.start},(_,i)=>row.start+i)),[0,1,2,3,4,5]);
  assert.throws(()=>onlineAttention({block:0}));assert.throws(()=>onlineAttention({offset:Infinity}));
});
