import assert from "node:assert/strict";
import { test } from "node:test";
import { servingTrace } from "../src/scripts/features/serving-scheduler.js";

test("all teaching configurations complete within reservations and token budget", () => {
  for (const capacity of [3,4,6,8,12]) for (const budget of [2,4,8,16]) for(const chunk of [1,4,8]) for(const prefix of [false,true]) for(const mode of ["continuous","wave"]) {
    const t=servingTrace({capacity,budget,chunk,prefix,mode});
    assert.equal(t.computed,prefix?18:26); assert.equal(t.reused,prefix?8:0);
    assert.equal(t.requests.reduce((n,r)=>n+r.generated,0),9);
    assert.ok(t.peak<=capacity);
    for(const e of t.events){
      assert.ok(e.work<=budget); assert.ok(e.reserved<=capacity);
      assert.ok(e.blocks.filter(b=>b.owner).length<=e.reserved);
      for(const r of e.requests){
        if(r.status==='active') assert.equal(r.cached,r.generated?r.prompt+r.generated-1:r.processedPrompt);
        if(r.first!==null) assert.ok(r.first>=r.arrival+1);
      }
    }
    assert.equal(t.events.at(-1).blocks.filter(b=>b.owner).length,Number(prefix));
  }
});
test("shared prefix is referenced by active compatible requests only",()=>{
  const t=servingTrace();
  assert.ok(t.events.some(e=>e.blocks[0].references.length===2));
  assert.ok(t.events.every(e=>!e.blocks[0].references.includes('C')));
  assert.deepEqual(t.events.at(-1).blocks[0].references,[]);
});
test("completion releases private blocks and next arrivals reuse physical ids",()=>{
  const t=servingTrace({capacity:3,prefix:false});
  assert.ok(t.requests[1].admitted>=t.requests[0].finish);
  assert.ok(t.events.some(e=>e.blocks.some(p=>p.id===0&&p.owner==='A')));
  assert.ok(t.events.some(e=>e.blocks.some(p=>p.id===0&&p.owner==='B')));
  assert.equal(t.requests[0].cached,11);
});
test("wave mode admits a new group only with an empty active set",()=>{
  const t=servingTrace({capacity:8,prefix:false,mode:'wave'});
  assert.ok(t.requests[1].admitted>=t.requests[0].finish);
  assert.equal(t.requests[1].admitted,t.requests[2].admitted);
});
test("invalid scheduling inputs fail before mutation",()=>{
  for(const config of [{capacity:2},{budget:0},{chunk:0},{mode:'other'},{prefix:'yes'}]) assert.throws(()=>servingTrace(config));
});
