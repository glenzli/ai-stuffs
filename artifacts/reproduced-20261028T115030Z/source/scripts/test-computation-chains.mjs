import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile, readdir } from "node:fs/promises";
import { parse, parseFragment } from "parse5";
import path from "node:path";
import { pages } from "../src/pages.mjs";
import { ropeRotate, rotaryRelation, cacheConsistency, positionCacheParameters, initPositionCache } from "../src/scripts/features/position-cache.js";
import { scaledGradientWindow, mixedPrecisionRun, initMixedPrecisionStep } from "../src/scripts/features/mixed-precision-step.js";
import { initialTrainingState, runTraining, restoreTrainingState, checkpointExperiment, validateTrainingState, initCheckpointReplay } from "../src/scripts/features/checkpoint-replay.js";
import { blockBackward, blockForward } from "../src/scripts/features/backpropagation.js";
import { fsdpLifecycle, fsdpReference, initFsdpLifecycle } from "../src/scripts/features/fsdp-lifecycle.js";
import { dispatchExample, initMoeDispatch } from "../src/scripts/features/moe-dispatch.js";
import { speculativeMass, speculativeToken, speculativeRound, initSpeculativeProbability } from "../src/scripts/features/speculative-probability.js";

const near = (a, b, tolerance = 1e-11) => assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tolerance, `${a} versus ${b}`);
const vectorNear = (a, b, tolerance) => { assert.equal(a.length, b.length); a.forEach((v,i)=>near(v,b[i],tolerance)); };

test("rotary norm, inverse and relative dot identity hold for both frequencies", () => {
  const input = [.4,-.6,.1,.8];
  for (const base of [100,10000]) for (const m of [-3,0,2,12]) for (const n of [-2,0,5,12]) {
    const relation = rotaryRelation(m,n,base);
    near(relation.beforeNorm,relation.afterNorm); near(relation.product,relation.relative);
    vectorNear(ropeRotate(ropeRotate(input,m,base),-m,base),input);
    const shifted = rotaryRelation(m+4,n+4,base); near(shifted.product,relation.product);
  }
  vectorNear(ropeRotate([1,0,1,0],Math.PI/2,100),[0,1,Math.cos(Math.PI/20),Math.sin(Math.PI/20)]);
});

test("full and incremental logits match across prefix, suffix chunk, padding and frequency settings", () => {
  for (const prefix of [1,2,3,4,5]) for (const chunk of [1,2,3]) for (const padding of [0,1,2,3]) for (const base of [100,10000]) {
    const result=cacheConsistency({prefix,chunk,padding,base});
    assert.equal(result.rows.length,6);assert.equal(result.invalidRows,0);assert.equal(result.firstDivergence,null);near(result.maxError,0);
    assert.equal(result.events.at(-1).cache.length,6+padding);
    for (const row of result.rows) {
      row.allowed.forEach((a,index)=>assert.equal(a,index>=padding&&index-padding<=row.position));
      near(row.probabilities.reduce((a,b)=>a+b,0),1);
    }
    for (const key of result.events.at(-1).cache.filter(k=>k.valid)) assert.equal(key.position,key.rotation);
  }
});

test("wrong positions, double rotation and local masks expose actual numeric differences", () => {
  for (const fault of ["reset","double","mask"]) {
    const result=cacheConsistency({fault});assert.equal(result.firstDivergence,2);assert.ok(result.maxError>1e-3);
  }
  const allMasked=cacheConsistency({padding:2,fault:"mask"});
  assert.ok(allMasked.invalidRows>0);assert.ok(allMasked.rows.some(r=>r.logits===null&&r.error===null));
  assert.deepEqual(cacheConsistency({padding:3}).reference,cacheConsistency().reference);
});

test("cache logits agree with an independently assembled dense causal attention calculation", () => {
  const {inputs,weights}=positionCacheParameters(),base=10000;
  const project=(x,w)=>Array.from({length:w[0].length},(_,j)=>x.map((v,i)=>v*w[i][j]).reduce((a,b)=>a+b,0));
  const rotated=(x,w,position)=>{
    const value=project(x,w),result=[...value];
    for(let j=0;j<4;j+=2){const angle=position/(base**(j/4));result[j]=value[j]*Math.cos(angle)-value[j+1]*Math.sin(angle);result[j+1]=value[j]*Math.sin(angle)+value[j+1]*Math.cos(angle);}
    return result;
  };
  const actual=cacheConsistency().reference;
  inputs.forEach((x,position)=>{
    const q=rotated(x,weights.q,position),keys=inputs.slice(0,position+1).map((v,i)=>rotated(v,weights.k,i));
    const scores=keys.map(k=>k.map((v,i)=>v*q[i]).reduce((a,b)=>a+b,0)/2),ex=scores.map(Math.exp),sum=ex.reduce((a,b)=>a+b,0);
    const context=[0,0,0,0];inputs.slice(0,position+1).forEach((v,i)=>project(v,weights.v).forEach((value,j)=>context[j]+=ex[i]/sum*value));
    vectorNear(actual[position],project(x.map((v,j)=>v+context[j]),weights.out));
  });
});

test("invalid position/cache contracts are rejected", () => {
  for(const fn of [()=>ropeRotate([1],2),()=>ropeRotate([1,Infinity],2),()=>ropeRotate([1,2],0,1),()=>cacheConsistency({prefix:6}),()=>cacheConsistency({chunk:0}),()=>cacheConsistency({padding:-1}),()=>cacheConsistency({fault:"unknown"})]) assert.throws(fn,RangeError);
});

test("unscale-before-clip preserves the declared clipping threshold", () => {
  const correct=scaledGradientWindow();vectorNear(correct.applied,[.6,.8]);near(correct.norm,5);
  const wrong=scaledGradientWindow({order:"clip-first"});vectorNear(wrong.applied,[.075,.1]);near(wrong.norm,40);
  for(const scale of [1,2,8,1024]) vectorNear(scaledGradientWindow({scale}).applied,[.6,.8]);
  const zero=scaledGradientWindow({gradients:[[0,0],[0,0]]});near(zero.coefficient,1);vectorNear(zero.applied,[0,0]);
});

test("changing scale during accumulation weights earlier gradients incorrectly", () => {
  const result=scaledGradientWindow({order:"change-scale"});
  vectorNear(result.unscaled,[2.5,3]);assert.ok(result.error>.03);
  vectorNear(result.rows[0].accumulated,[8,16]);vectorNear(result.rows[1].accumulated,[40,48]);
});

test("overflow skips parameters and schedule while adjusting only the next scale", () => {
  const run=mixedPrecisionRun({mode:"overflow"});
  assert.equal(run.updates,3);assert.ok(run.rows[0].skipped);
  assert.deepEqual(run.rows[0].before,run.rows[0].after);assert.equal(run.rows[0].updates,0);
  near(run.rows[0].nextScale,4);near(run.rows[1].lr,run.rows[0].lr);
  assert.equal(run.rows[1].updates,1);assert.ok(run.rows[2].lr<run.rows[1].lr);
});

test("scaling rescues tiny FP16 contributions but FP32 parameter updates may still round away", () => {
  const gradients=[[2e-8,-1e-8],[2e-8,-1e-8]];
  vectorNear(scaledGradientWindow({gradients,scale:1}).applied,[0,0]);
  assert.ok(scaledGradientWindow({gradients,scale:1024}).applied[0]>0);
  const run=mixedPrecisionRun({mode:"tiny",scale:1024});assert.deepEqual(run.parameters,[.8,-.5].map(Math.fround));
  assert.throws(()=>scaledGradientWindow({scale:0}));assert.throws(()=>scaledGradientWindow({gradients:[]}));assert.throws(()=>mixedPrecisionRun({mode:"invalid"}));
});

test("first CPU training step follows the block gradients and AdamW equation", () => {
  const state=initialTrainingState(), before=structuredClone(state.parameters);
  const [row]=runTraining(state,1), result=blockBackward({...before,X:row.input},row.target);
  near(row.loss,result.forward.loss);
  for(const key of Object.keys(before)) for(let i=0;i<before[key].length;i++) for(let j=0;j<before[key][i].length;j++) {
    const g=result.gradients[key][i][j], expected=before[key][i][j]*(1-.015*.01)-.015*g/(Math.abs(g)+1e-8);
    near(state.parameters[key][i][j],expected);
  }
  assert.equal(state.cursor,1);assert.equal(state.scheduler,1);assert.equal(state.optimizer.t,1);assert.notEqual(state.rng,initialTrainingState().rng);
});

test("training input perturbation still admits independent finite-difference gradients", () => {
  const state=initialTrainingState(), parameters=structuredClone(state.parameters), [row]=runTraining(state,1), p={...parameters,X:row.input};
  const analytic=blockBackward(p,row.target).gradients, epsilon=1e-5;
  for(const key of Object.keys(parameters)) {
    const original=p[key][0][0];p[key][0][0]=original+epsilon;const plus=blockForward(p,row.target).loss;
    p[key][0][0]=original-epsilon;const minus=blockForward(p,row.target).loss;p[key][0][0]=original;
    near(analytic[key][0][0],(plus-minus)/(2*epsilon),2e-8);
  }
});

test("serialized complete checkpoints exactly reproduce uninterrupted trajectories", () => {
  for(let cut=1;cut<8;cut++) {
    const result=checkpointExperiment({cut});assert.equal(result.maxError,0);assert.equal(result.firstDivergence,null);
    assert.deepEqual(result.restoredState,result.baselineState);
    assert.deepEqual(restoreTrainingState(result.checkpoint),result.checkpoint);
  }
});

test("omitted state has distinct observable effects after restoration", () => {
  for(const omission of ["optimizer","scheduler","rng","data","weights-only"]) {
    const result=checkpointExperiment({omission});assert.equal(result.firstDivergence,4);assert.ok(result.maxError>1e-4);
  }
  const optimizer=checkpointExperiment({omission:"optimizer"}).rows[0];
  near(optimizer.baseline.loss,optimizer.resumed.loss);near(optimizer.baseline.gradientNorm,optimizer.resumed.gradientNorm);
  assert.ok(optimizer.difference>0);
  const rng=checkpointExperiment({omission:"rng"}).rows[0];assert.equal(rng.baseline.sample,rng.resumed.sample);assert.notDeepEqual(rng.baseline.input,rng.resumed.input);
  const coincident=checkpointExperiment({cut:4,omission:"data"});assert.equal(coincident.maxError,0);
});

test("checkpoint validation rejects mismatched shape, schema and nonfinite state", () => {
  const bad=initialTrainingState();bad.parameters.q[0].pop();assert.throws(()=>validateTrainingState(bad));
  const nonfinite=initialTrainingState();nonfinite.optimizer.v.q[0][0]=-1;assert.throws(()=>validateTrainingState(nonfinite));
  assert.throws(()=>restoreTrainingState({}));assert.throws(()=>restoreTrainingState(initialTrainingState(),"invalid"));assert.throws(()=>checkpointExperiment({cut:8}));
});

test("FSDP execution policies reconstruct the same gradients and optimizer update", () => {
  for(const reshard of [true,false])for(const prefetch of [true,false]) {
    const r=fsdpLifecycle({reshard,prefetch});near(r.error,0);assert.equal(r.gathers,reshard?4:2);assert.equal(r.scatters,2);
    for(const key of ["W1","W2"]) {vectorNear(r.gradients[key],r.reference.gradients[key]);vectorNear(r.updated[key],r.reference.updated[key]);}
    assert.equal(r.events[0].ranks[0].elements,8);assert.equal(r.events.at(-1).ranks[0].elements,8);
    for(const event of r.events)for(const rank of event.ranks) {
      const count=Object.values(rank.parameters).flat().length+Object.values(rank.momentum).flat().length+Object.values(rank.full).flat().length+Object.values(rank.localGradients).flat().length+Object.values(rank.gradients).flat().length;
      assert.equal(count,rank.elements);assert.ok(event.peak>=rank.elements);
    }
  }
  assert.ok(fsdpLifecycle({reshard:false}).peak>fsdpLifecycle({reshard:true}).peak);
});

test("FSDP averaged gradients match finite differences of global mean loss", () => {
  const p={W1:[.3,-.2,.5,.4],W2:[.6,.1,-.3,.7]},r=fsdpLifecycle(),eps=1e-5;
  for(const key of Object.keys(p))for(let i=0;i<4;i++) {
    const x=p[key][i];p[key][i]=x+eps;const plus=fsdpReference(p).loss;p[key][i]=x-eps;const minus=fsdpReference(p).loss;p[key][i]=x;
    near(r.gradients[key][i],(plus-minus)/(2*eps),1e-9);
  }
});

test("MoE dispatch/combine conserves kept branches and original token identity", () => {
  for(const pattern of ["balanced","skewed"])for(const capacity of [0,1,2,3]) {
    const r=dispatchExample({pattern,capacity});near(r.error,0);assert.equal(r.routes.length,8);
    assert.equal(r.sendCounts.flat().reduce((a,b)=>a+b,0),r.dispatched.length);
    assert.equal(r.received.reduce((a,b)=>a+b,0)+r.dropped,8);
    if(capacity)assert.ok(r.received.every(n=>n<=capacity));
    for(let token=0;token<4;token++) near(r.routes.filter(x=>x.token===token).reduce((s,x)=>s+x.gate,0),1);
  }
  const empty=dispatchExample({pattern:"skewed"});assert.deepEqual(empty.received,[4,4,0,0]);assert.deepEqual(empty.dw.slice(2),[[0,0,0,0],[0,0,0,0]]);
  const dropped=dispatchExample({pattern:"skewed",capacity:1});assert.deepEqual(dropped.output.slice(1),[[0,0],[0,0],[0,0]]);
});

test("MoE input and expert gradients agree with finite differences including capacity drops", () => {
  for(const pattern of ["balanced","skewed"])for(const capacity of [0,1,2]) {
    const input=[[1,.5],[-.3,.8],[.7,-.2],[.1,-.6]],expertWeights=[[.2,.4,-.1,.6],[-.3,.5,.7,.2],[.6,-.2,.1,.8],[.4,.3,-.5,.2]],options={pattern,capacity,input,expertWeights};
    const r=dispatchExample(options),eps=1e-5;
    for(const [matrices,gradients] of [[input,r.dx],[expertWeights,r.dw]])for(let i=0;i<matrices.length;i++)for(let j=0;j<matrices[i].length;j++) {
      const original=matrices[i][j];matrices[i][j]=original+eps;const plus=dispatchExample(options).loss;matrices[i][j]=original-eps;const minus=dispatchExample(options).loss;matrices[i][j]=original;
      near(gradients[i][j],(plus-minus)/(2*eps),1e-9);
    }
  }
});

test("speculative mass restores each target distribution including zeros and disjoint support", () => {
  const distributions=[];for(let a=0;a<=4;a++)for(let b=0;b<=4-a;b++)distributions.push([a/4,b/4,(4-a-b)/4]);
  for(const p of distributions)for(const q of distributions) {
    const r=speculativeMass(p,q);vectorNear(r.output,p);near(r.rejection,1-r.accepted.reduce((a,b)=>a+b,0));
    if(r.correction)near(r.correction.reduce((a,b)=>a+b,0),1);else assert.deepEqual(p,q);
    p.forEach((v,i)=>{if(q[i]===0)assert.equal(r.acceptance[i],null);else near(q[i]*r.acceptance[i],r.accepted[i]);});
  }
});

test("sampling exercises accepted and rejected branches without selecting zero-probability tokens", () => {
  const p=[.5,.3,.2],q=[.2,.5,.3];
  assert.deepEqual(speculativeToken(p,q,{draft:.4,accept:.5,repair:.2}),{candidate:1,accepted:true,token:1});
  assert.deepEqual(speculativeToken(p,q,{draft:.4,accept:.8,repair:.2}),{candidate:1,accepted:false,token:0});
  assert.ok(speculativeToken(p,p,{accept:.999}).accepted);
  assert.equal(speculativeToken([1,0,0],[0,.4,.6],{draft:0}).token,0);
  assert.throws(()=>speculativeMass([.5,.2],[.5,.5]));assert.throws(()=>speculativeToken(p,q,{accept:1}));
});

test("a two-draft round stops at the first rejection or emits an extra target token", () => {
  const p=[.5,.3,.2],q=[.2,.5,.3];
  const first=speculativeRound(p,q,{accepts:[.9,.1]});assert.equal(first.rejectedAt,0);assert.deepEqual(first.emitted,[0]);assert.deepEqual(first.discarded,[2]);
  const second=speculativeRound(p,q,{accepts:[.1,.9]});assert.equal(second.rejectedAt,1);assert.deepEqual(second.emitted,[1,0]);
  const all=speculativeRound(p,q,{accepts:[.1,.1]});assert.equal(all.rejectedAt,null);assert.deepEqual(all.emitted,[1,2,0]);assert.ok(all.extraToken);
  assert.ok(speculativeRound(p,p,{accepts:[.99,.99]}).extraToken);
  assert.throws(()=>speculativeRound(p,q,{accepts:[.1]}));
});

async function mountPage(t, file) {
  const nodes=new Map(), source=await readFile(new URL(`../src/content/${file}.html`,import.meta.url),"utf8");
  function walk(node) {
    const attrs=Object.fromEntries((node.attrs||[]).map(a=>[a.name,a.value]));
    const data=Object.keys(attrs).filter(k=>k.startsWith("data-"));
    if(data.length) {
      const options=(node.childNodes||[]).filter(n=>n.tagName==="option");
      const selected=options.find(n=>n.attrs?.some(a=>a.name==="selected"))||options[0];
      const text=n=>n.nodeName==="#text"?n.value:(n.childNodes||[]).map(text).join("");
      const optionAttrs=Object.fromEntries((selected?.attrs||[]).map(a=>[a.name,a.value]));
      const value=selected?(optionAttrs.value??text(selected)):(attrs.value??"");
      const listeners=new Map(),element={value,checked:Object.hasOwn(attrs,"checked"),textContent:"",innerHTML:"",disabled:false,
        addEventListener(type,fn){listeners.set(type,[...(listeners.get(type)||[]),fn]);},emit(type){for(const fn of listeners.get(type)||[])fn();}};
      for(const key of data)nodes.set(`[${key}]`,element);
    }
    for(const child of node.childNodes||[])walk(child);
  }
  walk(parseFragment(source));
  const old=Object.getOwnPropertyDescriptor(globalThis,"document");
  Object.defineProperty(globalThis,"document",{configurable:true,value:{querySelector:s=>nodes.get(s)||null}});
  t.after(()=>{if(old)Object.defineProperty(globalThis,"document",old);else delete globalThis.document;});
  return key=>{const node=nodes.get(`[data-${key}]`);assert.ok(node,`missing ${key}`);return node;};
}

test("position controls update logits, mask and invalid-row display",async t=>{
  const node=await mountPage(t,"position-cache");initPositionCache();assert.equal(node("rc-first").textContent,"全序列一致");
  node("rc-fault").value="reset";node("rc-fault").emit("input");assert.equal(node("rc-first").textContent,"位置 2");
  node("rc-fault").value="mask";node("rc-padding").value="2";node("rc-padding").emit("input");assert.equal(node("rc-error").textContent,"存在无定义行");
});

test("AMP UI reports skipped updates and wrong-order gradients",async t=>{
  const node=await mountPage(t,"numerical-computing");initMixedPrecisionStep();assert.equal(node("mp-updates").textContent,"4 / 4");
  node("mp-order").value="clip-first";node("mp-order").emit("change");assert.equal(node("mp-applied").textContent,"[0.075, 0.1]");
  node("mp-mode").value="overflow";node("mp-mode").emit("change");assert.equal(node("mp-updates").textContent,"3 / 4");
});

test("recovery UI changes actual trajectory evidence",async t=>{
  const node=await mountPage(t,"model-artifacts");initCheckpointReplay();assert.equal(node("cr-first").textContent,"轨迹一致");
  node("cr-omission").value="optimizer";node("cr-omission").emit("change");assert.equal(node("cr-first").textContent,"第 4 步");
  assert.ok(Number(node("cr-error").textContent)>0);
});

test("FSDP UI steps through state without showing future peak as current",async t=>{
  const node=await mountPage(t,"training-systems-deep");initFsdpLifecycle();assert.equal(node("fl-peak").textContent,"8 元素");
  node("fl-next").emit("click");assert.equal(node("fl-live").textContent,"12 元素");assert.equal(node("fl-collectives").textContent,"1 / 0");
  node("fl-strategy").value="keep";node("fl-strategy").emit("change");assert.equal(node("fl-live").textContent,"8 元素");
});

test("MoE UI updates capacity counts and output rows",async t=>{
  const node=await mountPage(t,"transformer-anatomy");initMoeDispatch();assert.equal(node("md-kept").textContent,"8 / 8");
  node("md-pattern").value="skewed";node("md-capacity").value="1";node("md-capacity").emit("change");assert.equal(node("md-kept").textContent,"2 / 8");assert.equal(node("md-load").textContent,"1 / 1 / 0 / 0");
});

test("speculative UI separates mass calculation from a single trial",async t=>{
  const node=await mountPage(t,"efficient-inference");initSpeculativeProbability();const rows=node("sp-rows").innerHTML;
  node("sp-accept").value="0.9";node("sp-accept").emit("input");assert.equal(node("sp-rows").innerHTML,rows);assert.match(node("sp-trial").textContent,/拒绝/);
  node("sp-scenario").value="equal";node("sp-scenario").emit("change");assert.equal(node("sp-correction").textContent,"不需要修正");
});

test("built feature dependency imports resolve to hashed assets",async()=>{
  const directory=new URL("../dist/assets/js/features/",import.meta.url),files=new Set(await readdir(directory));
  for(const file of files) {
    const text=await readFile(new URL(file,directory),"utf8");
    for(const match of text.matchAll(/from "\.\/([^\"]+)"/g)) {
      assert.ok(/\.[a-f0-9]{10}\.js$/.test(match[1]),`${file}: unversioned ${match[1]}`);
      assert.ok(files.has(match[1]),`${file}: missing ${match[1]}`);
    }
  }
});

test("generated pages preserve local links, anchor targets and labelled headings",async()=>{
  const descendants=n=>[n,...(n.childNodes||[]).flatMap(descendants)],documents=new Map();
  for(const page of pages){
    const nodes=descendants(parse(await readFile(new URL(`../dist/${page.file}`,import.meta.url),"utf8")));
    const ids=nodes.flatMap(n=>(n.attrs||[]).filter(a=>a.name==="id").map(a=>a.value));
    assert.equal(new Set(ids).size,ids.length,`${page.file}: duplicate IDs`);documents.set(page.file,{nodes,ids:new Set(ids)});
  }
  for(const [file,{nodes,ids}] of documents)for(const node of nodes){
    const attrs=Object.fromEntries((node.attrs||[]).map(a=>[a.name,a.value]));
    for(const key of ["href","src"]){
      const value=attrs[key];if(!value||/^(https?:|data:|mailto:)/.test(value))continue;
      const [target,fragment]=value.split("#"),relative=target||file;
      if(target)await readFile(new URL(path.posix.join("../dist/",target),import.meta.url));
      if(fragment&&documents.has(relative))assert.ok(documents.get(relative).ids.has(fragment),`${file}: missing ${value}`);
    }
    if(attrs["aria-labelledby"])for(const id of attrs["aria-labelledby"].split(/\s+/))assert.ok(ids.has(id),`${file}: missing label ${id}`);
  }
});
