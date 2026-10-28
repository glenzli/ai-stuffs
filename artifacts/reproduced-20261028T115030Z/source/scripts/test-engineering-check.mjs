import assert from "node:assert/strict";
import { test } from "node:test";
import { engineeringQuestions, gradeEngineeringAnswer } from "../src/scripts/features/engineering-check.js";
test("each check has one correct option and a local exploration route",()=>{
  assert.equal(new Set(engineeringQuestions.map(q=>q.id)).size,7);
  for(const q of engineeringQuestions){
    assert.equal(q.options.filter((_,i)=>gradeEngineeringAnswer(q.id,i).correct).length,1);
    assert.match(q.link,/^[a-z-]+\.html#/);assert.ok(q.why.length>20);
  }
});
test("answers match the stated integer accounting",()=>{
  assert.equal(engineeringQuestions[0].options[engineeringQuestions[0].correct],`${Math.ceil(10/3)*3} 次`);
  assert.equal(Number(engineeringQuestions[1].options[engineeringQuestions[1].correct]),3*2*2*2);
  assert.equal(Number(engineeringQuestions[4].options[engineeringQuestions[4].correct]),8+4-1);
});
test("invalid question and answer fail explicitly",()=>{
  assert.throws(()=>gradeEngineeringAnswer('unknown',0));assert.throws(()=>gradeEngineeringAnswer('delivery',-1));
});
