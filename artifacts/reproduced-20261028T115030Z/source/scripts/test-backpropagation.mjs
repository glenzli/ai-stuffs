import assert from "node:assert/strict";
import { test } from "node:test";
import { blockParameters, blockForward, blockBackward, gradientCheck } from "../src/scripts/features/backpropagation.js";

test("all block weights and input gradients match finite differences for all targets", () => {
  for (let target = 0; target < 3; target++) {
    for (const [name, matrix] of Object.entries(blockParameters())) {
      for (let i = 0; i < matrix.flat().length; i++) {
        const result = gradientCheck(name, i, target);
        assert.ok(result.error < 1e-7, `${name}[${i}] target=${target}: ${JSON.stringify(result)}`);
      }
    }
  }
});
test("causal attention excludes future values and unsupervised logits have zero gradient", () => {
  const { forward: f, activationGradients: d } = blockBackward();
  assert.deepEqual(f.P[0], [1, 0]);
  assert.equal(d.S[0][1], 0);
  assert.deepEqual(d.logits[0], [0, 0, 0]);
  assert.ok(Math.hypot(...d.X[0]) > 0);
  assert.ok(Math.abs(d.S[1].reduce((a, b) => a + b)) < 1e-12);
});
test("changing a future input leaves earlier hidden output unchanged", () => {
  const p = blockParameters(), before = blockForward(p);
  p.X[1] = [-2, 3]; const after = blockForward(p);
  assert.deepEqual(before.h2[0], after.h2[0]);
  assert.notDeepEqual(before.h2[1], after.h2[1]);
});
test("small gradient step lowers the supervised loss", () => {
  const p = blockParameters(), result = blockBackward(p);
  for (const name of Object.keys(p)) p[name] = p[name].map((row, i) => row.map((v, j) => v - .01 * result.gradients[name][i][j]));
  assert.ok(blockForward(p).loss < result.forward.loss);
});
test("parameters are isolated and invalid finite difference selections fail", () => {
  const p = blockParameters(); p.q[0][0] = 100;
  assert.equal(blockParameters().q[0][0], .3);
  for (const name of ["missing", "__proto__"]) assert.throws(() => gradientCheck(name));
  assert.throws(() => gradientCheck("q", 4));
  assert.throws(() => gradientCheck("q", 0, 3));
  assert.throws(() => gradientCheck("q", 0, 1, 0));
});
