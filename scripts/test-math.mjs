import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { parse, parseFragment } from "parse5";
import { pages } from "../src/pages.mjs";
import { renderMath } from "./render-math.mjs";
import { softenedDistribution } from "../src/scripts/features/distillation-data.js";

const nodes = (node, predicate) => {
  const children = [...(node.childNodes || []), ...(node.content ? [node.content] : [])];
  return (predicate(node) ? [node] : []).concat(children.flatMap(child => nodes(child, predicate)));
};
const attr = (node, name) => node.attrs?.find(item => item.name === name)?.value;

test("typesets fractions and subscripts while preserving unrelated source byte-for-byte", () => {
  const prefix = '<p id="same">a &amp; b</p>\n<code>const x = "$not_math";</code>\n';
  const suffix = '\n<button data-kept="a">Done</button>';
  const tex = String.raw`p_\theta(a\mid x)=\frac{1}{Z}`;
  const result = renderMath(`${prefix}<div data-math class="math-display">${tex}</div>${suffix}`);
  assert.equal(result.count, 1);
  assert.ok(result.html.startsWith(prefix)); assert.ok(result.html.endsWith(suffix));
  const tree = parseFragment(result.html);
  assert.equal(nodes(tree, node => node.tagName === "math").length, 1);
  assert.equal(nodes(tree, node => node.tagName === "mfrac").length, 1);
  const annotation = nodes(tree, node => node.tagName === "annotation")[0];
  assert.equal(annotation.childNodes[0].value, tex);
});

test("renders HTML entities, inline math and hidden variants without breaking attributes", () => {
  const result = renderMath(String.raw`<span data-math="inline">x_{&lt;t}</span><div data-math hidden data-mode="b">\begin{aligned}a&amp;=b\\c&amp;=d\end{aligned}</div>`);
  assert.equal(result.count, 2);
  const tree = parseFragment(result.html);
  const hidden = nodes(tree, node => attr(node, "data-mode") === "b")[0];
  assert.ok(hidden.attrs.some(item => item.name === "hidden"));
  assert.equal(nodes(tree, node => node.tagName === "math" && attr(node, "display") === "block").length, 1);
  assert.ok(nodes(tree, node => node.tagName === "annotation")[0].childNodes[0].value.includes("<t"));
});

test("invalid or ambiguous TeX fails with context instead of silently showing broken math", () => {
  assert.throws(() => renderMath(String.raw`<div data-math>\frac{1}</div>`, "bad.html"), /bad.html:1/);
  assert.throws(() => renderMath('<div data-math><b>x</b></div>'), /plain TeX/);
  assert.throws(() => renderMath('<div data-math>x'), /closed div or span/);
  assert.throws(() => renderMath('<div data-math></div>'), /empty math/);
  assert.throws(() => renderMath('<div data-math="unknown">x</div>'), /invalid math mode/);
  const ordinary = '<pre><code>const p = "x^2";</code></pre>';
  assert.deepEqual(renderMath(ordinary), { html: ordinary, count: 0 });
});

test("every authored formula has HTML and MathML in the generated page", async () => {
  let count = 0, mathPages = 0;
  for (const page of pages) {
    const source = await readFile(`src/content/${page.slug}.html`, "utf8");
    const rendered = renderMath(source, page.file);
    count += rendered.count;
    if (rendered.count) mathPages++;
    const built = await readFile(`dist/${page.file}`, "utf8");
    const tree = parse(built);
    assert.equal(nodes(tree, node => node.attrs?.some(item => item.name === "data-math")).length, rendered.count, page.file);
    assert.equal(nodes(tree, node => node.tagName === "math").length, rendered.count, page.file);
    const mathCss = nodes(tree, node => node.tagName === "link" && attr(node, "href")?.includes("katex.min.css"));
    assert.equal(mathCss.length, Number(rendered.count > 0), page.file);
    for (const css of mathCss) await readFile(`dist/${attr(css, "href")}`);
  }
  assert.ok(count >= 50 && mathPages >= 10, `Unexpected formula coverage: ${count}/${mathPages}`);
});

test("marginal answer example sums paths; score-function gradient does not require reward derivatives", () => {
  const conditional = [0.9, 0.1], logit = 0.4, reward = [2, -1];
  assert.ok(Math.abs(0.6 * conditional[0] + 0.4 * conditional[1] - 0.58) < 1e-12);
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  const objective = theta => sigmoid(theta) * reward[0] + (1 - sigmoid(theta)) * reward[1];
  const numeric = (objective(logit + 1e-5) - objective(logit - 1e-5)) / 2e-5;
  const p = sigmoid(logit);
  const policyGradient = p * reward[0] * (1 - p) + (1 - p) * reward[1] * (-p);
  assert.ok(Math.abs(numeric - policyGradient) < 1e-8);
});

test("distillation derivative includes temperature and the KL direction", () => {
  for (const t of [1, 2, 8]) {
    const teacher = softenedDistribution([4,2,1,0], t);
    const logits = [1.4,1.2,0.7,0.2];
    const student = softenedDistribution(logits, t);
    for (let j = 0; j < logits.length; j++) {
      const objective = value => {
        const changed = [...logits]; changed[j] = value;
        const probs = softenedDistribution(changed, t);
        return t*t*teacher.reduce((sum,p,i) => sum+p*Math.log(p/probs[i]), 0);
      };
      const numeric = (objective(logits[j]+1e-5)-objective(logits[j]-1e-5))/2e-5;
      assert.ok(Math.abs(numeric-t*(student[j]-teacher[j])) < 1e-7);
    }
  }
});

test("fixed-compute scaling stationary point satisfies the derivative and local minimum", () => {
  const A=2, B=3, alpha=0.4, beta=0.3, C=600;
  const optimum = ((alpha*A)/(beta*B)*(C/6)**beta)**(1/(alpha+beta));
  const loss = n => A*n**(-alpha)+B*(6*n/C)**beta;
  const derivative = -alpha*A*optimum**(-alpha-1)+beta*B*(6/C)**beta*optimum**(beta-1);
  assert.ok(Math.abs(derivative) < 1e-10);
  assert.ok(loss(optimum)<loss(optimum*0.9) && loss(optimum)<loss(optimum*1.1));
});
