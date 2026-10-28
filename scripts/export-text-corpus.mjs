import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { parseFragment } from "parse5";

const output = process.argv[2];
if (!output || process.argv.length !== 3) throw new Error("Usage: node scripts/export-text-corpus.mjs <new corpus.json>");
const splits = {
  train: ["index", "training", "training-deep", "transformer-anatomy", "transformer-evolution", "architecture-variants", "data-tokenizer", "packing-correctness", "representation", "representation-deep", "backpropagation", "numerical-computing", "tensor-layout", "optimization-dynamics", "initialization-width", "batch-noise", "scaling-laws", "continual-learning", "posttraining", "posttraining-deep", "finetuning-recipe", "lora-mechanics", "distillation-data", "model-artifacts"],
  validation: ["inference-engine", "position-cache", "decoding-algorithms", "efficient-inference", "quantized-linear", "lab"],
  test: ["model-evaluation", "paired-evaluation", "uncertainty-calibration", "agents-context", "training-environments", "multimodal", "reasoning-compute", "learning-mechanisms"],
};
const hash = text => createHash("sha256").update(text).digest("hex");
const walk = node => [node, ...(node.childNodes || []).flatMap(walk)];
const text = node => node.attrs?.some(a => a.name === "data-math") ? "" : node.nodeName === "#text" ? node.value : (node.childNodes || []).map(text).join("");
const documents = [];
for (const [split, slugs] of Object.entries(splits)) for (const id of slugs) {
  const source_path = `src/content/${id}.html`, bytes = await readFile(source_path);
  const paragraphs = walk(parseFragment(bytes.toString("utf8"))).filter(n => n.tagName === "p")
    .map(text).map(value => value.replace(/\{\{[^}]+\}\}/g, "").replace(/\s+/g, " ").trim())
    .filter(value => [...value].length >= 20 && [...value].length <= 800)
    .map(value => ({ id: hash(value), text: value }));
  documents.push({ id, split, source_path, source_sha256: hash(bytes), paragraphs });
}
const counts = new Map();
for (const doc of documents) for (const p of doc.paragraphs) counts.set(p.id, (counts.get(p.id) || 0) + 1);
let removed = 0;
for (const doc of documents) { removed += doc.paragraphs.filter(p => counts.get(p.id) > 1).length; doc.paragraphs = doc.paragraphs.filter(p => counts.get(p.id) === 1); }
const corpus = { schema: "wiki-natural-text-v1", scope: "Chinese natural-language engineering paragraphs authored in this repository; no external corpus, synthetic story generator, pretrained model or downloaded tokenizer. Narrow-domain document holdout, not a general Chinese language benchmark.",
  transforms: "parse p elements; remove data-math subtrees and unresolved build markers; collapse whitespace; retain 20..800 Unicode code points; remove ALL globally repeated exact normalized paragraphs; never cross document boundaries",
  split_policy: "explicit document lists fixed before tokenizer/training; tokenizer sees train only; exact paragraph hashes unique globally; semantic/near-duplicate contamination is not ruled out",
  removed_repeated_paragraphs: removed, documents };
await writeFile(output, JSON.stringify(corpus, null, 2) + "\n", { flag: "wx" });
console.log(`Frozen ${documents.length} natural-text documents; removed ${removed} duplicate paragraphs.`);
