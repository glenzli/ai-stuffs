import { writeFile } from "node:fs/promises";
import { attentionBackwardInputs, tiledAttentionBackward } from "../src/scripts/features/attention-backward.js";
import { gradientCacheInputs, gradientCacheExperiment } from "../src/scripts/features/gradient-cache.js";
import { shardedClipping } from "../src/scripts/features/sharded-clipping.js";

const attention = [], cache = [], clipping = [];
for (const queryBlock of [1, 2, 3, 5]) for (const keyBlock of [1, 2, 3, 5]) for (const mask of ["causal", "full", "empty-first"]) for (const fault of ["none", "local-softmax", "overwrite-kv"]) {
  const options = { queryBlock, keyBlock, mask, fault }, r = tiledAttentionBackward(options);
  attention.push({ options, O: r.O, dQ: r.dQ, dK: r.dK, dV: r.dV, storage: r.storage });
}
for (const chunk of [1, 2, 3, 4]) for (const dropout of [0, .25, .5]) for (const fault of ["none", "fresh-rng"]) {
  const options = { chunk, dropout, fault }, r = gradientCacheExperiment(options);
  cache.push({ options, loss: r.global.loss, gradients: r.actual, reference: r.reference, replayError: r.replayError, gradientError: r.gradientError, storage: r.storage });
}
for (const scenario of ["shards", "mixed", "replicated", "zero", "overflow"]) for (const mode of ["global", "local", "duplicate", "local-skip"]) for (const threshold of [0, 1, 20]) {
  const options = { scenario, mode, threshold }, r = shardedClipping(options);
  clipping.push({ options, norm: r.norm, outputNorm: r.outputNorm, consistentDecision: r.consistentDecision, ranks: r.ranks });
}
await writeFile(process.argv[2], JSON.stringify({ schema: 1, attentionInputs: attentionBackwardInputs(), cacheInputs: gradientCacheInputs(), attention, cache, clipping }, (_, value) => value === Infinity ? "Infinity" : value === -Infinity ? "-Infinity" : value) + "\n", { flag: "wx" });
console.log(`Exported ${attention.length} Attention, ${cache.length} cache and ${clipping.length} clipping configurations.`);
