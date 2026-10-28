import { writeFile } from "node:fs/promises";
import { blockParameters, blockBackward } from "../src/scripts/features/backpropagation.js";
import { initialTrainingState, trainingStep } from "../src/scripts/features/checkpoint-replay.js";
import { packingRecords, packingParameters, packingExperiment } from "../src/scripts/features/packing-correctness.js";
import { loraExperiment } from "../src/scripts/features/lora-mechanics.js";
import { contrastiveCalculation } from "../src/scripts/features/contrastive-distributed.js";

const packing = [];
for (const limit of [32, 10, 6, 3, 1]) for (const supervise of ["all", "last"]) for (const changeA of [false, true]) for (const mode of ["separate", "padded", "packed", "leaky", "double-shift"]) {
  const options = { mode, limit, supervise, changeA }, result = packingExperiment(options);
  packing.push({ options, records: packingRecords(changeA), expected: { loss: result.loss, denominator: result.denominator, gradients: result.gradients, outputs: result.outputs.map(row => ({ key: row.key, logits: row.logits, label: row.label })) } });
}
const lora = [];
for (const initialization of ["zero-b", "both-zero"]) for (const precision of ["full", "int4"]) for (const graph of ["frozen", "detach-base", "detach-all"]) for (const steps of [0, 1, 3, 5]) {
  const options = { initialization, precision, graph, steps }, r = loraExperiment(options);
  lora.push({ options, expected: { output: r.output, loss: r.loss, gradients: r.gradients, A: r.parameters.A, B: r.parameters.B, merged: r.merged, mergedOutput: r.mergedOutput, requantizedOutput: r.requantizedOutput } });
}
const contrastive = [];
for (const temperature of [.5, 1, 2]) for (const mode of ["global", "detached", "micro"]) contrastive.push(contrastiveCalculation({ mode, temperature }));
const state = initialTrainingState(), trajectory = [], checkpoints = [];
for (let index = 0; index < 8; index++) {
  const parameters = structuredClone(state.parameters), row = trainingStep(state);
  trajectory.push({ ...row, gradients: blockBackward({ ...parameters, X: row.input }, row.target).gradients });
  if (index === 2) checkpoints.push(structuredClone(state));
}
const fixture = { schema: 1, block: [0, 1, 2].map(target => ({ target, parameters: blockParameters(), expected: blockBackward(blockParameters(), target) })),
  training: { initial: initialTrainingState(), trajectory, checkpoint: checkpoints[0], final: state }, packingParameters: packingParameters(), packing, lora, contrastive };
if (process.argv.length !== 3) throw new Error("Usage: node scripts/export-framework-fixtures.mjs <new-output.json>");
await writeFile(process.argv[2], `${JSON.stringify(fixture)}\n`, { flag: "wx" });
console.log(`Exported ${packing.length} packing, ${lora.length} LoRA and ${contrastive.length} contrastive references.`);
