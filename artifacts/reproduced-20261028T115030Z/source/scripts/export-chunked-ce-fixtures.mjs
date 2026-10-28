import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { chunkedCEInputs, chunkedCrossEntropy, validateChunkedCEReport } from "../src/scripts/features/chunked-cross-entropy.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const expected = r => ({ loss: r.loss, dH: r.dH, dW: r.dW, db: r.db, validCount: r.validCount, skipped: r.skipped, work: r.work, storage: r.storage });

export function makeChunkedCEFixtures() {
  const inputs = {}, cases = [], fault_cases = [];
  for (const bias of [true, false]) for (const mask of ["all", "mixed", "empty", "ignore-all"]) inputs[`${mask === "all" ? "base" : mask}-${bias ? "bias" : "no-bias"}`] = chunkedCEInputs({ bias, mask });
  inputs["zero-positions"] = { ...chunkedCEInputs(), H: [], labels: [], mask: [] };
  inputs["single-position"] = { ...chunkedCEInputs(), H: [[.2, -.3, .5]], labels: [6], mask: [true] };
  inputs["all-valid"] = { ...chunkedCEInputs(), labels: [0, 2, 3, 6, 4] };
  const add = (input, options, kind = "sweep") => {
    const r = chunkedCrossEntropy({ inputs: inputs[input], snapshots: false, ...options });
    cases.push({ id: `${kind}/${input}/t${options.tokenBlock}/v${options.vocabBlock}/s${options.scoreShift || 0}/u${options.upstream ?? 1}/${cases.length}`, input, options, expected: expected(r) });
  };
  for (const input of Object.keys(inputs).slice(0, 8)) for (const tokenBlock of [1, 2, 3, 4, 5, 8]) for (const vocabBlock of [1, 2, 3, 4, 7, 9]) add(input, { tokenBlock, vocabBlock });
  for (const input of ["base-bias", "mixed-no-bias", "empty-bias", "ignore-all-no-bias", "zero-positions", "single-position", "all-valid"]) for (const upstream of [0, .375, -2.25, 3.5]) add(input, { tokenBlock: 3, vocabBlock: 2, upstream }, "upstream");
  for (const scoreShift of [1e4, -1e4, 1e6, -1e6]) for (const input of ["base-bias", "mixed-no-bias"]) for (const upstream of [1, -2.25]) add(input, { tokenBlock: 3, vocabBlock: 2, scoreShift, upstream }, "stable-shift");
  for (let target = 0; target < 7; target++) {
    const key = `target-${target}`; inputs[key] = chunkedCEInputs({ target });
    for (const tokenBlock of [2, 3]) for (const vocabBlock of [2, 3]) add(key, { tokenBlock, vocabBlock, upstream: -2.25 }, "target-every-shard");
  }
  for (const fault of ["local-softmax", "overwrite-dh"]) for (const upstream of [1, -2.25, 0]) for (const vocabBlock of [2, 7]) {
    const options = { tokenBlock: 3, vocabBlock, upstream, fault }, r = chunkedCrossEntropy({ inputs: inputs["base-bias"], snapshots: false, ...options });
    fault_cases.push({ id: `fault/${fault}/v${vocabBlock}/u${upstream}`, input: "base-bias", options, detectable: vocabBlock < 7 && upstream !== 0, expected: expected(r) });
  }
  return { schema: "chunked-ce-fixtures-v1", scope: "Fixed binary64 CPU teaching matrices; independent Python autograd oracle receives identical H/W/bias/labels/masks/options. No random/model data downloads.", inputs, cases, fault_cases };
}

export async function extractChunkedCESnapshot(reportPath, outputPath) {
  const bytes = await readFile(reportPath), report = validateChunkedCEReport(JSON.parse(bytes));
  const snapshot = structuredClone(report);
  snapshot.provenance.report_sha256 = digest(bytes);
  snapshot.provenance.extraction = "Copied actual saved run values; timings and reference counts unchanged. Per-save payload metadata omitted from this UI snapshot; full data remain in report.json.";
  for (const row of snapshot.measurements) delete row.saved_tensor_storage.saved_tensors;
  const relative = path.relative(root, path.resolve(reportPath));
  await writeFile(outputPath, `// Generated from ${relative}\n// Report SHA256 ${snapshot.provenance.report_sha256}; see full saved run for checks and metadata.\nexport const chunkedCEReportSnapshot = ${JSON.stringify(snapshot, null, 2)};\n`);
  return snapshot;
}

async function main() {
  const args = process.argv.slice(2);
  if (args[0] === "--snapshot" && args.length === 3) {
    await extractChunkedCESnapshot(args[1], args[2]);
    console.log(`Extracted actual saved report snapshot: ${args[2]}`); return;
  }
  if (args.length !== 1) throw new Error("Usage: node scripts/export-chunked-ce-fixtures.mjs <new-reference.json> | --snapshot <report.json> <owned-data.js>");
  const fixture = makeChunkedCEFixtures();
  fixture.provenance = { exporter_sha256: digest(await readFile(fileURLToPath(import.meta.url))), feature_sha256: digest(await readFile(path.join(root, "src/scripts/features/chunked-cross-entropy.js"))), created_at: new Date().toISOString() };
  await writeFile(args[0], JSON.stringify(fixture) + "\n", { flag: "wx" });
  console.log(`Exported ${fixture.cases.length} exact configurations and ${fixture.fault_cases.length} expected-fault cases.`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
