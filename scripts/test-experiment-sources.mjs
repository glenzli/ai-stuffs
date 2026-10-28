import test from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, mkdir, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { tmpdir } from "node:os";
import path from "node:path";
import { verifyExperimentSources, learningRunSummaries } from "./experiment-sources.mjs";
import { learningRunSnapshot } from "../src/scripts/features/learning-run-data.js";
import { trainingFailuresSnapshot } from "../src/scripts/features/training-failures-data.js";
import { initializationWidthData } from "../src/scripts/features/initialization-width-data.js";
import { batchNoiseData } from "../src/scripts/features/batch-noise-data.js";
import { chunkedCEReportSnapshot } from "../src/scripts/features/chunked-cross-entropy-data.js";

const manifest = JSON.parse(await readFile("src/experiment-sources.json", "utf8"));
const experiments = await verifyExperimentSources({ manifest });
const sha = bytes => createHash("sha256").update(bytes).digest("hex");

// A display projection may omit rows/fields, but cannot alter any retained value.
function isProjection(view, saved) {
  if (view === null || typeof view !== "object") return Object.is(view, saved);
  if (Array.isArray(view)) {
    if (!Array.isArray(saved)) return false;
    let cursor = 0;
    for (const row of view) {
      while (cursor < saved.length && !isProjection(row, saved[cursor])) cursor++;
      if (cursor === saved.length) return false;
      cursor++;
    }
    return true;
  }
  if (!saved || typeof saved !== "object" || Array.isArray(saved)) return false;
  return Object.entries(view).every(([key, value]) => key === "prediction_total"
    ? value === saved.predictions?.length : key === "selected_for_transfer"
      ? value === Boolean(saved[key]) : key in saved && isProjection(value, saved[key]));
}

test("all five page snapshots are faithful projections of their pinned complete reports", async () => {
  const saved = Object.fromEntries(experiments.map(item => [item.id, item.reportData]));
  assert.deepEqual(learningRunSnapshot, JSON.parse(await readFile(path.join(path.dirname(manifest.experiments.learning.report.path), "page-snapshot.json"), "utf8")));
  assert.ok(isProjection(learningRunSnapshot.runs, saved.learning.runs));
  assert.ok(isProjection(learningRunSnapshot.baselines, saved.learning.baselines));
  assert.deepEqual(trainingFailuresSnapshot, JSON.parse(await readFile(path.join(path.dirname(manifest.experiments.failures.report.path), "snapshot.json"), "utf8")));
  assert.ok(isProjection(initializationWidthData.width, saved.dynamics.width));
  assert.ok(isProjection(initializationWidthData.depth, saved.dynamics.depth));
  assert.ok(isProjection(batchNoiseData.noise, saved.dynamics.noise));
  const { page_prediction_subset, ...batch } = batchNoiseData.batch;
  assert.ok(page_prediction_subset);
  assert.ok(isProjection(batch, saved.dynamics.batch));
  assert.equal(chunkedCEReportSnapshot.measurements.length, saved.output.measurements.length);
  assert.ok(isProjection(chunkedCEReportSnapshot.measurements, saved.output.measurements));
  for (const [snapshot, id] of [[learningRunSnapshot, "learning"], [trainingFailuresSnapshot, "failures"], [initializationWidthData, "dynamics"], [batchNoiseData, "dynamics"], [chunkedCEReportSnapshot, "output"]]) {
    assert.equal(snapshot.provenance.report_sha256, manifest.experiments[id].report.sha256);
  }
});

test("modified measurements cannot satisfy the projection contract", () => {
  const changed = structuredClone(initializationWidthData.width);
  changed.selected_runs[0].coordinates[0].rows[0].rms += .25;
  assert.equal(isProjection(changed, experiments.find(item => item.id === "dynamics").reportData.width), false);
});

test("missing files, changed hashes and wrong page ownership reject the build inputs", async () => {
  const badHash = structuredClone(manifest); badHash.experiments.learning.report.sha256 = "0".repeat(64);
  await assert.rejects(verifyExperimentSources({ manifest: badHash }), /hash mismatch/);
  const missing = structuredClone(manifest); missing.experiments.learning.report.path += ".absent";
  await assert.rejects(verifyExperimentSources({ manifest: missing }), /ENOENT/);
  const badOwner = structuredClone(manifest); badOwner.experiments.learning.pages = ["posttraining"];
  await assert.rejects(verifyExperimentSources({ manifest: badOwner }), /ownership/);
});

test("raw artifact checksum lists still match every saved file", async () => {
  let count = 0;
  for (const experiment of experiments) {
    const directory = path.dirname(experiment.report.path);
    for (const line of (await readFile(experiment.checksums.path, "utf8")).trim().split("\n")) {
      const [digest, file] = line.split("  ");
      assert.equal(sha(await readFile(path.join(directory, file))), digest, file);
      count++;
    }
  }
  assert.ok(count >= 450, `${count} raw experiment files`);
});

test("download archives unpack into fully checksummed raw runs and run-time source", async t => {
  const directory = await mkdtemp(path.join(tmpdir(), "ai-stuffs-bundle-audit-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const run = promisify(execFile);
  for (const experiment of experiments) {
    const destination = path.join(directory, experiment.id);
    await mkdir(destination);
    await run("tar", ["-xzf", path.resolve(experiment.archive.path), "-C", destination]);
    await run("shasum", ["-a", "256", "-c", "SHA256SUMS"], { cwd: path.join(destination, experiment.id) });
    await run("shasum", ["-a", "256", "-c", "SOURCE-SHA256SUMS"], { cwd: destination });
    assert.equal(sha(await readFile(path.join(destination, experiment.id, "report.json"))), experiment.report.sha256);
  }
});

test("built pages expose byte-identical full reports, source archives and checksums", async () => {
  for (const experiment of experiments) {
    for (const asset of Object.values(experiment.assets)) {
      assert.equal(sha(await readFile(`dist/${asset.href}`)), asset.sha256, asset.href);
    }
    for (const page of experiment.pages) {
      const html = await readFile(`dist/${page}.html`, "utf8");
      assert.ok(html.includes('aria-label="本页实测结果的完整来源"'), page);
      for (const asset of Object.values(experiment.assets)) assert.ok(html.includes(`href="${asset.href}" download`), page);
    }
  }
});

test("learning prose is generated from this report's capability results", async () => {
  const summary = learningRunSummaries(experiments);
  const html = await readFile("dist/learning-run.html", "utf8");
  assert.ok(html.includes(summary.intro));
  assert.ok(html.includes(summary.limits));
  assert.ok(html.includes(summary.reportPath));
  assert.ok(!html.includes("seed 47 未达到"));
  assert.ok(summary.limits.includes("长度外推为 0.00%"));
});
