import test from "node:test";
import assert from "node:assert/strict";
import { readFile, writeFile, mkdir, mkdtemp, rm, readdir, symlink } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { hash } from "./experiment-bundles.mjs";
import { experimentPrerequisites, checkExperimentReadiness, parsePreflightArguments, pinnedRequirements } from "./experiment-preflight.mjs";
import { experimentReplayPlan } from "./experiment-sources.mjs";

const manifest = JSON.parse(await readFile("src/experiment-sources.json", "utf8"));
const command = fileURLToPath(new URL("./check-experiment.mjs", import.meta.url));
const referenceProbe = async ({ packages }) => ({ python: "3.12.13", gloo: true, packages: Object.fromEntries(Object.entries(packages).map(([name, version]) => [name, { version }])) });
async function fixture(t, id = "learning") {
  const root = await mkdtemp(path.join(tmpdir(), "ai-stuffs-preflight-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, "src")); await mkdir(path.join(root, "scripts")); await mkdir(path.join(root, "artifacts/reference"), { recursive: true });
  const saved = structuredClone(manifest), entry = saved.experiments[id];
  const report = Buffer.from(JSON.stringify({ schema: entry.schema, status: entry.status })), payload = Buffer.from("fixed tiny input\n");
  const checksums = Buffer.from(`${hash(payload)}  weights.bin\n`);
  entry.report = { path: "artifacts/reference/report.json", sha256: hash(report), bytes: report.length };
  entry.checksums = { path: "artifacts/reference/SHA256SUMS", sha256: hash(checksums), bytes: checksums.length };
  await writeFile(path.join(root, entry.report.path), report);
  await writeFile(path.join(root, entry.checksums.path), checksums);
  await writeFile(path.join(root, "artifacts/reference/weights.bin"), payload);
  await writeFile(path.join(root, "src/experiment-sources.json"), JSON.stringify(saved));
  await writeFile(path.join(root, "scripts/framework-requirements.txt"), "torch==2.9.0\nnumpy==2.3.4\n");
  return { root, saved, entry };
}

test("preflight covers thirteen source groups and identifies actual upstream inputs", () => {
  assert.deepEqual(Object.keys(experimentPrerequisites).sort(), Object.keys(manifest.experiments).sort());
  assert.deepEqual(experimentPrerequisites.practice.inputs, ["learning"]);
  assert.deepEqual(experimentPrerequisites["matched-budget"].inputs, ["practice", "learning"]);
  assert.deepEqual(experimentPrerequisites["text-evaluation-audit"].inputs, ["practice", "language-baselines"]);
  assert.equal(experimentPrerequisites.retrieval.environment, "RETRIEVAL_PYTHON");
  assert.equal(experimentPrerequisites.accumulation.gloo, true);
});

test("CLI rejects unknown, ambiguous and unsafe requests before any probing", () => {
  for (const args of [["--only","unknown"], ["--only"], ["--only","learning","--only","failures"], ["--out-dir","artifacts/a"], ["--only","learning","--out-dir","../old"], ["--only","learning","--out-dir","artifacts/a/../../src"], ["--install"]]) assert.throws(() => parsePreflightArguments(args));
  assert.deepEqual(parsePreflightArguments(["--only","learning","--out-dir","artifacts/new","--json"]), { only: "learning", "out-dir": "artifacts/new", json: true });
});

test("a ready check preserves reference files and does not create its output directory", async t => {
  const { root } = await fixture(t), before = await readFile(path.join(root, "src/experiment-sources.json"));
  const result = await checkExperimentReadiness({ root, only: "learning", outDir: "artifacts/new-run", probe: referenceProbe });
  assert.equal(result.status, "ready"); assert.equal(result.studies.length, 1);
  assert.equal(result.studies[0].checks.find(row => row.code === "output.available").status, "passed");
  assert.equal(result.manifest_sha256, hash(before));
  assert.deepEqual(await readFile(path.join(root, "src/experiment-sources.json")), before);
  assert.deepEqual(await readdir(path.join(root, "artifacts")), ["reference"]);
});

test("missing, mismatched and unloadable packages receive distinct blocking diagnostics", async t => {
  const { root } = await fixture(t);
  for (const [kind, edit] of [["package.missing", value => delete value.packages.torch], ["package.version", value => value.packages.torch.version = "0.0.0"], ["package.import", value => value.packages.torch.importError = "Missing binary library"]]) {
    const result = await checkExperimentReadiness({ root, only: "learning", probe: async args => { const value = await referenceProbe(args); edit(value); return value; } });
    assert.equal(result.status, "blocked");
    assert.equal(result.studies[0].checks.filter(row => row.status === "failed")[0].code, kind);
  }
});

test("Python reference differences are warnings; failed interpreters block readiness", async t => {
  const { root } = await fixture(t);
  const different = await checkExperimentReadiness({ root, only: "learning", probe: async args => ({ ...await referenceProbe(args), python: "3.12.14" }) });
  assert.equal(different.status, "ready-with-warnings");
  assert.equal(different.studies[0].checks.find(row => row.status === "warning").code, "python.reference-differs");
  const blocked = await checkExperimentReadiness({ root, only: "learning", probe: async () => { throw Object.assign(new Error("absent"), { code: "ENOENT" }); } });
  assert.equal(blocked.status, "blocked"); assert.equal(blocked.studies[0].checks[0].code, "python.probe");
});

test("altered, absent and traversing raw input files cannot pass a preflight", async t => {
  const { root, saved, entry } = await fixture(t);
  await writeFile(path.join(root, "artifacts/reference/weights.bin"), "changed input");
  assert.equal((await checkExperimentReadiness({ root, only: "learning", probe: referenceProbe })).status, "blocked");
  await rm(path.join(root, "artifacts/reference/weights.bin"));
  assert.equal((await checkExperimentReadiness({ root, only: "learning", probe: referenceProbe })).status, "blocked");
  const unsafe = Buffer.from(`${"0".repeat(64)}  ../outside\n`);
  await writeFile(path.join(root, entry.checksums.path), unsafe);
  entry.checksums.bytes = unsafe.length; entry.checksums.sha256 = hash(unsafe);
  await writeFile(path.join(root, "src/experiment-sources.json"), JSON.stringify(saved));
  const traversal = await checkExperimentReadiness({ root, only: "learning", probe: referenceProbe });
  assert.ok(traversal.studies[0].checks.some(row => row.status === "failed" && row.detail.includes("Unsafe relative path")));
});

test("existing files, directories and dangling links are not available output locations", async t => {
  const { root } = await fixture(t);
  await writeFile(path.join(root, "artifacts/old-file"), "preserve");
  await symlink("missing-target", path.join(root, "artifacts/old-link"));
  for (const outDir of ["artifacts/reference", "artifacts/old-file", "artifacts/old-link"]) {
    const result = await checkExperimentReadiness({ root, only: "learning", outDir, probe: referenceProbe });
    assert.equal(result.status, "blocked");
    assert.equal(result.studies[0].checks.find(row => row.code === "output.exists").status, "failed");
  }
  assert.equal(await readFile(path.join(root, "artifacts/old-file"), "utf8"), "preserve");
});

test("missing upstream inputs and changed retrieval questions remain blockers even when the selected report is intact", async t => {
  const padding = await fixture(t, "padding-inference");
  const missing = await checkExperimentReadiness({ root: padding.root, only: "padding-inference", probe: referenceProbe });
  assert.equal(missing.status, "blocked");
  assert.equal(missing.studies[0].checks.find(row => row.label === "practice 固定来源").status, "failed");
  assert.equal(missing.studies[0].checks.find(row => row.label === "padding-inference 固定来源").status, "passed");
  const retrieval = await fixture(t, "retrieval");
  await writeFile(path.join(retrieval.root, "scripts/retrieval-requirements.txt"), "numpy==2.3.4\n");
  await mkdir(path.join(retrieval.root, "src/experiments"));
  await writeFile(path.join(retrieval.root, "src/experiments/retrieval-queries.json"), "changed questions");
  const questions = await checkExperimentReadiness({ root: retrieval.root, only: "retrieval", probe: referenceProbe });
  assert.equal(questions.studies[0].checks.find(row => row.code === "input.queries").status, "failed");
});

test("NumPy-only filtering does not request Torch; Gloo capability and environment overrides are explicit", async () => {
  let filteringProbe;
  const filtering = await checkExperimentReadiness({ only: "retrieval-filtering", env: { FRAMEWORK_PYTHON: "/tmp/custom-python" }, probe: async args => { filteringProbe = args; return referenceProbe(args); } });
  assert.equal(filtering.status, "ready");
  assert.deepEqual(Object.keys(filteringProbe.packages), ["numpy"]);
  assert.equal(filteringProbe.executable, "/tmp/custom-python");
  const accumulation = await checkExperimentReadiness({ only: "accumulation", probe: async args => ({ ...await referenceProbe(args), gloo: false }) });
  assert.equal(accumulation.status, "blocked");
  assert.equal(accumulation.studies[0].checks.find(row => row.code === "gloo.build").status, "failed");
});

test("bulk inspection reuses equal probes and checks each source without changing any manifest", async () => {
  let calls = 0;
  const result = await checkExperimentReadiness({ probe: async args => { calls++; return referenceProbe(args); } });
  assert.equal(result.status, "ready"); assert.equal(result.studies.length, 13);
  assert.equal(calls, 4, "framework, dynamics, retrieval and NumPy-only environments");
  assert.equal(result.manifest_sha256, hash(await readFile("src/experiment-sources.json")));
});

test("every source card exposes a preflight for the same source group and output as its replay", async () => {
  for (const entry of Object.values(manifest.experiments)) {
    const recipe = experimentReplayPlan(entry), tokens = recipe.preflightCommand.split(" ");
    assert.equal(tokens.shift(), "node"); assert.equal(tokens.shift(), "scripts/check-experiment.mjs");
    const options = parsePreflightArguments(tokens);
    assert.equal(options.only, entry.id); assert.equal(options["out-dir"], recipe.outDir);
    for (const page of entry.pages.concat("deep-dive")) {
      const html = await readFile(`dist/${page}.html`, "utf8");
      assert.ok(html.includes(recipe.preflightCommand), `${page}: ${entry.id}`);
    }
  }
});

test("requirement pins reject unsupported syntax rather than silently skipping a dependency", async t => {
  const { root } = await fixture(t);
  await writeFile(path.join(root, "scripts/framework-requirements.txt"), "numpy>=2\n");
  await assert.rejects(pinnedRequirements(root, "scripts/framework-requirements.txt"), /Unsupported/);
});

test("real CLI returns structured failure for a missing interpreter without writing outputs", async t => {
  const { root } = await fixture(t);
  let failure;
  try { await promisify(execFile)(process.execPath, [command, "--only", "learning", "--out-dir", "artifacts/new-run", "--json"], { cwd: root, env: { ...process.env, FRAMEWORK_PYTHON: path.join(root, "absent-python") } }); }
  catch (error) { failure = error; }
  assert.equal(failure.code, 1);
  const report = JSON.parse(failure.stdout);
  assert.equal(report.status, "blocked"); assert.equal(report.studies[0].checks[0].code, "python.probe");
  assert.deepEqual(await readdir(path.join(root, "artifacts")), ["reference"]);
});
