import { copyFile, cp, mkdir, readFile, rm, readdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { pages, site } from "../src/pages.mjs";
import { searchMetadata } from "../src/search.mjs";
import { pageEvidence, sources } from "../src/references.mjs";
import { wikiGroups, pageLabel, icon } from "../src/wiki.mjs";
import { renderMath } from "./render-math.mjs";
import katex from "katex";
import { renderLearningPath, renderLearningBranches, renderReadingContext, renderLearningContinuation } from "./render-learning.mjs";
import { verifyExperimentSources, copyExperimentSources, renderExperimentSources, learningRunSummaries } from "./experiment-sources.mjs";

const root = process.cwd();
const outDir = path.join(root, "dist");
const globalStyles = ["tokens", "layout", "shared", "reading"];
const featureExports = {
  "accumulation": "initAccumulation",
  "retrieval-comparison": "initRetrievalComparison",
  "text-evaluation-audit": "initTextEvaluationAudit",
  "language-baselines": "initLanguageBaselines",
  "matched-budget": "initMatchedBudget",
  "retrieval-project": "initRetrievalProject",
  "experiment-check": "initExperimentCheck",
  "practice-reports": "initPracticeReports",
  "home": "initHome",
  "representation": "initRepresentation",
  "representation-deep": "initRepresentationDeep",
  "training": "initTraining",
  "training-deep": "initTrainingDeep",
  "posttraining": "initPosttraining",
  "posttraining-deep": "initPosttrainingDeep",
  "retrieval-lab": "initRetrievalLab",
  "engineering": "initEngineering",
  "simulator": "initSimulator",
  "transformer-evolution": "initTransformerEvolution",
  "transformer-anatomy": "initTransformerAnatomy",
  "inference-engine": "initInferenceEngine",
  "data-tokenizer": "initDataTokenizer",
  "data-curation": "initDataCuration",
  "evaluation": "initEvaluation",
  "model-artifacts": "initModelArtifacts",
  "training-systems": "initTrainingSystems",
  "architecture-variants": "initArchitectureVariants",
  "scaling-laws": "initScalingLaws",
  "rl-posttraining": "initRlPosttraining",
  "model-cases": "initModelCases",
  "engineer-guide": "initEngineerGuide",
  "site-index": "initSiteIndex",
  "wiki-shell": "initWikiShell",
  "reasoning-compute": "initReasoningCompute",
  "generation-mechanisms": "initGenerationMechanisms",
  "multimodal": "initMultimodal",
  "agents-context": "initAgentsContext",
  "distillation-data": "initDistillationData",
  "efficient-inference": "initEfficientInference",
  "optimization-dynamics": "initOptimizationDynamics",
  "parallel-correctness": "initParallelCorrectness",
  "performance-analysis": "initPerformanceAnalysis",
  "engineering-cases": "initEngineeringCases",
  "finetuning-recipe": "initFinetuningRecipe",
  "rl-dynamics": "initRlDynamics",
  "on-policy-distillation": "initOnPolicyDistillation",
  "agent-memory": "initAgentMemory",
  "evaluation-reliability": "initEvaluationReliability",
  "multimodal-stream": "initMultimodalStream",
  "uncertainty-calibration": "initUncertaintyCalibration",
  "continual-learning": "initContinualLearning",
  "learning-mechanisms": "initLearningMechanisms",
  "training-environments": "initTrainingEnvironments",
  "reasoning-evidence": "initReasoningEvidence",
  "matrix-optimizer": "initMatrixOptimizer",
  "backpropagation": "initBackpropagation",
  "numerical-computing": "initNumericalComputing",
  "attention-kernels": "initAttentionKernels",
  "distributed-execution": "initDistributedExecution",
  "activation-checkpointing": "initActivationCheckpointing",
  "decoding-algorithms": "initDecodingAlgorithms",
  "data-order": "initDataOrder",
  "bpe-mechanics": "initBpeMechanics",
  "tensor-layout": "initTensorLayout",
  "vocab-parallel": "initVocabParallel",
  "pipeline-schedule": "initPipelineSchedule",
  "serving-scheduler": "initServingScheduler",
  "engineering-check": "initEngineeringCheck",
  "position-cache": "initPositionCache",
  "mixed-precision-step": "initMixedPrecisionStep",
  "checkpoint-replay": "initCheckpointReplay",
  "fsdp-lifecycle": "initFsdpLifecycle",
  "moe-dispatch": "initMoeDispatch",
  "speculative-probability": "initSpeculativeProbability",
  "packing-correctness": "initPackingCorrectness",
  "lora-mechanics": "initLoraMechanics",
  "contrastive-distributed": "initContrastiveDistributed",
  "attention-backward": "initAttentionBackward",
  "gradient-cache": "initGradientCache",
  "sharded-clipping": "initShardedClipping",
  "memory-report": "initMemoryReport",
  "context-parallel": "initContextParallel",
  "rl-update": "initRlUpdate",
  "quantized-linear": "initQuantizedLinear",
  "paired-evaluation": "initPairedEvaluation",
  "ann-index": "initAnnIndex",
  "learning-run": "initLearningRun",
  "training-failures": "initTrainingFailures",
  "initialization-width": "initInitializationWidth",
  "batch-noise": "initBatchNoise",
  "chunked-cross-entropy": "initChunkedCrossEntropy",
};

const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

const read = (file) => readFile(path.join(root, file), "utf8");
const hash = (content) => createHash("sha256").update(content).digest("hex").slice(0, 10);

const layout = await read("src/layouts/base.html");
const coreSource = await read("src/scripts/core.js");
const featureFiles = await readdir(path.join(root, "src/scripts/features"));
const cssCache = new Map();

for (const style of [...new Set(pages.flatMap((page) => page.styles).concat(globalStyles, "overrides", "wiki", "math"))]) {
  cssCache.set(style, await read(`src/styles/${style}.css`));
}

const slugs = new Set();
const renderedContent = new Map();
for (const page of pages) {
  if (slugs.has(page.slug)) throw new Error(`Duplicate page slug: ${page.slug}`);
  slugs.add(page.slug);
  renderedContent.set(page.slug, renderMath(await read(`src/content/${page.slug}.html`), `${page.slug}.html`));
  for (const style of page.styles) {
    if (!cssCache.has(style)) throw new Error(`Missing stylesheet for ${page.slug}: ${style}`);
  }
  for (const feature of page.features) {
    if (!featureExports[feature]) throw new Error(`Missing feature export for ${page.slug}: ${feature}`);
    if (!featureFiles.includes(`${feature}.js`)) throw new Error(`Missing feature module for ${page.slug}: ${feature}`);
  }
}

// Validate authored formulas before replacing the currently served output.
const experimentSources = await verifyExperimentSources();
const learningSummaries = learningRunSummaries(experimentSources);
await rm(outDir, { recursive: true, force: true });
await mkdir(path.join(outDir, "assets/js/features"), { recursive: true });
await mkdir(path.join(outDir, "assets/js/data"), { recursive: true });
await mkdir(path.join(outDir, "assets"), { recursive: true });
await copyExperimentSources(experimentSources, outDir);

await copyFile(path.join(root, "src/assets/favicon.svg"), path.join(outDir, "assets/favicon.svg"));
await copyFile(path.join(root, "src/assets/lucide-LICENSE.txt"), path.join(outDir, "assets/lucide-LICENSE.txt"));
const katexDir = `assets/katex-${katex.version}`;
await mkdir(path.join(outDir, katexDir), { recursive: true });
await copyFile(path.join(root, "node_modules/katex/dist/katex.min.css"), path.join(outDir, katexDir, "katex.min.css"));
await cp(path.join(root, "node_modules/katex/dist/fonts"), path.join(outDir, katexDir, "fonts"), { recursive: true });
await copyFile(path.join(root, "node_modules/katex/LICENSE"), path.join(outDir, katexDir, "LICENSE"));
const coreName = `core.${hash(coreSource)}.js`;
await writeFile(path.join(outDir, "assets/js", coreName), coreSource);
const searchIndex = pages.map((page) => {
  const metadata = searchMetadata[page.file];
  if (!metadata) throw new Error(`Missing search metadata for ${page.file}`);
  return {
    href: page.file,
    title: page.title.replace(` · ${site.name}`, ""),
    summary: metadata.summary,
    terms: metadata.terms,
    priority: metadata.priority,
  };
});
const searchSource = `export const searchIndex = ${JSON.stringify(searchIndex, null, 2)};\n`;
const searchName = `search-index.${hash(searchSource)}.js`;
await writeFile(path.join(outDir, "assets/js/data", searchName), searchSource);
const featureNames = new Map();
const pendingFeatures = new Set();
const reportSharedNames = new Map();
for (const slug of ["experiment-ui", "experiment-view"]) {
  const source = await read(`src/scripts/${slug}.js`), name = `${slug}.${hash(source)}.js`;
  reportSharedNames.set(slug, name);
  await writeFile(path.join(outDir, "assets/js", name), source);
}
async function writeFeature(slug) {
  if (featureNames.has(slug)) return featureNames.get(slug);
  if (pendingFeatures.has(slug)) throw new Error(`Circular feature dependency: ${slug}`);
  pendingFeatures.add(slug);
  let versioned = (await read(`src/scripts/features/${slug}.js`))
    .replaceAll('from "../core.js"', `from "../${coreName}"`)
    .replaceAll('from "../data/search-index.js"', `from "../data/${searchName}"`);
  for (const [shared, name] of reportSharedNames) {
    for (const quote of ['"', "'"]) versioned = versioned.replaceAll(`from ${quote}../${shared}.js${quote}`, `from "../${name}"`);
  }
  for (const match of [...versioned.matchAll(/from (["'])\.\/([\w-]+)\.js\1/g)]) {
    const dependency = await writeFeature(match[2]);
    versioned = versioned.replaceAll(match[0], `from "./${dependency}"`);
  }
  const name = `${slug}.${hash(versioned)}.js`;
  featureNames.set(slug, name);
  pendingFeatures.delete(slug);
  await writeFile(path.join(outDir, "assets/js/features", name), versioned);
  return name;
}
for (const file of featureFiles) await writeFeature(file.replace(/\.js$/, ""));

let equationCount = 0;
for (const page of pages) {
  const math = renderedContent.get(page.slug);
  const content = math.html
    .replaceAll("{{learningPath}}", renderLearningPath())
    .replaceAll("{{learningPathDetailed}}", renderLearningPath(true))
    .replaceAll("{{learningBranches}}", renderLearningBranches())
    .replaceAll("{{experimentSources}}", renderExperimentSources(page.slug, experimentSources))
    .replaceAll("{{learningRunSummary}}", escapeHtml(learningSummaries.intro))
    .replaceAll("{{learningRunLimitSummary}}", escapeHtml(learningSummaries.limits))
    .replaceAll("{{learningReportPath}}", escapeHtml(learningSummaries.reportPath));
  equationCount += math.count;
  const evidence = pageEvidence[page.slug];
  if (!evidence) throw new Error(`Missing evidence metadata: ${page.slug}`);
  const sourceLinks = evidence.refs.map((key) => {
    const source = sources[key];
    if (!source) throw new Error(`Unknown source: ${key}`);
    return `<li><a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.title)}</a><span>${escapeHtml(source.note)}</span></li>`;
  }).join("\n");
  const evidenceHtml = `<section class="page-evidence" aria-labelledby="page-evidence-title" id="page-evidence">
    <h2 id="page-evidence-title">适用范围与一手资料</h2>
    <p>${escapeHtml(evidence.scope)}</p>
    <details><summary>公式与机制的原始依据 · ${evidence.refs.length} 项</summary><ul>${sourceLinks}</ul></details>
  </section>`;
  const activeNavigation = site.navigation.map((item) =>
    `<a href="${item.href}"${item.href === page.nav ? ' aria-current="page"' : ""}>${escapeHtml(item.label)}</a>`,
  ).join("\n");
  const library = wikiGroups.map((group) => `<div class="wiki-nav-group"><h2>${escapeHtml(group.title)}</h2>${group.pages.map(([slug, label]) => {
    const target = pages.find(item => item.slug === slug);
    if (!target) throw new Error(`Missing library page: ${slug}`);
    return `<a href="${target.file}"${slug === page.slug ? ' aria-current="page"' : ""}>${escapeHtml(label)}</a>`;
  }).join("\n")}</div>`).join("\n");
  const footer = page.footer.map((item) => `<p>${item}</p>`).join("\n");
  const styles = [...globalStyles, ...page.styles, "overrides", "wiki", "math"];
  const css = styles.map((name) => `/* ${name}.css */\n${cssCache.get(name)}`).join("\n");
  const cssName = `${page.slug}.${hash(css)}.css`;
  await writeFile(path.join(outDir, "assets", cssName), css);

  const imports = [`import { initCore } from "./js/${coreName}";`];
  const calls = ["initCore();"];
  ["wiki-shell", ...page.features].forEach((feature, index) => {
    const exportName = featureExports[feature];
    if (!exportName) throw new Error(`Unknown feature: ${feature}`);
    const featureName = featureNames.get(feature);
    if (!featureName) throw new Error(`Missing feature module: ${feature}`);
    imports.push(`import { ${exportName} as initFeature${index} } from "./js/features/${featureName}";`);
    calls.push(`initFeature${index}();`);
  });
  const entry = `${imports.join("\n")}\n\n${calls.join("\n")}\ndocument.documentElement.dataset.wikiReady = "true";\n`;
  const jsName = `${page.slug}.${hash(entry)}.js`;
  await writeFile(path.join(outDir, "assets", jsName), entry);

  const html = layout
    .replaceAll("{{title}}", escapeHtml(page.title))
    .replaceAll("{{description}}", escapeHtml(page.description))
    .replaceAll("{{bodyPage}}", escapeHtml(page.bodyPage))
    .replaceAll("{{navigation}}", activeNavigation)
    .replaceAll("{{library}}", library)
    .replaceAll("{{pageLabel}}", escapeHtml(pageLabel(page.slug)))
    .replaceAll("{{pageCount}}", String(pages.length))
    .replaceAll("{{iconPanel}}", icon("panel"))
    .replaceAll("{{iconSearch}}", icon("search"))
    .replaceAll("{{iconTheme}}", icon("theme"))
    .replaceAll("{{iconClose}}", icon("close"))
    .replaceAll("{{iconUp}}", icon("up"))
    .replaceAll("{{footer}}", footer)
    .replaceAll("{{content}}", content)
    .replaceAll("{{readingContext}}", renderReadingContext(page.slug))
    .replaceAll("{{learningContinuation}}", renderLearningContinuation(page.slug))
    .replaceAll("{{mathStyles}}", math.count ? `<link rel="stylesheet" href="${katexDir}/katex.min.css">` : "")
    .replaceAll("{{evidence}}", evidenceHtml)
    .replaceAll("{{cssPath}}", `assets/${cssName}`)
    .replaceAll("{{jsPath}}", `assets/${jsName}`);
  await writeFile(path.join(outDir, page.file), html);
}

const baseUrl = process.env.SITE_URL || site.url;
const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${pages.map((page) => `  <url><loc>${new URL(page.file, baseUrl).href}</loc></url>`).join("\n")}
</urlset>
`;
await writeFile(path.join(outDir, "sitemap.xml"), sitemap);
await writeFile(path.join(outDir, "robots.txt"), `User-agent: *\nAllow: /\nSitemap: ${new URL("sitemap.xml", baseUrl).href}\n`);

console.log(`Built ${pages.length} pages and ${equationCount} equations into dist/.`);
