import { readFile, mkdir, copyFile, access } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";

const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const esc = value => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
export const experimentOwners = { learning: ["learning-run"], failures: ["training-failures"], dynamics: ["initialization-width", "batch-noise"], output: ["chunked-cross-entropy"], practice: ["generalization-analysis", "text-training-project", "reproducibility-lab", "training-step-profile"], "language-baselines": ["text-training-project"], "matched-budget": ["generalization-analysis"], retrieval: ["retrieval-project"], "retrieval-comparison": ["retrieval-project", "paired-evaluation"], "text-evaluation-audit": ["text-training-project"], accumulation: ["parallel-correctness"] };
experimentOwners["retrieval-filtering"] = ["ann-index"];
experimentOwners["padding-inference"] = ["position-cache"];
const owners = experimentOwners;

// Validate before replacing dist: an absent or modified source must fail the build.
export async function verifyExperimentSources({ root = process.cwd(), manifest } = {}) {
  try { await access(path.join(root, ".cache/experiment-adoption.lock")); throw new Error("An experiment adoption is incomplete. Finish or recover it before building."); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  manifest ??= JSON.parse(await readFile(path.join(root, "src/experiment-sources.json"), "utf8"));
  if (manifest?.schema !== "model-engineering-experiments-v1" || JSON.stringify(Object.keys(manifest.experiments || {}).sort()) !== JSON.stringify(Object.keys(owners).sort())) throw new Error("Incomplete experiment source manifest");
  const experiments = [];
  for (const [id, entry] of Object.entries(manifest.experiments)) {
    if (entry.id !== id || JSON.stringify(entry.pages) !== JSON.stringify(owners[id])) throw new Error(`Invalid experiment ownership: ${id}`);
    const assets = {}; let reportData;
    for (const kind of ["report", "archive", "checksums"]) {
      const file = entry[kind];
      if (!file?.path?.startsWith("artifacts/") || file.path.split("/").includes("..") || !/^[a-f0-9]{64}$/.test(file.sha256)) throw new Error(`Invalid ${kind} source: ${id}`);
      const bytes = await readFile(path.join(root, file.path));
      if (bytes.length !== file.bytes || sha(bytes) !== file.sha256) throw new Error(`Experiment ${kind} hash mismatch: ${file.path}`);
      if (kind === "report") {
        const report = JSON.parse(bytes);
        if (report.schema !== entry.schema || report.status !== entry.status) throw new Error(`Incomplete saved report: ${id}`);
        reportData = report;
      }
      const extension = kind === "report" ? "json" : kind === "archive" ? "tar.gz" : "txt";
      assets[kind] = { ...file, href: `assets/experiments/${id}-${kind}.${file.sha256.slice(0, 10)}.${extension}` };
    }
    if (entry.projection) {
      for (const file of [entry.projection.path, entry.projection.target]) {
        if (file.split("/").includes("..") || path.isAbsolute(file)) throw new Error("Invalid projection path");
        const bytes = await readFile(path.join(root, file));
        if (sha(bytes) !== entry.projection.sha256 || bytes.length !== entry.projection.bytes) throw new Error(`Display projection does not match accepted evidence: ${file}`);
      }
    }
    experiments.push({ ...entry, assets, reportData });
  }
  return experiments;
}

export function learningRunSummaries(experiments) {
  const report = experiments.find(item => item.id === "learning").reportData;
  const count = report.runs.length, fitted = report.runs.filter(run => run.fit_passed).length;
  const range = split => {
    const values = report.runs.map(run => 100 * run.learned[split].metrics.exact_match);
    const min = Math.min(...values).toFixed(2), max = Math.max(...values).toFixed(2);
    return min === max ? `${min}%` : `${min}%…${max}%`;
  };
  return {
    intro: `固定 ${count} 个 seed、每个 ${report.frozen.training.steps.toLocaleString("en-US")} 次更新。本次 ${fitted}/${count} 达到 95% 训练拟合阈值，独立测试完整匹配为 ${range("test")}；长度外推完整匹配为 ${range("length")}。`,
    limits: `本次训练 EM 为 ${range("train")}，独立测试为 ${range("test")}，长度外推为 ${range("length")}。训练拟合、独立测试与更长输入的能力分别报告；配置不因测试或外推结果继续调参。这些结果属于当前保存的运行环境与固定预算。`,
    reportPath: report.provenance.report_path,
  };
}

export async function copyExperimentSources(experiments, outDir, root = process.cwd()) {
  await mkdir(path.join(outDir, "assets/experiments"), { recursive: true });
  for (const experiment of experiments) for (const file of Object.values(experiment.assets)) {
    await copyFile(path.join(root, file.path), path.join(outDir, file.href));
  }
}

export function renderExperimentSources(slug, experiments) {
  return experiments.filter(item => item.pages.includes(slug)).map(experiment => renderSource(experiment)).join("\n");
}
function renderSource(experiment) {
  const { assets, runtime } = experiment;
  return `<aside class="experiment-source" aria-label="本页实测结果的完整来源">
    <p><strong>核验：${esc(experiment.title)}</strong></p>
    <p>图表与表格取自本次保存的完整报告。工件包同时提供运行时源码、配置、环境及对应的 checkpoint、张量或固定输入。</p>
    <div class="step-actions"><a class="button secondary" href="${assets.report.href}" download>下载完整报告</a><a class="button secondary" href="${assets.archive.href}" download>下载源码与原始工件 · ${(assets.archive.bytes / 1048576).toFixed(1)} MiB</a><a href="${assets.checksums.href}" download>文件校验表</a></div>
    <details><summary>报告身份与复现实验</summary>
    <p>${esc(experiment.created_at)} · Python ${esc(runtime.python)} / PyTorch ${esc(runtime.torch)} / NumPy ${esc(runtime.numpy)} · ${esc(runtime.platform)}</p>
    <p>报告 SHA-256：<code>${assets.report.sha256}</code><br>工件包 SHA-256：<code>${assets.archive.sha256}</code></p>
    <p>在仓库根目录使用固定依赖环境，输出到一个尚不存在的新目录：</p><pre class="decision-code" tabindex="0" aria-label="复现实验命令"><code>${esc(experiment.command)}</code></pre>
    <p>解压工件包后，可在实验目录运行 <code>shasum -a 256 -c SHA256SUMS</code>；在解压根目录运行 <code>shasum -a 256 -c SOURCE-SHA256SUMS</code> 核对运行时源码。相同配置在不同系统上可能产生不同浮点轨迹，CPU 耗时也只描述本次环境。</p>
    </details>
  </aside>`;
}
