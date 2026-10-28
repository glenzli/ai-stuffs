// Readiness diagnostics own no experiment outputs or adoption state.
import { readFile, lstat, access, constants } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";
import { checkAsset, checkChecksumFile, hash } from "./experiment-bundles.mjs";

const framework = "scripts/framework-requirements.txt";
export const experimentPrerequisites = {
  learning: { requirements: framework, inputs: [] },
  failures: { requirements: framework, inputs: [] },
  dynamics: { requirements: "scripts/dynamics-requirements.txt", inputs: [] },
  output: { requirements: framework, inputs: [] },
  practice: { requirements: framework, inputs: ["learning"] },
  "language-baselines": { requirements: framework, inputs: ["practice"] },
  "matched-budget": { requirements: framework, inputs: ["practice", "learning"] },
  retrieval: { requirements: "scripts/retrieval-requirements.txt", inputs: ["practice"], environment: "RETRIEVAL_PYTHON" },
  "retrieval-comparison": { requirements: framework, inputs: ["retrieval"] },
  "text-evaluation-audit": { requirements: framework, inputs: ["practice", "language-baselines"] },
  accumulation: { requirements: framework, inputs: ["practice"], gloo: true },
  "retrieval-filtering": { requirements: "scripts/retrieval-filtering-requirements.txt", inputs: ["retrieval"] },
  "padding-inference": { requirements: framework, inputs: ["practice"] },
};
const moduleNames = { torch: "torch", numpy: "numpy", mup: "mup", PyYAML: "yaml", "faiss-cpu": "faiss", transformers: "transformers", tokenizers: "tokenizers", safetensors: "safetensors", "huggingface-hub": "huggingface_hub" };
const pythonProbe = `import contextlib, importlib, importlib.metadata, json, platform, sys
requested = json.loads(sys.argv[1])
result = {"python": platform.python_version(), "packages": {}, "gloo": False}
for distribution, module in requested.items():
    entry = {}
    try:
        entry["version"] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        entry["version"] = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            loaded = importlib.import_module(module)
        if distribution == "torch":
            result["gloo"] = bool(loaded.distributed.is_available() and loaded.distributed.is_gloo_available())
    except Exception as error:
        entry["importError"] = type(error).__name__ + ": " + str(error)
    result["packages"][distribution] = entry
print(json.dumps(result))
`;

export async function probePython({ executable, packages, root, env }) {
  const requested = Object.fromEntries(Object.keys(packages).map(name => {
    if (!moduleNames[name]) throw new Error(`Unreviewed probe package: ${name}`);
    return [name, moduleNames[name]];
  }));
  const { stdout } = await promisify(execFile)(executable, ["-I", "-B", "-c", pythonProbe, JSON.stringify(requested)], {
    cwd: root, timeout: 20000, maxBuffer: 2 * 1024 * 1024,
    env: { ...env, HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", TOKENIZERS_PARALLELISM: "false" },
  });
  return JSON.parse(stdout);
}

export async function pinnedRequirements(root, file) {
  const packages = {};
  for (const raw of (await readFile(path.join(root, file), "utf8")).split("\n")) {
    const line = raw.trim(); if (!line || line.startsWith("#")) continue;
    const match = /^([A-Za-z0-9-]+)==([\w.+-]+)$/.exec(line);
    if (!match || Object.hasOwn(packages, match[1]) || !moduleNames[match[1]]) throw new Error(`Unsupported pinned requirement in ${file}: ${line}`);
    packages[match[1]] = match[2];
  }
  if (!Object.keys(packages).length) throw new Error(`Empty pinned requirements: ${file}`);
  return packages;
}

export function parsePreflightArguments(args) {
  const options = {}, seen = new Set();
  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    if (seen.has(flag)) throw new Error(`重复参数：${flag}`); seen.add(flag);
    if (["--json", "--help"].includes(flag)) options[flag.slice(2)] = true;
    else if (["--only", "--out-dir"].includes(flag) && args[i + 1] && !args[i + 1].startsWith("--")) options[flag.slice(2)] = args[++i];
    else throw new Error(`未知参数或缺少值：${flag}`);
  }
  if (options.only && !Object.hasOwn(experimentPrerequisites, options.only)) throw new Error(`未知来源组：${options.only}`);
  if (options["out-dir"] && !options.only) throw new Error("检查输出目录时须用 --only 指定一组实验。");
  if (options["out-dir"] && !/^artifacts\/[\w-]+$/.test(options["out-dir"])) throw new Error("输出目录须为 artifacts/<新目录名>。");
  return options;
}

export async function checkExperimentReadiness({ root = process.cwd(), only, outDir, env = process.env, probe = probePython } = {}) {
  root = path.resolve(root);
  // Validate API callers as strictly as CLI callers before probing an interpreter.
  parsePreflightArguments([...(only ? ["--only", only] : []), ...(outDir ? ["--out-dir", outDir] : [])]);
  const manifestBytes = await readFile(path.join(root, "src/experiment-sources.json"));
  const manifest = JSON.parse(manifestBytes);
  if (manifest.schema !== "model-engineering-experiments-v1" || JSON.stringify(Object.keys(manifest.experiments || {}).sort()) !== JSON.stringify(Object.keys(experimentPrerequisites).sort())) throw new Error("来源清单与复现前检查的覆盖范围不一致。");
  const sourceCache = new Map(), probeCache = new Map(), studies = [];
  const inspectSource = id => {
    if (!sourceCache.has(id)) sourceCache.set(id, (async () => {
      const entry = manifest.experiments[id];
      if (entry.id !== id) throw new Error(`来源组身份不匹配：${id}`);
      const report = JSON.parse(await checkAsset(entry.report, root));
      if (report.schema !== entry.schema || report.status !== entry.status) throw new Error(`报告状态或 schema 不匹配：${id}`);
      const checksums = await checkAsset(entry.checksums, root);
      await checkChecksumFile(path.join(root, entry.checksums.path), path.dirname(path.join(root, entry.report.path)));
      return { files: checksums.toString("utf8").trim().split("\n").length, report };
    })());
    return sourceCache.get(id);
  };
  for (const id of only ? [only] : Object.keys(experimentPrerequisites)) {
    const definition = experimentPrerequisites[id], entry = manifest.experiments[id], checks = [];
    const record = (status, code, label, detail, hint = "") => checks.push({ status, code, label, detail, ...(hint ? { hint } : {}) });
    const variable = definition.environment || "FRAMEWORK_PYTHON";
    const setting = env[variable] || (variable === "RETRIEVAL_PYTHON" ? ".venv-retrieval/bin/python" : ".venv-framework/bin/python");
    const executable = setting.includes("/") || setting.includes("\\") ? path.resolve(root, setting) : setting;
    let packages;
    try {
      packages = await pinnedRequirements(root, definition.requirements);
      const cacheKey = JSON.stringify([executable, packages]);
      if (!probeCache.has(cacheKey)) probeCache.set(cacheKey, Promise.resolve().then(() => probe({ executable, packages, root, env })));
      const runtime = await probeCache.get(cacheKey);
      record("passed", "python.available", "Python 可启动", `${executable} · ${runtime.python}`);
      if (entry.runtime?.python && runtime.python !== entry.runtime.python) record("warning", "python.reference-differs", "Python 与报告不同", `本机 ${runtime.python}；报告 ${entry.runtime.python}。需另行核验数值差异。`);
      for (const [name, version] of Object.entries(packages)) {
        const installed = runtime.packages?.[name];
        if (!installed?.version) record("failed", "package.missing", `缺少 ${name}`, `需要 ${name}==${version}`, `在 ${variable} 指向的环境准备 ${definition.requirements}。`);
        else if (installed.version !== version) record("failed", "package.version", `${name} 版本不符`, `本机 ${installed.version}；需要 ${version}`, `使用固定依赖 ${definition.requirements}；不要改写已登记报告。`);
        else if (installed.importError) record("failed", "package.import", `${name} 无法导入`, installed.importError, "包已登记但导入失败，检查该环境的二进制依赖。");
        else record("passed", "package.available", `${name} 版本与导入`, version);
      }
      if (definition.gloo) record(runtime.gloo ? "passed" : "failed", "gloo.build", "Gloo 构建支持", runtime.gloo ? "构建可用；本检查不启动双进程或绑定端口。" : "当前 PyTorch 未提供 Gloo。", runtime.gloo ? "" : "使用含 Gloo 的 PyTorch CPU 环境。");
    } catch (error) {
      const detail = error.code === "ENOENT" ? `找不到 ${executable}` : error.code === "EACCES" ? `不能执行 ${executable}` : error.killed ? "Python 探测超时。" : String(error.stderr || error.message || error).trim().slice(0,600);
      record("failed", "python.probe", "环境探测失败", detail, `准备 ${definition.requirements}，或设置 ${variable} 为对应 Python 程序的路径。`);
    }
    for (const source of [...new Set([id, ...definition.inputs])]) {
      try {
        const saved = await inspectSource(source);
        record("passed", "input.source", `${source} 固定来源`, `${saved.files} 个清单文件与报告哈希一致。`);
        if (id === "retrieval" && source === "retrieval") {
          try {
            const queries = await readFile(path.join(root, "src/experiments/retrieval-queries.json"));
            if (hash(queries) !== saved.report.dataset?.queries_sha256) throw new Error("当前检索问题与固定报告的 queries_sha256 不同。");
            record("passed", "input.queries", "检索问题身份", saved.report.dataset.queries_sha256);
          } catch (error) { record("failed", "input.queries", "检索问题身份", error.message, "核对 src/experiments/retrieval-queries.json 与登记的固定问题文件。"); }
        }
      } catch (error) { record("failed", "input.source", `${source} 固定来源`, error.message, "根据来源清单和完整工件包恢复缺失或变化的文件；先核对身份再重跑。"); }
    }
    if (outDir) {
      try {
        await lstat(path.join(root, outDir));
        record("failed", "output.exists", "输出目录已存在", outDir, "换一个尚不存在的 artifacts/<目录名>。");
      } catch (error) {
        if (error.code !== "ENOENT") record("failed", "output.inspect", "输出目录无法检查", error.message);
        else {
          try { await access(path.join(root, "artifacts"), constants.W_OK); record("passed", "output.available", "输出目录尚不存在", `${outDir}；artifacts 父目录可写。`); }
          catch (accessError) { record("failed", "output.parent", "输出父目录不可写或不存在", accessError.message, "先准备可写的 artifacts 目录。"); }
        }
      }
    }
    studies.push({ id, title: entry.title, python: executable, requirements: definition.requirements,
      status: checks.some(row => row.status === "failed") ? "blocked" : checks.some(row => row.status === "warning") ? "ready-with-warnings" : "ready", checks });
  }
  return { schema: "experiment-preflight-v1", created_at: new Date().toISOString(), manifest_sha256: hash(manifestBytes),
    status: studies.some(study => study.status === "blocked") ? "blocked" : studies.some(study => study.status === "ready-with-warnings") ? "ready-with-warnings" : "ready", studies,
    limits: ["仅检查列出的固定依赖、模块导入、登记报告与原始文件清单；不展开下载压缩包，不检查所有系统库或当前源码与历史源码的逐字节一致性。", "不启动训练、不下载模型、不创建实验输出目录。通过预检不等于数值复现、Gloo 通信或性能测试通过。", "输入检查覆盖所选组与其直接依赖的完整来源组，可能包含本次重跑不会直接读取的文件。"] };
}
