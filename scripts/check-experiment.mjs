import { checkExperimentReadiness, parsePreflightArguments, experimentPrerequisites } from "./experiment-preflight.mjs";

let options;
try { options = parsePreflightArguments(process.argv.slice(2)); }
catch (error) { console.error(error.message); process.exitCode = 2; }

if (options?.help) {
  console.log("复现前检查（不运行训练）：node scripts/check-experiment.mjs [--only <来源组>] [--out-dir artifacts/<新目录>] [--json]\n来源组：" + Object.keys(experimentPrerequisites).join(", "));
} else if (options) {
  try {
    const report = await checkExperimentReadiness({ only: options.only, outDir: options["out-dir"] });
    if (options.json) console.log(JSON.stringify(report, null, 2));
    else {
      for (const study of report.studies) {
        console.log(`\n${study.id} · ${study.title} · ${study.status}`);
        for (const check of study.checks) console.log(`  ${check.status === "passed" ? "通过" : check.status === "warning" ? "差异" : "阻塞"} | ${check.label}：${check.detail}${check.hint ? `\n    下一步：${check.hint}` : ""}`);
      }
      console.log(`\n预检结果：${report.status}\n${report.limits.join("\n")}`);
    }
    process.exitCode = report.status === "blocked" ? 1 : 0;
  } catch (error) {
    if (options.json) console.log(JSON.stringify({ schema: "experiment-preflight-v1", status: "blocked", error: error.message }));
    else console.error(`预检无法完成：${error.message}`);
    process.exitCode = 1;
  }
}
