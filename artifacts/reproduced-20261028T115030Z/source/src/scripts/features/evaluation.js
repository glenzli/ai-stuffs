import { $, $$ } from "../core.js";

export function initEvaluation() {
    const lensButtons = $$("[data-eval-lens]");
    if (lensButtons.length) {
      const lenses = {
        pretrain: {
          kicker: "NEXT-TOKEN OBJECTIVE",
          title: "用 held-out loss 与下游任务评估模型",
          copy: "报告 bits/token 或 bits/byte 时要说明 tokenizer；perplexity 在同一 tokenizer/数据切分下可比较，跨 tokenizer 不能直接横向比。",
          primary: "held-out bits/byte、领域任务、长上下文、多语言",
          slices: "语言、领域、长度、训练阶段、重复度",
          risk: "测试集污染、tokenizer 改变、训练数据重复导致的记忆",
        },
        embedding: {
          kicker: "RETRIEVAL OBJECTIVE",
          title: "评估完整索引的召回与排序",
          copy: "Recall@k 决定候选上限，MRR/nDCG 看位置与相关程度。必须按真实 query 分布、文档长度和领域分片。",
          primary: "Recall@k、MRR、nDCG、top-k 上下文可回答率",
          slices: "语言、领域、短/长 query、hard negatives、文档长度",
          risk: "假负例、只测同源 query、新 query 查旧索引、normalize/prefix 不一致",
        },
        sft: {
          kicker: "INSTRUCTION FOLLOWING",
          title: "token loss 与端到端生成质量分开测",
          copy: "除了任务正确率，还要看格式、停止、工具调用、拒答、长度和 held-out prompt 的真实生成。",
          primary: "任务准确率、pass@k、rubric、人工/模型 pairwise、格式遵循",
          slices: "单轮/多轮、长上下文、工具、拒答、语言、提示长度",
          risk: "训练 loss 下降但生成退化；judge 与训练偏好同源；只看平均值",
        },
        preference: {
          kicker: "PREFERENCE & SAFETY",
          title: "同时评估偏好胜率、基础能力与安全",
          copy: "要同时测 chosen/rejected held-out 胜率、人工复核、安全底线、拒答、长度偏置与基础能力回归。",
          primary: "pairwise win rate、safety pass、指令遵循、回归 suite",
          slices: "偏好类型、风险类别、语言、长度、用户群",
          risk: "奖励投机、judge 偏差、训练集与评估集共享相同 annotator/judge",
        },
      };
      const render = (key) => {
        const data = lenses[key];
        lensButtons.forEach((button) => {
          button.setAttribute("aria-pressed", String(button.dataset.evalLens === key));
        });
        $("[data-eval-kicker]").textContent = data.kicker;
        $("[data-eval-title]").textContent = data.title;
        $("[data-eval-copy]").textContent = data.copy;
        $("[data-eval-primary]").textContent = data.primary;
        $("[data-eval-slices]").textContent = data.slices;
        $("[data-eval-risk]").textContent = data.risk;
      };
      lensButtons.forEach((button) => {
        button.addEventListener("click", () => render(button.dataset.evalLens));
      });
      render("pretrain");
    }

    const sliceInputs = $$("[data-slice]");
    if (sliceInputs.length) {
      const values = {
        zh: { name: "中文短问", score: 0.86, weight: 0.6 },
        en: { name: "英文检索", score: 0.72, weight: 0.2 },
        long: { name: "长文档", score: 0.61, weight: 0.15 },
        code: { name: "代码问题", score: 0.54, weight: 0.05 },
      };
      const render = () => {
        const selected = sliceInputs.filter((input) => input.checked).map((input) => input.dataset.slice);
        const items = selected.map((key) => values[key]);
        const macro = items.length ? items.reduce((sum, item) => sum + item.score, 0) / items.length : 0;
        const weightSum = items.reduce((sum, item) => sum + item.weight, 0);
        const micro = weightSum ? items.reduce((sum, item) => sum + item.score * item.weight, 0) / weightSum : 0;
        const weakest = items.length ? items.reduce((weak, item) => item.score < weak.score ? item : weak) : null;
        $("[data-slice-macro]").textContent = items.length ? macro.toFixed(3) : "无数据";
        $("[data-slice-micro]").textContent = items.length ? micro.toFixed(3) : "无数据";
        $("[data-slice-weak]").textContent = weakest ? `${weakest.name} · ${weakest.score.toFixed(2)}` : "无切片";
        $("[data-slice-bars]").innerHTML = items.map((item) => `
          <div><span>${item.name}</span><i style="width:${item.score * 100}%"></i><code>${item.score.toFixed(2)}</code></div>
        `).join("");
      };
      sliceInputs.forEach((input) => input.addEventListener("change", render));
      render();
    }
}
