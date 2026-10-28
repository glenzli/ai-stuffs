import { $, $$ } from "../core.js";

export function initDataTokenizer() {
    const stepButtons = $$("[data-data-step]");
    if (stepButtons.length) {
      const steps = [
        {
          name: "规范化",
          title: "统一可比较的文本形式",
          body: "Unicode 规范化、编码修复、HTML 清理、语言识别与元数据补全。保留原始文本，并记录清洗规则及输出摘要，以便复查不可逆改写。",
          io: "原始文档 → 规范化文档 + 语言/来源元数据",
          metric: "语言分布、截断率、编码错误率、每来源文档/token 数量",
          risk: "全角/半角或代码缩进被误改；语言识别把混合文本误判。",
        },
        {
          name: "质量与安全",
          title: "过滤低信息、重复模板与高风险内容",
          body: "质量分类器、规则过滤、PII 检测与来源信任度应分层记录。别只保存一个最终“通过”标记，要能解释被丢掉的数量和原因。",
          io: "规范化文档 → 保留文档 + reject reason",
          metric: "各来源通过率、分类器覆盖率、PII / safety 命中率",
          risk: "过滤规则偏向某种文体；多语言模型被英语质量分类器系统性低召回。",
        },
        {
          name: "去重",
          title: "控制模型实际看到每段内容的次数",
          body: "精确哈希、MinHash/LSH 近重复与语义相似去重解决不同层级的问题。重复文本等于隐式提高采样权重，会扭曲 token 预算。",
          io: "文档集合 → canonical docs + duplicate clusters",
          metric: "去重率、cluster 大小、按来源/语言重复 token 数",
          risk: "阈值过松误删不同事实；模板去重把少量差异的关键字段一起清掉。",
        },
        {
          name: "污染隔离",
          title: "检查训练集与评估集的重叠",
          body: "对规范化文本做 exact、long n-gram 与语义近邻扫描。验证集/测试集一旦进入训练，loss 和 benchmark 就不再提供独立证据。",
          io: "train + eval suites → decontaminated train",
          metric: "exact / n-gram / semantic overlap、删除 token 数",
          risk: "只消除原题哈希，忽略翻译、改写或同答案不同问法。",
        },
        {
          name: "混合与采样",
          title: "按 token 配置数据来源比例",
          body: "不同来源和 tokenizer 的 token 密度不同。需要记录 mixture weights、采样温度、重复上限和课程/阶段安排。",
          io: "source shards + weights → sampled stream",
          metric: "实际 token 占比、重复次数、阶段曲线、来源覆盖",
          risk: "网页占 60% 文件，却可能占 80% token；训练后期某小来源被完全饿死。",
        },
        {
          name: "Tokenize",
          title: "用固定 tokenizer 版本把文本变成整数",
          body: "Tokenizer 的 normalization、BPE merges、special tokens 与 byte fallback 都是模型接口。版本变化会让同一句文本变成不同长度和 ID。",
          io: "clean docs → token IDs + document boundaries",
          metric: "chars/token、fertility、序列长度分位数、byte fallback 率",
          risk: "代码缩进或中文被过度切碎；换了 tokenizer 却继续用旧数据 shard。",
        },
        {
          name: "Pack / Shard",
          title: "形成可流式读取的训练格式",
          body: "把 token 打包成固定长度序列并保存边界、样本 mask 与 shard hash。分词结果可提前保存，训练时按需打乱和读取 shard。",
          io: "token stream → binary / arrow shards",
          metric: "packing 填充率、跨样本边界、shard 大小、读取吞吐",
          risk: "packing 让 attention 跨样本泄漏；shard 太大导致随机性与恢复粒度变差。",
        },
      ];
      const render = (index) => {
        const data = steps[index];
        stepButtons.forEach((button, buttonIndex) => {
          button.setAttribute("aria-pressed", String(index === buttonIndex));
        });
        $("[data-data-step-number]").textContent = String(index + 1).padStart(2, "0");
        $("[data-data-step-name]").textContent = data.name;
        $("[data-data-title]").textContent = data.title;
        $("[data-data-body]").textContent = data.body;
        $("[data-data-io]").textContent = data.io;
        $("[data-data-metric]").textContent = data.metric;
        $("[data-data-risk]").textContent = data.risk;
      };
      stepButtons.forEach((button) => {
        button.addEventListener("click", () => render(Number(button.dataset.dataStep)));
      });
      render(0);
    }

    const tokenButtons = $$("[data-token-mode]");
    if (tokenButtons.length) {
      const sentence = "负样本让模型学会区分错误候选。";
      const chars = Array.from(sentence);
      const bpe = ["负样本", "让", "模型", "学会", "区分", "错误", "候选", "。"];
      const bytes = Array.from(new TextEncoder().encode(sentence)).map((value) =>
        value.toString(16).toUpperCase().padStart(2, "0"),
      );
      const modes = {
        char: {
          tokens: chars,
          note: "按字符切分时，每个汉字约一个 token。序列很长，但词表与未见词处理比较直接。",
        },
        bpe: {
          tokens: bpe,
          note: "子词把常见组合压成一个 token，减少序列长度；但不同 tokenizer 的词表不可直接比较 token 数、loss 或推理速度。",
        },
        byte: {
          tokens: bytes,
          note: "字节回退保证任意文本都能编码，但一个汉字通常变成 3 个 byte token，长上下文与 KV Cache 成本明显增加。",
        },
      };
      const renderTokens = (key) => {
        tokenButtons.forEach((button) => {
          button.setAttribute("aria-pressed", String(button.dataset.tokenMode === key));
        });
        const data = modes[key];
        const count = data.tokens.length;
        $("[data-tokenizer-count]").textContent = String(count);
        $("[data-tokenizer-ratio]").textContent = (count / chars.length).toFixed(2);
        $("[data-tokenizer-capacity]").textContent = Math.floor(4096 * chars.length / count).toLocaleString();
        $("[data-tokenizer-note]").textContent = data.note;
        $("[data-tokenizer-chips]").innerHTML = data.tokens.map((token) =>
          `<span>${token}</span>`,
        ).join("");
        $("[data-tokenizer-seq]").textContent = `S = ${count} for this sentence`;
        $("[data-tokenizer-vocab]").textContent = key === "byte" ? "V × d (byte fallback)" : "V × d";
        $("[data-tokenizer-kv]").textContent = `∝ ${count} · L · Hkv`;
      };
      tokenButtons.forEach((button) => {
        button.addEventListener("click", () => renderTokens(button.dataset.tokenMode));
      });
      renderTokens("bpe");
    }

    const mixtureInputs = $$("[data-mixture]");
    if (mixtureInputs.length) {
      const labels = { web: "网页", code: "代码", math: "数学 / 科学", multi: "多语种", book: "书籍 / 长文" };
      const colors = { web: "var(--purple)", code: "var(--blue)", math: "var(--yellow)", multi: "var(--coral)", book: "var(--line-strong)" };
      const renderMixture = () => {
        const values = Object.fromEntries(mixtureInputs.map((input) => [input.dataset.mixture, Number(input.value)]));
        const total = Object.values(values).reduce((sum, value) => sum + value, 0);
        if (!total) {
          $("[data-mixture-bars]").textContent = "所有来源权重为零，无法分配 100B token 预算。";
          return;
        }
        $("[data-mixture-bars]").innerHTML = mixtureInputs.map((input) => {
          const key = input.dataset.mixture;
          const pct = total ? values[key] / total : 0;
          const tokens = pct * 100;
          return `
            <div class="mixture-row">
              <span>${labels[key]}</span>
              <div><i style="width:${pct * 100}%; background:${colors[key]}"></i></div>
              <code>${Math.round(pct * 100)}% · ${tokens.toFixed(1)}B tokens</code>
            </div>
          `;
        }).join("");
      };
      mixtureInputs.forEach((input) => input.addEventListener("input", renderMixture));
      renderMixture();
    }
}
