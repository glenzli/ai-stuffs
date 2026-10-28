import { $, $$ } from "../core.js";

export function initRetrievalLab() {
    const form = $("[data-lab-form]");
    const list = $("[data-retrieval-list]");
    if (!form || !list) return;

    const queries = {
      training: {
        title: "负样本怎样产生梯度？",
        vec: [0.92, 0.84, 0.32, 0.08, 0.03],
        keywords: ["负样本", "正样本", "梯度", "损失", "自动微分", "查表", "更新", "内积", "token", "embedding"],
      },
      context: {
        title: "句向量为什么要池化？",
        vec: [0.34, 0.08, 0.94, 0.2, 0.02],
        keywords: ["句向量", "上下文", "池化", "隐藏状态", "归一化", "token", "表示", "编码"],
      },
      posttrain: {
        title: "SFT 与偏好训练有什么区别？",
        vec: [0.16, 0.1, 0.28, 0.94, 0.04],
        keywords: ["SFT", "偏好", "chosen", "rejected", "回答", "后训练", "交叉熵", "示范", "排序"],
      },
    };

    const documents = [
      {
        id: "d1",
        title: "Embedding 表与前向传播",
        chunks: [
          {
            text: "训练样本先是一串 token ID。前向传播时，用 ID 从 embedding 表取出对应行，得到 [T,d] 的浮点张量。",
            vec: [0.8, 0.22, 0.68, 0.02, 0.12],
          },
          {
            text: "真实上下文作为正样本，采样出的噪声词作为负样本。模型提高正样本对的内积，降低负样本对的内积。",
            vec: [0.96, 0.93, 0.26, 0.03, 0.08],
          },
          {
            text: "损失是一个标量。自动微分沿计算图把梯度传回被取出的查表行，优化器再更新这些参数。",
            vec: [0.9, 0.77, 0.38, 0.05, 0.1],
          },
        ],
      },
      {
        id: "d2",
        title: "上下文与句向量",
        chunks: [
          {
            text: "BERT 为每个位置输出一个随上下文变化的隐藏状态。",
            vec: [0.4, 0.08, 0.96, 0.14, 0.03],
          },
          {
            text: "句子级检索通常需要 pooling 与对比训练，把 token 状态压成可比较的句向量。",
            vec: [0.55, 0.12, 0.88, 0.22, 0.04],
          },
          {
            text: "池化与归一化都是模型接口的一部分；换掉它们，旧索引与新 query 可能不在同一标尺。",
            vec: [0.32, 0.06, 0.7, 0.09, 0.08],
          },
        ],
      },
      {
        id: "d3",
        title: "SFT 与偏好后训练",
        chunks: [
          {
            text: "SFT 使用 prompt 与示范 response，目标仍常是对 response token 的交叉熵。",
            vec: [0.12, 0.04, 0.25, 0.95, 0.06],
          },
          {
            text: "偏好训练比较 chosen 与 rejected 两个回答，优化相对概率或奖励，对应的监督来自回答偏好。",
            vec: [0.2, 0.24, 0.18, 0.92, 0.04],
          },
          {
            text: "这些后训练会改变模型权重，借用 hidden states 作为句向量时，需要选择读出方式并重新评估。",
            vec: [0.3, 0.14, 0.3, 0.82, 0.14],
          },
        ],
      },
      {
        id: "d4",
        title: "城市散步与咖啡",
        chunks: [
          {
            text: "周末去街角咖啡店坐一会儿，顺便看看新开的书店。",
            vec: [0.02, 0.01, 0.05, 0.02, 0.98],
          },
          {
            text: "今天的天气适合散步，但傍晚可能下雨。",
            vec: [0.05, 0.02, 0.12, 0.04, 0.91],
          },
        ],
      },
      {
        id: "d5",
        title: "难负例需要验证",
        chunks: [
          {
            text: "难负例通常是关键词相似、却没有答案的段落。错误标注会把可回答段落也推远。",
            vec: [0.78, 0.67, 0.18, 0.04, 0.1],
          },
          {
            text: "负样本可从随机候选或难负候选中选择，随后按任务标准检查 query 与段落的关系。",
            vec: [0.84, 0.7, 0.24, 0.03, 0.07],
          },
        ],
      },
      {
        id: "d6",
        title: "Tokenizer 与表示边界",
        chunks: [
          {
            text: "Tokenizer 决定文字如何切成 token，词表大小也决定 embedding 表的行数。",
            vec: [0.52, 0.08, 0.62, 0.02, 0.18],
          },
          {
            text: "中文可能按字或子词切分，同一个可见词在不同 tokenizer 中不一定对应同一个 ID。",
            vec: [0.43, 0.05, 0.54, 0.04, 0.22],
          },
        ],
      },
    ];

    const range = $("[name='topk']", form);
    const querySelect = $("[name='query']", form);
    const rerankInput = $("[name='rerank']", form);
    const chunkHelp = $("[data-chunk-help]");
    const topkValue = $("[data-topk-value]");
    const vectorBars = $("[data-query-vector]");
    const chunkLabels = ["训练信号", "负样本", "上下文", "行为", "噪声"];

    const normalize = (vec) => {
      const norm = Math.sqrt(vec.reduce((sum, value) => sum + value * value, 0)) || 1;
      return vec.map((value) => value / norm);
    };

    const cosine = (a, b) => {
      const left = normalize(a);
      const right = normalize(b);
      return left.reduce((sum, value, index) => sum + value * right[index], 0);
    };

    const averageVectors = (vectors) =>
      vectors[0].map((_, index) => vectors.reduce((sum, vec) => sum + vec[index], 0) / vectors.length);

    const makeChunks = (granularity) => {
      if (granularity === "coarse") {
        return documents.map((doc) => ({
          id: `${doc.id}-coarse`,
          docTitle: doc.title,
          label: "整篇文档",
          text: doc.chunks.map((chunk) => chunk.text).join(" "),
          vec: averageVectors(doc.chunks.map((chunk) => chunk.vec)),
        }));
      }

      if (granularity === "medium") {
        return documents.flatMap((doc) => {
          const groups = [];
          for (let index = 0; index < doc.chunks.length; index += 2) {
            const chunkGroup = doc.chunks.slice(index, index + 2);
            groups.push({
              id: `${doc.id}-m${index / 2 + 1}`,
              docTitle: doc.title,
              label: chunkGroup.length > 1 ? "相邻两块合并" : "单个短块",
              text: chunkGroup.map((chunk) => chunk.text).join(" "),
              vec: averageVectors(chunkGroup.map((chunk) => chunk.vec)),
            });
          }
          return groups;
        });
      }

      return documents.flatMap((doc) =>
        doc.chunks.map((chunk, index) => ({
          id: `${doc.id}-c${index + 1}`,
          docTitle: doc.title,
          label: "单句 / 短块",
          text: chunk.text,
          vec: chunk.vec,
        })),
      );
    };

    const lexicalScore = (query, text) => {
      const hits = query.keywords.filter((keyword) => text.includes(keyword)).length;
      return hits / Math.max(query.keywords.length * 0.45, 1);
    };

    const getState = () => ({
      queryKey: querySelect.value,
      chunking: $("[name='chunking']:checked", form).value,
      topk: Number(range.value),
      rerank: rerankInput.checked,
    });

    const renderQueryVector = (vec) => {
      vectorBars.innerHTML = vec.map((value, index) =>
        `<div>
          <span>${chunkLabels[index]}</span>
          <i style="width:${Math.round(value * 100)}%"></i>
          <code>${value.toFixed(2)}</code>
        </div>`,
      ).join("");
    };

    const render = () => {
      const state = getState();
      const query = queries[state.queryKey];
      const chunks = makeChunks(state.chunking);
      const candidates = chunks
        .map((chunk) => ({
          ...chunk,
          cosine: Math.max(0, cosine(query.vec, chunk.vec)),
        }))
        .sort((a, b) => b.cosine - a.cosine);

      let ordered = candidates;
      let rerankPool = [];
      if (state.rerank) {
        const poolSize = Math.min(candidates.length, Math.max(6, state.topk + 2));
        rerankPool = candidates.slice(0, poolSize).map((candidate) => {
          const lexical = Math.min(1, lexicalScore(query, candidate.text));
          return {
            ...candidate,
            lexical,
            rerankScore: 0.58 * candidate.cosine + 0.42 * lexical,
          };
        }).sort((a, b) => b.rerankScore - a.rerankScore);
        ordered = [...rerankPool, ...candidates.slice(poolSize)];
      }

      const displayed = ordered.slice(0, state.topk);
      $("[data-query-title]").textContent = query.title;
      topkValue.textContent = String(state.topk);
      $("[data-pipeline-state]").textContent = state.rerank ? "召回 → 模拟重排" : "仅 bi-encoder 召回";
      $("[data-result-summary]").textContent = `${displayed.length} 个块进入上下文包 · 共 ${candidates.length} 个候选`;
      $("[data-rerank-note]").textContent = state.rerank
        ? `对前 ${rerankPool.length} 个候选做关键词近似重排`
        : "未启用重排";
      renderQueryVector(query.vec);

      list.innerHTML = displayed.map((item, index) => {
        const visibleScore = state.rerank && item.rerankScore !== undefined
          ? item.rerankScore
          : item.cosine;
        const scoreLabel = state.rerank && item.rerankScore !== undefined
          ? `重排 ${item.rerankScore.toFixed(3)} · cosine ${item.cosine.toFixed(3)}`
          : `cosine ${item.cosine.toFixed(3)}`;
        const scoreClass = state.rerank && item.rerankScore !== undefined ? "reranked" : "embedding";
        const similarityTag = item.cosine >= 0.8 ? "高相似" : item.cosine >= 0.6 ? "中相似" : "低相似";
        return `
          <article class="retrieval-item">
            <div class="rank-number">${String(index + 1).padStart(2, "0")}</div>
            <div class="retrieval-body">
              <div class="retrieval-meta">
                <span>${item.docTitle}</span>
                <span>${item.label}</span>
                <span class="tag ${item.cosine >= 0.8 ? "tag-teal" : ""}">${similarityTag}</span>
              </div>
              <p>${item.text}</p>
              <div class="score-line">
                <div class="score-track"><i class="${scoreClass}" style="width:${Math.max(4, Math.min(100, visibleScore * 100))}%"></i></div>
                <code>${scoreLabel}</code>
              </div>
            </div>
          </article>
        `;
      }).join("");

      if (candidates.length > displayed.length) {
        list.insertAdjacentHTML("beforeend", `
          <div class="topk-cut">
            <span>Top-${state.topk} 截断</span>
            <i></i>
            <small>其余候选不进入上下文包</small>
          </div>
        `);
      }
    };

    const helpers = {
      fine: "逐短块编码：召回更精确，但可能丢掉前后文，生成阶段也会看到碎片化证据。",
      medium: "相邻句子合并成中等块：召回更稳，但可能混入一些无关细节。",
      coarse: "整篇文档压成一个点：上下文完整，主题信号容易被平均稀释。",
    };

    form.addEventListener("change", () => {
      const chunking = $("[name='chunking']:checked", form).value;
      chunkHelp.textContent = helpers[chunking];
      render();
    });
    form.addEventListener("input", (event) => {
      if (event.target === range) render();
    });
    $("[data-sandbox-reset]").addEventListener("click", () => {
      querySelect.value = "training";
      $("[name='chunking'][value='medium']", form).checked = true;
      range.value = "4";
      rerankInput.checked = false;
      chunkHelp.textContent = helpers.medium;
      render();
    });
    render();
}
