import { $, $$ } from "../core.js";

export function initRepresentation() {
    const timelineData = [
      {
        year: "1960s–80s",
        era: "符号与 one-hot",
        kicker: "离散符号表示",
        title: "词是原子符号，词典很大但几乎全零",
        what: "每个词占词表的一个位置。one-hot 可以表示“是哪个词”，却没有内建的相似性；两个不同词的向量点积总是 0。",
        objective: "没有从语境学习表示的目标",
        solved: "唯一标识、可做规则和统计",
        limit: "无法泛化到没见过的词，也无法表达近义关系",
        shapes: ["|V| ≈ 10⁴–10⁶", "稀疏度 ≈ 100%", "距离≈符号差异"],
      },
      {
        year: "1990",
        era: "LSA / 共现分解",
        kicker: "分布假设进入工程",
        title: "先统计词—文档关系，再做低秩压缩",
        what: "经典 LSA 对加权的“词 × 文档”矩阵做截断 SVD，得到低秩表示。词在文档中的分布模式决定表示；直接共现不保证两个词的向量更近。",
        objective: "低秩近似词—文档矩阵；不依赖神经网络式的逐样本训练",
        solved: "全局统计、主题结构、降低维度",
        limit: "矩阵很大、动态语境弱，投影与任务目标并不总一致",
        shapes: ["词数 × 文档数", "SVD → k 维", "全局分布统计"],
      },
      {
        year: "2013",
        era: "Word2Vec",
        kicker: "预测任务替代数矩阵",
        title: "用上下文预测，把相似性压进几百维",
        what: "Skip-gram 用中心词预测邻近词；CBOW 用上下文预测中心词。负采样只为本次输入、正例与采样负例产生数据梯度；完整 softmax 通常涉及全部输出词行，层次 softmax 则更新树路径。",
        objective: "Skip-gram/CBOW 可配不同目标；SGNS 对真实搭配与噪声独立二分类，不等同完整 softmax",
        solved: "稠密、可迁移的词向量；高效训练",
        limit: "同一表内每个词只有一个静态向量；多义和句级语义不足",
        shapes: ["V × d", "d ≈ 100–300", "静态词向量"],
      },
      {
        year: "2014–16",
        era: "GloVe / fastText",
        kicker: "全局统计与子词结构",
        title: "一部分回到共现，一部分看向词内部",
        what: "GloVe 的加权最小二乘目标拟合 log 共现次数，概率比提供推导动机；fastText 用字符 n-gram 组成词表示，能覆盖部分未登录词。",
        objective: "GloVe 拟合共现；fastText 预测上下文并聚合子词",
        solved: "全局统计与局部预测折中；形态变化更鲁棒",
        limit: "仍基本是静态词向量；句级任务需要额外聚合",
        shapes: ["d ≈ 100–300", "subword n-gram", "静态"],
      },
      {
        year: "2018",
        era: "ELMo / BERT",
        kicker: "表示开始依赖上下文",
        title: "根据上下文生成各位置的表示",
        what: "ELMo 用双向语言模型产生上下文词表示；BERT 用 Transformer 和掩码语言建模。同一个 token 在不同句子中会得到不同隐藏状态。",
        objective: "双向语言建模 / masked token prediction",
        solved: "多义、句法和上下文敏感表示",
        limit: "用于句向量时，还需选择池化方式和训练目标",
        shapes: ["T × d", "d ≈ 768–1024", "逐位置变化"],
      },
      {
        year: "2019–21",
        era: "SBERT / SimCSE",
        kicker: "句向量被显式优化",
        title: "直接训练“两句话该多近”",
        what: "SBERT 用句对数据训练孪生编码器；无监督 SimCSE 用同一句话在两次 dropout 下的表示作正对，监督版使用 NLI 蕴含对。它们显式优化句级接口，语义相似与检索可回答性需要按各自标准训练和评估。",
        objective: "句对相似度、cross-entropy、对比学习 InfoNCE",
        solved: "高效句级相似度、语义检索、聚类",
        limit: "对负样本设计、领域和池化方式敏感；可回答性需单独标注",
        shapes: ["1 × d / 句", "pair / triplet", "cosine"],
      },
      {
        year: "2022–now",
        era: "检索专用与可伸缩",
        kicker: "从通用语义到任务几何",
        title: "模型、指令、维度和重排一起设计",
        what: "BGE、E5、GTE 等检索模型加入大规模弱监督、难负样本和指令前缀；Matryoshka 表示允许同一向量截断到不同维度；多阶段检索常把 embedding 与 cross-encoder reranker 组合。",
        objective: "大规模对比训练 + 难负样本 + 任务指令；有时再加蒸馏",
        solved: "跨语言/多领域检索、灵活成本、可控维度",
        limit: "目标任务需检查数据污染、长文档和块边界的影响",
        shapes: ["d ≈ 128–4096", "query / passage", "retrieve + rerank"],
      },
    ];

    const buttons = $$("[data-era]");
    if (!buttons.length) return;
    const fields = {
      year: $("[data-era-year]"),
      label: $("[data-era-label]"),
      kicker: $("[data-era-kicker]"),
      title: $("[data-era-title]"),
      what: $("[data-era-what]"),
      objective: $("[data-era-objective]"),
      solved: $("[data-era-solved]"),
      limit: $("[data-era-limit]"),
      shapes: $("[data-era-shapes]"),
    };
    buttons.forEach((button, index) => {
      button.addEventListener("click", () => {
        const data = timelineData[index];
        buttons.forEach((item) => {
          item.classList.toggle("is-current", item === button);
          item.setAttribute("aria-pressed", String(item === button));
        });
        fields.year.textContent = data.year;
        fields.label.textContent = data.era;
        fields.kicker.textContent = data.kicker;
        fields.title.textContent = data.title;
        fields.what.textContent = data.what;
        fields.objective.textContent = data.objective;
        fields.solved.textContent = data.solved;
        fields.limit.textContent = data.limit;
        fields.shapes.replaceChildren(
          ...data.shapes.map((text) => {
            const span = document.createElement("span");
            span.textContent = text;
            return span;
          }),
        );
      });
    });

    const objectiveData = {
      context: {
        label: "预测邻近词",
        title: "上下文分布塑造词表示",
        copy: "Skip-gram 提高真实搭配的输入词—输出上下文得分，不保证两个输入词向量直接靠近。静态“苹果”会混合水果与公司语境，各义项的影响取决于语料与采样；下图仅示意这些语境。",
        groups: [
          { color: "teal", label: "食物语境", points: [[18, 36], [25, 28], [33, 40], [30, 53], [16, 52]] },
          { color: "blue", label: "科技语境", points: [[73, 65], [80, 54], [90, 63], [84, 76], [69, 79]] },
          { color: "coral", label: "目标词", points: [[49, 42]] },
        ],
      },
      retrieval: {
        label: "query ↔ passage",
        title: "“能不能检索到”定义邻居",
        copy: "对比目标促使可回答段落相对负例获得更高得分，实际排名仍取决于训练结果。单位向量的点积增大对应欧氏距离减小；未归一化的点积不能直接这样解释。",
        groups: [
          { color: "teal", label: "可回答段落", points: [[35, 48], [44, 53], [29, 60], [48, 40]] },
          { color: "coral", label: "query", points: [[56, 47]] },
          { color: "blue", label: "难负样本", points: [[65, 36], [74, 43], [67, 57]] },
        ],
      },
      classification: {
        label: "类别可分",
        title: "决策边界定义邻居",
        copy: "分类目标强调类间可分离，对类内各句的距离约束取决于具体目标。类内方差可以被压得很小，也可能保留很多与标签无关的信息。",
        groups: [
          { color: "teal", label: "类别 A", points: [[20, 69], [28, 62], [37, 70], [29, 77]] },
          { color: "blue", label: "类别 B", points: [[70, 28], [80, 35], [77, 22], [64, 37]] },
          { color: "coral", label: "边界附近", points: [[52, 51], [56, 46]] },
        ],
      },
    };

    const objectiveButtons = $$("[data-objective]");
    const objectiveSvg = $("[data-objective-svg]");
    if (objectiveButtons.length && objectiveSvg) {
      const colors = { teal: "var(--purple)", blue: "var(--blue)", coral: "var(--coral)" };
      const renderObjective = (key) => {
        const data = objectiveData[key];
        objectiveButtons.forEach((button) => {
          button.setAttribute("aria-selected", String(button.dataset.objective === key));
        });
        $("[data-objective-title]").textContent = data.title;
        $("[data-objective-copy]").textContent = data.copy;
        const circles = [];
        data.groups.forEach((group) => {
          group.points.forEach(([x, y], pointIndex) => {
            circles.push(
              `<circle cx="${x}" cy="${y}" r="${group.label === "目标词" || group.label === "query" ? 8 : 5}" fill="${colors[group.color]}" />`,
            );
            if (pointIndex === 0) {
              circles.push(
                `<text x="${Math.min(x + 10, 84)}" y="${Math.max(y - 10, 9)}" fill="var(--ink-soft)" font-size="9">${group.label}</text>`,
              );
            }
          });
        });
        objectiveSvg.innerHTML = `
          <rect x="0.5" y="0.5" width="299" height="199" rx="6" fill="none" stroke="var(--line)" />
          <line x1="150" y1="0" x2="150" y2="200" stroke="var(--line)" stroke-width="1" opacity=".55" />
          <line x1="0" y1="100" x2="300" y2="100" stroke="var(--line)" stroke-width="1" opacity=".55" />
          ${circles.join("")}
        `;
      };
      objectiveButtons.forEach((button) => {
        button.addEventListener("click", () => renderObjective(button.dataset.objective));
      });
      renderObjective("context");
    }

    const pickerAdvice = {
      word: {
        title: "固定词向量：fastText 或领域词向量",
        copy: "只有词级相似性、拼写/形态很重要时，子词模型仍有价值。若下游是句级语义，可优先测试句编码器，并与词向量平均的基线比较。",
        boundary: "适合：词典扩展、词聚类、轻量近邻",
      },
      sentence: {
        title: "句向量：SBERT / BGE / E5 一类编码器",
        copy: "先确认是语义相似、聚类还是分类。相似度与检索的标注标准不同；用你的正样本定义测试集，再比较池化、指令和维度。",
        boundary: "适合：语义检索、聚类、去重、分类特征",
      },
      pair: {
        title: "检索：bi-encoder 粗召回 + cross-encoder 精排",
        copy: "bi-encoder 可预先编码文档；cross-encoder 联合编码 query 与候选，增加交互也增加逐对计算成本。两阶段分别检查召回覆盖和候选排序，是否更准需在同一任务上实测。",
        boundary: "关注：Recall@k、MRR、延迟、索引成本",
      },
      multimodal: {
        title: "跨模态：对齐同一空间的图文编码器",
        copy: "图文配对数据与对比目标训练两个模态之间的匹配关系。查图、查文、图像聚类要分别评估，视觉细节可能需要专用模型或 reranker。",
        boundary: "关注：配对数据、模态缺失、检索方向",
      },
    };
    const picker = $("[data-picker-form]");
    if (picker) {
      const render = () => {
        const data = pickerAdvice[$("[name='embedding-unit']", picker).value];
        $("[data-picker-title]").textContent = data.title;
        $("[data-picker-copy]").textContent = data.copy;
        $("[data-picker-boundary]").textContent = data.boundary;
      };
      picker.addEventListener("change", render);
      render();
    }
}
