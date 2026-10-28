import { $, $$ } from "../core.js";

export function initHome() {
  const pipelineSteps = [
    {
      kicker: "原始材料",
      title: "文本先被切成离散符号",
      body: "分词器按照自己的词表，把文本切成 token；中文可能切成字、子词或多字词，取决于 tokenizer。",
      input: "字符串 / 字节",
      action: "tokenize",
      next: "整数 ID",
    },
    {
      kicker: "离散索引",
      title: "token ID 标记词表中的位置",
      body: "token ID 只是词表位置。比如“负样本”可能是两个 ID，也可能是一个 ID。ID 的大小没有距离含义，17 并不比 16 更接近 18。",
      input: "token 列表",
      action: "查词表映射",
      next: "形状 [B, T]",
    },
    {
      kicker: "查表与上下文",
      title: "每个 ID 取出一行可训练参数",
      body: "E[id] 等价于 one-hot(id) × E。取出后得到 [B, T, d] 的浮点张量；如果经过 Transformer，每个位置还会融合上下文。",
      input: "整数 ID",
      action: "gather + context",
      next: "浮点张量",
    },
    {
      kicker: "形成预测",
      title: "从坐标变成分数或概率",
      body: "语言模型把隐藏状态投到词表，得到每个下一 token 的 logits；对比学习则计算 query 与候选的点积或余弦相似度。",
      input: "隐藏状态 h",
      action: "线性投影 / cosine",
      next: "logits / score",
    },
    {
      kicker: "负反馈",
      title: "目标函数把“对错”变成可优化数字",
      body: "交叉熵会压低正确 token 之外的概率；对比损失会拉近正样本、推开负样本。损失越小，当前参数下目标越被满足。",
      input: "预测 + 目标",
      action: "cross entropy / InfoNCE",
      next: "标量 loss",
    },
    {
      kicker: "参数更新",
      title: "自动微分沿原路返回梯度",
      body: "反向传播沿计算图求各可训练参数的梯度，优化器据此更新。lookup 分支只访问部分 E 行；完整词表输出通常有稠密梯度，权重绑定后也会加回 E。",
      input: "loss + 计算图",
      action: "autograd + optimizer",
      next: "新的参数",
    },
  ];
  
  const pipelineButtons = $$("[data-pipeline-step]");
  if (pipelineButtons.length) {
    const fields = {
      index: $("[data-pipeline-index]"),
      kicker: $("[data-pipeline-kicker]"),
      title: $("[data-pipeline-title]"),
      body: $("[data-pipeline-body]"),
      input: $("[data-pipeline-input]"),
      action: $("[data-pipeline-action]"),
      next: $("[data-pipeline-next]"),
    };
    pipelineButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const step = Number(button.dataset.pipelineStep);
        const data = pipelineSteps[step];
        pipelineButtons.forEach((item) => {
          item.setAttribute("aria-selected", String(item === button));
        });
        fields.index.textContent = String(step + 1).padStart(2, "0");
        fields.kicker.textContent = data.kicker;
        fields.title.textContent = data.title;
        fields.body.textContent = data.body;
        fields.input.textContent = data.input;
        fields.action.textContent = data.action;
        fields.next.textContent = data.next;
      });
    });
  }
  
  const scaleData = {
    token: {
      kicker: "静态查表表示",
      title: "同一词，查到同一行",
      body: "Word2Vec、GloVe 和 LLM 的输入 embedding 表都是这样：词表是 V 行，每行是 d 维参数，查表等价于 one-hot 乘矩阵。区别在于 LLM 之后还会经过上下文层，形成随输入语境变化的隐藏状态。",
      shape: "1 × d",
      data: "1 × 768",
      context: "不随句境改变",
      reduce: "还不足以表示整句",
    },
    context: {
      kicker: "上下文状态",
      title: "同一词，在不同句里各有坐标",
      body: "经过注意力层后，每个位置都有一个随上下文变化的状态。BERT 的“苹果”在“吃苹果”和“苹果发布会”里会不同；但它仍然是每个 token 一个向量。",
      shape: "T × d",
      data: "T × 768",
      context: "随整句上下文改变",
      reduce: "需要 pooled / mean / 专用 head",
    },
    sentence: {
      kicker: "句子级表示",
      title: "把整段意思压成一个检索单位",
      body: "句向量模型需要一个聚合规则，例如 CLS、mean pooling，或在对比训练中学习如何读这些 token 状态。聚合后保留哪些信息，取决于表示维度、聚合规则和训练目标。",
      shape: "1 × d",
      data: "1 × 1024",
      context: "代表整段输入",
      reduce: "池化 + 归一化 + 投影",
    },
  };
  
  const scaleButtons = $$("[data-scale]");
  if (scaleButtons.length) {
    const fields = {
      kicker: $("[data-scale-kicker]"),
      title: $("[data-scale-title]"),
      body: $("[data-scale-body]"),
      shape: $("[data-scale-shape]"),
      data: $("[data-scale-data]"),
      context: $("[data-scale-context]"),
      reduce: $("[data-scale-reduce]"),
    };
    scaleButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const data = scaleData[button.dataset.scale];
        scaleButtons.forEach((item) => {
          item.setAttribute("aria-selected", String(item === button));
        });
        fields.kicker.textContent = data.kicker;
        fields.title.textContent = data.title;
        fields.body.textContent = data.body;
        fields.shape.textContent = data.shape;
        fields.data.textContent = data.data;
        fields.context.textContent = data.context;
        fields.reduce.textContent = data.reduce;
      });
    });
  }
}
