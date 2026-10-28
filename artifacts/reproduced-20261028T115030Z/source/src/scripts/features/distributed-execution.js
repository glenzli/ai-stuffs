import { $ } from "../core.js";

const transpose = matrix => matrix[0].map((_, column) => matrix.map(row => row[column]));
const multiply = (left, right) => left.map(row => right[0].map((_, column) =>
  row.reduce((sum, value, index) => sum + value * right[index][column], 0)
));
const add = (left, right) => left.map((row, i) => row.map((value, j) => value + right[i][j]));
const activate = matrix => matrix.map(row => row.map(value => Math.tanh(value)));
const activationGradient = (gradient, hidden) => gradient.map((row, i) =>
  row.map((value, j) => value * (1 - hidden[i][j] ** 2))
);
const matrixError = (actual, expected) => {
  const values = expected.flat();
  return Math.max(...actual.flat().map((value, index) => Math.abs(value - values[index])));
};
const formatMatrix = matrix => matrix.map(row => `[${row.map(value => value.toFixed(4)).join(", ")}]`).join("；");

export function tensorParallelTrace() {
  const X = [[1, 2], [3, -1]];
  const W1 = [[0.2, -0.3, 0.5, 0.1], [0.4, 0.6, -0.2, 0.3]];
  const W2 = [[0.7, -0.2], [0.1, 0.5], [-0.4, 0.8], [0.3, 0.2]];
  const G = [[0.2, -0.4], [0.6, 0.1]];
  const hidden = activate(multiply(X, W1));
  const dHidden = multiply(G, transpose(W2));
  const dZ = activationGradient(dHidden, hidden);
  const reference = {
    Y: multiply(hidden, W2),
    dX: multiply(dZ, transpose(W1)),
    dW1: multiply(transpose(X), dZ),
    dW2: multiply(transpose(hidden), G),
  };
  const ranks = [0, 1].map(rank => {
    const start = rank * 2;
    const localW1 = W1.map(row => row.slice(start, start + 2));
    const localW2 = W2.slice(start, start + 2).map(row => row.slice());
    const Z = multiply(X, localW1);
    const H = activate(Z);
    const dH = multiply(G, transpose(localW2));
    const localDZ = activationGradient(dH, H);
    return {
      rank, W1: localW1, W2: localW2, Z, H,
      Y: multiply(H, localW2),
      dH, dZ: localDZ,
      dW1: multiply(transpose(X), localDZ),
      dW2: multiply(transpose(H), G),
      dX: multiply(localDZ, transpose(localW1)),
    };
  });
  const Y = add(ranks[0].Y, ranks[1].Y);
  const dX = add(ranks[0].dX, ranks[1].dX);
  const dW1 = W1.map((_, row) => ranks.flatMap(rank => rank.dW1[row]));
  const dW2 = ranks.flatMap(rank => rank.dW2);
  const error = Math.max(
    matrixError(Y, reference.Y), matrixError(dX, reference.dX),
    matrixError(dW1, reference.dW1), matrixError(dW2, reference.dW2)
  );
  return { X, W1, W2, G, reference, ranks, Y, dX, error };
}

export function collectiveExample(mode) {
  const input = [[1, 2, 3, 4], [10, 20, 30, 40]];
  const sum = input[0].map((value, index) => value + input[1][index]);
  let output;
  if (mode === "allreduce") {
    output = [sum.slice(), sum.slice()];
  } else if (mode === "allgather") {
    const gathered = input.flat();
    output = [gathered.slice(), gathered.slice()];
  } else if (mode === "reducescatter") {
    output = [sum.slice(0, 2), sum.slice(2)];
  } else if (mode === "alltoall") {
    output = [
      [...input[0].slice(0, 2), ...input[1].slice(0, 2)],
      [...input[0].slice(2), ...input[1].slice(2)],
    ];
  } else {
    throw new RangeError("集合通信模式无效。");
  }
  return { input, output };
}

export function initDistributedExecution() {
  const stage = $("[data-dx-stage]");
  if (!stage) return;
  const collective = $("[data-dx-collective]");
  const trace = tensorParallelTrace();
  const renderStage = () => {
    const backward = stage.value === "backward";
    const key = backward ? "dX" : "Y";
    $("[data-dx-ranks]").innerHTML = trace.ranks.map(rank =>
      `<tr><td>进程 ${rank.rank}</td><td>${formatMatrix(rank.W1)}</td><td>${formatMatrix(backward ? rank.dZ : rank.H)}</td><td>${formatMatrix(rank[key])}</td></tr>`
    ).join("");
    $("[data-dx-reference]").textContent = formatMatrix(trace.reference[key]);
    $("[data-dx-combined]").textContent = formatMatrix(trace[key]);
    $("[data-dx-error]").textContent = trace.error === 0 ? "0" : trace.error.toExponential(3);
    const explanation = backward
      ? "上游梯度复制到两个进程，各自按链式法则求导。第一层权重梯度按列拼接，第二层权重梯度按行拼接；两个部分输入梯度逐元素求和。"
      : "第一层权重按隐藏维度分列，第二层权重按相同维度分行。两个进程各计算局部激活与部分输出，再逐元素求和得到完整输出。";
    $("[data-dx-note]").textContent = `${explanation}误差取完整输出、输入梯度和两层权重梯度重建值的最大绝对差。这里只执行数组数值运算。`;
  };
  const renderCollective = () => {
    const { input, output } = collectiveExample(collective.value);
    $("[data-dx-collective-rows]").innerHTML = input.map((vector, rank) =>
      `<tr><td>进程 ${rank}</td><td>[${vector.join(", ")}]</td><td>[${output[rank].join(", ")}]</td></tr>`
    ).join("");
  };
  for (const event of ["input", "change"]) {
    stage.addEventListener(event, renderStage);
    collective.addEventListener(event, renderCollective);
  }
  renderStage();
  renderCollective();
}
