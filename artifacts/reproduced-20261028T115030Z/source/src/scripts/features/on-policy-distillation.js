import { $ } from "../core.js";

export function prefixDistillation(mix = 0.8) {
  if (!Number.isFinite(mix) || mix < 0 || mix > 1) {
    throw new RangeError("mix must be a finite number in [0, 1].");
  }
  const teacher = [[0.9, 0.1], [0.8, 0.2]];
  const student = [[0.6, 0.4], [0.2, 0.8]];
  const teacherWeights = [1, 0];
  const studentWeights = [mix, 1 - mix];
  const kl = (p, q) => p.reduce((sum, probability, index) => sum + probability * Math.log(probability / q[index]), 0);
  const states = teacher.map((distribution, index) => ({
    name: index === 0 ? "teacher" : "student",
    teacher: distribution,
    student: student[index],
    occupancy: studentWeights[index],
    teacherWeight: teacherWeights[index],
    studentWeight: studentWeights[index],
    forwardKL: kl(distribution, student[index]),
    reverseKL: kl(student[index], distribution),
  }));
  return {
    states,
    teacherLoss: states.reduce((sum, state) => sum + state.teacherWeight * state.forwardKL, 0),
    onPolicyLoss: states.reduce((sum, state) => sum + state.occupancy * state.reverseKL, 0),
    recoveryCoverage: 1 - mix,
  };
}

export function initOnPolicyDistillation() {
  const input = $("[data-opd-mix]");
  if (!input) return;
  const percent = value => `${(value * 100).toFixed(0)}%`;
  const distribution = values => `[${values.map(value => value.toFixed(2)).join(", ")}]`;
  const render = () => {
    const mix = Number(input.value) / 100;
    const result = prefixDistillation(mix);
    $("[data-opd-mix-value]").textContent = percent(mix);
    $("[data-opd-recovery]").textContent = percent(result.recoveryCoverage);
    $("[data-opd-teacher-loss]").textContent = result.teacherLoss.toFixed(3);
    $("[data-opd-student-loss]").textContent = result.onPolicyLoss.toFixed(3);
    $("[data-opd-rows]").innerHTML = result.states.map(state =>
      `<tr><td>${state.name === "teacher" ? "常见前缀" : "需要恢复的前缀"}</td><td>${percent(state.teacherWeight)}</td><td>${percent(state.studentWeight)}</td><td>${distribution(state.teacher)}</td><td>${distribution(state.student)}</td><td>${state.forwardKL.toFixed(3)}</td><td>${state.reverseKL.toFixed(3)}</td></tr>`
    ).join("");
    $("[data-opd-note]").textContent = `需要恢复的前缀占访问权重的 ${percent(result.recoveryCoverage)}；恢复成功率需单独测试。改变前缀比例会改变学生目标中两类状态的权重，教师轨迹目标保持不变。`;
  };
  input.addEventListener("input", render);
  render();
}
