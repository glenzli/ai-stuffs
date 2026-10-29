"""Finite state/gradient ledger. Python 3.10+ standard library; no model runtime."""
from pathlib import Path
from html.parser import HTMLParser
import copy
import itertools
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INPUTS = json.loads((HERE / "inputs.json").read_text())


def zeros(rows, columns):
    return [[0.0] * columns for _ in range(rows)]


def eye(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    assert len(a[0]) == len(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(a, s):
    return [[x * s for x in row] for row in a]


def outer(a, b):
    return [[x * y for y in b] for x in a]


def mv(a, v):
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def diagonal(a):
    return [[x if i == j else 0.0 for j in range(len(a))] for i, x in enumerate(a)]


def flatten(a):
    return [v for row in a for v in row]


def norm(a):
    return math.sqrt(sum(v * v for v in flatten(a)))


def distance(a, b):
    return max((abs(x - y) for x, y in zip(flatten(a), flatten(b))), default=0.0)


def close(a, b, tolerance=1e-12):
    assert abs(a - b) <= tolerance * max(1.0, abs(a), abs(b)), (a, b)


def affine_step(k, v, alpha, beta, wrong_order=False):
    erase = add(eye(len(k)), scale(outer(k, k), -beta))
    decay = diagonal(alpha)
    transition = matmul(decay, erase) if wrong_order else matmul(erase, decay)
    return transition, scale(outer(k, v), beta)


def forward(data, variant="KDA", reset_at=None):
    state = copy.deepcopy(data["initialState"])
    states, outputs, maps = [copy.deepcopy(state)], [], []
    for t, (k, q, v, alpha, beta) in enumerate(zip(data["keys"], data["queries"], data["values"], data["channelDecay"], data["writeStrength"])):
        if reset_at == t:
            state = zeros(len(k), len(v))
        if variant == "GDN":
            alpha = [data["scalarDecay"][t]] * len(k)
        transition, write = affine_step(k, v, alpha, beta, variant == "wrong-order")
        # An independent evaluation of the same recurrence: decay, predict, correct.
        decayed = matmul(diagonal(alpha), state)
        error = [x - y for x, y in zip(v, mv(transpose(decayed), k))]
        corrected = add(decayed, scale(outer(k, error), beta))
        state = add(matmul(transition, state), write)
        if variant != "wrong-order":
            assert distance(state, corrected) < 1e-12
        states.append(copy.deepcopy(state))
        outputs.append(mv(transpose(state), q))
        maps.append((transition, write))
    return states, outputs, maps


def compose(later, earlier):
    p2, r2 = later
    p1, r1 = earlier
    return matmul(p2, p1), add(matmul(p2, r1), r2)


def chunk_forward(data, sizes):
    assert sum(sizes) == len(data["keys"]) and all(size > 0 for size in sizes)
    dk, dv = len(data["initialState"]), len(data["initialState"][0])
    state = copy.deepcopy(data["initialState"])
    states, outputs, chunk_maps, cursor = [copy.deepcopy(state)], [], [], 0
    for size in sizes:
        incoming = copy.deepcopy(state)
        prefix_map = (eye(dk), zeros(dk, dv))
        for t in range(cursor, cursor + size):
            local = affine_step(data["keys"][t], data["values"][t], data["channelDecay"][t], data["writeStrength"][t])
            prefix_map = compose(local, prefix_map)
            state = add(matmul(prefix_map[0], incoming), prefix_map[1])
            states.append(copy.deepcopy(state))
            outputs.append(mv(transpose(state), data["queries"][t]))
        chunk_maps.append(prefix_map)
        cursor += size
    return states, outputs, chunk_maps


def terminal_loss(data):
    outputs = forward(data)[1]
    return 0.5 * sum((x - y) ** 2 for x, y in zip(outputs[-1], data["terminalTarget"]))


def empty_grad(data):
    result = {}
    for name in ["initialState", "values", "channelDecay", "writeStrength"]:
        value = data[name]
        result[name] = zeros(len(value), len(value[0])) if isinstance(value[0], list) else [0.0] * len(value)
    return result


def vjp(data, states, end_gradient, start=0, end=None, detach_at=None):
    """Adjoint of supplied k/q-free inputs; upstream projections are excluded."""
    end = len(data["keys"]) if end is None else end
    grad = empty_grad(data)
    adjoints = [zeros(len(states[0]), len(states[0][0])) for _ in states]
    current = copy.deepcopy(end_gradient)
    adjoints[end] = copy.deepcopy(current)
    for t in range(end - 1, start - 1, -1):
        k, v, alpha, beta = (data[name][t] for name in ["keys", "values", "channelDecay", "writeStrength"])
        previous = states[t]
        decayed = matmul(diagonal(alpha), previous)
        error = [x - y for x, y in zip(v, mv(transpose(decayed), k))]
        grad["values"][t] = [beta * x for x in mv(transpose(current), k)]
        grad["writeStrength"][t] = sum(x * y for x, y in zip(flatten(current), flatten(outer(k, error))))
        erase = add(eye(len(k)), scale(outer(k, k), -beta))
        decayed_grad = matmul(erase, current)
        grad["channelDecay"][t] = [sum(x * y for x, y in zip(row, old)) for row, old in zip(decayed_grad, previous)]
        current = matmul(diagonal(alpha), decayed_grad)
        if t == detach_at:
            current = zeros(len(k), len(v))
        adjoints[t] = copy.deepcopy(current)
    if start == 0:
        grad["initialState"] = copy.deepcopy(current)
    return grad, adjoints, current


def gradient_slots(data):
    slots = []
    for name in ["initialState", "values", "channelDecay", "writeStrength"]:
        for i, row in enumerate(data[name]):
            if isinstance(row, list):
                slots.extend((name, i, j) for j in range(len(row)))
            else:
                slots.append((name, i, None))
    return slots


def get_slot(data, slot):
    name, i, j = slot
    return data[name][i] if j is None else data[name][i][j]


def set_slot(data, slot, value):
    name, i, j = slot
    if j is None:
        data[name][i] = value
    else:
        data[name][i][j] = value


for vector in INPUTS["keys"] + INPUTS["queries"]:
    close(sum(x * x for x in vector), 1)
assert all(0 <= a <= 1 for a in flatten(INPUTS["channelDecay"]) + INPUTS["writeStrength"])
states, outputs, maps = forward(INPUTS)
dk, dv, n = len(states[0]), len(states[0][0]), len(outputs)
partition_checks = []
for bits in itertools.product([0, 1], repeat=n - 1):
    cuts = [0] + [i + 1 for i, bit in enumerate(bits) if bit] + [n]
    sizes = [b - a for a, b in zip(cuts, cuts[1:])]
    chunk_states, chunk_outputs, _ = chunk_forward(INPUTS, sizes)
    error = max(distance(a, b) for a, b in zip(states, chunk_states))
    error = max(error, distance(outputs, chunk_outputs))
    assert error < 1e-12
    partition_checks.append({"sizes": sizes, "maxStateAndOutputError": error})

loss = terminal_loss(INPUTS)
terminal_error = [x - y for x, y in zip(outputs[-1], INPUTS["terminalTarget"])]
terminal_gradient = outer(INPUTS["queries"][-1], terminal_error)
full_grad, full_adjoints, _ = vjp(INPUTS, states, terminal_gradient)
detached_grad, detached_adjoints, _ = vjp(INPUTS, states, terminal_gradient, detach_at=INPUTS["detachBoundary"])
chunk_states, chunk_outputs, chunk_maps = chunk_forward(INPUTS, INPUTS["chunkSizesForReadout"])
chunk_grad = empty_grad(INPUTS)
incoming, end = copy.deepcopy(terminal_gradient), n
for size, block_map in reversed(list(zip(INPUTS["chunkSizesForReadout"], chunk_maps))):
    start = end - size
    local_grad, _, boundary_gradient = vjp(INPUTS, states, incoming, start, end)
    assert distance(boundary_gradient, matmul(transpose(block_map[0]), incoming)) < 1e-12
    for slot in gradient_slots(INPUTS):
        set_slot(chunk_grad, slot, get_slot(chunk_grad, slot) + get_slot(local_grad, slot))
    incoming, end = boundary_gradient, start
finite_differences = []
for slot in gradient_slots(INPUTS):
    plus, minus = copy.deepcopy(INPUTS), copy.deepcopy(INPUTS)
    value, step = get_slot(INPUTS, slot), INPUTS["finiteDifferenceStep"]
    set_slot(plus, slot, value + step)
    set_slot(minus, slot, value - step)
    finite = (terminal_loss(plus) - terminal_loss(minus)) / (2 * step)
    analytic = get_slot(full_grad, slot)
    close(analytic, get_slot(chunk_grad, slot))
    assert abs(analytic - finite) <= INPUTS["finiteDifferenceTolerance"], slot
    finite_differences.append({"slot": slot, "analytic": analytic, "chunkAnalytic": get_slot(chunk_grad, slot), "finiteDifference": finite, "absoluteError": abs(analytic - finite)})
assert len(finite_differences) == 30
assert abs(full_grad["values"][0][0]) > 1e-6 and detached_grad["values"][0][0] == 0

variants = [{"name": "KDA：逐通道衰减", "output": outputs[-1]},
            {"name": "GDN：每头标量衰减", "output": forward(INPUTS, "GDN")[1][-1]},
            {"name": "错序：先修正再衰减旧状态", "output": forward(INPUTS, "wrong-order")[1][-1]},
            {"name": "误把计算分块当新文档清零", "output": forward(INPUTS, reset_at=2)[1][-1]}]
assert distance([variants[0]["output"]], [variants[2]["output"]]) > 1e-4
training = [{"name": "逐 token 递推", "loss": loss, "earlyValueGradient": full_grad["values"][0][0], "boundaryAdjointNorm": norm(full_adjoints[2])},
            {"name": "仿射分块 [2,2]", "loss": 0.5 * sum((x - y) ** 2 for x, y in zip(chunk_outputs[-1], INPUTS["terminalTarget"])), "earlyValueGradient": chunk_grad["values"][0][0], "boundaryAdjointNorm": norm(matmul(transpose(chunk_maps[-1][0]), terminal_gradient))},
            {"name": "块边界 detach", "loss": loss, "earlyValueGradient": detached_grad["values"][0][0], "boundaryAdjointNorm": norm(detached_adjoints[2])}]

mem = INPUTS["memoryLedger"]
memory = [{"length": length,
           "full": mem["layers"] * mem["headsPerLayer"] * length * (mem["keyDimension"] + mem["valueDimension"]),
           "recurrent": mem["layers"] * mem["headsPerLayer"] * mem["keyDimension"] * mem["valueDimension"],
           "hybrid": mem["headsPerLayer"] * (mem["linearLayers"] * mem["keyDimension"] * mem["valueDimension"] + mem["fullLayers"] * length * (mem["keyDimension"] + mem["valueDimension"]))}
          for length in mem["lengths"]]

boundary = INPUTS["overwriteBoundary"]
a, b = affine_step(boundary["key"], boundary["value"], boundary["channelDecay"], boundary["writeStrength"])
after_a = add(matmul(a, boundary["previousA"]), b)
after_b = add(matmul(a, boundary["previousB"]), b)
assert distance(after_a, after_b) == 0 and distance(boundary["previousA"], boundary["previousB"]) > 0
future = copy.deepcopy(INPUTS)
future["values"][-1][0] += 10
future_states = forward(future)[0]
assert all(distance(a, b) == 0 for a, b in zip(states[:-1], future_states[:-1]))
assert distance(states[-1], future_states[-1]) > 0
empty = copy.deepcopy(INPUTS)
for name in ["keys", "queries", "values", "channelDecay", "scalarDecay", "writeStrength"]:
    empty[name] = []
assert forward(empty)[:2] == ([INPUTS["initialState"]], [])
assert chunk_forward(empty, [])[:2] == ([INPUTS["initialState"]], [])
for beta, alpha in [(0, [0.6, 0.8]), (0.5, [0, 0]), (0.5, [1, 1])]:
    a, b = affine_step([1, 0], [0.4, -0.2, 0.1], alpha, beta)
    state = add(matmul(a, INPUTS["initialState"]), b)
    if beta == 0:
        assert distance(state, matmul(diagonal(alpha), INPUTS["initialState"])) == 0
    if alpha == [0, 0]:
        assert distance(state, b) == 0

results = {"schema": 1, "states": states, "outputs": outputs, "variants": variants, "training": training,
           "partitionChecks": partition_checks, "finiteDifferences": finite_differences, "memory": memory,
           "overwriteBoundary": {"afterA": after_a, "afterB": after_b, "maxDifference": distance(after_a, after_b)},
           "checks": {"finiteDerivatives": 30, "chunkPartitions": 8, "causalSuffixIsolation": True, "emptySequence": True, "gateEndpoints": True}}


class Readouts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.scalars = {}, {}
        self.table, self.row, self.cell, self.scalar = None, None, None, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.table = next((k for k in attrs if k.startswith("data-delta-")), None)
            if self.table:
                self.tables[self.table] = []
        if tag == "tr" and self.table:
            self.row = []
        if tag == "td" and self.row is not None:
            self.cell = ""
        if tag == "strong" and "data-delta-overwrite" in attrs:
            self.scalar = "data-delta-overwrite"
            self.scalars[self.scalar] = ""

    def handle_data(self, value):
        if self.cell is not None:
            self.cell += value
        if self.scalar:
            self.scalars[self.scalar] += value

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        if tag == "tr" and self.row is not None:
            if self.row:
                self.tables[self.table].append(self.row)
            self.row = None
        if tag == "table":
            self.table = None
        if tag == "strong":
            self.scalar = None


f = lambda x: f"{x:.6f}"
page = Readouts()
page.feed((ROOT / "src/content/architecture-variants.html").read_text())
expected_tables = {
    "data-delta-outputs": [[str(i + 1), *[f"{x:.4f}" for x in out]] for i, out in enumerate(outputs)],
    "data-delta-variants": [[row["name"], *map(f, row["output"])] for row in variants],
    "data-delta-training": [[row["name"], f(row["loss"]), f(row["earlyValueGradient"]), f(row["boundaryAdjointNorm"])] for row in training],
    "data-delta-memory": [[str(row[k]) for k in ["length", "full", "recurrent", "hybrid"]] for row in memory],
}


def compare(actual, expected, path=()):
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected
    elif isinstance(expected, int):
        assert type(actual) is int and actual == expected
    elif isinstance(expected, float):
        tolerance = INPUTS["finiteDifferenceTolerance"] if path and path[0] == "finiteDifferences" and path[-1] in ["finiteDifference", "absoluteError"] else 1e-12
        close(actual, expected, tolerance)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, path + (i,))
    else:
        assert actual.keys() == expected.keys()
        for k in expected:
            compare(actual[k], expected[k], path + (k,))


flags = sys.argv[1:]
assert flags in [[], ["--write"], ["--preview"]], "use no argument, --write once, or --preview"
if flags == ["--preview"]:
    print(json.dumps({"tables": expected_tables, "maxFiniteDifferenceError": max(row["absoluteError"] for row in finite_differences)}, ensure_ascii=False, indent=2))
    sys.exit(0)
assert page.tables == expected_tables, (page.tables, expected_tables)
assert page.scalars["data-delta-overwrite"].strip() == "0.000000"
if flags == ["--write"]:
    with (HERE / "results.json").open("x") as stream:
        stream.write(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
else:
    compare(results, json.loads((HERE / "results.json").read_text()))
print(json.dumps({"ok": True, "finiteDerivatives": 30, "chunkPartitions": 8, "authoredTables": 4, "scalarReadouts": 1,
                  "maxFiniteDifferenceError": max(row["absoluteError"] for row in finite_differences),
                  "earlyValueGradient": full_grad["values"][0][0], "detachedEarlyGradient": detached_grad["values"][0][0],
                  "lastOutput": outputs[-1], "mode": "created" if flags else "verified"}, ensure_ascii=False, indent=2))
