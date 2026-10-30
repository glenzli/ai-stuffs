"""Single-token depth attention; Python 3.10+ standard library, no model code."""
from pathlib import Path
from html.parser import HTMLParser
import copy
import itertools
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = json.loads((HERE / "inputs.json").read_text())


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def close(a, b, tol=1e-12):
    assert abs(a - b) <= tol * max(1.0, abs(a), abs(b)), (a, b)


def vector_close(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        close(x, y)


def candidates(data, block=False):
    groups = data["blockSourceGroups"] if block else [[i] for i in range(data["eligibleSourceCount"]) ]
    values = [[sum(data["sources"][i][j] for i in group) for j in range(len(data["query"]))] for group in groups]
    return groups, values


def attention(values, query, epsilon):
    if not values:
        raise ValueError("Depth attention requires at least one eligible source")
    assert epsilon > 0 and all(len(v) == len(query) for v in values)
    rms = [math.sqrt(epsilon + dot(v, v) / len(v)) for v in values]
    keys = [[x / r for x in v] for v, r in zip(values, rms)]
    scores = [dot(query, k) for k in keys]
    maximum = max(scores)
    weights = [math.exp(s - maximum) for s in scores]
    denominator = sum(weights)
    numerator = [sum(w * v[j] for w, v in zip(weights, values)) for j in range(len(query))]
    output = [x / denominator for x in numerator]
    return {"rms": rms, "keys": keys, "scores": scores, "weights": [w / denominator for w in weights],
            "output": output, "maximum": maximum, "denominator": denominator, "numerator": numerator,
            "logZ": maximum + math.log(denominator)}


def merge(a, b):
    maximum = max(a["maximum"], b["maximum"])
    x, y = math.exp(a["maximum"] - maximum), math.exp(b["maximum"] - maximum)
    denominator = x * a["denominator"] + y * b["denominator"]
    numerator = [x * p + y * q for p, q in zip(a["numerator"], b["numerator"])]
    return {"maximum": maximum, "denominator": denominator, "numerator": numerator,
            "output": [p / denominator for p in numerator], "logZ": maximum + math.log(denominator)}


def forward(data, block=False):
    return attention(candidates(data, block)[1], data["query"], data["rmsEpsilon"])


def loss(data, block=False):
    error = [x - y for x, y in zip(forward(data, block)["output"], data["target"])]
    return 0.5 * dot(error, error)


def gradients(data, block=False):
    groups, values = candidates(data, block)
    readout = attention(values, data["query"], data["rmsEpsilon"])
    error = [x - y for x, y in zip(readout["output"], data["target"])]
    width = len(error)
    direct, routed = [[0.0] * width for _ in data["sources"]], [[0.0] * width for _ in data["sources"]]
    query_grad = [0.0] * width
    for group, value, r, key, weight in zip(groups, values, readout["rms"], readout["keys"], readout["weights"]):
        score_grad = weight * dot(error, [x - y for x, y in zip(value, readout["output"])])
        local_direct = [weight * e for e in error]
        local_route = [score_grad * (q / r - dot(data["query"], value) * x / (width * r ** 3)) for q, x in zip(data["query"], value)]
        query_grad = [a + score_grad * b for a, b in zip(query_grad, key)]
        for i in group:
            direct[i], routed[i] = local_direct[:], local_route[:]
    source_grad = [[a + b for a, b in zip(x, y)] for x, y in zip(direct, routed)]
    return {"sources": source_grad, "query": query_grad, "direct": direct, "routed": routed}


full, block = forward(DATA), forward(DATA, True)
group_checks = []
for name, is_block in [("Full", False), ("Block", True)]:
    values = candidates(DATA, is_block)[1]
    direct = attention(values, DATA["query"], DATA["rmsEpsilon"])
    for cuts in itertools.product([0, 1], repeat=len(values) - 1):
        endpoints = [0] + [i + 1 for i, cut in enumerate(cuts) if cut] + [len(values)]
        summaries = [attention(values[a:b], DATA["query"], DATA["rmsEpsilon"]) for a, b in zip(endpoints, endpoints[1:])]
        merged = summaries[0]
        for summary in summaries[1:]:
            merged = merge(merged, summary)
        vector_close(merged["output"], direct["output"])
        close(merged["logZ"], direct["logZ"])
        group_checks.append({"kind": name, "sizes": [b - a for a, b in zip(endpoints, endpoints[1:])],
                             "maxOutputError": max(abs(x - y) for x, y in zip(merged["output"], direct["output"]))})
assert len(group_checks) == 12
block_values = candidates(DATA, True)[1]
phases = [attention([block_values[i] for i in group], DATA["query"], DATA["rmsEpsilon"]) for group in DATA["blockPhaseGroups"]]
two_phase = merge(*phases)
bad_average = [(x + y) / 2 for x, y in zip(phases[0]["output"], phases[1]["output"])]
wrong_values = [sum(a * k[j] for a, k in zip(full["weights"], full["keys"])) for j in range(len(DATA["query"]))]
assert max(abs(x - y) for x, y in zip(full["output"], block["output"])) > 1e-3
assert max(abs(x - y) for x, y in zip(bad_average, block["output"])) > 1e-3
vector_close(two_phase["output"], block["output"])

slots = [("sources", i, j) for i, v in enumerate(DATA["sources"]) for j in range(len(v))] + [("query", j, None) for j in range(len(DATA["query"]))]
finite_differences, gradient_rows = [], []
for name, is_block in [("Full", False), ("Block", True)]:
    analytic = gradients(DATA, is_block)
    for field, i, j in slots:
        plus, minus = copy.deepcopy(DATA), copy.deepcopy(DATA)
        if j is None:
            plus[field][i] += DATA["finiteDifferenceStep"]
            minus[field][i] -= DATA["finiteDifferenceStep"]
            expected = analytic[field][i]
        else:
            plus[field][i][j] += DATA["finiteDifferenceStep"]
            minus[field][i][j] -= DATA["finiteDifferenceStep"]
            expected = analytic[field][i][j]
        finite = (loss(plus, is_block) - loss(minus, is_block)) / (2 * DATA["finiteDifferenceStep"])
        assert abs(finite - expected) <= DATA["finiteDifferenceTolerance"], (name, field, i, j)
        finite_differences.append({"kind": name, "slot": [field, i, j], "analytic": expected, "finiteDifference": finite, "absoluteError": abs(finite - expected)})
    chosen = next(row for row in finite_differences if row["kind"] == name and row["slot"] == ["sources", 1, 0])
    gradient_rows.append({"kind": name, "direct": analytic["direct"][1][0], "routed": analytic["routed"][1][0], "total": analytic["sources"][1][0], "finiteDifference": chosen["finiteDifference"]})

updated = copy.deepcopy(DATA)
updated["query"] = [q - DATA["queryLearningRate"] * g for q, g in zip(DATA["query"], gradients(DATA)["query"])]
assert loss(updated) < loss(DATA)
query_update = [{"stage": "更新前", "query": DATA["query"], "loss": loss(DATA)}, {"stage": "一次 query SGD 后", "query": updated["query"], "loss": loss(updated)}]
uniform = attention(DATA["sources"], [0.0] * len(DATA["query"]), DATA["rmsEpsilon"])
vector_close(uniform["output"], [sum(v[j] for v in DATA["sources"]) / len(DATA["sources"]) for j in range(len(DATA["query"]))])
first_layer = attention(block_values[:-1], DATA["query"], DATA["rmsEpsilon"])
phantom_partial = attention(block_values[:-1] + [[0.0] * len(DATA["query"])], DATA["query"], DATA["rmsEpsilon"])
assert max(abs(x - y) for x, y in zip(first_layer["output"], phantom_partial["output"])) > 1e-3
try:
    attention([], DATA["query"], DATA["rmsEpsilon"])
except ValueError:
    pass
else:
    raise AssertionError("empty source list must fail")
single = attention([DATA["sources"][0]], DATA["query"], DATA["rmsEpsilon"])
vector_close(single["output"], DATA["sources"][0])
vector_close(attention(list(reversed(DATA["sources"])), DATA["query"], DATA["rmsEpsilon"])["output"], full["output"])
future = copy.deepcopy(DATA)
future["sources"].append(DATA["futureSource"])
vector_close(forward(future)["output"], full["output"])
large = attention(DATA["sources"], [q * 10000 for q in DATA["query"]], DATA["rmsEpsilon"])
assert all(math.isfinite(x) for x in large["output"]) and math.isfinite(large["logZ"])

readouts = [{"name": "Full：四个原始来源", "output": full["output"]},
            {"name": "Block：embedding / f1+f2 / f3", "output": block["output"]},
            {"name": "Block：两阶段精确合并", "output": two_phase["output"]},
            {"name": "错误：两组结果各取一半", "output": bad_average},
            {"name": "错误：把归一化键当作值", "output": wrong_values}]
results = {"schema": 1, "full": full, "block": block, "phaseSummaries": phases, "twoPhase": two_phase,
           "readouts": readouts, "gradientRows": gradient_rows, "finiteDifferences": finite_differences,
           "groupChecks": group_checks, "queryUpdate": query_update,
           "boundaries": {"queryZero": uniform["output"], "firstLayerWithoutPartial": first_layer["output"], "phantomZeroPartial": phantom_partial["output"], "singleSource": single["output"], "futureExcluded": True, "largeScoresFinite": True}}


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.table, self.row, self.cell = {}, None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = next((name for name, _ in attrs if name.startswith("data-attnres-")), None)
            if self.table:
                self.tables[self.table] = []
        if tag == "tr" and self.table:
            self.row = []
        if tag == "td" and self.row is not None:
            self.cell = ""

    def handle_data(self, value):
        if self.cell is not None:
            self.cell += value

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


f = lambda value: f"{value:.6f}"
tables = {
    "data-attnres-weights": [[name, f(score), f(weight)] for name, score, weight in zip(DATA["sourceNames"], full["scores"], full["weights"])],
    "data-attnres-readouts": [[row["name"], *map(f, row["output"])] for row in readouts],
    "data-attnres-gradients": [[row["kind"], *[f(row[k]) for k in ["direct", "routed", "total", "finiteDifference"]]] for row in gradient_rows],
    "data-attnres-update": [[row["stage"], *map(f, row["query"]), f(row["loss"])] for row in query_update],
}


def compare(actual, expected, path=()):
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected
    elif isinstance(expected, int):
        assert type(actual) is int and actual == expected
    elif isinstance(expected, float):
        tolerance = DATA["finiteDifferenceTolerance"] if path[-1] in ["finiteDifference", "absoluteError"] else 1e-12
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
assert flags in [[], ["--write"], ["--preview"]]
if flags == ["--preview"]:
    print(json.dumps({"tables": tables, "boundaries": results["boundaries"]}, ensure_ascii=False, indent=2))
    sys.exit(0)
page = Tables()
page.feed((ROOT / "src/content/architecture-variants.html").read_text())
assert page.tables == tables, (page.tables, tables)
if flags == ["--write"]:
    with (HERE / "results.json").open("x") as stream:
        stream.write(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
else:
    compare(results, json.loads((HERE / "results.json").read_text()))
print(json.dumps({"ok": True, "finiteDerivatives": len(finite_differences), "executionPartitions": len(group_checks), "authoredTables": len(tables),
                  "maxFiniteDifferenceError": max(row["absoluteError"] for row in finite_differences), "fullLossBefore": loss(DATA), "fullLossAfterQueryStep": loss(updated),
                  "mode": "created" if flags else "verified"}, ensure_ascii=False, indent=2))
