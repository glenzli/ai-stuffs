"""One-query SDPA output gate; row vectors, Python 3.10+ standard library."""
from html.parser import HTMLParser
from pathlib import Path
import copy
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = json.loads((HERE / "inputs.json").read_text())


def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))


def rowmat(x, matrix):
    assert len(x) == len(matrix) and len({len(row) for row in matrix}) == 1
    return [sum(x[i] * matrix[i][j] for i in range(len(x))) for j in range(len(matrix[0]))]


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def softmax(logits):
    if not logits:
        raise ValueError("At least one causally eligible key is required")
    maximum = max(logits)
    weights = [math.exp(x - maximum) for x in logits]
    return [w / sum(weights) for w in weights]


def attend(logits, values):
    assert len(logits) == len(values)
    weights = softmax(logits)
    return weights, rowmat(weights, values)


def forward(data):
    weights, head = attend(data["logits"], data["values"])
    z = rowmat(data["input"], data["gateWeight"])
    gate = list(map(sigmoid, z))
    gated = [h * g for h, g in zip(head, gate)]
    output = rowmat(gated, data["outputWeight"])
    error = [a - b for a, b in zip(output, data["target"])]
    return {"attentionWeights": weights, "head": head, "gateLogits": z, "gate": gate,
            "gatedHead": gated, "output": output, "error": error, "loss": 0.5 * dot(error, error)}


def gradients(data):
    f = forward(data)
    # y = r W_o, so dL/dr = e W_o^T.
    t = [dot(f["error"], row) for row in data["outputWeight"]]
    dh = [g * q for g, q in zip(f["gate"], t)]
    dz = [h * g * (1 - g) * q for h, g, q in zip(f["head"], f["gate"], t)]
    return {
        "logits": [a * dot(dh, [v - h for v, h in zip(value, f["head"])]) for a, value in zip(f["attentionWeights"], data["values"])],
        "values": [[a * g for g in dh] for a in f["attentionWeights"]],
        "gateWeight": [[x * z for z in dz] for x in data["input"]],
        "outputWeight": [[r * e for e in f["error"]] for r in f["gatedHead"]],
        "head": dh, "gateLogits": dz,
    }


def close(a, b, tol=None):
    tol = DATA["algebraTolerance"] if tol is None else tol
    assert abs(a - b) <= tol * max(1, abs(a), abs(b)), (a, b, tol)


def vector_close(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        close(x, y)


full = forward(DATA)
analytic = gradients(DATA)
slots = [("logits", i, None) for i in range(3)] + [(field, i, j) for field, matrix in [("values", DATA["values"]), ("gateWeight", DATA["gateWeight"]), ("outputWeight", DATA["outputWeight"])] for i, row in enumerate(matrix) for j in range(len(row))]
finite = []
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
    measured = (forward(plus)["loss"] - forward(minus)["loss"]) / (2 * DATA["finiteDifferenceStep"])
    assert abs(measured - expected) <= DATA["finiteDifferenceTolerance"], (field, i, j)
    finite.append({"slot": [field, i, j], "analytic": expected, "finiteDifference": measured, "absoluteError": abs(measured - expected)})
assert len(finite) == 17

updated = copy.deepcopy(DATA)
updated["gateWeight"] = [[w - DATA["gateLearningRate"] * g for w, g in zip(row, gradient)] for row, gradient in zip(DATA["gateWeight"], analytic["gateWeight"])]
after = forward(updated)
assert after["loss"] < full["loss"]
# A gate-only update at fixed logits/values leaves attention probabilities unchanged.
vector_close(full["attentionWeights"], after["attentionWeights"])

plain = rowmat(full["head"], DATA["outputWeight"])
wrong_after_projection = [y * g for y, g in zip(plain, full["gate"])]
assert max(abs(a - b) for a, b in zip(full["output"], wrong_after_projection)) > 1e-3
scalar = sigmoid(DATA["sharedGateLogit"])
scalar_before = rowmat([scalar * h for h in full["head"]], DATA["outputWeight"])
scalar_after = [scalar * y for y in plain]
vector_close(scalar_before, scalar_after)
folded_weight = [[g * w for w in row] for g, row in zip(full["gate"], DATA["outputWeight"])]
vector_close(rowmat(full["head"], folded_weight), full["output"])
alternate = copy.deepcopy(DATA)
alternate["input"] = [-0.8, 0.9]
alternate_readout = forward(alternate)
assert max(abs(a - b) for a, b in zip(rowmat(alternate_readout["head"], folded_weight), alternate_readout["output"])) > 1e-3

cache_input = DATA["cacheCounterexample"]
current_gate = sigmoid(cache_input["currentGateLogit"])
assert 0 < current_gate < 1e-8
current = rowmat([current_gate * h for h in full["head"]], DATA["outputWeight"])
next_weights, next_head = attend(cache_input["laterLogits"], DATA["values"])
keep = [i for i in range(3) if i != cache_input["wronglyRemovedPosition"]]
pruned_weights, pruned_head = attend([cache_input["laterLogits"][i] for i in keep], [DATA["values"][i] for i in keep])
later_gate = sigmoid(cache_input["laterGateLogit"])
next_output = rowmat([later_gate * h for h in next_head], DATA["outputWeight"])
pruned_output = rowmat([later_gate * h for h in pruned_head], DATA["outputWeight"])
assert max(abs(a - b) for a, b in zip(next_output, pruned_output)) > 0.1
cache = {"currentGate": current_gate, "currentOutput": current, "laterGate": later_gate,
         "allAttentionWeights": next_weights, "prunedAttentionWeights": pruned_weights,
         "allOutput": next_output, "prunedOutput": pruned_output,
         "maxOutputDifference": max(abs(a - b) for a, b in zip(next_output, pruned_output))}

saturation = [{"logit": z, "gate": sigmoid(z), "slope": sigmoid(z) * (1 - sigmoid(z))} for z in DATA["saturationLogits"]]
for row in saturation:
    assert 0 < row["gate"] < 1 and 0 < row["slope"] <= 0.25
close(sigmoid(0), 0.5)
assert all(math.isfinite(sigmoid(x)) for x in [-1000, 1000])
vector_close(softmax([x + 10000 for x in DATA["logits"]]), full["attentionWeights"])
vector_close(attend([0.5], [DATA["values"][0]])[1], DATA["values"][0])
try:
    softmax([])
except ValueError:
    pass
else:
    raise AssertionError("empty eligible set must fail")

readouts = [{"name": "无输出门", "output": plain}, {"name": "G1：逐通道门", "output": full["output"]},
            {"name": "G1：整头标量门", "output": scalar_before}, {"name": "移到 W_O 后（不同机制）", "output": wrong_after_projection}]
selected = [["logits", 0, None], ["values", 0, 0], ["gateWeight", 0, 0], ["outputWeight", 1, 0]]
gradient_rows = [next(row for row in finite if row["slot"] == slot) for slot in selected]
results = {"schema": 1, "forward": full, "gradients": analytic, "finiteDifferences": finite,
           "updatedGateWeight": updated["gateWeight"], "afterGateStep": after, "readouts": readouts,
           "gradientRows": gradient_rows, "cacheCounterexample": cache, "saturation": saturation,
           "boundaries": {"sharedScalarCommutesWithoutBias": True, "fixedGateFoldsIntoOutputWeight": True,
                          "oneFoldDoesNotMatchAlternateInput": True, "attentionWeightsUnchangedByGateOnlyUpdate": True,
                          "singleKey": True, "emptyEligibleSetRejected": True, "stableExtremeSigmoid": True}}

f = lambda x: f"{x:.6f}"
gradient_names = ["logit₀", "V₀[0]", "Wg[0,0]", "Wo[1,0]"]
tables = {
    "data-output-gate-readouts": [[r["name"], *map(f, r["output"])] for r in readouts],
    "data-output-gate-gradients": [[name, f"{r['analytic']:.9f}", f"{r['finiteDifference']:.9f}"] for name, r in zip(gradient_names, gradient_rows)],
    "data-output-gate-cache": [["完整保留 K/V", *map(f, next_output)], ["因前一小门值误删第 0 项", *map(f, pruned_output)]],
    "data-output-gate-saturation": [[f"{r['logit']:g}", f"{r['gate']:.9g}", f"{r['slope']:.9g}"] for r in saturation],
}
scalars = {"data-output-gate-loss-before": f(full["loss"]), "data-output-gate-loss-after": f(after["loss"])}


class Readouts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.scalars = {}, {}
        self.table = self.row = self.cell = self.scalar = None

    def handle_starttag(self, tag, attrs):
        names = [name for name, _ in attrs]
        if tag == "table":
            self.table = next((n for n in names if n in tables), None)
            if self.table:
                self.tables[self.table] = []
        if tag == "tr" and self.table:
            self.row = []
        if tag == "td" and self.row is not None:
            self.cell = ""
        if tag == "strong":
            self.scalar = next((n for n in names if n in scalars), None)
            if self.scalar:
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


def compare(actual, expected, path=()):
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert type(actual) is type(expected) and actual == expected, path
    elif isinstance(expected, int):
        assert type(actual) is int and actual == expected, path
    elif isinstance(expected, float):
        tolerance = DATA["finiteDifferenceTolerance"] if path[-1] in ["finiteDifference", "absoluteError"] else DATA["algebraTolerance"]
        close(actual, expected, tolerance)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, path + (i,))
    else:
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], path + (key,))


flags = sys.argv[1:]
assert flags in [[], ["--write"], ["--preview"]]
if flags == ["--preview"]:
    print(json.dumps({"tables": tables, "scalars": scalars, "forward": full}, ensure_ascii=False, indent=2))
    sys.exit(0)
page = Readouts()
page.feed((ROOT / "src/content/transformer-anatomy.html").read_text())
assert page.tables == tables, (page.tables, tables)
assert page.scalars == scalars, (page.scalars, scalars)
if flags == ["--write"]:
    with (HERE / "results.json").open("x") as stream:
        stream.write(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
else:
    compare(results, json.loads((HERE / "results.json").read_text()))
print(json.dumps({"ok": True, "finiteDerivatives": len(finite), "authoredTables": len(tables), "authoredScalars": len(scalars),
                  "maxFiniteDifferenceError": max(r["absoluteError"] for r in finite), "lossBefore": full["loss"], "lossAfterGateStep": after["loss"],
                  "cacheOutputDifference": cache["maxOutputDifference"], "scope": DATA["scope"]}, ensure_ascii=False))
