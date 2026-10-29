"""Fixed teaching arrays; no model, GPU or distributed runtime is executed.

Run at repository root with Python 3.10+. Default recomputes and verifies saved
results and authored readouts. --write creates results.json once, after checks.
"""
import copy
import itertools
import json
import math
import sys
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def near(a, b, tol=1e-8):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol, (a, b)


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def lookup_forward(cfg, tables=None, tokens=None):
    tables = tables or [cfg["table0"], cfg["table1"]]
    tokens = cfg["tokens"] if tokens is None else tokens
    assert all(isinstance(x, int) and x >= 0 for x in tokens)
    assert len(tables) == 2 and all(all(math.isfinite(x) for x in a) for a in tables)
    rows, total = [], 0
    for t, token in enumerate(tokens):
        total += token
        context = [token/10, total/10]
        if t == 0:
            rows.append({"position":t, "ngram":None, "addresses":None, "context":context, "memory":[0,0], "gate":0, "residual":[0,0], "output":context[:]})
            continue
        pair = [tokens[t-1], token]
        addresses = [(cfg["hash"+str(h)]["previousCoefficient"]*pair[0]+cfg["hash"+str(h)]["currentCoefficient"]*pair[1]) % cfg["hash"+str(h)]["modulus"] for h in range(2)]
        assert all(len(tables[h]) == cfg["hash"+str(h)]["modulus"] for h in range(2))
        memory = [tables[h][addresses[h]] for h in range(2)]
        gate = 1/(1+math.exp(-dot(context, memory)/math.sqrt(2)))
        residual = [gate*x for x in memory]
        rows.append({"position":t, "ngram":pair, "addresses":addresses, "context":context, "memory":memory, "gate":gate, "residual":residual, "output":[x+y for x, y in zip(context, residual)]})
    return rows


def lookup_checks(cfg):
    rows = lookup_forward(cfg)
    contributions, gradients = {}, [[0.0]*len(cfg["table0"]), [0.0]*len(cfg["table1"])]
    for row, upstream in zip(rows[1:], cfg["upstream"][1:]):
        a, memory, context = row["gate"], row["memory"], row["context"]
        for h in range(2):
            local = a*upstream[h]+dot(upstream, memory)*a*(1-a)*context[h]/math.sqrt(2)
            address = row["addresses"][h]
            gradients[h][address] += local
            contributions.setdefault((h,address), []).append({"position":row["position"],"gradient":local})
    def loss(tables):
        return sum(dot(row["output"], u) for row,u in zip(lookup_forward(cfg,tables), cfg["upstream"]))
    checks = 0
    for h, table in enumerate([cfg["table0"],cfg["table1"]]):
        for i, value in enumerate(table):
            plus, minus = copy.deepcopy([cfg["table0"],cfg["table1"]]), copy.deepcopy([cfg["table0"],cfg["table1"]])
            plus[h][i] = value+1e-5; minus[h][i] = value-1e-5
            near(gradients[h][i], (loss(plus)-loss(minus))/2e-5); checks += 1
    touched = [{"head":h,"row":i,"contributions":parts,"gradient":gradients[h][i]} for (h,i),parts in sorted(contributions.items())]
    assert len(touched) == 4
    assert rows[1]["addresses"] == rows[3]["addresses"] and rows[1]["gate"] != rows[3]["gate"]
    suffix_rows = lookup_forward(cfg,tokens=cfg["suffixVariant"])
    assert rows[:2] == suffix_rows[:2], "prefix reads must not depend on the suffix"
    assert lookup_forward(cfg,tokens=[]) == []
    assert lookup_forward(cfg,tokens=[1])[0]["addresses"] is None
    updated = [[v-cfg["learningRate"]*gradients[h][i] for i,v in enumerate(table)] for h,table in enumerate([cfg["table0"],cfg["table1"]])]
    before, after = loss([cfg["table0"],cfg["table1"]]), loss(updated)
    assert after < before
    def addresses(pair):
        return [(cfg["hash"+str(h)]["previousCoefficient"]*pair[0]+cfg["hash"+str(h)]["currentCoefficient"]*pair[1]) % cfg["hash"+str(h)]["modulus"] for h in range(2)]
    first, second = cfg["partialCollisionPairs"]
    assert addresses(first)[0] == addresses(second)[0] and addresses(first)[1] != addresses(second)[1]
    lo, hi = cfg["collisionSearchDomain"]
    collisions = [list(pair) for pair in itertools.product(range(lo,hi+1),repeat=2) if list(pair) != first and addresses(pair) == addresses(first)]
    assert collisions
    collision_rows = [{"pair":pair,"addresses":addresses(pair)} for pair in [first,second,collisions[0]]]
    return {"inputs":cfg,"rows":rows,"touchedRows":touched,"gradients":gradients,"finiteDifferenceChecks":checks,"beforeLoss":before,"afterLoss":after,"updatedTables":updated,"collisionRows":collision_rows,"prefixAndEmptyGuardsPassed":True}


def headwise_checks(cfg):
    matrix = cfg["matrix"]
    assert len(matrix) == 4 and all(len(row)==2 for row in matrix)
    def polar(a):
        # The fixed columns are orthogonal and nonzero, so this is the exact
        # M(M^T M)^(-1/2), without a general SVD implementation.
        near(sum(row[0]*row[1] for row in a),0)
        scales = [norm([row[j] for row in a]) for j in range(2)]
        assert min(scales)>0
        q = [[row[j]/scales[j] for j in range(2)] for row in a]
        for i in range(2):
            for j in range(2): near(sum(row[i]*row[j] for row in q),float(i==j))
        return q
    full = polar(matrix)
    by_head = [None]*len(matrix)
    for indices in cfg["headRows"]:
        for i,row in zip(indices,polar([matrix[i] for i in indices])): by_head[i] = row
    def unit(a):
        scale = norm([x for row in a for x in row])
        return [[x/scale for x in row] for row in a]
    full_unit, head_unit = unit(full), unit(by_head)
    cosine = dot([x for row in full_unit for x in row],[x for row in head_unit for x in row])
    assert cosine < 0.95
    return {"inputs":cfg,"fullPolar":full,"headPolars":by_head,"fullUnit":full_unit,"headUnit":head_unit,"cosine":cosine}


def balanced_step(cfg, gradient=None, previous=None):
    g = copy.deepcopy(cfg["gradient"] if gradient is None else gradient)
    old = cfg["previousMomentum"] if previous is None else previous
    m,n = len(g),len(g[0])
    assert m>=n and n>0 and all(len(row)==n and all(math.isfinite(v) for v in row) for row in g+old+cfg["weights"])
    assert cfg["weightDecay"]==0 and cfg["learningRate"]>=0 and cfg["learningRateCorrection"]>=0
    beta, eps, tau, steps = cfg["beta"],cfg["epsilon"],cfg["tau"],cfg["normalizationSteps"]
    assert 0<=beta<1 and eps>0 and tau>=0 and isinstance(steps,int) and steps>0 and steps%2==1
    momentum = [[beta*old[i][j]+(1-beta)*g[i][j] for j in range(n)] for i in range(m)]
    lookahead = [[beta*momentum[i][j]+(1-beta)*g[i][j] for j in range(n)] for i in range(m)]
    row_norms = [norm(row) for row in lookahead]; average = sum(row_norms)/m
    masked = [r<=tau*average for r in row_norms]
    u = [[0.0]*n if masked[i] else row[:] for i,row in enumerate(lookahead)]
    stages = []
    for k in range(1,steps+1):
        if k%2:
            u = [[v/(norm(row)+eps) for v in row] for row in u]
        else:
            denominators = [norm([row[j] for row in u])+eps for j in range(n)]
            u = [[v/denominators[j] for j,v in enumerate(row)] for row in u]
        stages.append({"step":k,"axis":"行" if k%2 else "列","matrix":copy.deepcopy(u),"rowNorms":[norm(row) for row in u],"columnNorms":[norm([row[j] for row in u]) for j in range(n)]})
    delta = [[math.sqrt(n)*v for v in row] for row in u]
    rate = cfg["learningRate"]*cfg["learningRateCorrection"]
    after = [[w-rate*delta[i][j] for j,w in enumerate(row)] for i,row in enumerate(cfg["weights"])]
    return {"momentum":momentum,"nesterov":lookahead,"initialRowNorms":row_norms,"rowThreshold":tau*average,"masked":masked,"stages":stages,"delta":delta,"effectiveRate":rate,"afterWeights":after}


def balanced_checks(cfg):
    step = balanced_step(cfg)
    assert step["masked"] == [False,False,False,True]
    assert cfg["gradient"][1] == [0,0] and norm(step["delta"][1]) > 1
    assert step["delta"][3] == [0,0] and step["afterWeights"][3] == cfg["weights"][3]
    assert any(v<0 for row in step["delta"] for v in row), "signed update is not a nonnegative stochastic matrix"
    zero = [[0,0] for _ in cfg["gradient"]]
    assert all(v==0 for row in balanced_step(cfg,zero,zero)["delta"] for v in row)
    tiny = copy.deepcopy(cfg["gradient"]); tiny[3] = [1e-9,-1e-9]
    near_zero = balanced_step(cfg,tiny)
    assert near_zero["masked"][3] and near_zero["delta"][3] == [0,0]
    bad = dict(cfg,normalizationSteps=2)
    try: balanced_step(bad)
    except AssertionError: pass
    else: raise AssertionError("even normalization count must be rejected")
    return {"inputs":cfg,**step,"zeroAndShapeGuardsPassed":True}


def schedule_checks(cfg):
    nodes = cfg["nodes"]; ids = [n["id"] for n in nodes]; index = {name:i for i,name in enumerate(ids)}
    assert len(index)==len(ids) and all(n["lane"] in ["compute","network"] and n["duration"]>0 for n in nodes)
    base_parents = [set(index[d] for d in node["deps"]) for node in nodes]
    compute = [i for i,n in enumerate(nodes) if n["lane"]=="compute"]
    network = [i for i,n in enumerate(nodes) if n["lane"]=="network"]
    cases = {duration:{"networkDuration":duration,"orderPairsChecked":0,"validOrders":0,"histogram":{},"best":None,"bestPeakOne":None} for duration in cfg["networkDurations"]}

    def plan(parents, durations):
        pending = set(range(len(nodes))); end=[0]*len(nodes); start=[0]*len(nodes)
        while pending:
            ready = [i for i in sorted(pending) if not(parents[i]&pending)]
            if not ready: return None
            for i in ready:
                start[i] = max((end[p] for p in parents[i]),default=0)
                end[i] = start[i]+durations[i]; pending.remove(i)
        old_done = max(end[index[name]] for name in ["BMI","BMW","BAI","BAW"])
        peak = 2 if start[index["FA"]] < old_done else 1
        events = [{"id":n["id"],"lane":n["lane"],"start":start[i],"finish":end[i],"deps":n["deps"]} for i,n in enumerate(nodes)]
        finish=max(end); busy={lane:sum(durations[i] for i,n in enumerate(nodes) if n["lane"]==lane) for lane in ["compute","network"]}
        return {"finish":finish,"activationPeak":peak,"oldBundleReleasedAt":old_done,"newBundleAcquiredAt":start[index["FA"]],"events":events,"busy":busy,"idle":{lane:finish-work for lane,work in busy.items()},"workLowerBound":max(busy.values())}

    def verify(p,durations):
        events = {x["id"]:x for x in p["events"]}
        assert len(events)==len(nodes)
        for i,node in enumerate(nodes):
            event=events[node["id"]]
            assert event["finish"]-event["start"]==durations[i] and event["start"]>=0
            assert all(events[d]["finish"]<=event["start"] for d in node["deps"])
        for lane in ["compute","network"]:
            intervals=sorted((e["start"],e["finish"]) for e in p["events"] if e["lane"]==lane)
            assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
        assert p["finish"]>=p["workLowerBound"]

    for corder in itertools.permutations(compute):
        for norder in itertools.permutations(network):
            parents=[s.copy() for s in base_parents]
            for order in [corder,norder]:
                for a,b in zip(order,order[1:]): parents[b].add(a)
            # Feasibility is independent of durations.
            feasible = plan(parents,[1]*len(nodes))
            for duration,case in cases.items():
                case["orderPairsChecked"]+=1
                if feasible is None: continue
                durations=[duration if n["lane"]=="network" else n["duration"] for n in nodes]
                p=plan(parents,durations);verify(p,durations)
                p["laneOrders"]={"compute":[ids[i] for i in corder],"network":[ids[i] for i in norder]}
                case["validOrders"]+=1
                key=str(p["finish"]);case["histogram"][key]=case["histogram"].get(key,0)+1
                if case["best"] is None or p["finish"]<case["best"]["finish"]: case["best"]=p
                if p["activationPeak"]==1 and (case["bestPeakOne"] is None or p["finish"]<case["bestPeakOne"]["finish"]): case["bestPeakOne"]=p
    for duration,case in cases.items():
        durations=[duration if n["lane"]=="network" else n["duration"] for n in nodes]
        serial_parents=[s.copy() for s in base_parents]
        for a,b in zip(cfg["serialOrder"],cfg["serialOrder"][1:]): serial_parents[index[b]].add(index[a])
        serial=plan(serial_parents,durations);verify(serial,durations);case["serial"]=serial
        assert case["orderPairsChecked"]==math.factorial(len(compute))*math.factorial(len(network))==17280
        assert sum(case["histogram"].values())==case["validOrders"]>0
        assert case["best"]["finish"]<serial["finish"] and case["bestPeakOne"]["finish"]>case["best"]["finish"]
        assert case["best"]["activationPeak"]==2 and case["bestPeakOne"]["activationPeak"]==1
        wrong=copy.deepcopy(case["best"])
        wrong["events"][index["FD"]].update(start=0,finish=durations[index["FD"]])
        overlap=copy.deepcopy(case["best"])
        start=overlap["events"][index["BMI"]]["start"]
        overlap["events"][index["BMW"]].update(start=start,finish=start+durations[index["BMW"]])
        for bad_plan in [wrong,overlap]:
            try: verify(bad_plan,durations)
            except AssertionError: pass
            else: raise AssertionError("invalid dependency/resource timing accepted")
    return {"inputs":cfg,"cases":[cases[x] for x in cfg["networkDurations"]],"exhaustiveOrderPairsPerCase":17280,"totalOrderPairs":17280*len(cases),"negativeTimingGuardPassed":True}


def compute():
    inputs=json.loads((HERE/"inputs.json").read_text())
    return {"schema":"lookup-updates-overlap-results-v1","boundary":inputs["boundary"],"lookup":lookup_checks(inputs["lookup"]),"headwise":headwise_checks(inputs["headwise"]),"balancedUpdate":balanced_checks(inputs["balancedUpdate"]),"schedule":schedule_checks(inputs["schedule"])}


class Readouts(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables={};self.scalars={};self.events={};self.table=None;self.tbody=False;self.cells=None;self.cell=None;self.scalar=None
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="rect" and "data-luo-event" in attrs:
            assert attrs["data-luo-event"] not in self.events
            self.events[attrs["data-luo-event"]]={k:float(attrs[k]) for k in ["x","y","width","height"]}
        if tag=="table":
            self.table=next((k for k in attrs if k.startswith("data-luo-")),None)
            if self.table:
                assert self.table not in self.tables
                self.tables[self.table]=[]
        if tag=="tbody" and self.table:self.tbody=True
        if tag=="tr" and self.table and self.tbody:self.cells=[]
        if tag=="td" and self.cells is not None:self.cell=""
        if tag=="span":
            key=next((k for k in attrs if k.startswith("data-luo-")),None)
            if key:
                assert key not in self.scalars
                self.scalar=key;self.scalars[key]=""
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data
        if self.scalar:self.scalars[self.scalar]+=data
    def handle_endtag(self,tag):
        if tag=="td" and self.cell is not None:self.cells.append(self.cell.strip().replace("−","-"));self.cell=None
        if tag=="tr" and self.cells is not None:
            self.tables[self.table].append(self.cells);self.cells=None
        if tag=="tbody":self.tbody=False
        if tag=="table":self.table=None
        if tag=="span":self.scalar=None


def verify_authored(result):
    # Read the same marker tables rendered on the pages, not a duplicate fixture.
    readers=[]
    for name in ["model-case-studies","optimization-dynamics","training-systems-deep"]:
        reader=Readouts();reader.feed((ROOT/"src/content"/(name+".html")).read_text());readers.append(reader)
    tables={k:v for reader in readers for k,v in reader.tables.items()};scalars={k:v.strip().replace("−","-") for reader in readers for k,v in reader.scalars.items()}
    assert len(tables)==sum(len(reader.tables) for reader in readers)
    assert len(scalars)==sum(len(reader.scalars) for reader in readers)
    fmt=lambda x:format(x,".6f")
    vec=lambda a:" / ".join(fmt(x) for x in a)
    lookup=result["lookup"];heads=result["headwise"];balanced=result["balancedUpdate"];schedule=result["schedule"]
    expected={
        "data-luo-lookup":[[str(r["position"]),"—" if r["ngram"] is None else " / ".join(map(str,r["ngram"])),"不读表" if r["addresses"] is None else " / ".join(map(str,r["addresses"])),fmt(r["gate"]),vec(r["residual"])] for r in lookup["rows"]],
        "data-luo-lookup-grad":[["E"+str(r["head"])+"["+str(r["row"])+"]"," / ".join(str(x["position"]) for x in r["contributions"])," + ".join(fmt(x["gradient"]) for x in r["contributions"]),fmt(r["gradient"])] for r in lookup["touchedRows"]],
        "data-luo-collisions":[[" / ".join(map(str,r["pair"]))," / ".join(map(str,r["addresses"]))] for r in lookup["collisionRows"]],
        "data-luo-headwise":[[str(i),vec(row),vec(heads["fullUnit"][i]),vec(heads["headUnit"][i])] for i,row in enumerate(heads["inputs"]["matrix"])],
        "data-luo-balanced":[[str(i),vec(row),vec(balanced["inputs"]["previousMomentum"][i]),vec(balanced["delta"][i]),vec(balanced["afterWeights"][i])] for i,row in enumerate(balanced["inputs"]["gradient"])],
        "data-luo-stages":[[str(s["step"]),s["axis"],vec(s["rowNorms"]),vec(s["columnNorms"])] for s in balanced["stages"]],
        "data-luo-schedules":[[str(c["networkDuration"]),str(c["serial"]["finish"]),str(c["best"]["finish"]),str(c["bestPeakOne"]["finish"]),str(c["best"]["workLowerBound"])] for c in schedule["cases"]],
    }
    case=next(c for c in schedule["cases"] if c["networkDuration"]==3)
    expected["data-luo-events"]=[[e["id"],"计算" if e["lane"]=="compute" else "通信",str(e["start"]),str(e["finish"])," / ".join(e["deps"]) or "—"] for e in sorted(case["best"]["events"],key=lambda e:(e["start"],e["id"]))]
    expected["data-luo-memory"]=[[label,str(p["finish"]),str(p["activationPeak"]),str(p["newBundleAcquiredAt"]),str(p["oldBundleReleasedAt"])] for label,p in [("显式串行",case["serial"]),("双资源最短",case["best"]),("激活峰值限 1",case["bestPeakOne"])]]
    assert tables==expected,(tables,expected)
    events={k:v for reader in readers for k,v in reader.events.items()}
    expected_events={e["id"]:{"x":80+50*e["start"],"y":34 if e["lane"]=="compute" else 98,"width":50*(e["finish"]-e["start"]),"height":42} for e in case["best"]["events"]}
    assert events==expected_events,(events,expected_events)
    expected_scalars={"data-luo-lookup-before":fmt(lookup["beforeLoss"]),"data-luo-lookup-after":fmt(lookup["afterLoss"]),"data-luo-cosine":fmt(heads["cosine"])}
    assert scalars==expected_scalars,(scalars,expected_scalars)
    result.update(passed=True,authoredTablesMatched=len(expected),authoredScalarValuesMatched=len(expected_scalars),authoredTimelineRectanglesMatched=len(expected_events))


def compare_saved(actual,saved):
    if isinstance(actual,float) and isinstance(saved,(int,float)):
        near(actual,saved,1e-12*max(1,abs(actual),abs(saved)))
    elif isinstance(actual,dict):
        assert actual.keys()==saved.keys()
        for k in actual:compare_saved(actual[k],saved[k])
    elif isinstance(actual,list):
        assert len(actual)==len(saved)
        for a,b in zip(actual,saved):compare_saved(a,b)
    else:assert actual==saved,(actual,saved)


if __name__=="__main__":
    assert all(arg=="--write" for arg in sys.argv[1:]),"unknown flag"
    result=compute();verify_authored(result)
    if "--write" in sys.argv:
        with (HERE/"results.json").open("x") as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write("\n")
    else:compare_saved(result,json.loads((HERE/"results.json").read_text()))
    print(json.dumps(result,ensure_ascii=False,indent=2))
