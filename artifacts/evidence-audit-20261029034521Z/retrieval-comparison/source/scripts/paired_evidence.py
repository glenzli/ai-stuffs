"""Deterministic paired resampling of whole units, retaining ratio denominators."""
import math


def bootstrap_ratio(units, repetitions=2000, random_seed=73):
    if not units or not isinstance(repetitions, int) or repetitions < 100:
        raise ValueError("Need nonempty independent units and at least 100 resamples")
    if not isinstance(random_seed, int) or not 0 <= random_seed < 2**32:
        raise ValueError("Seed must be uint32")
    for total, count in units:
        if not math.isfinite(total) or not isinstance(count, int) or count <= 0:
            raise ValueError("Each unit needs a finite paired total and a positive integer denominator")
    state = random_seed; distribution = []
    for _ in range(repetitions):
        total = 0.; count = 0
        for _ in units:
            state = (1664525 * state + 1013904223) % 2**32
            value, size = units[math.floor(state / 2**32 * len(units))]
            total += value; count += size
        distribution.append(total / count)
    distribution.sort()
    def quantile(p):
        position = (len(distribution) - 1) * p; lower = math.floor(position); fraction = position - lower
        return distribution[lower] * (1 - fraction) + distribution[min(lower + 1, len(distribution) - 1)] * fraction
    return {"estimate": sum(t for t, _ in units) / sum(n for _, n in units), "interval": [quantile(.025), quantile(.975)],
            "units": len(units), "denominator": sum(n for _, n in units), "distribution": distribution,
            "repetitions": repetitions, "random_seed": random_seed, "method": "paired percentile; linear quantiles; LCG uint32"}


def paired_rows(rows, mode="group"):
    if mode not in ("group", "row"): raise ValueError("Unknown resampling unit")
    groups = {}; seen = set()
    for row in rows:
        if row["id"] in seen: raise ValueError("Duplicate paired observation")
        seen.add(row["id"])
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in ("old", "new")):
            raise ValueError("Paired scores must lie in [0,1]")
        key = row["group"] if mode == "group" else row["id"]
        values = groups.setdefault(key, [0., 0]); values[0] += row["new"] - row["old"]; values[1] += 1
    result = bootstrap_ratio(list(groups.values()))
    result.update({"mode": mode, "wins": sum(r["new"] > r["old"] for r in rows),
                   "losses": sum(r["new"] < r["old"] for r in rows), "ties": sum(r["new"] == r["old"] for r in rows)})
    return result
