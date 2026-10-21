"""Recompute Section 4.9 from the 20 recorded outcomes per main configuration."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "recomputed"
RESULTS = ROOT / "results"

METHODS = ["KM", "FKM", "RSDE", "JSESR", "FRSDE", "HSCA", "MKKM", "HKFSC"]
METRICS = ["ACC", "NMI", "ARI"]
DATASETS = ["interleaved_moons", "circle_with_blob", "swiss_roll", "concentric_circles",
            "aggregation", "ds850", "ds577", "blobs", "zoo", "3mc", "atom", "iris",
            "coil20", "yale", "jaffe", "usps"]
NUMERICAL = DATASETS[4:12]
IMAGE_MAP = {"COIL20": "coil20", "Yale": "yale", "JAFFE": "jaffe", "USPS": "usps"}
TIE_TOL = 5e-13
NEMENYI_Q = 3.031  # alpha=.05, eight methods; studentized-range q/sqrt(2)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rankdata(values: list[float], reverse: bool = False) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(values):
        end = pos + 1
        while end < len(values) and abs(values[order[end]] - values[order[pos]]) <= TIE_TOL:
            end += 1
        rank = ((pos + 1) + end) / 2
        for j in range(pos, end):
            ranks[order[j]] = rank
        pos = end
    return ranks


def tie_sizes(values: list[float]) -> list[int]:
    sorted_vals = sorted(values)
    out = []
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and abs(sorted_vals[j] - sorted_vals[i]) <= TIE_TOL:
            j += 1
        if j - i > 1:
            out.append(j - i)
        i = j
    return out


def gammp_series(a: float, x: float) -> float:
    summation = term = 1.0 / a
    ap = a
    for _ in range(10000):
        ap += 1.0
        term *= x / ap
        summation += term
        if abs(term) < abs(summation) * 1e-15:
            break
    return summation * math.exp(-x + a * math.log(x) - math.lgamma(a))


def gammq_cf(a: float, x: float) -> float:
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / max(abs(b), tiny) * (1 if b >= 0 else -1)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(stat: float, df: int) -> float:
    a, x = df / 2, stat / 2
    return 1 - gammp_series(a, x) if x < a + 1 else gammq_cf(a, x)


def betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
                     + a*math.log(x)+b*math.log1p(-x))
    if x < (a+1)/(a+b+2):
        return front * betacf(a, b, x) / a
    return 1 - front * betacf(b, a, 1-x) / b


def f_sf(stat: float, df1: int, df2: int) -> float:
    return betai(df2 / 2, df1 / 2, df2 / (df2 + df1 * stat))


def wilcoxon_exact(delta: list[float]) -> tuple[float, float, int, int, int]:
    delta = [x for x in delta if abs(x) > TIE_TOL]
    if not delta:
        return 1.0, 0.0, 0, 0, 0
    ranks = rankdata([abs(x) for x in delta])
    scaled = [int(round(2*r)) for r in ranks]
    assert all(abs(2*r-s) < 1e-8 for r, s in zip(ranks, scaled))
    observed = sum(s for s, x in zip(scaled, delta) if x > 0)
    total = sum(scaled)
    counts = {0: 1}
    for r in scaled:
        next_counts = counts.copy()
        for summation, count in counts.items():
            next_counts[summation+r] = next_counts.get(summation+r, 0) + count
        counts = next_counts
    lower = sum(c for s, c in counts.items() if s <= observed)
    upper = sum(c for s, c in counts.items() if s >= observed)
    p = min(1.0, 2 * min(lower, upper) / 2**len(delta))
    wplus = observed / 2
    wminus = (total-observed) / 2
    effect = (wplus-wminus)/(wplus+wminus)
    return p, effect, sum(x > 0 for x in delta), sum(x < 0 for x in delta), len(delta)


def holm(pvals: list[float]) -> list[float]:
    sorted_indices = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted = [0.0] * len(pvals)
    current = 0.0
    for j, i in enumerate(sorted_indices):
        current = max(current, (len(pvals)-j)*pvals[i])
        adjusted[i] = min(1.0, current)
    return adjusted


def load_final_inputs() -> dict:
    import statistics
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in read_csv(RESULTS / "main_runs.csv"):
        grouped.setdefault((row["dataset"], row["method"]), []).append(row)
    lookup = {}
    for d in DATASETS:
        for m in METHODS:
            rows = grouped[(d, m)]
            if len(rows) != 20:
                raise ValueError((d, m, len(rows)))
            for metric in METRICS:
                lookup[d, m, metric] = statistics.mean(float(r[metric]) for r in rows)
    expected = set(itertools.product(DATASETS, METHODS, METRICS))
    if set(lookup) != expected:
        raise ValueError(f"Incomplete input: {len(lookup)} of {len(expected)} metric cells")
    return lookup


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lookup = load_final_inputs()
    means = [{"dataset": d, "metric": metric, **{m: format(lookup[d,m,metric], ".17g") for m in METHODS}}
             for metric in METRICS for d in DATASETS]
    write_csv(OUT/"final_dataset_means.csv", means, ["dataset", "metric", *METHODS])
    omnibus, pairwise, ranks_out, cd_groups = [], [], [], []
    n, k = len(DATASETS), len(METHODS)
    cd = NEMENYI_Q * math.sqrt(k*(k+1)/(6*n))
    for metric in METRICS:
        matrix = [[lookup[d,m,metric] for m in METHODS] for d in DATASETS]
        ranks = [rankdata(row, reverse=True) for row in matrix]
        avgr = [sum(row[j] for row in ranks)/n for j in range(k)]
        total_ties = sum(t**3-t for row in matrix for t in tie_sizes(row))
        correction = 1 - total_ties/(n*(k**3-k))
        q_uncorrected = 12*n/(k*(k+1))*sum((r-(k+1)/2)**2 for r in avgr)
        q = q_uncorrected/correction
        ff = (n-1)*q/(n*(k-1)-q)
        omnibus.append({"metric": metric, "n_datasets": n, "n_methods": k,
                        "tie_correction": correction, "friedman_q_uncorrected": q_uncorrected,
                        "friedman_q": q, "friedman_p": chi2_sf(q,k-1),
                        "iman_davenport_f": ff, "iman_davenport_p": f_sf(ff,k-1,(k-1)*(n-1)),
                        "hkfsc_average_rank": avgr[-1], "nemenyi_cd": cd})
        for m, r in zip(METHODS, avgr):
            ranks_out.append({"metric": metric, "method": m, "average_rank": r})
        target = [row[-1] for row in matrix]
        local = []
        for j, m in enumerate(METHODS[:-1]):
            diff = [a-row[j] for a,row in zip(target,matrix)]
            p, effect, wins, losses, nonzero = wilcoxon_exact(diff)
            local.append({"metric": metric, "baseline": m, "n_datasets": n,
                          "n_nonzero": nonzero, "wins": wins,
                          "ties": n-nonzero, "losses": losses,
                          "p_raw": p, "rank_biserial": effect,
                          "average_rank_gap": avgr[j]-avgr[-1],
                          "nemenyi_exceeds_cd": avgr[j]-avgr[-1] > cd})
        adjusted = holm([row["p_raw"] for row in local])
        for row,p in zip(local, adjusted):
            row["p_holm"] = p
            pairwise.append(row)
        ordered = sorted(zip(METHODS,avgr), key=lambda pair: pair[1])
        cd_groups.append({"metric": metric, "ordered": ordered})
    write_csv(OUT/"Table17_recomputed.csv", omnibus, list(omnibus[0]))
    write_csv(OUT/"Table18_recomputed.csv", pairwise, list(pairwise[0]))
    write_csv(OUT/"Fig17_average_ranks.csv", ranks_out, list(ranks_out[0]))
    (OUT/"figure_data.json").write_text(json.dumps({"cd":cd,"groups":cd_groups},indent=2),encoding="utf-8")
    (OUT/"source_hashes.json").write_text(json.dumps({"main_runs.csv":
        sha256(RESULTS / "main_runs.csv")},indent=2),encoding="utf-8")
    for row in omnibus:
        print(row)
    for row in pairwise:
        print(row)


if __name__ == "__main__":
    main()
