"""Recalculate means and sample standard deviations from the paper run files."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "recomputed"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def summarize(name: str, keys: tuple[str, ...]) -> None:
    groups = defaultdict(list)
    for row in read_csv(ROOT / "results" / f"{name}_runs.csv"):
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, rows in sorted(groups.items()):
        assert len(rows) == 20 and len({r["seed"] for r in rows}) == 20, values
        record = dict(zip(keys, values))
        record["runs"] = 20
        for metric in ("ACC", "NMI", "ARI"):
            scores = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = statistics.mean(scores)
            record[f"{metric}_sd"] = statistics.stdev(scores)
        output.append(record)
    fields = [*keys, "runs", "ACC_mean", "ACC_sd", "NMI_mean", "NMI_sd",
              "ARI_mean", "ARI_sd"]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(output)
    print(f"{name}: {len(output)} configurations")


if __name__ == "__main__":
    for experiment in ("main", "kernel", "ablation"):
        summarize(experiment, ("dataset", "method"))
    summarize("outlier", ("dataset", "outlier_fraction", "method"))
