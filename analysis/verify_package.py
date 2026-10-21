"""Verify file hashes, 20-run coverage, reported means and statistical tables."""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path

import statistical_tests as stats


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("interleaved_moons", "circle_with_blob", "swiss_roll", "concentric_circles",
            "aggregation", "ds850", "ds577", "blobs", "zoo", "3mc", "atom", "iris",
            "coil20", "yale", "jaffe", "usps")
METHODS = ("KM", "FKM", "RSDE", "JSESR", "FRSDE", "HSCA", "MKKM", "HKFSC")


def rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a: float | str, b: float | str, tol: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol)


def check_run_file(name: str, expected: int, keys: tuple[str, ...]) -> list[dict[str, str]]:
    data = rows(f"results/{name}_runs.csv")
    assert len(data) == expected, (name, len(data))
    groups = defaultdict(list)
    for row in data:
        groups[tuple(row[key] for key in keys)].append(row)
        assert int(row["run_id"]) == int(row["seed"]) - 1000
        assert all(math.isfinite(float(row[metric])) for metric in ("ACC", "NMI", "ARI"))
        assert row["termination_status"]
    summary = {tuple(r[key] for key in keys): r for r in rows(f"results/{name}_summary.csv")}
    assert set(groups) == set(summary)
    for key, group in groups.items():
        assert len(group) == 20 and {int(r["seed"]) for r in group} == set(range(1001, 1021)), key
        assert int(summary[key]["runs"]) == 20
        for metric in ("ACC", "NMI", "ARI"):
            values = [float(r[metric]) for r in group]
            assert close(statistics.mean(values), summary[key][f"{metric}_mean"])
            assert close(statistics.stdev(values), summary[key][f"{metric}_sd"])
    return data


def main() -> None:
    manifest = rows("SHA256SUMS.csv")
    for record in manifest:
        path = ROOT / record["file"]
        assert path.is_file() and int(record["bytes"]) == path.stat().st_size, path
        assert sha256(path) == record["sha256"], path
    datasets = rows("data/manifest.csv")
    assert len(datasets) == 22
    assert {r["file"] for r in datasets} == {p.name for p in (ROOT / "data").glob("*.mat")}
    for row in datasets:
        assert sha256(ROOT / "data" / row["file"]) == row["export_sha256"]

    main_runs = check_run_file("main", 2560, ("dataset", "method"))
    check_run_file("kernel", 1920, ("dataset", "method"))
    check_run_file("ablation", 480, ("dataset", "method"))
    check_run_file("outlier", 3200, ("dataset", "outlier_fraction", "method"))
    assert {r["dataset"] for r in main_runs} == set(DATASETS)
    assert {r["method"] for r in main_runs} == set(METHODS)
    assert len(list((ROOT / "results/convergence").rglob("runs/*.mat"))) == 120

    means = {(r["dataset"], r["metric"]): r for r in
             rows("results/statistics/final_dataset_means.csv")}
    assert len(means) == 48
    main_summary = {(r["dataset"], r["method"]): r for r in
                    rows("results/main_summary.csv")}
    for dataset in DATASETS:
        for metric in ("ACC", "NMI", "ARI"):
            for method in METHODS:
                assert close(means[dataset, metric][method],
                             main_summary[dataset, method][f"{metric}_mean"])
    kernel_summary = {(r["dataset"], r["method"]): r for r in
                      rows("results/kernel_summary.csv")}
    ablation_summary = {(r["dataset"], r["method"]): r for r in
                        rows("results/ablation_summary.csv")}
    for dataset in ("coil20", "yale", "jaffe", "usps"):
        for metric in ("ACC", "NMI", "ARI"):
            values = (main_summary[dataset, "HKFSC"][f"{metric}_mean"],
                      kernel_summary[dataset, "HKFSC_HGK"][f"{metric}_mean"],
                      ablation_summary[dataset, "HKFSC_FULL"][f"{metric}_mean"])
            assert close(values[0], values[1]) and close(values[0], values[2]), \
                (dataset, metric, values)

    timing = rows("results/runtime/complexity_runs.csv")
    timing_summary = {(r["dataset"], r["method"]): r for r in
                      rows("results/runtime/complexity_runtime_summary.csv")}
    timing_groups = defaultdict(list)
    runtime_names = {"KM": "K-means", "MKKM": "MKKM-RK"}
    for row in timing:
        timing_groups[row["dataset"], runtime_names.get(row["method"], row["method"])].append(row)
        assert math.isfinite(float(row["seconds"])) and float(row["seconds"]) > 0
        assert not row["error"]
    assert len(timing) == 400 and len(timing_groups) == len(timing_summary) == 40
    for key, group in timing_groups.items():
        assert len(group) == 10 and {int(r["seed"]) for r in group} == set(range(1001, 1011))
        times = [float(r["seconds"]) for r in group]
        assert int(timing_summary[key]["runs"]) == 10
        assert close(statistics.mean(times), timing_summary[key]["seconds_mean"])
        assert close(statistics.stdev(times), timing_summary[key]["seconds_std"])
        assert close(statistics.median(times), timing_summary[key]["seconds_median"])

    extra = "results/additional_validation/"
    holdout_runs = rows(extra + "raw_runs.csv")
    holdout_means = {r["metric"]: r for r in rows(extra + "holdout_dataset_means.csv")}
    holdout_groups = defaultdict(list)
    for row in holdout_runs:
        holdout_groups[row["method"]].append(row)
        assert row["success"] == "1" and row["dataset"] == "swiss_roll_file_holdout"
    assert len(holdout_runs) == 160 and set(holdout_groups) == set(METHODS)
    for method, group in holdout_groups.items():
        assert len(group) == 20 and {int(r["seed"]) for r in group} == set(range(2001, 2021))
        for metric in ("ACC", "NMI", "ARI"):
            assert close(statistics.mean(float(r[metric]) for r in group),
                         holdout_means[metric][method])

    scenarios = {
        "exclude_spiral_15": [d for d in DATASETS if d != "swiss_roll"],
        "replace_with_file_holdout_16": list(DATASETS),
        "exclude_all_development_overlap_10": [d for d in DATASETS if d not in
            {"blobs", "ds577", "jaffe", "swiss_roll", "yale", "zoo"}],
        "file_holdout_and_other_nonoverlap_11": [d for d in DATASETS if d not in
            {"blobs", "ds577", "jaffe", "swiss_roll", "yale", "zoo"}]
            + ["swiss_roll_file_holdout"],
    }
    omnibus = {(r["scenario"], r["metric"]): r for r in rows(extra + "A4_omnibus.csv")}
    pairwise = {(r["scenario"], r["metric"], r["baseline"]): r for r in
                rows(extra + "A4_hkfsc_pairwise.csv")}
    ranks_file = {(r["scenario"], r["metric"], r["method"]): r for r in
                  rows(extra + "A4_average_ranks.csv")}
    assert len(omnibus) == 12 and len(pairwise) == 84 and len(ranks_file) == 96
    for scenario, selected in scenarios.items():
        for metric in ("ACC", "NMI", "ARI"):
            matrix = []
            for dataset in selected:
                source = (holdout_means[metric] if dataset == "swiss_roll_file_holdout" or
                          (dataset == "swiss_roll" and scenario == "replace_with_file_holdout_16")
                          else means[dataset, metric])
                matrix.append([float(source[method]) for method in METHODS])
            n, k = len(matrix), len(METHODS)
            ranked = [stats.rankdata(row, reverse=True) for row in matrix]
            average = [statistics.mean(r[j] for r in ranked) for j in range(k)]
            ties = sum(t**3 - t for row in matrix for t in stats.tie_sizes(row))
            correction = 1 - ties / (n * (k**3-k))
            q = (12*n/(k*(k+1)) * sum((r-(k+1)/2)**2 for r in average)) / correction
            f_value = (n-1)*q/(n*(k-1)-q)
            record = omnibus[scenario, metric]
            assert int(record["n_datasets"]) == n
            assert close(record["friedman_q"], q)
            assert close(record["friedman_p"], stats.chi2_sf(q, k-1))
            assert close(record["iman_davenport_f"], f_value)
            assert close(record["iman_davenport_p"], stats.f_sf(f_value, k-1, (k-1)*(n-1)))
            assert close(record["hkfsc_average_rank"], average[-1])
            for method, average_rank in zip(METHODS, average):
                assert close(ranks_file[scenario, metric, method]["average_rank"], average_rank)
            tests = [stats.wilcoxon_exact([row[-1]-row[j] for row in matrix])
                     for j in range(k-1)]
            adjusted = stats.holm([test[0] for test in tests])
            for method, test, p_holm in zip(METHODS[:-1], tests, adjusted):
                record = pairwise[scenario, metric, method]
                assert close(record["p_raw"], test[0])
                assert close(record["rank_biserial"], test[1])
                assert close(record["p_holm"], p_holm)

    print("PASS: 22 datasets; main/kernel/ablation/outlier 20-run records; 120 "
          "convergence traces; Fig. 16's 400 timings; the 160-run file-instance "
          "evaluation and all A4 tests match their recorded inputs.")


if __name__ == "__main__":
    main()
