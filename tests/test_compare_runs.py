import csv

import numpy as np

from glioma3d import compare_runs as cr


def test_holm_adjust():
    assert cr.holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_bootstrap_ci_reproducible_and_brackets_mean():
    d = np.random.default_rng(1).normal(0.05, 0.1, 100)
    lo1, hi1 = cr.paired_bootstrap_ci(d, 2000, 0)
    lo2, hi2 = cr.paired_bootstrap_ci(d, 2000, 0)
    assert (lo1, hi1) == (lo2, hi2) and lo1 < d.mean() < hi1


def _write(path, rows, metric):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "patient_id", "dataset", metric])
        w.writeheader()
        w.writerows(rows)


def test_hd95_direction_aware(tmp_path):
    rows_a = [{"sample_id": f"s{i}", "patient_id": f"p{i}", "dataset": "X", "hd95_TC": 5.0} for i in range(12)]
    rows_b = [{"sample_id": f"s{i}", "patient_id": f"p{i}", "dataset": "X", "hd95_TC": 9.0} for i in range(12)]
    _write(tmp_path / "a.csv", rows_a, "hd95_TC")
    _write(tmp_path / "b.csv", rows_b, "hd95_TC")
    a, b = cr.read_per_sample(str(tmp_path / "a.csv")), cr.read_per_sample(str(tmp_path / "b.csv"))
    r = cr.compare_metric(a, b, "hd95_TC", "ALL", None, 500, 0)
    assert r["a_better_patients"] == 12 and r["b_better_patients"] == 0
