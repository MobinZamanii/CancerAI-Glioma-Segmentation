"""
Paired comparison of two evaluation runs (per_sample.csv files written by evaluate.py or baseline_monai.py).

Protocol
--------
* The two files are paired on `sample_id`; only cases with a finite value in BOTH files are used.
* Values are averaged within each patient first (patient = unit of analysis), so cases of the same
  patient (e.g. several sessions) never count as independent observations.
* For every metric the per-patient difference d = a - b is summarised by
    - the mean difference with a 95% percentile bootstrap interval over PATIENTS (default 10,000 resamples),
    - the median difference,
    - the number of patients in which a is better / b is better / tied
      (direction-aware: for HD95, "better" means lower; for Dice, higher),
    - the two-sided Wilcoxon signed-rank p-value on the per-patient differences.
* Holm correction is applied across the metrics of each group (the group is "ALL" or one cohort).
* A difference is called significant only if the bootstrap interval excludes 0 AND the Holm-adjusted p < 0.05
  (`ci_excludes_0` and `p_holm` are both reported; combine them as stated).

Usage
-----
  python -m glioma3d.compare_runs \
      --a eval_results/internal_best/per_sample.csv --label-a "Full model" \
      --b eval_results/internal_monai_segresnet/per_sample.csv --label-b "MONAI SegResNet" \
      --metrics dice_WT dice_TC dice_ET --by-dataset --out compare_internal_monai

  python -m glioma3d.compare_runs \
      --a eval_results/cfb_best/per_sample.csv --b eval_results/cfb_monai_segresnet/per_sample.csv \
      --metrics dice_TC hd95_TC --out compare_cfb_monai

Outputs: <out>.json and <out>.csv (group ALL); with --by-dataset also <out>_by_dataset.json/.csv
(group ALL plus one group per cohort, taken from the `dataset` column).

Optional: --check-against <reference.json> prints every numeric field that differs from a previously
written result (useful to confirm that a re-run reproduces stored numbers).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

LOWER_IS_BETTER_PREFIXES = ("hd95_",)


def read_per_sample(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {r["sample_id"]: r for r in rows}


def _to_float(x: Optional[str]) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def patient_level_pairs(a: Dict[str, Dict], b: Dict[str, Dict], metric: str, dataset: Optional[str] = None):
    """Per-patient means of metric in a and b over the cases that are finite in both runs."""
    by_pat: Dict[str, List[tuple]] = defaultdict(list)
    n_cases = 0
    for sid, ra in a.items():
        rb = b.get(sid)
        if rb is None:
            continue
        if dataset is not None and ra.get("dataset") != dataset:
            continue
        va, vb = _to_float(ra.get(metric)), _to_float(rb.get(metric))
        if math.isnan(va) or math.isnan(vb):
            continue
        by_pat[ra["patient_id"]].append((va, vb))
        n_cases += 1
    pats = sorted(by_pat)
    mean_a = np.array([np.mean([x[0] for x in by_pat[p]]) for p in pats], dtype=float)
    mean_b = np.array([np.mean([x[1] for x in by_pat[p]]) for p in pats], dtype=float)
    return mean_a, mean_b, n_cases


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boots = diffs[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def wilcoxon_p(diffs: np.ndarray) -> float:
    from scipy.stats import wilcoxon

    if diffs.size == 0 or not np.any(diffs != 0):
        return 1.0
    return float(wilcoxon(diffs).pvalue)


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaN stays NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.where(~np.isnan(p))[0]
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out.tolist()


def compare_metric(a, b, metric: str, group: str, dataset: Optional[str], n_boot: int, seed: int) -> Dict:
    ma, mb, n_cases = patient_level_pairs(a, b, metric, dataset)
    n = int(ma.size)
    res: Dict = {"group": group, "metric": metric, "n_patients": n, "n_cases": n_cases}
    if n == 0:
        res.update({k: float("nan") for k in ("mean_a", "mean_b", "diff_a_minus_b", "ci95_lo", "ci95_hi",
                                               "median_diff", "p_wilcoxon")})
        res.update({"a_better_patients": 0, "b_better_patients": 0, "ties": 0})
        return res
    d = ma - mb
    lo, hi = paired_bootstrap_ci(d, n_boot, seed)
    lower_better = metric.startswith(LOWER_IS_BETTER_PREFIXES)
    a_better = int((d < 0).sum() if lower_better else (d > 0).sum())
    b_better = int((d > 0).sum() if lower_better else (d < 0).sum())
    res.update({
        "mean_a": float(ma.mean()), "mean_b": float(mb.mean()), "diff_a_minus_b": float(d.mean()),
        "ci95_lo": lo, "ci95_hi": hi, "median_diff": float(np.median(d)),
        "a_better_patients": a_better, "b_better_patients": b_better, "ties": int((d == 0).sum()),
        "p_wilcoxon": wilcoxon_p(d),
    })
    return res


def run(args, groups: Sequence[Optional[str]], a, b) -> List[Dict]:
    results: List[Dict] = []
    for ds in groups:
        group = "ALL" if ds is None else ds
        block = [compare_metric(a, b, m, group, ds, args.n_boot, args.seed) for m in args.metrics]
        adj = holm_adjust([r["p_wilcoxon"] for r in block])
        for r, p_holm in zip(block, adj):
            r["p_holm"] = p_holm
            r["ci_excludes_0"] = bool(r["ci95_lo"] > 0 or r["ci95_hi"] < 0) if r["n_patients"] else False
        results.extend(block)
    return results


def write_outputs(path_stem: str, args, results: List[Dict]) -> None:
    payload = {"a": args.a, "b": args.b, "label_a": args.label_a, "label_b": args.label_b,
               "n_boot": args.n_boot, "seed": args.seed, "results": results}
    Path(path_stem + ".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    cols = ["group", "metric", "label_a", "label_b", "n_patients", "n_cases", "mean_a", "mean_b", "diff_a_minus_b",
            "ci95_lo", "ci95_hi", "median_diff", "a_better_patients", "b_better_patients", "ties",
            "p_wilcoxon", "p_holm", "ci_excludes_0"]
    with open(path_stem + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({**r, "label_a": args.label_a, "label_b": args.label_b})


def check_against(reference: str, results: List[Dict], tol: float = 1e-9) -> None:
    ref = json.loads(Path(reference).read_text(encoding="utf-8"))["results"]
    index = {(r["group"], r["metric"]): r for r in results}
    n_diff = 0
    for r in ref:
        new = index.get((r["group"], r["metric"]))
        if new is None:
            print(f"  missing in new run: {r['group']} / {r['metric']}")
            n_diff += 1
            continue
        for k, v in r.items():
            if k in ("group", "metric"):
                continue
            nv = new.get(k)
            if isinstance(v, (int, float)) and isinstance(nv, (int, float)):
                if abs(v - nv) > tol * max(1.0, abs(v)):
                    print(f"  {r['group']:6s} {r['metric']:8s} {k:18s} reference {v!r} | new {nv!r}")
                    n_diff += 1
    print("check-against: identical" if n_diff == 0 else f"check-against: {n_diff} field(s) differ")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired patient-level comparison of two per_sample.csv files")
    ap.add_argument("--a", required=True, help="per_sample.csv of run A")
    ap.add_argument("--b", required=True, help="per_sample.csv of run B")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--metrics", nargs="+", default=["dice_WT", "dice_TC", "dice_ET"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-dataset", action="store_true", help="also write results per cohort")
    ap.add_argument("--out", default="compare")
    ap.add_argument("--check-against", default=None, help="reference .json to compare the new numbers with")
    args = ap.parse_args()

    a, b = read_per_sample(args.a), read_per_sample(args.b)
    print(f"A: {args.label_a} ({len(a)} cases) | B: {args.label_b} ({len(b)} cases) | paired: {len(set(a) & set(b))}")
    overall = run(args, [None], a, b)
    write_outputs(args.out, args, overall)
    results = overall
    if args.by_dataset:
        datasets = sorted({r.get("dataset", "") for r in a.values() if r["sample_id"] in b} - {""})
        by_ds = run(args, [None] + datasets, a, b)
        write_outputs(args.out + "_by_dataset", args, by_ds)
        results = by_ds

    for r in results:
        print(f"{r['group']:6s} {r['metric']:8s} n={r['n_patients']:4d} | {args.label_a} {r['mean_a']:.4f} vs "
              f"{args.label_b} {r['mean_b']:.4f} | diff {r['diff_a_minus_b']:+.4f} "
              f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | p_holm {r['p_holm']:.3g} | CI excludes 0: {r['ci_excludes_0']}")
    if args.check_against:
        check_against(args.check_against, results)


if __name__ == "__main__":
    main()
