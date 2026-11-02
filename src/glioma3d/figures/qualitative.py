"""
Qualitative figures (MRI + reference mask + prediction + error map) for the paper.

Run it from the repository root after `pip install -e .` (see README).

It re-uses the SAME loading and sliding-window inference as evaluate.py, so the
pictures are exactly what was scored. Cases are chosen from the per_sample.csv that
evaluate.py already wrote: for each cohort the BEST, MEDIAN and WORST case (by the
primary metric). Nothing is cherry-picked by eye.

Examples
--------
  # internal test set, one figure per cohort + one overview figure
  python -m glioma3d.figures.qualitative --set internal ^
      --checkpoint runs/final_c24/checkpoints/best.pth ^
      --per-sample eval_results/internal_best/per_sample.csv

  # CFB-GBM external set (strict tier)
  python -m glioma3d.figures.qualitative --set cfb ^
      --checkpoint runs/final_c24/checkpoints/best.pth ^
      --per-sample eval_results/cfb_best/per_sample.csv

  # quick test with 1 case per cohort
  python -m glioma3d.figures.qualitative ... --picks median

Outputs (PNG 600 dpi + PDF) go to  qualitative_figs/
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# colour-blind-safe palette (Okabe-Ito)
C_NCR, C_ED, C_ET = "#0072B2", "#F0E442", "#D55E00"
C_TP, C_FP, C_FN = "#009E73", "#CC79A7", "#56B4E9"
LABEL_CMAP = ListedColormap(["none", C_NCR, C_ED, C_ET])


# ---------------------------------------------------------------------------
def to_ras(vol: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Re-orient the last 3 axes of vol ([...,D,H,W]) to RAS so that axis 2 is axial."""
    import nibabel as nib
    from nibabel.orientations import apply_orientation, io_orientation, ornt_transform

    src = io_orientation(affine)
    dst = nib.orientations.axcodes2ornt(("R", "A", "S"))
    tr = ornt_transform(src, dst)
    if vol.ndim == 3:
        return apply_orientation(vol, tr)
    return np.stack([apply_orientation(c, tr) for c in vol])


def read_rows(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_cases(rows, metric: str, picks):
    """Return {label: row} for best / median / worst by `metric` (NaN rows ignored)."""
    ok = [r for r in rows if r.get(metric) not in ("", "nan", None) and float(r[metric]) == float(r[metric])]
    ok.sort(key=lambda r: float(r[metric]))
    if not ok:
        return {}
    out = {}
    if "worst" in picks:
        out["worst"] = ok[0]
    if "median" in picks:
        out["median"] = ok[len(ok) // 2]
    if "best" in picks:
        out["best"] = ok[-1]
    return out


def best_slice(mask_bool: np.ndarray) -> int:
    area = mask_bool.sum(axis=(0, 1))
    return int(area.argmax()) if area.max() > 0 else mask_bool.shape[2] // 2


def crop_box(mask_bool: np.ndarray, img: np.ndarray, margin: int = 6):
    """Bounding box of the brain (non-zero voxels) in the displayed slice."""
    nz = np.argwhere(img != 0)
    if len(nz) == 0:
        return slice(None), slice(None)
    (x0, y0), (x1, y1) = nz.min(0), nz.max(0)
    return slice(max(0, x0 - margin), x1 + margin), slice(max(0, y0 - margin), y1 + margin)


def show(ax, arr2d, cmap="gray", vmin=None, vmax=None):
    ax.imshow(np.rot90(arr2d), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def window(img2d):
    v = img2d[img2d != 0]
    if v.size == 0:
        return 0, 1
    return np.percentile(v, 1), np.percentile(v, 99.5)


def overlay_labels(ax, lab2d):
    ax.imshow(np.rot90(np.ma.masked_equal(lab2d, 0)), cmap=LABEL_CMAP, vmin=0, vmax=3,
              alpha=0.75, interpolation="nearest")


def error_rgba(pred_b, gt_b):
    tp, fp, fn = pred_b & gt_b, pred_b & ~gt_b, ~pred_b & gt_b
    rgba = np.zeros(pred_b.shape + (4,), dtype=float)
    for m, c in ((tp, C_TP), (fp, C_FP), (fn, C_FN)):
        rgba[m] = matplotlib.colors.to_rgba(c, 0.85)
    return rgba


# ---------------------------------------------------------------------------
def draw_case_row(axs, img, gt, pred, mode, caption):
    """img [4,X,Y,Z] in RAS, gt/pred [X,Y,Z]. mode 'internal' (4 classes) or 'cfb' (binary GTV)."""
    if mode == "internal":
        k = best_slice(gt > 0) if (gt > 0).any() else best_slice(pred > 0)
        gt_b, pr_b = gt > 0, pred > 0       # error map on whole tumour
    else:
        k = best_slice(gt > 0)
        gt_b, pr_b = gt > 0, (pred == 1) | (pred == 3)   # TC vs GTV (primary metric)
    sl = lambda a: a[:, :, k]
    bx, by = crop_box(gt_b, sl(img[3]))
    c = lambda a: sl(a)[bx, by]

    t1ce, flair = c(img[1]), c(img[3])
    for ax, im in ((axs[0], t1ce), (axs[1], flair)):
        lo, hi = window(im)
        show(ax, im, vmin=lo, vmax=hi)

    show(axs[2], t1ce, vmin=window(t1ce)[0], vmax=window(t1ce)[1])
    show(axs[3], t1ce, vmin=window(t1ce)[0], vmax=window(t1ce)[1])
    if mode == "internal":
        overlay_labels(axs[2], c(gt))
        overlay_labels(axs[3], c(pred))
    else:  # binary GTV: contour vs predicted TC
        axs[2].contour(np.rot90(c(gt_b).astype(float)), levels=[0.5], colors=[C_TP], linewidths=1.4)
        overlay_labels(axs[3], np.where(c(pr_b), 3, 0))
    show(axs[4], t1ce, vmin=window(t1ce)[0], vmax=window(t1ce)[1])
    axs[4].imshow(np.rot90(error_rgba(c(pr_b), c(gt_b)), axes=(0, 1)), interpolation="nearest")
    axs[0].set_ylabel(caption, color="white", fontsize=7, rotation=0, ha="right", va="center", labelpad=6)


def make_figure(cases, mode, out_base, titles):
    """cases: list of (caption, img, gt, pred)."""
    n = len(cases)
    fig, axes = plt.subplots(n, 5, figsize=(10.5, 2.15 * n + 0.7), facecolor="black")
    axes = np.atleast_2d(axes)
    for r, (cap, img, gt, pred) in enumerate(cases):
        draw_case_row(axes[r], img, gt, pred, mode, cap)
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, color="white", fontsize=9)
    # legend
    from matplotlib.patches import Patch
    if mode == "internal":
        h = [Patch(color=C_NCR, label="NCR"), Patch(color=C_ED, label="ED"), Patch(color=C_ET, label="ET"),
             Patch(color=C_TP, label="TP (WT)"), Patch(color=C_FP, label="FP"), Patch(color=C_FN, label="FN")]
    else:
        h = [Patch(color=C_TP, label="GTV contour / TP"), Patch(color=C_ET, label="Predicted TC"),
             Patch(color=C_FP, label="FP"), Patch(color=C_FN, label="FN")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, labelcolor="white", fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.94 if n > 1 else 0.88, bottom=0.07, wspace=0.02, hspace=0.04)
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {})):
        fig.savefig(f"{out_base}.{ext}", facecolor="black", **kw)
    plt.close(fig)
    print("saved", out_base)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--set", choices=["internal", "cfb"], required=True)
    ap.add_argument("--per-sample", required=True, help="per_sample.csv written by evaluate.py for this checkpoint/set")
    ap.add_argument("--samples", default="samples_clean.json")
    ap.add_argument("--external", default="external_cfb.json")
    ap.add_argument("--cfb-tier", default="strict")
    ap.add_argument("--picks", nargs="+", default=["best", "median", "worst"])
    ap.add_argument("--out-dir", default="qualitative_figs")
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()

    import torch

    import evaluate as ev
    from glioma3d.data import load_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    model = ev._build_model(ckpt["config"], device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    rows = read_rows(args.per_sample)
    if args.set == "internal":
        samples = {s["sample_id"]: s for s in load_samples(args.samples, "test")}
        loader, metric = ev.load_internal, "dice_WT"
    else:
        samples = {s["sample_id"]: s for s in ev.select_cfb(args.external, args.cfb_tier, False, True)}
        loader, metric = ev.load_cfb, "dice_TC"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def run(row):
        s = samples[row["sample_id"]]
        img, gt = loader(s)
        pred, _ = ev.predict_volume(model, img, device, ckpt["config"].get("num_classes", 4),
                                    ev.PATCH, args.overlap, True)
        if args.set == "internal":      # raw array axes -> RAS for display
            import nibabel as nib
            aff = nib.load(s["image_paths"][0]).affine
            img, gt, pred = to_ras(img, aff), to_ras(gt, aff), to_ras(pred, aff)
        return img, gt, pred

    titles = ["T1ce", "FLAIR", "Reference mask", "Prediction", "Error map"]
    cohorts = sorted({r["dataset"] for r in rows})
    overview = []
    for ds in cohorts:
        chosen = pick_cases([r for r in rows if r["dataset"] == ds], metric, args.picks)
        cases = []
        for tag, r in chosen.items():
            print(f"{ds} {tag}: {r['sample_id']}  {metric}={float(r[metric]):.3f}")
            img, gt, pred = run(r)
            cap = f"{ds}\n{tag}\n{metric.replace('dice_', 'Dice ')} = {float(r[metric]):.2f}"
            cases.append((cap, img, gt, pred))
            if tag == "median":
                overview.append((cap, img, gt, pred))
        if cases:
            make_figure(cases, args.set, str(out / f"qual_{args.set}_{ds}"), titles)
    if overview:
        make_figure(overview, args.set, str(out / f"qual_{args.set}_overview_median"), titles)


if __name__ == "__main__":
    main()
