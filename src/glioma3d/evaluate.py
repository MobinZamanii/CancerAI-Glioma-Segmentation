"""
Full-volume evaluation of the trained 3D multi-task U-Net.

Unlike the validation Dice printed during training (computed on tumor-centred 96^3
patches), this script segments the WHOLE volume with Gaussian-weighted sliding-window
inference, so false positives anywhere in the brain are penalised.

Usage
-----
  # internal held-out test split (patient-level split from samples_clean.json)
  python -m glioma3d.evaluate --checkpoint runs/final_c24/checkpoints/best.pth --set internal --split test

  # external validation on CFB-GBM (primary protocol: strict tier, one group per patient)
  python -m glioma3d.evaluate --checkpoint runs/final_c24/checkpoints/best.pth --set cfb --cfb-tier strict

  # sensitivity analysis on CFB (T2* instead of T2-TSE)
  python -m glioma3d.evaluate --checkpoint runs/final_c24/checkpoints/best.pth --set cfb --cfb-tier t2star

  # smoke test first (2 cases)
  python -m glioma3d.evaluate --checkpoint ...best.pth --set internal --limit 2

Protocol
--------
internal : regions ET={3}, TC={1,3}, WT={1,2,3}, NCR={1}, ED={2}. Metrics: Dice and HD95
           (HD95 in voxels, because training volumes are NOT resampled to a common spacing).
cfb      : external set with a binary radiotherapy GTV. Volumes are resampled to 1 mm
           isotropic, skull-stripped with the provided brain mask, z-scored on non-zero voxels.
           PRIMARY metric (fixed a priori): Dice of TC (predicted classes 1 u 3) vs GTV.
           Secondary: Dice of WT (predicted classes > 0) vs GTV. HD95 in mm.
           A radiotherapy GTV is not the same structure as a BraTS tumor region, so absolute
           Dice values are not directly comparable with internal results.

Statistics
----------
Per-case values are averaged within each patient first; the 95% CI is a percentile
bootstrap over PATIENTS (2000 resamples). Cases whose ground-truth region is empty have
NaN Dice (excluded from the mean and counted in `n_nan`). For internal runs a second
set of columns, `dicebc_*`, follows the BraTS convention (empty GT & empty prediction -> 1,
empty GT & non-empty prediction -> 0) so that false positives on empty regions are not hidden.

Outputs
-------
eval_results/<set>_<checkpoint stem>/per_sample.csv   (rewritten every 5 cases)
eval_results/<set>_<checkpoint stem>/summary.json
eval_results/<set>_<checkpoint stem>/errors.csv       (only if some cases failed)
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PATCH = (96, 96, 96)
REGIONS_INTERNAL = ("ET", "TC", "WT", "NCR", "ED")


# ---------------------------------------------------------------------------
# Metrics (numpy / scipy only)
# ---------------------------------------------------------------------------
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient. NaN when the ground truth is empty."""
    g = int(gt.sum())
    if g == 0:
        return float("nan")
    return float(2.0 * np.logical_and(pred, gt).sum() / (int(pred.sum()) + g))


def dice_brats(pred: np.ndarray, gt: np.ndarray) -> float:
    """BraTS-convention Dice: empty GT -> 1 if prediction is also empty, else 0."""
    g = int(gt.sum())
    p = int(pred.sum())
    if g == 0:
        return 1.0 if p == 0 else 0.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (p + g))


def hd95(a: np.ndarray, b: np.ndarray) -> float:
    """95th-percentile symmetric surface distance in voxels. NaN if either mask is empty."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if not a.any() or not b.any():
        return float("nan")
    idx = np.argwhere(a | b)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = idx.max(0) + 3
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    a, b = a[sl], b[sl]
    sa, sb = a ^ binary_erosion(a), b ^ binary_erosion(b)
    da = distance_transform_edt(~sb)[sa]
    db = distance_transform_edt(~sa)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def internal_regions(m: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "ET": m == 3,
        "TC": (m == 1) | (m == 3),
        "WT": m > 0,
        "NCR": m == 1,
        "ED": m == 2,
    }


def bootstrap_ci(values_by_patient: Dict[str, List[float]], n_boot: int = 2000, seed: int = 0):
    """Patient-level percentile bootstrap. Returns (mean, lo, hi, n_patients)."""
    means = []
    for v in values_by_patient.values():
        arr = np.asarray(v, dtype=float)
        if np.any(~np.isnan(arr)):
            means.append(float(np.nanmean(arr)))
    means = np.asarray(means, dtype=float)
    if means.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, means.size, size=(n_boot, means.size))
    boots = means[idx].mean(axis=1)
    return (float(means.mean()), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)), int(means.size))


def summarize(rows: List[Dict], keys: List[str]) -> Dict:
    out = {}
    for k in keys:
        by_pat: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            by_pat[r["patient_id"]].append(r[k])
        mean, lo, hi, n = bootstrap_ci(by_pat)
        vals = np.array([r[k] for r in rows], dtype=float)
        has = bool(np.any(~np.isnan(vals)))
        out[k] = {
            "n_patients": n,
            "mean": mean,
            "ci95": [lo, hi],
            "std": float(np.nanstd(vals)) if has else float("nan"),
            "median": float(np.nanmedian(vals)) if has else float("nan"),
            "n_nan": int(np.isnan(vals).sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------
def gaussian_window(patch: Sequence[int], sigma_scale: float = 0.125) -> np.ndarray:
    ws = []
    for p in patch:
        x = np.arange(p) - (p - 1) / 2.0
        ws.append(np.exp(-(x ** 2) / (2 * (p * sigma_scale) ** 2)))
    w = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
    return np.maximum(w / w.max(), 1e-3).astype(np.float32)


def window_starts(size: int, patch: int, stride: int) -> List[int]:
    if size <= patch:
        return [0]
    s = list(range(0, size - patch + 1, stride))
    if s[-1] != size - patch:
        s.append(size - patch)
    return s


def predict_volume(model, img: np.ndarray, device, num_classes: int = 4,
                   patch: Sequence[int] = PATCH, overlap: float = 0.5,
                   amp: bool = True) -> Tuple[np.ndarray, int]:
    """
    img: [C, D, H, W], already normalised, background == 0.
    Returns (label volume [D, H, W] uint8, number of windows that needed an fp32 retry).
    Windows that are pure background are skipped (their probabilities stay 0 -> label 0).
    """
    import torch

    C, D, H, W = img.shape
    pad = [(0, max(0, p - s)) for p, s in zip(patch, (D, H, W))]
    imgp = np.pad(img, [(0, 0)] + pad) if any(b for _, b in pad) else img
    _, Dp, Hp, Wp = imgp.shape
    strides = [max(1, int(p * (1 - overlap))) for p in patch]

    prob = torch.zeros((num_classes, Dp, Hp, Wp), dtype=torch.float32, device=device)
    weight = torch.from_numpy(gaussian_window(patch)).to(device)
    is_cuda = getattr(device, "type", str(device)) == "cuda"
    use_amp = amp and is_cuda
    n_fallback = 0

    with torch.inference_mode():
        for sd in window_starts(Dp, patch[0], strides[0]):
            for sh in window_starts(Hp, patch[1], strides[1]):
                for sw in window_starts(Wp, patch[2], strides[2]):
                    block = imgp[:, sd:sd + patch[0], sh:sh + patch[1], sw:sw + patch[2]]
                    if not block.any():
                        continue
                    x = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))[None].to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        logits = model(x)["segmentation"]
                    if use_amp and not bool(torch.isfinite(logits).all()):
                        # fp16 overflow in this window: recompute it in full precision
                        n_fallback += 1
                        with torch.autocast(device_type="cuda", enabled=False):
                            logits = model(x)["segmentation"]
                    p = torch.softmax(logits.float(), dim=1)[0]
                    prob[:, sd:sd + patch[0], sh:sh + patch[1], sw:sw + patch[2]] += p * weight

    # No normalisation by the summed weights: they are identical for every class at a voxel,
    # so argmax is unaffected.
    pred = prob.argmax(0)[:D, :H, :W].to("cpu").numpy().astype(np.uint8)
    del prob
    if is_cuda:
        torch.cuda.empty_cache()
    return pred, n_fallback


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _axcodes(path: str) -> str:
    import nibabel as nib

    return "".join(nib.aff2axcodes(nib.load(path).affine))


def load_internal(sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Same loading/normalisation/label mapping as training (no resampling)."""
    from glioma3d.data import load_nifti_sample

    img, msk = load_nifti_sample(sample)
    return img, msk.astype(np.uint8)


def _resample_1mm(arr: np.ndarray, affine: np.ndarray, order: int) -> np.ndarray:
    import nibabel as nib
    from nibabel.processing import resample_to_output

    out = resample_to_output(nib.Nifti1Image(arr.astype(np.float32), affine),
                             voxel_sizes=(1.0, 1.0, 1.0), order=order)
    return out.get_fdata(dtype=np.float32)


def load_cfb(sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample to 1 mm isotropic, apply the brain mask (skull stripping), z-score over
    non-zero voxels. The binary GTV is resampled with linear interpolation and thresholded
    at 0.5. Channel order: [T1, T1gd, T2, FLAIR] (as stored in image_paths).
    """
    import nibabel as nib

    from glioma3d.data import normalize_nonzero

    bm_img = nib.load(sample["brain_mask_path"])
    bm = _resample_1mm(np.asanyarray(bm_img.dataobj), bm_img.affine, 1) > 0.5
    chans = []
    for p in sample["image_paths"]:
        im = nib.load(p)
        r = _resample_1mm(np.asanyarray(im.dataobj), im.affine, 1) * bm
        chans.append(normalize_nonzero(r))
    g_img = nib.load(sample["mask_path"])
    gtv = (_resample_1mm(np.asanyarray(g_img.dataobj) > 0, g_img.affine, 1) > 0.5).astype(np.uint8)
    if chans[0].shape != gtv.shape:
        raise ValueError(f"Shape mismatch after resampling: image {chans[0].shape} vs GTV {gtv.shape}")
    return np.stack(chans).astype(np.float32), gtv


def select_cfb(path: str, tier: str, t0_only: bool, one_per_patient: bool) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m glioma3d.prep.build_external_cfb` first.")
    samples = json.loads(p.read_text(encoding="utf-8"))
    if tier != "all":
        samples = [s for s in samples if s["tier"] == tier]
    if t0_only:
        samples = [s for s in samples if s["timepoint"] == "t0"]
    if one_per_patient:
        best: Dict[str, Dict] = {}
        for s in sorted(samples, key=lambda s: int(s["timepoint"][1:])):
            best.setdefault(s["patient_id"], s)  # earliest timepoint per patient
        samples = list(best.values())
    return samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return "  nan" if x != x else f"{x:.3f}"


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _build_model(cfg: Dict, device):
    from glioma3d.model import MultiTaskUNet3D

    return MultiTaskUNet3D(
        in_channels=cfg.get("in_channels", 4),
        num_classes=cfg.get("num_classes", 4),
        base_channels=cfg["base_channels"],
        dropout=cfg.get("dropout", 0.05),
        use_graph=cfg.get("use_graph", True),
    ).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Full-volume evaluation (internal test split or CFB external set)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--set", choices=["internal", "cfb"], required=True)
    ap.add_argument("--split", default="test", help="internal only: test | val")
    ap.add_argument("--samples", default="samples_clean.json")
    ap.add_argument("--external", default="external_cfb.json")
    ap.add_argument("--cfb-tier", default="strict", choices=["strict", "t2star", "all"])
    ap.add_argument("--t0-only", action="store_true", help="CFB: keep only timepoint t0")
    ap.add_argument("--all-timepoints", action="store_true",
                    help="CFB: evaluate every group instead of one (earliest) per patient")
    ap.add_argument("--overlap", type=float, default=0.5, help="sliding-window overlap fraction")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true", help="run inference in fp32")
    ap.add_argument("--no-hd95", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N cases")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import torch

    from glioma3d.data import load_samples

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    num_classes = cfg.get("num_classes", 4)
    model = _build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Checkpoint : {args.checkpoint} | epoch {ckpt.get('epoch')} | "
          f"best val Dice {ckpt.get('best_val_dice')} | base_channels {cfg['base_channels']}")
    print(f"Device     : {device} | AMP inference: {not args.no_amp and device.type == 'cuda'} | "
          f"overlap {args.overlap}")

    if args.set == "internal":
        samples = load_samples(args.samples, args.split)
        loader = load_internal
    else:
        samples = select_cfb(args.external, args.cfb_tier, args.t0_only, not args.all_timepoints)
        loader = load_cfb
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        sys.exit("No samples selected.")
    n_pat = len({s["patient_id"] for s in samples})
    print(f"Set        : {args.set} | {len(samples)} cases / {n_pat} patients")

    out_dir = Path(args.out_dir or f"eval_results/{args.set}_{Path(args.checkpoint).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Orientation sanity check: the network was trained on raw array axes (no reorientation).
    try:
        first_img = samples[0]["image_paths"][0]
        print(f"Axis codes of first case's raw image: {_axcodes(first_img)} "
              f"(CFB is resampled to a world-aligned RAS grid)")
    except Exception:
        pass

    rows: List[Dict] = []
    errors: List[Dict] = []
    t_start = time.time()
    main_key = "dice_WT" if args.set == "internal" else "dice_TC"

    for i, s in enumerate(samples, 1):
        try:
            img, gt = loader(s)
            pred, n_fb = predict_volume(model, img, device, num_classes, PATCH,
                                        args.overlap, not args.no_amp)
            row: Dict = {"sample_id": s["sample_id"], "dataset": s["dataset"],
                         "patient_id": s["patient_id"], "fp32_fallback_windows": n_fb}
            if args.set == "internal":
                pr, gr = internal_regions(pred), internal_regions(gt)
                row["mask_source"] = s.get("mask_source", "")
                for k in REGIONS_INTERNAL:
                    row[f"dice_{k}"] = dice_score(pr[k], gr[k])
                    row[f"dicebc_{k}"] = dice_brats(pr[k], gr[k])
                    row[f"hd95_{k}"] = float("nan") if args.no_hd95 else hd95(pr[k], gr[k])
                row["gt_WT_vox"] = int(gr["WT"].sum())
                row["pred_WT_vox"] = int(pr["WT"].sum())
                row["fp_when_gt_empty_WT"] = int(pr["WT"].any() and not gr["WT"].any())
            else:
                g = gt > 0
                tc, wt = (pred == 1) | (pred == 3), pred > 0
                row.update({
                    "tier": s["tier"], "timepoint": s["timepoint"],
                    "dice_TC": dice_score(tc, g), "dice_WT": dice_score(wt, g),
                    "hd95_TC": float("nan") if args.no_hd95 else hd95(tc, g),
                    "hd95_WT": float("nan") if args.no_hd95 else hd95(wt, g),
                    "gtv_ml": g.sum() / 1000.0,
                    "pred_TC_ml": tc.sum() / 1000.0,
                    "pred_WT_ml": wt.sum() / 1000.0,
                })
            rows.append(row)
            print(f"\r  {i}/{len(samples)} | {s['sample_id'][:40]:40s} | last {main_key}={_fmt(row[main_key])}"
                  f" | {time.time() - t_start:.0f}s", end="", flush=True)
        except Exception as e:  # keep going; record the failure
            errors.append({"sample_id": s.get("sample_id"), "error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc(limit=3)})
            print(f"\n  [ERROR] {s.get('sample_id')}: {type(e).__name__}: {e}")
        if i % 5 == 0:
            _write_csv(out_dir / "per_sample.csv", rows)
    print()

    if not rows:
        _write_csv(out_dir / "errors.csv", errors)
        sys.exit("All cases failed; see errors.csv")

    _write_csv(out_dir / "per_sample.csv", rows)
    _write_csv(out_dir / "errors.csv", errors)

    metric_keys = [k for k in rows[0] if k.startswith(("dice_", "dicebc_", "hd95_"))]
    summary = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "set": args.set,
        "args": vars(args),
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__},
        "n_cases_ok": len(rows),
        "n_cases_failed": len(errors),
        "n_patients": len({r["patient_id"] for r in rows}),
        "hd95_units": "voxels (native spacing)" if args.set == "internal" else "mm (1 mm isotropic)",
        "n_fp32_fallback_windows_total": int(sum(r["fp32_fallback_windows"] for r in rows)),
        "overall": summarize(rows, metric_keys),
    }
    if args.set == "internal":
        dk = [k for k in metric_keys if k.startswith(("dice_", "dicebc_", "hd95_"))]
        summary["by_dataset"] = {ds: summarize([r for r in rows if r["dataset"] == ds], dk)
                                 for ds in sorted({r["dataset"] for r in rows})}
        summary["by_mask_source"] = {m: summarize([r for r in rows if r["mask_source"] == m], dk)
                                     for m in sorted({r["mask_source"] for r in rows})}
        summary["n_empty_gt_wt_with_false_positive"] = int(sum(r["fp_when_gt_empty_WT"] for r in rows))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- console report ----
    def line(name: str, v: Dict) -> str:
        return (f"{name:12s} n={v['n_patients']:4d} | mean {_fmt(v['mean'])}  "
                f"CI95 [{_fmt(v['ci95'][0])}, {_fmt(v['ci95'][1])}] | "
                f"median {_fmt(v['median'])} | NaN {v['n_nan']}")

    print("\n" + "=" * 78)
    print(f"OVERALL ({args.set})  |  cases ok {len(rows)}, failed {len(errors)}  |  "
          f"fp32 fallback windows {summary['n_fp32_fallback_windows_total']}")
    print("-" * 78)
    for k, v in summary["overall"].items():
        if k.startswith("dice_") or k.startswith("dicebc_"):
            print(line(k, v))
    print(f"HD95 units: {summary['hd95_units']}")
    for k, v in summary["overall"].items():
        if k.startswith("hd95_"):
            print(line(k, v))
    if args.set == "internal":
        print("-" * 78 + "\nPer dataset (Dice WT / TC / ET):")
        for ds, d in summary["by_dataset"].items():
            print(f"  {ds:8s} n={d['dice_WT']['n_patients']:3d} | "
                  f"WT {_fmt(d['dice_WT']['mean'])}  TC {_fmt(d['dice_TC']['mean'])}  "
                  f"ET {_fmt(d['dice_ET']['mean'])}")
        print("Per mask source (Dice WT / TC / ET):")
        for m, d in summary["by_mask_source"].items():
            print(f"  {m or '(none)':14s} n={d['dice_WT']['n_patients']:3d} | "
                  f"WT {_fmt(d['dice_WT']['mean'])}  TC {_fmt(d['dice_TC']['mean'])}  "
                  f"ET {_fmt(d['dice_ET']['mean'])}")
    print("=" * 78)
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
