"""
Run the pretrained MONAI `brats_mri_segmentation` bundle (SegResNet, trained on BraTS 2018)
on the SAME test cases and with the SAME metrics as evaluate.py, and write a per_sample.csv
that can be compared with compare_runs.py.

Why: an external, published reference model for the comparison table.  This is INFERENCE
ONLY (no training, light on the GPU); it needs the bundle weights on this machine once.

Setup (once, on a machine with internet)
----------------------------------------
  pip install monai          # plain "monai" is enough: nibabel is already a dependency of this package
  python -m monai.bundle download "brats_mri_segmentation" --bundle_dir bundles
  # if that fails, download the repository https://huggingface.co/MONAI/brats_mri_segmentation
  # and keep its folder layout:  <bundle_dir>/brats_mri_segmentation/{configs,models,...}

Usage
-----
  python -m glioma3d.baseline_monai --bundle-dir bundles/brats_mri_segmentation --set internal --limit 2   # smoke test
  python -m glioma3d.baseline_monai --bundle-dir bundles/brats_mri_segmentation --set internal --split test
  python -m glioma3d.baseline_monai --bundle-dir bundles/brats_mri_segmentation --set cfb --cfb-tier strict

Then:
  python -m glioma3d.compare_runs --a eval_results/internal_best/per_sample.csv --label-a "Ours" \
      --b eval_results/internal_monai_segresnet/per_sample.csv --label-b "MONAI SegResNet" \
      --metrics dice_WT dice_TC dice_ET --out compare_internal_monai

Protocol notes (state these in the paper)
-----------------------------------------
* Bundle input channel order is [T1c, T1, T2, FLAIR]; our arrays are [T1, T1ce, T2, FLAIR],
  so channels 0 and 1 are swapped here.  Bundle outputs are three sigmoid channels
  [TC, WT, ET] thresholded at 0.5.  Metrics use the same region definitions as evaluate.py
  (ET={3}, TC={1,3}, WT>0), so NCR / ED are not available for this model.
* Same z-score over non-zero voxels as the bundle's own pre-processing.
* internal set: volumes are reoriented to the closest RAS+ orientation (flag --no-ras to
  disable); no resampling, exactly like evaluate.py.  CFB: identical loader to evaluate.py
  (1 mm, skull-stripped, RAS grid).
* The bundle was trained on BraTS 2018 (285 cases).  It never saw our training split, so
  this is not a same-data comparison: it quantifies how an off-the-shelf model performs,
  not how an architecture compares when trained identically.  Overlap of BraTS 2018 source
  data (TCIA) with our datasets has to be checked and reported.
* Sliding window: Gaussian, overlap 0.5, ROI given by --roi (the GTX 1650 cannot hold the
  bundle's native 240x240x160 window; on CUDA OOM the script retries with smaller ROIs).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from glioma3d.evaluate import (REGIONS_INTERNAL, _fmt, _write_csv, dice_brats, dice_score, hd95,
                      internal_regions, load_cfb, select_cfb, summarize)

PERM = [1, 0, 2, 3]            # [T1, T1ce, T2, FLAIR] -> [T1c, T1, T2, FLAIR]
ROI_FALLBACKS = [(192, 192, 128), (160, 160, 128), (128, 128, 128), (96, 96, 96)]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_bundle_model(bundle_dir: Path, device):
    import torch

    cfg_path = bundle_dir / "configs" / "inference.json"
    net = None
    if cfg_path.exists():
        try:
            from monai.bundle import ConfigParser

            parser = ConfigParser()
            parser.read_config(str(cfg_path))
            net = parser.get_parsed_content("network_def")
            print(f"Network built from {cfg_path}")
        except Exception as e:  # fall back to explicit definition
            print(f"[warn] could not build network from bundle config ({type(e).__name__}: {e})")
    if net is None:
        from monai.networks.nets import SegResNet

        print("[warn] using a hard-coded SegResNet definition (init_filters=16, blocks_down=1,2,2,4, "
              "blocks_up=1,1,1, dropout 0.2); strict loading below will fail if it does not match.")
        net = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16,
                        in_channels=4, out_channels=3, dropout_prob=0.2)

    wpath = None
    for cand in ("models/model.pt", "models/model.pth"):
        if (bundle_dir / cand).exists():
            wpath = bundle_dir / cand
            break
    if wpath is None:
        sys.exit(f"No weights found under {bundle_dir}/models/")
    state = torch.load(str(wpath), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    net.load_state_dict(state, strict=True)
    print(f"Weights loaded from {wpath}")
    return net.to(device).eval()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_internal_ras(sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Like dataset.load_nifti_sample but with every volume reoriented to RAS+."""
    import nibabel as nib

    from glioma3d.data import normalize_nonzero, sample_label_map

    chans = []
    for p in sample["image_paths"]:
        im = nib.as_closest_canonical(nib.load(p))
        chans.append(normalize_nonzero(im.get_fdata(dtype=np.float32)))
    img = np.stack(chans, axis=0)
    msk = np.rint(nib.as_closest_canonical(nib.load(sample["mask_path"])).get_fdata()).astype(np.int64)
    for src, dst in sample_label_map(sample).items():
        msk[msk == src] = dst
    if img.shape[1:] != msk.shape:
        raise ValueError(f"Shape mismatch {sample['sample_id']}: image {img.shape[1:]} vs mask {msk.shape}")
    return img, msk.astype(np.uint8)


def load_internal_raw(sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
    from glioma3d.data import load_nifti_sample

    img, msk = load_nifti_sample(sample)
    return img, msk.astype(np.uint8)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_regions(net, img: np.ndarray, device, rois: List[Tuple[int, int, int]],
                    overlap: float, amp: bool):
    """Return (tc, wt, et) boolean volumes and the ROI that was used."""
    import torch
    from monai.inferers import sliding_window_inference

    x = torch.from_numpy(np.ascontiguousarray(img[PERM], dtype=np.float32))[None]
    is_cuda = device.type == "cuda"
    last_err: Optional[Exception] = None
    for roi in rois:
        try:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and is_cuda):
                    out = sliding_window_inference(x, roi_size=roi, sw_batch_size=1, predictor=net,
                                                   overlap=overlap, mode="gaussian",
                                                   sw_device=device, device=torch.device("cpu"))
                if amp and is_cuda and not bool(torch.isfinite(out).all()):
                    with torch.autocast(device_type="cuda", enabled=False):
                        out = sliding_window_inference(x, roi_size=roi, sw_batch_size=1, predictor=net,
                                                       overlap=overlap, mode="gaussian",
                                                       sw_device=device, device=torch.device("cpu"))
            prob = torch.sigmoid(out.float())[0] > 0.5            # [3, D, H, W]
            tc, wt, et = (prob[i].numpy() for i in range(3))
            return tc, wt, et, roi
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            torch.cuda.empty_cache()
            print(f"\n  [OOM] ROI {roi} too large, trying a smaller one")
    raise RuntimeError(f"Out of memory for every ROI tried: {last_err}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the MONAI brats_mri_segmentation bundle")
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--set", choices=["internal", "cfb"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", default="samples_clean.json")
    ap.add_argument("--external", default="external_cfb.json")
    ap.add_argument("--cfb-tier", default="strict", choices=["strict", "t2star", "all"])
    ap.add_argument("--t0-only", action="store_true")
    ap.add_argument("--all-timepoints", action="store_true")
    ap.add_argument("--roi", type=int, nargs=3, default=None,
                    help="first ROI to try (default 192 192 128); smaller ones are tried on OOM")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--no-ras", action="store_true", help="internal: do not reorient to RAS+")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-hd95", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default=None,
                    help="internal only: keep a single dataset (UPENN, UCSF, UTSW, UCSD, RHUH); "
                         "useful for a smoke test on clean pre-operative cases")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import torch

    from glioma3d.data import load_samples

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = load_bundle_model(Path(args.bundle_dir), device)

    first = tuple(args.roi) if args.roi else ROI_FALLBACKS[0]
    rois = [first] + [r for r in ROI_FALLBACKS if r != first and np.prod(r) < np.prod(first)]

    if args.set == "internal":
        samples = load_samples(args.samples, args.split)
        loader = load_internal_raw if args.no_ras else load_internal_ras
    else:
        samples = select_cfb(args.external, args.cfb_tier, args.t0_only, not args.all_timepoints)
        loader = load_cfb
    if args.dataset:
        samples = [s for s in samples if s["dataset"] == args.dataset]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        sys.exit("No samples selected.")
    print(f"Set        : {args.set} | {len(samples)} cases / "
          f"{len({s['patient_id'] for s in samples})} patients | device {device} | ROI {rois[0]}")

    out_dir = Path(args.out_dir or f"eval_results/{args.set}_monai_segresnet")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    errors: List[Dict] = []
    t0 = time.time()
    main_key = "dice_WT" if args.set == "internal" else "dice_TC"
    roi_used: Dict[str, int] = {}

    for i, s in enumerate(samples, 1):
        try:
            img, gt = loader(s)
            tc, wt, et, roi = predict_regions(net, img, device, rois, args.overlap, not args.no_amp)
            roi_used[str(roi)] = roi_used.get(str(roi), 0) + 1
            row: Dict = {"sample_id": s["sample_id"], "dataset": s["dataset"],
                         "patient_id": s["patient_id"], "fp32_fallback_windows": 0}
            if args.set == "internal":
                pr = {"ET": et, "TC": tc, "WT": wt}
                gr = internal_regions(gt)
                row["mask_source"] = s.get("mask_source", "")
                for k in ("ET", "TC", "WT"):
                    row[f"dice_{k}"] = dice_score(pr[k], gr[k])
                    row[f"dicebc_{k}"] = dice_brats(pr[k], gr[k])
                    row[f"hd95_{k}"] = float("nan") if args.no_hd95 else hd95(pr[k], gr[k])
                row["gt_WT_vox"] = int(gr["WT"].sum())
                row["pred_WT_vox"] = int(wt.sum())
                row["fp_when_gt_empty_WT"] = int(bool(wt.any()) and not gr["WT"].any())
            else:
                g = gt > 0
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
                  f" | {time.time() - t0:.0f}s", end="", flush=True)
        except Exception as e:
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
        "model": "MONAI brats_mri_segmentation (SegResNet, BraTS 2018)",
        "bundle_dir": args.bundle_dir, "set": args.set, "args": vars(args),
        "roi_used": roi_used,
        "env": {"python": platform.python_version(), "torch": torch.__version__},
        "n_cases_ok": len(rows), "n_cases_failed": len(errors),
        "n_patients": len({r["patient_id"] for r in rows}),
        "hd95_units": "voxels (native/RAS grid)" if args.set == "internal" else "mm (1 mm isotropic)",
        "overall": summarize(rows, metric_keys),
    }
    if args.set == "internal":
        summary["by_dataset"] = {ds: summarize([r for r in rows if r["dataset"] == ds], metric_keys)
                                 for ds in sorted({r["dataset"] for r in rows})}
        summary["by_mask_source"] = {m: summarize([r for r in rows if r["mask_source"] == m], metric_keys)
                                     for m in sorted({r["mask_source"] for r in rows})}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"MONAI SegResNet | {args.set} | ok {len(rows)}, failed {len(errors)} | ROI used {roi_used}")
    for k, v in summary["overall"].items():
        if k.startswith(("dice_", "dicebc_")):
            print(f"{k:12s} n={v['n_patients']:4d} | mean {_fmt(v['mean'])}  "
                  f"CI95 [{_fmt(v['ci95'][0])}, {_fmt(v['ci95'][1])}] | median {_fmt(v['median'])}")
    print("=" * 78)
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()