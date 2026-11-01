"""
Evaluation of the LOCALIZATION head (full volume).

The localization head predicts a coarse (1/4 resolution) binary "tumor present" map. During training its
target is the whole-tumor mask (labels > 0) max-pooled by 4. This script runs sliding-window inference with
the localization output, stitches the 1/4-resolution probability maps, and compares them with the same
max-pooled target.

Per-case metrics
----------------
loc_dice          Dice of (prob > threshold) vs the pooled target
loc_sens/loc_prec voxel-level sensitivity / precision at the same threshold
peak_hit          1 if the arg-max of the probability map lies inside the target (detection success)
centroid_err      distance between the centroids of prediction and target (voxels of the native grid for
                  internal data, millimetres for CFB, i.e. 1/4-grid distance x 4); NaN if prediction empty
n_cc_pred         number of connected components in the thresholded prediction (extra blobs = false alarms)
pred_ratio        predicted volume / target volume

Usage
-----
  python -m glioma3d.evaluate_localization --checkpoint runs/final_c24/checkpoints/best.pth --set internal --split test
  python -m glioma3d.evaluate_localization --checkpoint ...best.pth --set cfb --cfb-tier strict
  python -m glioma3d.evaluate_localization --checkpoint ...best.pth --set internal --limit 2        # smoke test

Note on CFB: the GTV is not the whole tumor, so localization on CFB is evaluated against the GTV only as a
"is the predicted region centred on the treated target" check. Interpret loc_dice there with care.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from glioma3d.evaluate import (PATCH, _build_model, _fmt, _write_csv, dice_score, gaussian_window,
                      load_cfb, load_internal, select_cfb, summarize, window_starts)

STRIDE = 4  # localization head resolution relative to the input


def predict_localization(model, img: np.ndarray, device, patch: Sequence[int] = PATCH,
                         overlap: float = 0.25, amp: bool = True) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Returns (probability map at 1/4 resolution, padded full-resolution shape)."""
    import torch
    import torch.nn.functional as F

    C, D, H, W = img.shape
    tgt = [max(p, int(np.ceil(s / STRIDE) * STRIDE)) for p, s in zip(patch, (D, H, W))]
    pad = [(0, t - s) for t, s in zip(tgt, (D, H, W))]
    imgp = np.pad(img, [(0, 0)] + pad) if any(b for _, b in pad) else img
    _, Dp, Hp, Wp = imgp.shape
    lo = (Dp // STRIDE, Hp // STRIDE, Wp // STRIDE)
    ps = [p // STRIDE for p in patch]
    # strides must be multiples of 4 so that window origins stay aligned with the 1/4 grid
    strides = [max(STRIDE, int(p * (1 - overlap)) // STRIDE * STRIDE) for p in patch]

    acc = torch.zeros(lo, dtype=torch.float32, device=device)
    wsum = torch.zeros(lo, dtype=torch.float32, device=device)
    w = torch.from_numpy(gaussian_window(patch)).to(device)[None, None]
    w_lo = F.avg_pool3d(w, STRIDE)[0, 0]
    use_amp = amp and getattr(device, "type", str(device)) == "cuda"

    with torch.inference_mode():
        for sd in window_starts(Dp, patch[0], strides[0]):
            for sh in window_starts(Hp, patch[1], strides[1]):
                for sw in window_starts(Wp, patch[2], strides[2]):
                    block = imgp[:, sd:sd + patch[0], sh:sh + patch[1], sw:sw + patch[2]]
                    if not block.any():
                        continue
                    x = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))[None].to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        logit = model(x)["localization"]
                    if use_amp and not bool(torch.isfinite(logit).all()):
                        with torch.autocast(device_type="cuda", enabled=False):
                            logit = model(x)["localization"]
                    prob = torch.sigmoid(logit.float())[0, 0]
                    a, b, c = sd // STRIDE, sh // STRIDE, sw // STRIDE
                    acc[a:a + ps[0], b:b + ps[1], c:c + ps[2]] += prob * w_lo
                    wsum[a:a + ps[0], b:b + ps[1], c:c + ps[2]] += w_lo
    out = (acc / wsum.clamp(min=1e-6)).cpu().numpy()
    return out, (Dp, Hp, Wp)


def pooled_target(tumor: np.ndarray, padded_shape: Tuple[int, int, int]) -> np.ndarray:
    """Max-pool a binary tumor mask by 4 (same operation as the training target)."""
    D, H, W = tumor.shape
    full = np.zeros(padded_shape, dtype=bool)
    full[:D, :H, :W] = tumor > 0
    d, h, w = (s // STRIDE for s in padded_shape)
    return full.reshape(d, STRIDE, h, STRIDE, w, STRIDE).any(axis=(1, 3, 5))


def loc_metrics(prob: np.ndarray, target: np.ndarray, thr: float) -> Dict[str, float]:
    from scipy import ndimage as ndi

    pred = prob > thr
    t = int(target.sum())
    p = int(pred.sum())
    inter = int(np.logical_and(pred, target).sum())
    peak = np.unravel_index(int(np.argmax(prob)), prob.shape)
    out = {
        "loc_dice": dice_score(pred, target),
        "loc_sens": inter / t if t else float("nan"),
        "loc_prec": inter / p if p else float("nan"),
        "peak_hit": float(bool(target[peak])) if t else float("nan"),
        "pred_ratio": p / t if t else float("nan"),
        "n_cc_pred": float(ndi.label(pred)[1]),
        "centroid_err": float("nan"),
    }
    if p and t:
        c_pred = np.array(np.nonzero(pred)).mean(axis=1)
        c_tgt = np.array(np.nonzero(target)).mean(axis=1)
        out["centroid_err"] = float(np.linalg.norm(c_pred - c_tgt) * STRIDE)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the localization head (full volume)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--set", choices=["internal", "cfb"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", default="samples_clean.json")
    ap.add_argument("--external", default="external_cfb.json")
    ap.add_argument("--cfb-tier", default="strict", choices=["strict", "t2star", "all"])
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
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
    model = _build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Checkpoint {args.checkpoint} | epoch {ckpt.get('epoch')} | loc_weight {cfg.get('loc_weight', 0.3)} | "
          f"threshold {args.threshold} | overlap {args.overlap}")

    if args.set == "internal":
        samples, loader = load_samples(args.samples, args.split), load_internal
        unit = "voxels"
    else:
        samples, loader = select_cfb(args.external, args.cfb_tier, False, True), load_cfb
        unit = "mm"
    if args.limit:
        samples = samples[: args.limit]
    print(f"Set {args.set}: {len(samples)} cases / {len({s['patient_id'] for s in samples})} patients")

    out_dir = Path(args.out_dir or f"eval_results/loc_{args.set}_{Path(args.checkpoint).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    errors: List[Dict] = []
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        try:
            img, gt = loader(s)
            prob, padded = predict_localization(model, img, device, PATCH, args.overlap, not args.no_amp)
            target = pooled_target(gt, padded)
            m = loc_metrics(prob, target, args.threshold)
            row = {"sample_id": s["sample_id"], "dataset": s["dataset"], "patient_id": s["patient_id"],
                   "mask_source": s.get("mask_source", "")}
            row.update(m)
            rows.append(row)
            print(f"\r  {i}/{len(samples)} | loc_dice={_fmt(m['loc_dice'])} hit={m['peak_hit']:.0f} | "
                  f"{time.time() - t0:.0f}s", end="", flush=True)
        except Exception as e:
            errors.append({"sample_id": s.get("sample_id"), "error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc(limit=3)})
            print(f"\n  [ERROR] {s.get('sample_id')}: {type(e).__name__}: {e}")
        if i % 5 == 0:
            _write_csv(out_dir / "per_sample.csv", rows)
    print()
    if not rows:
        raise SystemExit("All cases failed.")
    _write_csv(out_dir / "per_sample.csv", rows)
    _write_csv(out_dir / "errors.csv", errors)

    keys = ["loc_dice", "loc_sens", "loc_prec", "peak_hit", "centroid_err", "n_cc_pred", "pred_ratio"]
    summary = {"checkpoint": args.checkpoint, "epoch": ckpt.get("epoch"), "set": args.set, "args": vars(args),
               "centroid_err_units": unit, "n_cases_ok": len(rows), "n_cases_failed": len(errors),
               "overall": summarize(rows, keys)}
    if args.set == "internal":
        summary["by_dataset"] = {ds: summarize([r for r in rows if r["dataset"] == ds], keys)
                                 for ds in sorted({r["dataset"] for r in rows})}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    for k, v in summary["overall"].items():
        print(f"{k:13s} n={v['n_patients']:4d} | mean {_fmt(v['mean'])}  CI95 [{_fmt(v['ci95'][0])}, "
              f"{_fmt(v['ci95'][1])}] | median {_fmt(v['median'])} | NaN {v['n_nan']}")
    print(f"centroid_err units: {unit}")
    if args.set == "internal":
        print("-" * 78 + "\nPer dataset (loc_dice / peak_hit):")
        for ds, d in summary["by_dataset"].items():
            print(f"  {ds:8s} n={d['loc_dice']['n_patients']:3d} | loc_dice {_fmt(d['loc_dice']['mean'])}  "
                  f"peak_hit {_fmt(d['peak_hit']['mean'])}")
    print("=" * 78 + f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
