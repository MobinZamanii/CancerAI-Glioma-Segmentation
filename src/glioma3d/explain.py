"""
Model-behaviour analysis ("where does the model look and what does it rely on").

Modes (--mode, can be combined with commas, default "gamma"):

  gamma     Prints the learned gate `gamma` of every local-graph message-passing module. gamma is initialised
            at 0, so |gamma| ~ 0 means the module contributes almost nothing. Seconds to run.

  ablation  Leave-one-modality-out analysis. For N test patients the full-volume prediction is recomputed with
            one MRI channel replaced by zeros (= the per-brain mean after z-scoring). Reports the drop in
            Dice (WT/TC/ET) per removed modality with patient-level bootstrap CIs, plus a bar figure.
            This is a quantitative statement about what the network relies on.

  graphoff  Inference-time test of the graph modules: the same patients are segmented with the learned gamma
            and with gamma forced to 0 (module switched off, no retraining). If Dice does not change, the
            modules contribute nothing at inference. This does NOT replace a retrained --no-graph ablation
            (the rest of the network may have co-adapted), but it is a cheap and honest first check.

  cam       Seg-Grad-CAM figures for selected cases: for each tumor class the gradient of the summed class
            logit over the voxels predicted as that class w.r.t. a chosen layer, plus the localization-head
            heat map and the local-graph attention statistics. QUALITATIVE only: saliency maps show where
            gradients are large, not causal evidence; use them as illustration next to the ablation numbers.

Usage
-----
  python -m glioma3d.explain --checkpoint runs/final_c24/checkpoints/best.pth --mode gamma
  python -m glioma3d.explain --checkpoint ...best.pth --mode ablation --n 60 --overlap 0.25
  python -m glioma3d.explain --checkpoint ...best.pth --mode cam --per-sample eval_results/internal_best/per_sample.csv
  python -m glioma3d.explain --checkpoint ...best.pth --mode cam --cases UCSF__xxx__0,UPENN__yyy__0
  python -m glioma3d.explain --checkpoint ...best.pth --mode ablation --n 2        # smoke test

Outputs in explain_results/<checkpoint stem>/ .
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from glioma3d.evaluate import (load_cfb, select_cfb, PATCH, _build_model, _fmt, _write_csv, dice_score, internal_regions, load_internal,
                      predict_volume, summarize)

CHANNELS = ["T1", "T1ce", "T2", "FLAIR"]


# ---------------------------------------------------------------------------------------------- gamma
def report_gamma(model) -> Dict[str, float]:
    out = {}
    for name, m in model.named_modules():
        if m.__class__.__name__ == "LocalGraphMessagePassing3D":
            g = float(m.gamma.detach().cpu().item())
            pn = float(m.project.weight.detach().float().norm().cpu().item())
            out[name] = g
            print(f"  {name:22s} gamma = {g:+.4f} | ||project weight|| = {pn:.3f}")
    if not out:
        print("  (no local-graph modules in this model, e.g. a --no-graph ablation)")
    return out


# ------------------------------------------------------------------------------------------- ablation
def run_ablation(model, device, args, out_dir: Path) -> None:
    from glioma3d.data import load_samples

    samples = load_samples(args.samples, args.split)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(samples))
    seen_patients, chosen = set(), []
    for i in order:  # one case per patient
        s = samples[int(i)]
        if s["patient_id"] in seen_patients:
            continue
        seen_patients.add(s["patient_id"])
        chosen.append(s)
        if len(chosen) >= args.n:
            break
    print(f"Ablation on {len(chosen)} test patients | overlap {args.overlap} | "
          f"5 passes per patient (baseline + 4 modalities)")

    rows: List[Dict] = []
    t0 = time.time()
    for i, s in enumerate(chosen, 1):
        img, gt = load_internal(s)
        gr = internal_regions(gt)
        base_pred, _ = predict_volume(model, img, device, 4, PATCH, args.overlap, not args.no_amp)
        bpr = internal_regions(base_pred)
        base = {k: dice_score(bpr[k], gr[k]) for k in ("WT", "TC", "ET")}
        for c, cname in enumerate(CHANNELS):
            x = img.copy()
            x[c] = 0.0
            pred, _ = predict_volume(model, x, device, 4, PATCH, args.overlap, not args.no_amp)
            pr = internal_regions(pred)
            row = {"sample_id": s["sample_id"], "dataset": s["dataset"], "patient_id": s["patient_id"],
                   "mask_source": s.get("mask_source", ""), "removed": cname}
            for k in ("WT", "TC", "ET"):
                d = dice_score(pr[k], gr[k])
                row[f"base_{k}"] = base[k]
                row[f"dice_{k}"] = d
                row[f"delta_{k}"] = d - base[k]  # negative = performance loss
            rows.append(row)
        print(f"\r  {i}/{len(chosen)} | {time.time() - t0:.0f}s", end="", flush=True)
        if i % 5 == 0:
            _write_csv(out_dir / "ablation_per_case.csv", rows)
    print()
    _write_csv(out_dir / "ablation_per_case.csv", rows)

    summary = {}
    print("\nMean change in Dice when one modality is removed (negative = the model relied on it)")
    print(f"{'removed':8s} " + " ".join(f"{'dDice_' + k:>26s}" for k in ("WT", "TC", "ET")))
    for cname in CHANNELS:
        sub = [r for r in rows if r["removed"] == cname]
        sm = summarize(sub, ["delta_WT", "delta_TC", "delta_ET"])
        summary[cname] = sm
        cells = []
        for k in ("WT", "TC", "ET"):
            v = sm[f"delta_{k}"]
            cells.append(f"{_fmt(v['mean']):>6s} [{_fmt(v['ci95'][0])},{_fmt(v['ci95'][1])}]")
        print(f"{cname:8s} " + " ".join(f"{c:>26s}" for c in cells))
    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
        width = 0.25
        for j, k in enumerate(("WT", "TC", "ET")):
            means = [summary[c][f"delta_{k}"]["mean"] for c in CHANNELS]
            lo = [summary[c][f"delta_{k}"]["mean"] - summary[c][f"delta_{k}"]["ci95"][0] for c in CHANNELS]
            hi = [summary[c][f"delta_{k}"]["ci95"][1] - summary[c][f"delta_{k}"]["mean"] for c in CHANNELS]
            ax.bar(np.arange(4) + (j - 1) * width, means, width, yerr=[lo, hi], capsize=2, label=k)
        ax.set_xticks(range(4))
        ax.set_xticklabels(CHANNELS)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("Change in Dice when modality removed")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_modalities.png")
        plt.close(fig)
        print(f"Figure: {out_dir / 'ablation_modalities.png'}")
    except Exception as e:  # matplotlib optional
        print(f"(figure skipped: {e})")


# ----------------------------------------------------------------------------------------- graph off
def run_graph_off(model, device, args, out_dir: Path) -> None:
    from glioma3d.data import load_samples

    mods = [m for m in model.modules() if m.__class__.__name__ == "LocalGraphMessagePassing3D"]
    if not mods:
        print("  no graph modules in this model")
        return
    samples = load_samples(args.samples, args.split)
    rng = np.random.default_rng(args.seed)
    seen, chosen = set(), []
    for i in rng.permutation(len(samples)):
        s = samples[int(i)]
        if s["patient_id"] in seen:
            continue
        seen.add(s["patient_id"])
        chosen.append(s)
        if len(chosen) >= args.n:
            break
    saved = [m.gamma.data.clone() for m in mods]
    rows: List[Dict] = []
    t0 = time.time()
    try:
        for i, s in enumerate(chosen, 1):
            img, gt = load_internal(s)
            gr = internal_regions(gt)
            res = {}
            for tag, off in (("on", False), ("off", True)):
                for m, g in zip(mods, saved):
                    m.gamma.data.copy_(g * 0 if off else g)
                pred, _ = predict_volume(model, img, device, 4, PATCH, args.overlap, not args.no_amp)
                pr = internal_regions(pred)
                res[tag] = {k: dice_score(pr[k], gr[k]) for k in ("WT", "TC", "ET")}
            row = {"sample_id": s["sample_id"], "dataset": s["dataset"], "patient_id": s["patient_id"]}
            for k in ("WT", "TC", "ET"):
                row[f"on_{k}"], row[f"off_{k}"] = res["on"][k], res["off"][k]
                row[f"delta_{k}"] = res["off"][k] - res["on"][k]
            rows.append(row)
            print(f"\r  {i}/{len(chosen)} | {time.time() - t0:.0f}s", end="", flush=True)
    finally:
        for m, g in zip(mods, saved):  # always restore the learned gates
            m.gamma.data.copy_(g)
    print()
    _write_csv(out_dir / "graph_off_per_case.csv", rows)
    sm = summarize(rows, ["on_WT", "off_WT", "delta_WT", "on_TC", "off_TC", "delta_TC", "on_ET", "off_ET", "delta_ET"])
    (out_dir / "graph_off_summary.json").write_text(json.dumps(sm, indent=2), encoding="utf-8")
    print("\nGraph modules switched off at inference (gamma -> 0); negative delta = Dice lost")
    for k in ("WT", "TC", "ET"):
        a, b, d = sm[f"on_{k}"], sm[f"off_{k}"], sm[f"delta_{k}"]
        print(f"  {k}: on {_fmt(a['mean'])} | off {_fmt(b['mean'])} | delta {_fmt(d['mean'])} "
              f"[{_fmt(d['ci95'][0])}, {_fmt(d['ci95'][1])}]  (n={d['n_patients']})")


# ------------------------------------------------------------------------------------------------ CAM
LAYERS = {"down3": lambda m: m.down3, "down4": lambda m: m.down4, "up1": lambda m: m.up1, "up0": lambda m: m.up0}


def crop_center(img: np.ndarray, gt: np.ndarray, size=PATCH) -> Tuple[np.ndarray, np.ndarray]:
    """96^3 patch centred on the tumor centroid (zero padded if the volume is smaller)."""
    C, D, H, W = img.shape
    pad = [(0, max(0, p - s)) for p, s in zip(size, (D, H, W))]
    if any(b for _, b in pad):
        img = np.pad(img, [(0, 0)] + pad)
        gt = np.pad(gt, pad)
    idx = np.argwhere(gt > 0)
    cen = idx.mean(0) if len(idx) else np.array(gt.shape) / 2.0
    start = [int(np.clip(round(c - p / 2), 0, s - p)) for c, p, s in zip(cen, size, gt.shape)]
    sl = tuple(slice(a, a + p) for a, p in zip(start, size))
    return img[(slice(None),) + sl], gt[sl]


def graph_attention_stats(model, x, device, gt_patch) -> Dict[str, Dict[str, float]]:
    """Mean attention weight per direction (+-D, +-H, +-W) inside vs outside the tumor for each graph module."""
    import torch
    import torch.nn.functional as F

    store: Dict[str, "torch.Tensor"] = {}
    hooks = []
    for name, m in model.named_modules():
        if m.__class__.__name__ == "LocalGraphMessagePassing3D":
            hooks.append(m.register_forward_hook(lambda mod, inp, out, n=name: store.__setitem__(n, inp[0].detach())))
    with torch.inference_mode():
        model(x)
    for h in hooks:
        h.remove()

    res = {}
    for name, m in model.named_modules():
        if name not in store:
            continue
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            f = store[name].float()
            q, k_map = m.query(f), m.key(f)
            keys, masks = [], []
            for off in m.offsets:
                sk, valid = m._shift_with_validity(k_map, off)
                keys.append(sk)
                masks.append(valid)
            keys = torch.stack(keys, 1)
            masks = torch.stack(masks, 1)
            scores = (q.unsqueeze(1) * keys).sum(2) * m.scale
            scores = scores.masked_fill(~masks.squeeze(2), torch.finfo(scores.dtype).min)
            w = torch.softmax(scores, dim=1)[0]  # [6, d, h, w]
            tumor = F.adaptive_max_pool3d(torch.from_numpy((gt_patch > 0).astype(np.float32))[None, None].to(device),
                                          w.shape[-3:])[0, 0] > 0.5
            inside = w[:, tumor].mean(1).cpu().numpy() if tumor.any() else np.full(6, np.nan)
            outside = w[:, ~tumor].mean(1).cpu().numpy() if (~tumor).any() else np.full(6, np.nan)
        names = ["-D", "+D", "-H", "+H", "-W", "+W"]
        res[name] = {"gamma": float(m.gamma.detach().cpu().item()),
                     **{f"in_{n}": float(v) for n, v in zip(names, inside)},
                     **{f"out_{n}": float(v) for n, v in zip(names, outside)}}
    return res


def seg_grad_cam(model, x, layer_name: str, cls: int, device, use_amp: bool) -> Optional[np.ndarray]:
    """Seg-Grad-CAM for one class at one layer. x: [1,C,96,96,96] tensor. Returns [96,96,96] in 0..1 or None."""
    import torch
    import torch.nn.functional as F

    layer = LAYERS[layer_name](model)
    acts: List["torch.Tensor"] = []
    grads: List["torch.Tensor"] = []

    def fwd_hook(mod, inp, out):
        acts.append(out)
        out.register_hook(lambda g: grads.append(g))

    h = layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
            logits = out["segmentation"]
            region = (logits.argmax(1) == cls)
            if not bool(region.any()):
                return None
            # sum rescaled to ~1000 voxels: keeps fp16 gradients away from under/overflow
            n_vox = region.float().sum()
            score = (logits[:, cls].float() * region.float()).sum() / (n_vox / 1000.0).clamp(min=1.0)
        score.backward()
    finally:
        h.remove()
    a, g = acts[0].detach().float(), grads[0].detach().float()
    weights = g.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((weights * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-3:], mode="trilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp(min=1e-8)
    model.zero_grad(set_to_none=True)
    return cam.cpu().numpy()


def pick_cases(args, samples: List[Dict]) -> List[Dict]:
    by_id = {s["sample_id"]: s for s in samples}
    if args.cases:
        return [by_id[c] for c in args.cases.split(",") if c in by_id]
    if not args.per_sample:
        raise SystemExit("--mode cam needs either --cases or --per-sample (eval_results/.../per_sample.csv)")
    with open(args.per_sample, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rank_col = "dice_TC" if args.set == "cfb" else "dice_WT"
    rows = [r for r in rows if r.get(rank_col) not in ("", "nan", None) and r["sample_id"] in by_id]
    if args.set == "cfb":
        # evenly spaced quality quantiles, from best to worst (includes both extremes)
        rows.sort(key=lambda r: -float(r[rank_col]))
        n = min(args.n_cam, len(rows))
        picks = sorted({round(i * (len(rows) - 1) / max(n - 1, 1)) for i in range(n)})
        return [by_id[rows[i]["sample_id"]] for i in picks]
    chosen: List[str] = []
    for ds in sorted({r["dataset"] for r in rows}):
        sub = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: float(r[rank_col]))
        if not sub:
            continue
        chosen += [sub[len(sub) // 2]["sample_id"], sub[0]["sample_id"]]  # median-quality and worst case
    return [by_id[c] for c in chosen[: args.n_cam] if c in by_id]


def run_cam(model, device, args, out_dir: Path) -> None:
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from glioma3d.data import load_samples

    is_cfb = args.set == "cfb"
    samples = select_cfb(args.cfb_json, args.cfb_tier, False, True) if is_cfb else load_samples(args.samples, args.split)
    cases = pick_cases(args, samples)
    layers = [l for l in args.layers.split(",") if l in LAYERS]
    use_amp = (not args.no_amp) and device.type == "cuda"
    lab_cmap = ListedColormap([(0, 0, 0, 0), (0.2, 0.4, 1, 0.55), (0.1, 0.8, 0.2, 0.45), (1, 0.9, 0.1, 0.65)])
    cls_names = {1: "NCR", 2: "ED", 3: "ET"}
    stats_all = {}
    print(f"Seg-Grad-CAM for {len(cases)} cases at layers {layers}")

    for s in cases:
        img, gt = load_cfb(s) if is_cfb else load_internal(s)
        patch, gpatch = crop_center(img, gt)
        x = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32))[None].to(device)
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
        pred = out["segmentation"].float().argmax(1)[0].cpu().numpy()
        loc = torch.sigmoid(out["localization"].float())
        loc = torch.nn.functional.interpolate(loc, size=PATCH, mode="trilinear", align_corners=False)[0, 0].cpu().numpy()
        stats_all[s["sample_id"]] = graph_attention_stats(model, x, device, gpatch)

        idx = np.argwhere(gpatch > 0)
        z = int(round(idx[:, 2].mean())) if len(idx) else PATCH[2] // 2
        cams = {}
        for ln in layers:
            for c in (1, 2, 3):
                cams[(ln, c)] = seg_grad_cam(model, x, ln, c, device, use_amp)
            torch.cuda.empty_cache()

        ncol = 4 + 3
        fig, axes = plt.subplots(len(layers) + 1, ncol, figsize=(2.2 * ncol, 2.3 * (len(layers) + 1)), dpi=170)
        axes = np.atleast_2d(axes)
        bg = patch[3][:, :, z]  # FLAIR
        bg2 = patch[1][:, :, z]  # T1ce

        def base(ax, im, title):
            ax.imshow(im.T, cmap="gray", origin="lower")
            ax.set_title(title, fontsize=7)
            ax.axis("off")

        base(axes[0, 0], bg, "FLAIR")
        base(axes[0, 1], bg2, "T1ce")
        if is_cfb:  # binary radiotherapy GTV: show as a contour
            base(axes[0, 2], bg, "GTV (contour)")
            if (gpatch[:, :, z] > 0).any():
                axes[0, 2].contour(gpatch[:, :, z].T, levels=[0.5], colors=["cyan"], linewidths=0.9, origin="lower")
        else:
            base(axes[0, 2], bg, "Ground truth")
            axes[0, 2].imshow(gpatch[:, :, z].T, cmap=lab_cmap, vmin=0, vmax=3, origin="lower")
        base(axes[0, 3], bg, "Prediction")
        axes[0, 3].imshow(pred[:, :, z].T, cmap=lab_cmap, vmin=0, vmax=3, origin="lower")
        base(axes[0, 4], bg, "Localization head")
        axes[0, 4].imshow(loc[:, :, z].T, cmap="jet", alpha=0.45, vmin=0, vmax=1, origin="lower")
        for j in (5, 6):
            axes[0, j].axis("off")
        for r, ln in enumerate(layers, start=1):
            for j in range(4):
                axes[r, j].axis("off")
            for j, c in enumerate((1, 2, 3)):
                ax = axes[r, 4 + j]
                base(ax, bg, f"{cls_names[c]} CAM @ {ln}")
                cam = cams[(ln, c)]
                if cam is None:
                    ax.text(0.5, 0.5, "class not\npredicted", transform=ax.transAxes, ha="center", fontsize=7,
                            color="w")
                else:
                    ax.imshow(cam[:, :, z].T, cmap="jet", alpha=0.45, vmin=0, vmax=1, origin="lower")
        fig.suptitle(f"{s['sample_id']}  (axial slice {z})", fontsize=8)
        fig.tight_layout()
        path = out_dir / f"cam_{s['sample_id']}.png"
        if is_cfb:
            fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path)
        plt.close(fig)
        print(f"  saved {path}")
    (out_dir / "graph_attention_stats.json").write_text(json.dumps(stats_all, indent=2), encoding="utf-8")
    print(f"Graph-attention statistics: {out_dir / 'graph_attention_stats.json'}")


# --------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Where does the model look / what does it rely on?")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", default="gamma", help="comma list of: gamma, ablation, graphoff, cam")
    ap.add_argument("--samples", default="samples_clean.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--set", default="internal", choices=["internal", "cfb"], help="cam: internal test split or external CFB")
    ap.add_argument("--cfb-json", default="external_cfb.json")
    ap.add_argument("--cfb-tier", default="strict", choices=["strict", "t2star", "all"])
    ap.add_argument("--n", type=int, default=60, help="ablation: number of test patients")
    ap.add_argument("--overlap", type=float, default=0.25, help="ablation sliding-window overlap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cases", default="", help="cam: comma-separated sample_ids")
    ap.add_argument("--per-sample", default="", help="cam: per_sample.csv from evaluate.py (auto picks median/worst per dataset)")
    ap.add_argument("--n-cam", type=int, default=8)
    ap.add_argument("--layers", default="down3,up1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    model = _build_model(ckpt["config"], device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    out_dir = Path(args.out_dir or f"explain_results/{Path(args.checkpoint).stem}" + ("_cfb" if args.set == "cfb" else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint {args.checkpoint} | epoch {ckpt.get('epoch')} | device {device}")

    modes = [m.strip() for m in args.mode.split(",")]
    if "gamma" in modes:
        print("\n[gamma] learned gates of the local-graph modules")
        g = report_gamma(model)
        (out_dir / "gamma.json").write_text(json.dumps(g, indent=2), encoding="utf-8")
    if "ablation" in modes:
        print("\n[ablation] leave-one-modality-out")
        run_ablation(model, device, args, out_dir)
    if "graphoff" in modes:
        print("\n[graphoff] graph modules disabled at inference")
        run_graph_off(model, device, args, out_dir)
    if "cam" in modes:
        print("\n[cam] Seg-Grad-CAM")
        run_cam(model, device, args, out_dir)
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()