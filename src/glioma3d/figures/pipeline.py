"""
Pipeline figure: real MRI slices -> network -> localisation / segmentation -> error map -> evaluation.

Run after evaluate.py has written per_sample.csv (needs a checkpoint and the samples/external json files):

  # one example case from the internal test set (median Dice WT of a cohort, default UPENN)
  python -m glioma3d.figures.pipeline --set internal --dataset UPENN --pick median ^
      --checkpoint runs/final_c24/checkpoints/best.pth ^
      --per-sample eval_results/internal_best/per_sample.csv

  # a specific case
  python -m glioma3d.figures.pipeline --set internal --sample-id <id> ...

  # external CFB-GBM example
  python -m glioma3d.figures.pipeline --set cfb --pick median --checkpoint ... --per-sample eval_results/cfb_best/per_sample.csv

Output: qualitative_figs/pipeline_<set>_<case>.png (600 dpi) and .pdf, white background.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

C_NCR, C_ED, C_ET = "#0072B2", "#F0E442", "#D55E00"
C_TP, C_FP, C_FN = "#009E73", "#CC79A7", "#56B4E9"
INK, GREY = "#222222", "#6b6b6b"
LABEL_CMAP = ListedColormap(["none", C_NCR, C_ED, C_ET])


# ---------------------------------------------------------------- small helpers
def window(a):
    v = a[a != 0]
    return (0, 1) if v.size == 0 else (np.percentile(v, 1), np.percentile(v, 99.5))


def prep_slice(vol, k, box):
    return np.rot90(vol[:, :, k][box])


def add_img(fig, rect, img2d, title=None, cmap="gray", vlim=None, title_size=8):
    ax = fig.add_axes(rect)
    ax.set_zorder(3)
    lo, hi = vlim if vlim else window(img2d)
    ax.imshow(img2d, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=title_size, color=INK, pad=2)
    return ax


def rbox(fig, x, y, w, h, fc="white", ec=INK, lw=1.2, ls="-", r=0.012, z=0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       transform=fig.transFigure, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    fig.add_artist(p)
    return p


def arrow(fig, x0, y0, x1, y1, color=INK, lw=1.6, style="-|>", ls="-", rad=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle=style,
                        mutation_scale=11, color=color, lw=lw, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=5)
    fig.add_artist(a)


def txt(fig, x, y, s, size=8, color=INK, ha="center", va="center", weight="normal", style="normal", **kw):
    fig.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, fontstyle=style, **kw)


def block(fig, x, y, w, h, label, fc="#e8eef6", ec=INK, ga=False):
    rbox(fig, x, y, w, h, fc=fc, ec=("#D55E00" if ga else ec), lw=(2.0 if ga else 0.9), r=0.004, z=2)
    txt(fig, x + w / 2, y + h / 2, label, size=6.5)


# ---------------------------------------------------------------- the figure
def draw_pipeline(img, gt, pred, loc, mode, meta, out_base):
    """
    img [4,X,Y,Z] (RAS, channel order T1,T1ce,T2,FLAIR); gt/pred [X,Y,Z]; loc [X,Y,Z] probability at full resolution.
    meta: dict(case=..., dice_label=..., dice_value=...)
    """
    if mode == "internal":
        gt_b, pr_b, ref_name = gt > 0, pred > 0, "Reference mask"
    else:
        gt_b, pr_b, ref_name = gt > 0, (pred == 1) | (pred == 3), "Reference (GTV)"
    area = gt_b.sum(axis=(0, 1))
    k = int(area.argmax()) if area.max() > 0 else img.shape[3] // 2
    nz = np.argwhere(img[3][:, :, k] != 0)
    (x0, y0), (x1, y1) = nz.min(0), nz.max(0)
    box = (slice(max(0, x0 - 4), x1 + 5), slice(max(0, y0 - 4), y1 + 5))
    S = lambda v: prep_slice(v, k, box)

    fig = plt.figure(figsize=(13.0, 6.6), facecolor="white")
    asp = S(img[0]).shape[0] / S(img[0]).shape[1]          # height / width of the cropped slice
    k_fig = 13.0 / 6.6                                      # figure width / height
    W_IMG = min(0.085, 0.21 / (k_fig * asp))
    H_IMG = W_IMG * k_fig * asp

    # ---- input panel
    rbox(fig, 0.008, 0.30, 0.205, 0.665, fc="#fbfbfb", ec=GREY, ls=(0, (5, 3)), z=0)
    txt(fig, 0.110, 0.945, "Input: four MRI sequences", size=9.5, weight="bold")
    names = ["T1", "T1ce", "T2", "FLAIR"]
    xs, ys = [0.022, 0.115], [0.905 - H_IMG - 0.03, 0.905 - 2 * H_IMG - 0.085]
    for i, nm in enumerate(names):
        cx, cy = xs[i % 2], ys[i // 2]
        for off in (0.010, 0.005):                         # stack effect: 3D volume
            fig.add_artist(Rectangle((cx + off * 0.6, cy - off * 1.2), W_IMG, H_IMG, transform=fig.transFigure,
                                     fc="#cfcfcf", ec="white", lw=0.8, zorder=1))
        add_img(fig, [cx, cy, W_IMG, H_IMG], S(img[i]))
        txt(fig, cx + W_IMG / 2, cy - 0.028, nm, size=8.5)
    txt(fig, 0.110, 0.325, "3D volumes, native grid", size=7.5, color=GREY, style="italic")

    # ---- arrow 1
    arrow(fig, 0.216, 0.63, 0.262, 0.63)
    txt(fig, 0.239, 0.695, "z-score\n96³ windows\n(Gaussian\nfusion)", size=7, color=GREY)

    # ---- network panel
    rbox(fig, 0.265, 0.30, 0.355, 0.665, fc="#f6f9fc", ec=INK, z=0)
    txt(fig, 0.4425, 0.945, "3D residual U-Net  (17.3 M parameters)", size=9.5, weight="bold")
    ch = [24, 48, 96, 192, 384]
    bw, bh = 0.045, 0.050
    ytop, dy = 0.835, 0.100
    enc, dec = [], []
    for l in range(5):
        y = ytop - dy * l
        xe = 0.285 + 0.022 * l
        xd = 0.500 - 0.022 * l
        ga = l >= 3
        block(fig, xe, y, bw, bh, f"{ch[l]}", ga=ga)
        if l < 4:
            block(fig, xd, y, bw, bh, f"{ch[l]}", ga=False)
        enc.append((xe, y)); dec.append((xd, y))
    for l in range(4):                                     # skip connections
        (xe, y), (xd, _) = enc[l], dec[l]
        arrow(fig, xe + bw, y + bh / 2, xd, y + bh / 2, color="#8a8a8a", lw=0.9, ls=(0, (3, 2)))
    for l in range(4):                                     # encoder path (down)
        arrow(fig, enc[l][0] + bw / 2, enc[l][1], enc[l + 1][0] + bw / 2, enc[l + 1][1] + bh, lw=1.0)
    arrow(fig, enc[4][0] + bw, enc[4][1] + bh / 2, dec[3][0] + bw / 2, dec[3][1], lw=1.0, rad=-0.3)
    for l in range(3, 0, -1):                              # decoder path (up)
        arrow(fig, dec[l][0] + bw / 2, dec[l][1] + bh, dec[l - 1][0] + bw / 2, dec[l - 1][1], lw=1.0)
    # heads
    rbox(fig, 0.556, 0.822, 0.058, 0.075, fc="#e1f0ea", ec=C_TP, lw=1.3, r=0.005, z=2)
    txt(fig, 0.585, 0.8595, "Segmentation\nhead\n(4 classes)", size=6.3)
    arrow(fig, dec[0][0] + bw, dec[0][1] + bh / 2, 0.556, dec[0][1] + bh / 2, color=C_TP)
    rbox(fig, 0.283, 0.335, 0.070, 0.062, fc="#f6e4ee", ec=C_FP, lw=1.3, r=0.005, z=2)
    txt(fig, 0.318, 0.366, "Localisation\nhead (1/4 res.)", size=6.3)
    arrow(fig, enc[2][0] + 0.010, enc[2][1], 0.318, 0.397, color=C_FP)
    # legend + loss
    rbox(fig, 0.460, 0.337, 0.012, 0.016, fc="#e8eef6", ec="#D55E00", lw=1.8, r=0.002, z=2)
    txt(fig, 0.477, 0.345, "block with local graph attention", size=6.8, ha="left")
    txt(fig, 0.4425, 0.318, "loss:  L = L_seg + 0.3 \u00b7 L_loc", size=7.5, color=GREY, style="italic")
    txt(fig, 0.4425, 0.40, "numbers = feature channels;  dashed = skip connections", size=6.3, color=GREY)

    # ---- arrow 2 (outputs)
    arrow(fig, 0.623, 0.63, 0.668, 0.63)

    # ---- outputs
    rbox(fig, 0.672, 0.30, 0.32, 0.665, fc="#fbfbfb", ec=GREY, ls=(0, (5, 3)), z=0)
    txt(fig, 0.832, 0.945, "Output for this case", size=9.5, weight="bold")
    t1ce = S(img[1]); lo, hi = window(t1ce)
    asp2 = t1ce.shape[0] / t1ce.shape[1]
    OW = min(0.125, 0.225 / (k_fig * asp2)); OH = OW * k_fig * asp2
    r1 = 0.895 - OH - 0.035
    r2 = r1 - OH - 0.060
    pos = {"loc": (0.690, r1), "seg": (0.838, r1), "ref": (0.690, r2), "err": (0.838, r2)}
    titles = {"loc": "Localisation map", "seg": "Segmentation", "ref": ref_name, "err": "Error map"}
    axes = {}
    for key, (cx, cy) in pos.items():
        axes[key] = add_img(fig, [cx, cy, OW, OH], t1ce, vlim=(lo, hi))
        txt(fig, cx + OW / 2, cy + OH + 0.016, titles[key], size=8.5)
    # localisation heat map
    axes["loc"].imshow(np.ma.masked_less(S(loc), 0.05), cmap="magma", vmin=0, vmax=1, alpha=0.75, interpolation="nearest")
    # segmentation overlay
    seg_s = S(pred if mode == "internal" else np.where(pr_b, 3, 0))
    axes["seg"].imshow(np.ma.masked_equal(seg_s, 0), cmap=LABEL_CMAP, vmin=0, vmax=3, alpha=0.75, interpolation="nearest")
    # reference
    if mode == "internal":
        axes["ref"].imshow(np.ma.masked_equal(S(gt), 0), cmap=LABEL_CMAP, vmin=0, vmax=3, alpha=0.75, interpolation="nearest")
    else:
        axes["ref"].contour(S(gt_b).astype(float), levels=[0.5], colors=[C_TP], linewidths=1.5)
    # error map
    tp, fp, fn = pr_b & gt_b, pr_b & ~gt_b, ~pr_b & gt_b
    rgba = np.zeros(S(tp).shape + (4,))
    for m, c in ((tp, C_TP), (fp, C_FP), (fn, C_FN)):
        rgba[S(m)] = matplotlib.colors.to_rgba(c, 0.85)
    axes["err"].imshow(rgba, interpolation="nearest")
    # legends under outputs
    def chip(x, y, c, s):
        fig.add_artist(Rectangle((x, y - 0.007), 0.011, 0.014, transform=fig.transFigure, fc=c, ec="none"))
        txt(fig, x + 0.015, y, s, size=7, ha="left")
    if mode == "internal":
        for i, (c, s) in enumerate(((C_NCR, "NCR"), (C_ED, "ED"), (C_ET, "ET"))):
            chip(0.690 + 0.040 * i, 0.322, c, s)
        for i, (c, s) in enumerate(((C_TP, "TP"), (C_FP, "FP"), (C_FN, "FN"))):
            chip(0.840 + 0.040 * i, 0.322, c, s)
    else:
        for i, (c, s) in enumerate(((C_ET, "Predicted TC"),)):
            chip(0.690, 0.322, c, s)
        for i, (c, s) in enumerate(((C_TP, "TP / GTV"), (C_FP, "FP"), (C_FN, "FN"))):
            chip(0.840 + 0.045 * i, 0.322, c, s)

    # ---- evaluation strip
    arrow(fig, 0.80, 0.298, 0.80, 0.238, color=GREY)
    arrow(fig, 0.72, 0.298, 0.40, 0.238, color=GREY)
    rbox(fig, 0.008, 0.03, 0.485, 0.205, fc="#eef4fa", ec=C_ET if False else "#0072B2", lw=1.3, z=0)
    txt(fig, 0.2505, 0.205, "Internal evaluation: held-out test set (patient-level split)", size=9, weight="bold")
    txt(fig, 0.2505, 0.150, "285 patients, five cohorts; whole-volume sliding-window inference\n"
                            "Dice and HD95 for WT, TC, ET; patient-level bootstrap intervals;\n"
                            "results by cohort and by reference-mask source", size=8)
    txt(fig, 0.2505, 0.060, "Paired comparison: same-code baseline and off-the-shelf MONAI SegResNet", size=7.5, color=GREY, style="italic")
    rbox(fig, 0.507, 0.03, 0.485, 0.205, fc="#fbf1ea", ec="#D55E00", lw=1.3, z=0)
    txt(fig, 0.7495, 0.205, "External validation: CFB-GBM (never used in training)", size=9, weight="bold")
    txt(fig, 0.7495, 0.150, "94 patients; resampled to 1 mm, skull-stripped\n"
                            "primary outcome fixed in advance: Dice of TC against the radiotherapy GTV;\n"
                            "secondary: HD95 in mm; sensitivity analysis with T2*", size=8)
    txt(fig, 0.7495, 0.060, "Absolute Dice not comparable with internal results (different reference)", size=7.5, color=GREY, style="italic")

    # case label
    txt(fig, 0.832, 0.918, f"{meta['case']}  |  {meta['dice_label']} = {meta['dice_value']:.2f}", size=7.5, color=GREY)

    for ext, kw in (("png", {"dpi": 600}), ("pdf", {})):
        fig.savefig(f"{out_base}.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print("saved", out_base)


# ---------------------------------------------------------------- main (runs on the user's computer)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--set", choices=["internal", "cfb"], required=True)
    ap.add_argument("--per-sample", required=True)
    ap.add_argument("--dataset", default=None, help="internal: cohort to pick from (default: all)")
    ap.add_argument("--pick", choices=["best", "median", "worst"], default="median")
    ap.add_argument("--sample-id", default=None)
    ap.add_argument("--samples", default="samples_clean.json")
    ap.add_argument("--external", default="external_cfb.json")
    ap.add_argument("--cfb-tier", default="strict")
    ap.add_argument("--out-dir", default="qualitative_figs")
    args = ap.parse_args()

    import torch

    import evaluate as ev
    import evaluate_localization as el
    from glioma3d.data import load_samples
    from glioma3d.figures.qualitative import pick_cases, read_rows, to_ras

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
        loader, metric, label = ev.load_internal, "dice_WT", "Dice WT"
    else:
        samples = {s["sample_id"]: s for s in ev.select_cfb(args.external, args.cfb_tier, False, True)}
        loader, metric, label = ev.load_cfb, "dice_TC", "Dice TC"
    if args.dataset:
        rows = [r for r in rows if r["dataset"] == args.dataset]
    if args.sample_id:
        row = next(r for r in rows if r["sample_id"] == args.sample_id)
    else:
        row = list(pick_cases(rows, metric, [args.pick]).values())[0]
    s = samples[row["sample_id"]]
    print("case:", s["sample_id"], f"{metric}={float(row[metric]):.3f}")

    img, gt = loader(s)
    pred, _ = ev.predict_volume(model, img, device, ckpt["config"].get("num_classes", 4), ev.PATCH, 0.5, True)
    prob, padded = el.predict_localization(model, img, device, el.PATCH, 0.25, True)
    D, H, W = gt.shape
    loc = np.kron(prob, np.ones((4, 4, 4), dtype=np.float32))[:D, :H, :W]

    if args.set == "internal":
        import nibabel as nib
        aff = nib.load(s["image_paths"][0]).affine
        img, gt, pred, loc = to_ras(img, aff), to_ras(gt, aff), to_ras(pred, aff), to_ras(loc, aff)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = {"case": f"{row['dataset']} ({args.pick})", "dice_label": label, "dice_value": float(row[metric])}
    draw_pipeline(img, gt, pred, loc, args.set, meta, str(out / f"pipeline_{args.set}_{row['dataset']}_{args.pick}"))


if __name__ == "__main__":
    main()