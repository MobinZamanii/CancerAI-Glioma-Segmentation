"""
Build the external evaluation set from CFB-GBM  ->  external_cfb.json

CFB is used for TESTING only and never enters training or model selection. Its mask is the binary
radiotherapy GTV. Every (patient, timepoint) group must provide:  FLAIR, T1gd, GTV, brain_mask  and
  T1 = t1eg (preferred) or t1tse        |   T2 = t2tse (tier=strict) or t2star (tier=t2star, sensitivity analysis)
The channel order is the same as in training: [T1, T1ce, T2, FLAIR].

Usage:  python -m glioma3d.prep.build_external_cfb --manifest all_datasets_manifest.csv [--check-t1 8]
  --check-t1 N : for N groups, print the mean intensity inside the GTV relative to the brain for t1eg and t1gd
                 (to confirm that t1eg really is the pre-contrast image; t1gd must be much brighter inside the GTV).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def load_groups(manifest: Path):
    groups = defaultdict(dict)
    with open(manifest, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if "cfb" not in r["Dataset"].lower() or "upenn" in r["Dataset"].lower():
                continue
            n = r["File_Name"].lower()
            if not n.endswith((".nii", ".nii.gz")):
                continue
            m = re.fullmatch(r"(\d+)_(t\d+)_(.+?)\.nii(\.gz)?", n)
            if m:
                groups[(m.group(1), m.group(2))][m.group(3)] = r["Full_Path"]
    return groups


def select(groups):
    out, rej = [], Counter()
    for (pid, tp), g in sorted(groups.items()):
        need = [k for k in ("flair", "t1gd", "gtv", "brain_mask") if k not in g]
        if need:
            rej["missing:" + ",".join(need)] += 1
            continue
        t1 = g.get("t1eg") or g.get("t1tse")
        if not t1:
            rej["missing:t1(eg/tse)"] += 1
            continue
        if "t2tse" in g:
            t2, tier = g["t2tse"], "strict"
        elif "t2star" in g:
            t2, tier = g["t2star"], "t2star"
        else:
            rej["missing:t2(tse/star)"] += 1
            continue
        out.append({
            "sample_id": f"CFB__{pid}__{tp}",
            "dataset": "CFB",
            "patient_id": f"CFB__{pid}",
            "timepoint": tp,
            "tier": tier,
            "t1_source": "t1eg" if g.get("t1eg") else "t1tse",
            "image_paths": [t1, g["t1gd"], t2, g["flair"]],   # T1, T1ce, T2, FLAIR
            "mask_path": g["gtv"],
            "brain_mask_path": g["brain_mask"],
        })
    return out, rej


def validate(s):
    import nibabel as nib
    try:
        imgs = [nib.load(p) for p in s["image_paths"] + [s["mask_path"], s["brain_mask_path"]]]
        if len({tuple(i.shape) for i in imgs}) != 1:
            return s, "shape_mismatch"
        if not all(np.allclose(i.affine, imgs[0].affine, atol=1e-2) for i in imgs):
            return s, "affine_mismatch"
        gtv = np.asanyarray(imgs[-2].dataobj)
        if not (gtv > 0).any():
            return s, "empty_gtv"
        return s, None
    except Exception as e:  # noqa: BLE001
        return s, f"read_error:{type(e).__name__}"


def check_t1(samples, n):
    import nibabel as nib
    print("\nt1eg vs t1gd: ratio of the mean intensity INSIDE the GTV to the mean brain intensity (the post-contrast t1gd should give a clearly larger ratio)")
    shown = 0
    for s in samples:
        if shown >= n or s["t1_source"] != "t1eg":
            continue
        shown += 1
        gtv = np.asanyarray(nib.load(s["mask_path"]).dataobj) > 0
        brain = np.asanyarray(nib.load(s["brain_mask_path"]).dataobj) > 0
        res = []
        for name, p in (("t1eg", s["image_paths"][0]), ("t1gd", s["image_paths"][1])):
            a = np.asanyarray(nib.load(p).dataobj).astype(np.float32)
            res.append(f"{name}: {a[gtv].mean() / max(a[brain].mean(), 1e-6):.2f}")
        print(f"  {s['sample_id']}: " + " | ".join(res))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="all_datasets_manifest.csv",
                    help="file manifest written by glioma3d.prep.make_file_manifest")
    ap.add_argument("--out", default="external_cfb.json")
    ap.add_argument("--report", default="external_cfb_report.txt")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--check-t1", type=int, default=0)
    args = ap.parse_args()

    mpath = Path(args.manifest)
    if not mpath.exists():
        raise FileNotFoundError(f"{mpath} not found. Run `python -m glioma3d.prep.make_file_manifest` first.")

    groups = load_groups(mpath)
    samples, rej = select(groups)
    rep = [f"CFB groups: {len(groups)} | initial selection: {len(samples)} | rejected during selection: {dict(rej)}"]

    kept, bad = [], Counter()
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (s, err) in enumerate(ex.map(validate, samples), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(samples)}", flush=True)
            if err:
                bad[err] += 1
            else:
                kept.append(s)
    rep.append(f"validation: healthy {len(kept)} | dropped: {dict(bad) or 0}")
    for tier in ("strict", "t2star"):
        t = [s for s in kept if s["tier"] == tier]
        rep.append(f"  tier={tier}: {len(t)} groups / {len({s['patient_id'] for s in t})} patients | timepoints: {dict(Counter(s['timepoint'] for s in t))}")
    rep.append(f"  T1 source: {dict(Counter(s['t1_source'] for s in kept))}")

    Path(args.out).write_text(json.dumps(kept, indent=1, ensure_ascii=False), encoding="utf-8")
    text = "\n".join(rep)
    print(text)
    if args.check_t1:
        check_t1(kept, args.check_t1)
    Path(args.report).write_text(text, encoding="utf-8")
    print(f"\n>> saved: {args.out}")


if __name__ == "__main__":
    main()
