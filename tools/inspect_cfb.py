"""Read-only inspection of CFB-GBM for use as the external evaluation set.

Prints which sequence combinations exist per (patient, timepoint) and, for a few
groups, the shape / voxel spacing of each file (to see whether images share a
space or need resampling). Output is also saved to ``--out``.

Usage:
    python tools/inspect_cfb.py --manifest all_datasets_manifest.csv --n 6
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="all_datasets_manifest.csv")
    ap.add_argument("--n", type=int, default=6, help="number of groups whose headers are read")
    ap.add_argument("--out", default="cfb_report.txt")
    args = ap.parse_args()

    groups = defaultdict(dict)
    with open(args.manifest, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if "cfb" not in r["Dataset"].lower() or "upenn" in r["Dataset"].lower():
                continue
            n = r["File_Name"].lower()
            if not n.endswith((".nii", ".nii.gz")):
                continue
            stem = re.sub(r"\.nii(\.gz)?$", "", n)
            m = re.fullmatch(r"(\d+)_(t\d+)_(.+)", stem)
            if m:
                groups[(m.group(1), m.group(2))][m.group(3)] = r["Full_Path"]

    out = [f"(patient, timepoint) groups in CFB: {len(groups)} | patients: {len({k[0] for k in groups})}"]
    out.append("timepoints: " + str(dict(Counter(k[1] for k in groups))))
    seq = Counter(s for g in groups.values() for s in g)
    out.append("number of groups containing each file: " + str(dict(sorted(seq.items()))))

    structural = {"t1tse", "t1eg", "t1gd", "t2tse", "t2star", "flair"}
    combos = Counter(tuple(sorted(s for s in g if s in structural)) for g in groups.values())
    out.append("\nstructural sequence combinations (number of groups):")
    for c, n in combos.most_common(12):
        out.append(f"  {n:4d} x {list(c)}")

    need_a = sum(1 for g in groups.values() if "gtv" in g and "flair" in g and "t1gd" in g and "t2tse" in g and ("t1tse" in g or "t1eg" in g))
    need_b = sum(1 for g in groups.values() if "gtv" in g and "flair" in g and "t1gd" in g and "t2tse" in g and "t1tse" in g)
    out.append(f"\ngroups with GTV + FLAIR + T1gd + T2tse + (T1tse or T1eg): {need_a}  | with T1tse only: {need_b}")

    try:
        import nibabel as nib
        import numpy as np

        out.append("\nshape / voxel spacing (mm) for a few sample groups:")
        shown = 0
        for key, g in sorted(groups.items()):
            if shown >= args.n or "gtv" not in g:
                continue
            shown += 1
            out.append(f"  --- patient {key[0]} timepoint {key[1]}")
            for name in sorted(g):
                if name in {"ct", "rtdose", "adc", "brain_mask", "gtv", "flair", "t1gd", "t1tse", "t1eg", "t2tse", "t2star"}:
                    img = nib.load(g[name])
                    zooms = tuple(round(float(z), 2) for z in img.header.get_zooms()[:3])
                    extra = ""
                    if name in {"gtv", "brain_mask"}:
                        arr = np.asanyarray(img.dataobj)
                        extra = f" | values: {np.unique(arr).tolist()[:6]} | fraction>0: {float((arr > 0).mean()):.3%}"
                    out.append(f"      {name:10s} shape={tuple(img.shape)} zooms={zooms}{extra}")
    except Exception as e:  # noqa: BLE001
        out.append(f"(could not read headers: {type(e).__name__}: {e})")

    text = "\n".join(out)
    print(text)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"\n>> saved: {args.out}")


if __name__ == "__main__":
    main()
