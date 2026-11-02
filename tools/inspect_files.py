"""Read-only: print the real file-name patterns of every dataset from the file manifest.

It never touches the NIfTI files. The report is also saved to ``--out``.

Usage:
    python tools/inspect_files.py --manifest all_datasets_manifest.csv --out inspect_report.txt
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


def norm(name: str) -> str:
    stem = name.lower()
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return re.sub(r"\d{2,}", "#", stem)  # multi-digit identifiers -> #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="all_datasets_manifest.csv")
    ap.add_argument("--examples", type=int, default=2)
    ap.add_argument("--out", default="inspect_report.txt")
    args = ap.parse_args()

    groups = defaultdict(lambda: {"files": [], "dataset": None, "dir": None})
    with open(args.manifest, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = f"{row['Dataset']}_{row['Directory_Structure']}"
            g = groups[key]
            g["dataset"], g["dir"] = row["Dataset"], row["Directory_Structure"]
            g["files"].append((row["File_Name"], row["Full_Path"]))

    out: list[str] = []
    by_ds = defaultdict(list)
    for g in groups.values():
        by_ds[g["dataset"]].append(g)

    for ds, items in sorted(by_ds.items()):
        out.append("=" * 72)
        out.append(f"Dataset: {ds} | groups (samples): {len(items)}")

        names = Counter()
        masks_per_group = Counter()
        for g in items:
            masks = [n for n, _ in g["files"]
                     if any(k in n.lower() for k in ("gtv", "mask", "seg", "label"))]
            masks_per_group[len(masks)] += 1
            for n, _ in g["files"]:
                names[norm(n)] += 1

        out.append(f"  mask-like files per sample: {dict(masks_per_group)}")
        out.append("  file-name patterns (number of samples that have it):")
        for n, c in sorted(names.items(), key=lambda x: (-x[1], x[0])):
            out.append(f"    {c:5d}  {n}")

        for g in items[: args.examples]:
            out.append(f"  --- example | Directory_Structure = {g['dir']}")
            for n, _ in sorted(g["files"]):
                out.append(f"      {n}")

    text = "\n".join(out)
    print(text)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"\n>> saved: {args.out}")


if __name__ == "__main__":
    main()
