"""
Step 0 of the data pipeline: list every NIfTI file under the data root in a flat CSV.

    python -m glioma3d.prep.make_file_manifest --data-root /path/to/Data --out all_datasets_manifest.csv

The data root holds one folder per cohort (the first folder level is stored as `Dataset`), e.g.

    Data/
      PKG - CFB-GBM version 4/CFB-GBM/001/t0/1_t0_flair.nii.gz ...
      PKG - RHUH-GBM-nii-v1/RHUH-GBM_nii_v1/RHUH-0001/0/RHUH-0001_0_flair.nii.gz ...
      PKG - UCSD-PTGBM-v1/UCSD-PTGBM/UCSD-PTGBM-0002_01/UCSD-PTGBM-0002_01_FLAIR.nii.gz ...
      PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5/UCSF-PDGM-0004_nifti/UCSF-PDGM-0004_FLAIR.nii.gz ...
      PKG - UTSW-Glioma/UTSW-Glioma/BT0001/brain_flair.nii.gz ...
      UPENN-GBM/UPENN-GBM-NIfTI/images_structural/.../UPENN-GBM-00001_11_FLAIR.nii.gz ...

Columns (identical to the manifest used for the reported results):
    Dataset, File_Name, Directory_Structure, File_Size_MB, Relative_Path, Full_Path

`glioma3d.prep.build_samples` and `glioma3d.prep.build_external_cfb` only read Dataset,
Directory_Structure, File_Name and Full_Path; the other columns are informational.

Only the file listing is read: no image data is opened.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

FIELDS = ["Dataset", "File_Name", "Directory_Structure", "File_Size_MB", "Relative_Path", "Full_Path"]
NIFTI_SUFFIXES = (".nii.gz", ".nii")


def iter_rows(data_root: Path):
    # Case-insensitive ordering with upper-casing (the NTFS collation), so the listing is identical on every platform.
    for dataset_dir in sorted((p for p in data_root.iterdir() if p.is_dir()), key=lambda p: p.name.upper()):
        for current, dirs, files in os.walk(dataset_dir):
            dirs.sort(key=str.upper)
            for name in sorted(files, key=str.upper):
                if not name.lower().endswith(NIFTI_SUFFIXES):
                    continue
                full = Path(current) / name
                rel_parts = full.relative_to(dataset_dir).parts
                yield {
                    "Dataset": dataset_dir.name,
                    "File_Name": name,
                    "Directory_Structure": " / ".join(rel_parts[:-1]),
                    "File_Size_MB": round(full.stat().st_size / (1024 * 1024), 2),
                    "Relative_Path": "\\".join(rel_parts),
                    "Full_Path": str(full),
                }


def main() -> None:
    ap = argparse.ArgumentParser(description="List all NIfTI files under the data root in a CSV manifest.")
    ap.add_argument("--data-root", required=True, help="folder that contains one sub-folder per cohort")
    ap.add_argument("--out", default="all_datasets_manifest.csv")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Data root not found: {root}")

    n = 0
    per_dataset: dict = {}
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in iter_rows(root):
            writer.writerow(row)
            n += 1
            per_dataset[row["Dataset"]] = per_dataset.get(row["Dataset"], 0) + 1
    for name, count in per_dataset.items():
        print(f"  {name}: {count} files")
    print(f"{n} NIfTI files -> {args.out}")


if __name__ == "__main__":
    main()
