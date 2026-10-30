# Data

No imaging data, patient identifiers beyond public dataset IDs, or model weights are distributed with this repository.
You must obtain each dataset from its original source and respect its licence / data-use agreement.

## Cohorts

| Cohort | Role | Sessions used | Patients | Mask source |
|---|---|---:|---:|---|
| RHUH-GBM | train/val/test | 120 | 40 | expert |
| UCSD-PTGBM | train/val/test | 184 | 136 | expert |
| UCSF-PDGM | train/val/test | 495 | 495 | expert |
| UPENN-GBM | train/val/test | 611 | 611 | 147 manual, 464 automated-approximate |
| UTSW-Glioma | train/val/test | 622 | 622 | 87 manual, 154 revised-manual, 381 automated (FeTS) |
| **Internal total** | | **2032** | **1904** | |
| CFB-GBM v4 | external test only | 97 (strict tier) / 412 (all tiers) | 94 (strict) | radiotherapy GTV (binary) |

Exclusions applied by `glioma3d.prep.build_samples`: 60 UPENN sessions without a mask, 3 UTSW sessions with an empty
mask. CFB groups missing a required sequence are dropped by `glioma3d.prep.build_external_cfb`
(554 groups found → 413 selected → 412 valid after one empty-GTV removal).

## Split

Patient-level 70 / 15 / 15 split with seed 42 (a patient never appears in two splits):

| Split | Sessions |
|---|---:|
| train | 1423 |
| val | 303 |
| test | 306 (285 patients) |

`splits/internal_samples.csv` lists every session with its split and `splits/external_cfb_samples.csv` lists the CFB
groups with their tier. These files contain no paths. After rebuilding your own `samples_clean.json` run
`python tools/verify_splits.py --samples samples_clean.json --external external_cfb.json` to confirm it matches.

## Pipeline

```
make_file_manifest   ->  all_datasets_manifest.csv   (flat listing of every NIfTI file; no image data read)
build_samples        ->  samples_clean.json          (per-cohort file rules, label schemes, mask validation, split)
preprocess_cache     ->  cache_v2/                   (float16 images, uint8 masks, foreground index)
build_external_cfb   ->  external_cfb.json           (CFB groups, tiers strict / t2star)
```

The manifest reader expects the directory layout documented in `glioma3d/prep/make_file_manifest.py`. If your
download layout differs, adapt the per-cohort rules in `glioma3d/prep/build_samples.py`.

## Evaluation protocols

* **Internal**: whole-volume sliding-window inference on the native grid; Dice for WT/TC/ET (and NCR/ED); HD95 in
  voxels (volumes are not resampled to a common spacing).
* **External (CFB)**: volumes resampled to 1 mm isotropic, skull-stripped with the supplied brain mask, z-scored.
  Primary metric (fixed a priori) is Dice of TC (classes 1∪3) vs the GTV; secondary WT vs GTV; HD95 in mm. A
  radiotherapy GTV is not the same structure as a BraTS tumour region, so absolute values are not comparable with
  internal numbers.
* Confidence intervals: patient-level bootstrap (2000 resamples, seed 0) in `glioma3d.evaluate`; paired comparisons in
  `glioma3d.compare_runs` (10 000 resamples, seed 0, Wilcoxon signed-rank, Holm correction within each group).
