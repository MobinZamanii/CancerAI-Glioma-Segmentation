"""
Build the list of clean samples (samples_clean.json) from the file manifest.

Every cohort has its own explicit rules:
  - the 4 input channels are selected in a fixed order: [T1, T1ce, T2, FLAIR];
  - the mask is selected explicitly (not "the last file whose name contains seg");
  - the train/val/test split is made per PATIENT (not per sample) and stored in the output file
    (reproducible, seed 42);
  - defective samples (shape mismatch, empty mask, labels outside the cohort's scheme) are
    dropped and the reason is reported.

Usage:
    python -m glioma3d.prep.build_samples --manifest all_datasets_manifest.csv
        (a few minutes: the masks are read to validate the labels)
    python -m glioma3d.prep.build_samples --skip-validate
        (file selection only, fast; the split then differs slightly because defective samples are kept)
Output: samples_clean.json and samples_report.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MODALITIES = ("t1", "t1ce", "t2", "flair")
# Label convention of each cohort -> allowed raw labels + map to {0: background, 1: necrosis/NCR, 2: oedema, 3: enhancing/ET}
SCHEMES = {
    # BraTS 2017-2021: 1=NCR, 2=ED, 4=ET
    "brats21": {"allowed": {0, 1, 2, 4}, "map": {"4": 3}},
    # RHUH-GBM (dataset paper): 1=necrosis, 2=peritumoral signal alteration, 3=enhancing tumor
    "rhuh": {"allowed": {0, 1, 2, 3}, "map": {}},
    # BraTS 2024 post-treatment: 1=NETC, 2=SNFH, 3=ET, 4=resection cavity  (resection cavity -> background)
    "brats24_post": {"allowed": {0, 1, 2, 3, 4}, "map": {"4": 0}},
}


# ----------------------------------------------------------------- helpers
def stem_of(name: str) -> str:
    s = name.lower()
    for ext in (".nii.gz", ".nii"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


def dataset_key(name: str) -> str:
    n = name.lower()
    for k in ("upenn", "rhuh", "ucsd", "ucsf", "utsw", "cfb"):
        if k in n:
            return k.upper()
    return "OTHER"


def dir_parts(ds: str) -> List[str]:
    return [p.strip() for p in ds.replace("\\", "/").split("/") if p.strip()]


Row = Tuple[str, str, str]  # (directory_structure, stem, full_path)
Result = Tuple[List[Dict], List[Tuple[str, str]]]


def find_one(files: Dict[str, str], pattern: str) -> Optional[str]:
    """Path of the file whose stem fully matches the regex; if there are several, the first in alphabetical order."""
    hits = sorted(s for s in files if re.fullmatch(pattern, s))
    return files[hits[0]] if hits else None


def group_by_dir(rows: List[Row]) -> Dict[str, Dict[str, str]]:
    g: Dict[str, Dict[str, str]] = defaultdict(dict)
    for d, stem, path in rows:
        g[d].setdefault(stem, path)
    return g


def make_sample(ds, patient, tp, imgs: Dict[str, Optional[str]], candidates, scheme: str):
    """candidates: list of (mask path, source) in order of priority; the first healthy one is chosen during validation."""
    candidates = [(p, src) for p, src in candidates if p]
    missing = [m for m in MODALITIES if not imgs.get(m)]
    sid = f"{ds}__{patient}__{tp}"
    if missing:
        return None, (sid, f"missing_modality:{','.join(missing)}")
    if not candidates:
        return None, (sid, "no_mask")
    return {
        "sample_id": sid,
        "dataset": ds,
        "patient_id": f"{ds}__{patient}",
        "timepoint": tp,
        "image_paths": [imgs[m] for m in MODALITIES],
        "mask_candidates": [{"path": p, "source": src} for p, src in candidates],
        "mask_path": candidates[0][0],
        "mask_source": candidates[0][1],
        "scheme": scheme,
        "label_map": SCHEMES[scheme]["map"],
    }, None


# ----------------------------------------------------------------- per-dataset rules
def build_rhuh(rows: List[Row], **_) -> Result:
    out, rej = [], []
    for d, files in sorted(group_by_dir(rows).items()):
        any_stem = next(iter(files))
        patient, tp = any_stem.split("_")[0], any_stem.split("_")[1]
        imgs = {m: find_one(files, rf".*_{m}") for m in MODALITIES}
        mask = find_one(files, r"(.*_)?segmentations?")
        s, r = make_sample("RHUH", patient, tp, imgs, [(mask, "expert")], "rhuh")
        out.append(s) if s else rej.append(r)
    return out, rej


def build_ucsd(rows: List[Row], **_) -> Result:
    out, rej = [], []
    for d, files in sorted(group_by_dir(rows).items()):
        any_stem = next(iter(files))
        patient, tp = any_stem.split("_")[0], any_stem.split("_")[1]
        imgs = {
            "t1": find_one(files, r".*_t1pre"),
            "t1ce": find_one(files, r".*_t1post"),
            "t2": find_one(files, r".*_t2"),
            "flair": find_one(files, r".*_flair"),
        }
        mask = find_one(files, r".*_brats_tumor_seg")
        s, r = make_sample("UCSD", patient, tp, imgs, [(mask, "expert")], "brats24_post")
        out.append(s) if s else rej.append(r)
    return out, rej


def build_ucsf(rows: List[Row], ucsf_bias: bool = False, **_) -> Result:
    out, rej = [], []
    suf = "_bias" if ucsf_bias else ""
    for d, files in sorted(group_by_dir(rows).items()):
        base = {s: p for s, p in files.items() if not re.search(r"_fu\d+d_", s)}  # follow-up scans are dropped
        if not base:
            continue
        patient = next(iter(base)).split("_")[0]
        imgs = {
            "t1": find_one(base, rf".*_t1{suf}"),
            "t1ce": find_one(base, rf".*_t1c{suf}"),
            "t2": find_one(base, rf".*_t2{suf}"),
            "flair": find_one(base, rf".*_flair{suf}"),
        }
        mask = find_one(base, r".*_tumor_segmentation")
        s, r = make_sample("UCSF", patient, "0", imgs, [(mask, "expert")], "brats21")
        out.append(s) if s else rej.append(r)
    return out, rej


def build_utsw(rows: List[Row], **_) -> Result:
    out, rej = [], []
    for d, files in sorted(group_by_dir(rows).items()):
        patient = dir_parts(d)[-1].lower()
        imgs = {
            "t1": find_one(files, r"brain_t1"),
            "t1ce": find_one(files, r"brain_t1ce"),
            "t2": find_one(files, r"brain_t2"),
            "flair": find_one(files, r"brain_flair"),
        }
        cands = [
            (find_one(files, r"tumorseg_manual_correction"), "manual"),
            (find_one(files, r"rtumorseg_manual_correction"), "manual_r"),
            (find_one(files, r"tumorseg_fets"), "auto_fets"),
        ]
        s, r = make_sample("UTSW", patient, "0", imgs, cands, "brats21")
        out.append(s) if s else rej.append(r)
    return out, rej


def build_upenn(rows: List[Row], **_) -> Result:
    # The masks live in a separate folder, so images and masks are paired through the "UPENN-GBM-xxxxx_yy" id in the file name.
    per: Dict[str, Dict[str, str]] = defaultdict(dict)
    for _d, stem, path in rows:
        m = re.fullmatch(r"(upenn-gbm-\d+_\d+)_(.+)", stem)
        if m:
            per[m.group(1)].setdefault(m.group(2), path)
    out, rej = [], []
    for sid in sorted(per):
        f = per[sid]
        imgs = {"t1": f.get("t1"), "t1ce": f.get("t1gd"), "t2": f.get("t2"), "flair": f.get("flair")}
        if not any(imgs.values()):
            continue  # DSC/DTI groups etc. without structural images
        cands = [(f.get("segm"), "manual"), (f.get("automated_approx_segm"), "auto_approx")]
        patient, tp = sid.rsplit("_", 1)
        s, r = make_sample("UPENN", patient, tp, imgs, cands, "brats21")
        out.append(s) if s else rej.append(r)
    return out, rej


BUILDERS = {"RHUH": build_rhuh, "UCSD": build_ucsd, "UCSF": build_ucsf, "UTSW": build_utsw, "UPENN": build_upenn}
SKIPPED_REASONS = {
    "CFB": "excluded from development: binary radiotherapy GTV mask, no standard T1/T2 (variable t1eg/t2star/t2tse) and it contains CT/RTDOSE; used only as the external test set (build_external_cfb.py)",
}


# ----------------------------------------------------------------- validation
def validate(s: Dict):
    """Select the first healthy mask among the candidates (matching shape/affine, non-empty, allowed labels)."""
    import nibabel as nib

    allowed = SCHEMES[s["scheme"]]["allowed"]
    try:
        imgs = [nib.load(p) for p in s["image_paths"]]
    except Exception as e:  # noqa: BLE001
        return s, f"read_error:{type(e).__name__}", None, False
    shapes = {tuple(i.shape) for i in imgs}
    if len(shapes) != 1 or len(next(iter(shapes))) != 3:
        return s, "image_shapes_differ_or_not_3d", None, False

    first_err, first_vals = None, None
    for cand in s["mask_candidates"]:
        try:
            m = nib.load(cand["path"])
            if tuple(m.shape) != next(iter(shapes)):
                err, vals = "shape_mismatch", None
            else:
                affine_ok = all(np.allclose(i.affine, m.affine, atol=1e-2) for i in imgs)
                vals = np.unique(np.rint(np.asanyarray(m.dataobj)).astype(np.int16)).tolist()
                if max(vals) == 0:
                    err = "empty_mask"
                elif any(v not in allowed for v in vals):
                    err = "labels_outside_scheme"
                elif not affine_ok:
                    err = "affine_mismatch"
                else:
                    s["mask_path"], s["mask_source"] = cand["path"], cand["source"]
                    return s, None, vals, True
        except Exception as e:  # noqa: BLE001
            err, vals = f"read_error:{type(e).__name__}", None
        if first_err is None:
            first_err, first_vals = err, vals
    return s, first_err, first_vals, False


# ----------------------------------------------------------------- split
def split_by_patient(samples: List[Dict], val_frac: float, test_frac: float, seed: int) -> None:
    """Assign train/val/test per patient inside every cohort (cohorts with < 6 patients stay in train)."""
    rng = random.Random(seed)
    by_ds: Dict[str, List[str]] = defaultdict(list)
    seen = set()
    for s in samples:
        if s["patient_id"] not in seen:
            seen.add(s["patient_id"])
            by_ds[s["dataset"]].append(s["patient_id"])
    assign: Dict[str, str] = {}
    for ds, pats in by_ds.items():
        pats = sorted(pats)
        rng.shuffle(pats)
        n = len(pats)
        n_test = max(1, round(n * test_frac)) if n >= 6 else 0
        n_val = max(1, round(n * val_frac)) if n >= 6 else 0
        for i, p in enumerate(pats):
            assign[p] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    for s in samples:
        s["split"] = assign[s["patient_id"]]


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="all_datasets_manifest.csv",
                    help="file manifest written by glioma3d.prep.make_file_manifest")
    ap.add_argument("--out", default="samples_clean.json")
    ap.add_argument("--report", default="samples_report.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--ucsf-bias", action="store_true", help="use the *_bias (bias-corrected) UCSF images")
    ap.add_argument("--exclude", default="", help="comma-separated cohort names to leave out, e.g. UCSD,UTSW")
    args = ap.parse_args()

    mpath = Path(args.manifest)
    if not mpath.exists():
        raise FileNotFoundError(f"{mpath} not found. Run `python -m glioma3d.prep.make_file_manifest` first.")
    excluded = {x.strip().upper() for x in args.exclude.split(",") if x.strip()}

    rows_by_ds: Dict[str, List[Row]] = defaultdict(list)
    with open(mpath, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["File_Name"].lower().endswith((".nii", ".nii.gz")):
                rows_by_ds[dataset_key(r["Dataset"])].append(
                    (r["Directory_Structure"], stem_of(r["File_Name"]), r["Full_Path"])
                )

    report: List[str] = []
    samples: List[Dict] = []
    for key, rows in sorted(rows_by_ds.items()):
        if key in SKIPPED_REASONS:
            report.append(f"[{key}] {SKIPPED_REASONS[key]}")
            continue
        if key in excluded or key not in BUILDERS:
            report.append(f"[{key}] left out ({'--exclude' if key in excluded else 'no rule defined'})")
            continue
        got, rej = BUILDERS[key](rows, ucsf_bias=args.ucsf_bias)
        reasons = Counter(r for _, r in rej)
        report.append(f"[{key}] initial selection: {len(got)} | rejected during file selection: {dict(reasons) or 0}")
        samples.extend(got)

    if not args.skip_validate and samples:
        print(f"Validating {len(samples)} samples (reading the masks)...")
        kept, label_sets, affine_bad, dropped = [], defaultdict(Counter), Counter(), defaultdict(Counter)
        with ThreadPoolExecutor(args.workers) as ex:
            for i, (s, err, vals, aff_ok) in enumerate(ex.map(validate, samples), 1):
                if i % 200 == 0:
                    print(f"  {i}/{len(samples)}", flush=True)
                if vals is not None:
                    label_sets[s["dataset"]][tuple(vals)] += 1
                if not err and s["mask_source"] != s["mask_candidates"][0]["source"]:
                    affine_bad[s["dataset"]] += 1  # i.e. the first candidate was rejected and a replacement was used
                if err:
                    dropped[s["dataset"]][err.split(":")[0]] += 1
                else:
                    kept.append(s)
        report.append("")
        report.append("--- validation result ---")
        for ds in sorted({s["dataset"] for s in samples}):
            report.append(f"[{ds}] dropped: {dict(dropped[ds]) or 0} | mask replaced (first candidate rejected): {affine_bad[ds]}")
            for ls, n in label_sets[ds].most_common(6):
                report.append(f"      label set {list(ls)}: {n} samples")
        samples = kept

    split_by_patient(samples, args.val_frac, args.test_frac, args.seed)

    report.append("")
    report.append("--- final counts (samples | patients) ---")
    tbl: Dict[str, Counter] = defaultdict(Counter)
    pat: Dict[Tuple[str, str], set] = defaultdict(set)
    for s in samples:
        tbl[s["dataset"]][s["split"]] += 1
        pat[(s["dataset"], s["split"])].add(s["patient_id"])
    for ds in sorted(tbl):
        parts = [f"{sp}: {tbl[ds][sp]} samples/{len(pat[(ds, sp)])} patients" for sp in ("train", "val", "test")]
        report.append(f"[{ds}] " + " | ".join(parts))
    tot = Counter(s["split"] for s in samples)
    report.append(f"[total] train {tot['train']} | val {tot['val']} | test {tot['test']}")
    src = Counter((s["dataset"], s["mask_source"]) for s in samples)
    report.append("mask source: " + ", ".join(f"{d}/{m}={n}" for (d, m), n in sorted(src.items())))

    Path(args.out).write_text(json.dumps(samples, indent=1, ensure_ascii=False), encoding="utf-8")
    text = "\n".join(report)
    Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    print(f"\n>> saved: {args.out} and {args.report}")


if __name__ == "__main__":
    main()