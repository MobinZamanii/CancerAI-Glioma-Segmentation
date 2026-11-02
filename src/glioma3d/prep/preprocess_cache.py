"""
Build the pre-processed training cache (cache_v2) from samples_clean.json.

For every sample three files are written:
    <key>_img.npy  (float16, [4, D, H, W], z-score over the non-zero voxels)
    <key>_msk.npy  (uint8, labels 0..3)
    <key>_fg.npz   (tumour voxel coordinates for fast patch sampling)
float16/uint8 reduce the disk footprint and the read time.

Usage:  python -m glioma3d.prep.preprocess_cache [--workers 4] [--splits train val test] [--cache-dir cache_v2]
(Resumable: files that already exist are not rebuilt.)
"""
from __future__ import annotations

import argparse
import functools
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from glioma3d.data import CACHE_DIR, cache_paths, compute_fg_info, fg_path, load_nifti_sample, load_samples


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def process(s: dict, cache_dir: Path = CACHE_DIR):
    img_p, msk_p = cache_paths(s["sample_id"], cache_dir)
    fg_p = fg_path(s["sample_id"], cache_dir)
    if img_p.exists() and msk_p.exists() and fg_p.exists():
        return s["sample_id"], "skip", None
    try:
        img, msk = load_nifti_sample(s)
        if msk.min() < 0 or msk.max() > 3:
            return s["sample_id"], "error", f"labels outside 0..3 after mapping: {np.unique(msk).tolist()}"
        coords, center = compute_fg_info(msk)
        _atomic_save(img_p, img.astype(np.float16))
        _atomic_save(msk_p, msk.astype(np.uint8))
        tmp = fg_p.with_name(f"{fg_p.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, coords=coords, center=center)
        os.replace(tmp, fg_p)
        return s["sample_id"], "ok", None
    except Exception as e:  # noqa: BLE001
        return s["sample_id"], "error", f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples_clean.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--cache-dir", default=None, help="default: ./cache_v2 or $GLIOMA3D_CACHE_DIR")
    ap.add_argument("--errors-file", default="cache_errors.txt")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples = [s for sp in args.splits for s in load_samples(args.samples, sp)]
    print(f"{len(samples)} samples -> {cache_dir}")
    stats = {"ok": 0, "skip": 0, "error": 0}
    errors = []
    with Pool(args.workers) as pool:
        worker = functools.partial(process, cache_dir=cache_dir)
        for i, (sid, status, msg) in enumerate(pool.imap_unordered(worker, samples), 1):
            stats[status] += 1
            if status == "error":
                errors.append((sid, msg))
                print(f"\n  error: {sid}: {msg}")
            if i % 25 == 0 or i == len(samples):
                print(f"\r  {i}/{len(samples)} | ok {stats['ok']} skip {stats['skip']} error {stats['error']}", end="", flush=True)
    print()
    if errors:
        Path(args.errors_file).write_text("\n".join(f"{a}\t{b}" for a, b in errors), encoding="utf-8")
        print(f"Samples with errors are listed in {args.errors_file} (they are not removed from training; they are read from NIfTI).")


if __name__ == "__main__":
    main()