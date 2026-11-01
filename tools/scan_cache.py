"""Cache check: NaN/Inf or implausible values in cached images. Read-only.

Usage:
    python tools/scan_cache.py --samples samples_clean.json --cache-dir cache_v2
"""
import argparse
from pathlib import Path

import numpy as np

import glioma3d.data as D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=D.SAMPLES_JSON)
    ap.add_argument("--cache-dir", default=str(D.CACHE_DIR))
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    samples = D.load_samples(args.samples)
    bad, missing = [], 0
    for i, s in enumerate(samples):
        ip = D.cache_paths(s["sample_id"], cache_dir)[0]
        if not Path(ip).exists():
            missing += 1
            continue
        a = np.asarray(np.load(ip, mmap_mode="r"), dtype=np.float32)
        fin = bool(np.isfinite(a).all())
        mx = float(np.abs(np.nan_to_num(a, nan=0, posinf=0, neginf=0)).max())
        if (not fin) or mx > 50:
            bad.append((s["sample_id"], fin, round(mx, 1)))
            print("BAD", bad[-1], flush=True)
        if i % 200 == 0:
            print(i, "/", len(samples), flush=True)
    print("missing from cache:", missing, "| suspicious:", len(bad))


if __name__ == "__main__":
    main()
