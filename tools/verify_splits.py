"""Check a locally rebuilt ``samples_clean.json`` / ``external_cfb.json`` against the shipped splits.

The CSVs in ``splits/`` list, for every session, the patient-level split it belongs to.
After rebuilding the sample lists from your own copy of the data (see README), run:

    python tools/verify_splits.py --samples samples_clean.json --external external_cfb.json

Exit status is non-zero if any sample id, split, or CFB tier differs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SPLITS = Path(__file__).resolve().parent.parent / "splits"


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def diff(name, expected, got):
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    changed = sorted(k for k in set(expected) & set(got) if expected[k] != got[k])
    ok = not (missing or extra or changed)
    print(f"[{'OK' if ok else 'MISMATCH'}] {name}: expected {len(expected)}, found {len(got)}"
          f" | missing {len(missing)} | extra {len(extra)} | changed {len(changed)}")
    for label, items in (("missing", missing), ("extra", extra), ("changed", changed)):
        for k in items[:5]:
            print(f"    {label}: {k}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples_clean.json")
    ap.add_argument("--external", default=None, help="optional external_cfb.json")
    args = ap.parse_args()

    exp = {r["sample_id"]: (r["split"], r["patient_id"], r["mask_source"])
           for r in read(SPLITS / "internal_samples.csv")}
    got = {s["sample_id"]: (s["split"], s["patient_id"], s["mask_source"])
           for s in json.loads(Path(args.samples).read_text(encoding="utf-8"))}
    ok = diff("internal splits", exp, got)

    if args.external:
        exp = {r["sample_id"]: (r["tier"], r["t1_source"]) for r in read(SPLITS / "external_cfb_samples.csv")}
        got = {s["sample_id"]: (s["tier"], s["t1_source"])
               for s in json.loads(Path(args.external).read_text(encoding="utf-8"))}
        ok = diff("external CFB", exp, got) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
