# glioma3d — multi-task 3D U-Net for glioma segmentation and localisation

Code for a 3D multi-task network that jointly segments glioma sub-regions (necrotic/non-enhancing core, oedema,
enhancing tumour) and predicts a coarse tumour-localisation map from four co-registered MRI sequences
(T1, T1ce, T2, FLAIR). A residual 3D U-Net with local graph message-passing modules (17.3 M parameters,
`base_channels = 24`) is trained on 2032 sessions from five public cohorts and evaluated on a held-out internal test split
and on an external radiotherapy cohort (CFB-GBM).

> **Research code, not a medical device.** It has not been validated for clinical use.

## Contents

```
src/glioma3d/
  model.py                 MultiTaskUNet3D
  data.py                  cache-based dataset, patch sampling, dataloaders
  train.py                 training (Dice+CE segmentation loss, weighted BCE localisation loss)
  evaluate.py              full-volume sliding-window evaluation (internal and CFB), bootstrap CIs
  evaluate_localization.py localisation metrics
  baseline_monai.py        MONAI SegResNet bundle baseline under the same protocol
  compare_runs.py          paired patient-level bootstrap + Wilcoxon + Holm comparison of two runs
  explain.py               gamma / ablation / Grad-CAM analyses
  figures/                 qualitative and pipeline figures
  prep/                    file manifest -> samples_clean.json -> cache; external CFB list
tools/                     dataset/cache inspection, split verification, GPU-memory profiling
splits/                    patient-level split and CFB tiers (session IDs only, no paths)
docs/                      data, loss/training and reproducibility notes
tests/                     unit tests (no data required)
```

## Installation

```bash
git clone https://github.com/MobinZamanii/CancerAI-Glioma-Segmentation.git && cd CancerAI-Glioma-Segmentation
python -m venv .venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate
# 1) install PyTorch for your CUDA version: https://pytorch.org/get-started/locally/
# 2) then:
pip install -e .            # extras: .[baseline] (MONAI), .[dev] (pytest)
```

Python ≥ 3.10. The reported model was trained on a 4 GB GPU (GTX 1650) with mixed precision.
For the MONAI baseline use plain `pip install monai` (not `monai[nibabel]`).

## Data

The data are **not** included and must be downloaded from the original providers under their own terms:
RHUH-GBM, UCSD-PTGBM, UCSF-PDGM, UPENN-GBM, UTSW-Glioma (internal cohorts) and CFB-GBM (external; TCIA collection version 4, i.e. the v2.0 release described in the CFB-GBM v2.0 data paper). Details,
counts and the split are in [`docs/data.md`](docs/data.md).

```bash
python -m glioma3d.prep.make_file_manifest --data-root /path/to/Data --out all_datasets_manifest.csv
python -m glioma3d.prep.build_samples      --manifest all_datasets_manifest.csv --out samples_clean.json
python -m glioma3d.prep.build_external_cfb --manifest all_datasets_manifest.csv --out external_cfb.json
python -m glioma3d.prep.preprocess_cache   --cache-dir cache_v2 --workers 4
python tools/verify_splits.py --samples samples_clean.json --external external_cfb.json
```

## Train

```bash
python -m glioma3d.train --output-dir runs/final_c24
```

Defaults are the reported settings. Ablations: `--no-graph`, `--loc-weight 0`, `--base-channels N`, `--seed S`.

## Evaluate

```bash
CK=runs/final_c24/checkpoints/best.pth
python -m glioma3d.evaluate --checkpoint $CK --set internal --split test
python -m glioma3d.evaluate --checkpoint $CK --set cfb --cfb-tier strict
```

Per-sample metrics are written to `per_sample.csv`; summaries include patient-level bootstrap 95 % CIs. The full
command list (localisation, baseline, paired comparison, ablations, figures) and the reference numbers are in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Pretrained weights

The checkpoint of the reported model is not stored in this repository. If a checkpoint is published, it is listed on the
repository's *Releases* page. Load only checkpoints you trust: `torch.load` unpickles files.

## Tests

```bash
pip install -e .[dev] && pytest
```

## Data and licence notes

Code: MIT (see `LICENSE`). Datasets keep their own licences and data-use agreements; redistributing them or models
derived from them may carry restrictions. Check each provider's terms before releasing weights.

## Citation

Add the paper citation here once published.
