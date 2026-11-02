# Reproducing the reported results

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -e .            # add [baseline] for MONAI, [dev] for pytest
```

## 2. Data preparation (see `docs/data.md`)

```bash
python -m glioma3d.prep.make_file_manifest --data-root /path/to/Data --out all_datasets_manifest.csv
python -m glioma3d.prep.build_samples      --manifest all_datasets_manifest.csv --out samples_clean.json
python -m glioma3d.prep.build_external_cfb --manifest all_datasets_manifest.csv --out external_cfb.json
python -m glioma3d.prep.preprocess_cache   --samples samples_clean.json --cache-dir cache_v2 --workers 4
python tools/verify_splits.py --samples samples_clean.json --external external_cfb.json
```

## 3. Training (or download the released checkpoint)

```bash
python -m glioma3d.train --output-dir runs/final_c24       # ~40 epochs; best.pth was selected at epoch 33
```

If the reported checkpoint is published (GitHub Releases), place it at
`runs/final_c24/checkpoints/best.pth`. It is a standard `torch.save` dict containing `model_state_dict` and a `config`
entry (`in_channels`, `num_classes`, `base_channels`, `dropout`, `use_graph`, `loc_weight`, `seed`) that the
evaluation scripts read. The model has 17,312,695 parameters.

> Only load checkpoints from sources you trust (`torch.load` unpickles).

## 4. Evaluation

```bash
CK=runs/final_c24/checkpoints/best.pth
python -m glioma3d.evaluate --checkpoint $CK --set internal --split test --out-dir eval_results/internal_best
python -m glioma3d.evaluate --checkpoint $CK --set cfb --cfb-tier strict --out-dir eval_results/cfb_best
python -m glioma3d.evaluate --checkpoint $CK --set cfb --cfb-tier t2star --out-dir eval_results/cfb_t2star
python -m glioma3d.evaluate_localization --checkpoint $CK --set internal --split test
# baseline: MONAI SegResNet bundle (pip install monai)
python -m glioma3d.baseline_monai --bundle-dir /path/to/brats_mri_segmentation --set internal --out-dir eval_results/internal_monai_segresnet
python -m glioma3d.baseline_monai --bundle-dir /path/to/brats_mri_segmentation --set cfb --out-dir eval_results/cfb_monai_segresnet
# paired comparison
python -m glioma3d.compare_runs --a eval_results/internal_best/per_sample.csv \
    --b eval_results/internal_monai_segresnet/per_sample.csv \
    --label-a "Full model" --label-b "MONAI SegResNet" --by-dataset --out compare_internal_monai
python -m glioma3d.compare_runs --a eval_results/cfb_best/per_sample.csv \
    --b eval_results/cfb_monai_segresnet/per_sample.csv --metrics dice_TC hd95_TC \
    --label-a "Full model" --label-b "MONAI SegResNet" --out compare_cfb_monai
```

Ablations: `python -m glioma3d.train --no-graph --output-dir runs/no_graph` and
`python -m glioma3d.train --loc-weight 0 --output-dir runs/seg_only`. Interpretability and figures:
`glioma3d.explain`, `glioma3d.figures.qualitative`, `glioma3d.figures.pipeline`.

## 5. Reference values (from the original run)

Full model vs MONAI SegResNet, paired patient-level bootstrap (10 000 resamples, seed 0):

| Set | Metric | n patients | Full model | SegResNet | Difference [95 % CI] |
|---|---|---:|---:|---:|---|
| Internal test | Dice WT | 285 | 0.9099 | 0.8834 | +0.0265 [+0.0131, +0.0402] |
| Internal test | Dice TC | 266 | 0.8768 | 0.8581 | +0.0187 [+0.0015, +0.0355] |
| Internal test | Dice ET | 255 | 0.8895 | 0.8507 | +0.0388 [+0.0257, +0.0517] |
| CFB strict | Dice TC | 94 | 0.7577 | 0.7721 | −0.0144 [−0.0419, +0.0098] |
| CFB strict | HD95 TC (mm) | 93 | 7.167 | 11.419 | −4.252 [−7.738, −1.482] |

Your numbers should agree to within the nondeterminism of GPU inference. Retraining from scratch will not reproduce
the checkpoint exactly (`cudnn.benchmark`, data-loader order).

## 6. Checking `compare_runs`

`glioma3d.compare_runs` was re-implemented for this release. To confirm that it matches the original analysis, run it
on the original `per_sample.csv` files with `--check-against compare_internal_monai.json`. Two columns,
`a_better_patients` and `b_better_patients`, are intentionally direction-aware for HD95 (lower is better); older output
files counted the sign of the difference only, so those two columns will differ for `hd95_*` rows.
