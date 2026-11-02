# Loss functions and training procedure

This note documents what `glioma3d.train` actually implements (it replaces the earlier design note, whose
generic loss module is not part of this repository).

## Labels

* Input `x ∈ R^{B×4×D×H×W}`, channel order **[T1, T1ce, T2, FLAIR]**, z-scored over non-zero voxels.
* Target `y ∈ {0,1,2,3}^{B×D×H×W}`: 0 background, 1 necrotic/non-enhancing core, 2 oedema, 3 enhancing tumour
  (BraTS label 4 is mapped to 3; see `glioma3d.prep.build_samples` for the per-cohort label schemes).
* The model returns `segmentation` logits `[B,4,D,H,W]` and `localization` logits at a coarser resolution.

## Segmentation loss

`L_seg = L_CE + L_Dice`

* `L_CE`: voxel-wise multiclass cross-entropy (unweighted).
* `L_Dice = 1 − mean_{c∈{1,2,3}} (2 Σ p_c g_c + s) / (Σ p_c + Σ g_c + s)`, with softmax probabilities `p`, one-hot
  target `g`, smoothing `s = 1e-5`, sums over the whole batch. Background is excluded from the mean.

## Localisation loss

The coarse target is `t = AdaptiveMaxPool3d(1[y > 0])` resized to the localisation output shape. The max-pool keeps
small foci that plain down-sampling would drop.

`L_loc = BCEWithLogits(a, t; pos_weight = 5)`

## Total objective

`L = 1.0 · L_seg + 0.3 · L_loc` (`--loc-weight 0` gives the segmentation-only ablation).

## Optimisation

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 3e-4, weight decay 1e-5 |
| Schedule | cosine annealing over 40 epochs, `eta_min = 1e-6` |
| Patches | 96³; training: 50 % centred on a random tumour voxel, 50 % random; validation: fixed patch at the tumour centre |
| Batch | 1 × 4 gradient-accumulation steps (effective batch 4) |
| Precision | mixed precision (AMP) on CUDA; gradient clipping at norm 1.0 |
| Early stopping | patience 15 epochs, min-delta 1e-4 |
| Model selection | `best.pth` = best validation **mean foreground Dice** (patch-based); `best_loss.pth` = best validation loss; `last.pth` = latest epoch |
| Seed | 42 |

`cudnn.benchmark` is enabled, so reruns are statistically but not bit-wise reproducible.

The validation Dice logged during training is computed on tumour-centred patches. It is *not* the reported
test metric; reported numbers come from full-volume sliding-window inference (`glioma3d.evaluate`).

## Memory

The reported configuration (`base_channels = 24`, 96³ patches, AMP) was trained on a 4 GB GPU (GTX 1650).
`tools/profile_gpu_memory.py` prints the peak memory of one forward/backward step for other widths and patch sizes.
