from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

# Default label map: in the BraTS convention label 4 (enhancing tumour) becomes class 3.
LABEL_MAP = {4: 3}
NUM_CLASSES = 4

# Patch size (D, H, W)
PATCH_SIZE = (96, 96, 96)

# Pre-processed cache directory (built by `python -m glioma3d.prep.preprocess_cache`).
# If the cache files of a sample are missing, that sample is read directly from the NIfTI files.
# Default: ./cache_v2 in the current working directory; override with the GLIOMA3D_CACHE_DIR
# environment variable or with the --cache-dir option of the command-line tools.
CACHE_DIR = Path(os.environ.get("GLIOMA3D_CACHE_DIR", "cache_v2"))
SAMPLES_JSON = "samples_clean.json"  # written by glioma3d.prep.build_samples


def cache_paths(sample_id: str, cache_dir: Path = CACHE_DIR) -> Tuple[Path, Path]:
    key = hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{key}_img.npy", cache_dir / f"{key}_msk.npy"


def sample_label_map(sample_info: Dict) -> Dict[int, int]:
    """Per-sample label map (written by build_samples.py); default: BraTS convention (4 -> 3)."""
    raw = sample_info.get("label_map")
    if raw is None:
        return dict(LABEL_MAP)
    return {int(k): int(v) for k, v in raw.items()}


def normalize_nonzero(img: np.ndarray) -> np.ndarray:
    """Z-score over the non-zero (brain) voxels only; the background stays at zero."""
    nz = img != 0
    if nz.sum() > 10:
        vals = img[nz]
        std = float(vals.std())
        out = np.zeros_like(img, dtype=np.float32)
        out[nz] = (vals - float(vals.mean())) / (std if std > 0 else 1.0)
        return out
    return img.astype(np.float32)


MAX_FG_COORDS = 20000


def fg_path(sample_id: str, cache_dir: Path = CACHE_DIR) -> Path:
    key = hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{key}_fg.npz"


def compute_fg_info(msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(Sub-sampled) coordinates of the tumour voxels + the bounding-box centre. Computed once per sample."""
    idx = np.argwhere(msk > 0)
    if len(idx) == 0:
        return np.zeros((0, 3), np.int32), np.array([-1, -1, -1], np.int32)
    center = (idx.min(axis=0) + idx.max(axis=0)) // 2
    if len(idx) > MAX_FG_COORDS:
        rng = np.random.default_rng(0)
        idx = idx[rng.choice(len(idx), MAX_FG_COORDS, replace=False)]
    return idx.astype(np.int32), center.astype(np.int32)


def load_fg_info(sample_id: str, msk: np.ndarray, cache_dir: Path = CACHE_DIR):
    """
    Read the tumour information from the small sidecar file; build and store it once if it is missing.
    This avoids running np.argwhere over the whole mask volume at every iteration (the main CPU bottleneck).
    """
    path = fg_path(sample_id, cache_dir)
    try:
        with np.load(path) as z:
            return z["coords"], z["center"]
    except Exception:
        pass
    coords, center = compute_fg_info(msk)
    try:
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, coords=coords, center=center)
        os.replace(tmp, path)  # atomic; safe with several workers
    except Exception:
        pass
    return coords, center


def crop_patch(
    img: np.ndarray,
    msk: np.ndarray,
    patch_size: Tuple[int, int, int],
    is_train: bool,
    tumor_crop_prob: float = 0.5,
    tumor_coords: Optional[np.ndarray] = None,
    tumor_center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop a patch from numpy arrays (they may be memory-mapped).
    Training: 50% of the patches are centred on a random tumour voxel, 50% are random.
    Validation: a fixed patch around the tumour centre (or the volume centre if there is no tumour).
    img: [C, D, H, W], msk: [D, H, W]
    """
    d, h, w = msk.shape
    pd, ph, pw = patch_size
    if tumor_coords is None:  # slow path (fallback)
        tumor_coords, tumor_center = compute_fg_info(msk)
    tumor_idx = tumor_coords

    if is_train:
        if len(tumor_idx) > 0 and random.random() < tumor_crop_prob:
            cd, ch, cw = tumor_idx[random.randint(0, len(tumor_idx) - 1)].tolist()
        else:
            cd = random.randint(pd // 2, max(pd // 2, d - pd // 2))
            ch = random.randint(ph // 2, max(ph // 2, h - ph // 2))
            cw = random.randint(pw // 2, max(pw // 2, w - pw // 2))
    else:
        if len(tumor_idx) > 0:
            cd, ch, cw = tumor_center.tolist()
        else:
            cd, ch, cw = d // 2, h // 2, w // 2

    sd = min(max(cd - pd // 2, 0), max(d - pd, 0))
    sh = min(max(ch - ph // 2, 0), max(h - ph, 0))
    sw = min(max(cw - pw // 2, 0), max(w - pw, 0))

    img_crop = np.array(img[:, sd:sd + pd, sh:sh + ph, sw:sw + pw], dtype=np.float32)
    msk_crop = np.array(msk[sd:sd + pd, sh:sh + ph, sw:sw + pw], dtype=np.int64)

    # If the volume is smaller than the patch, pad with zeros
    pad_d = pd - msk_crop.shape[0]
    pad_h = ph - msk_crop.shape[1]
    pad_w = pw - msk_crop.shape[2]
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        img_crop = np.pad(img_crop, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))
        msk_crop = np.pad(msk_crop, ((0, pad_d), (0, pad_h), (0, pad_w)))

    return img_crop, msk_crop


def load_nifti_sample(sample_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Slow path (no cache): read the NIfTI files, normalise them and map the labels."""
    modalities = []
    for path in sample_info["image_paths"]:
        data = nib.load(path).get_fdata(dtype=np.float32)
        modalities.append(normalize_nonzero(data))
    img = np.stack(modalities, axis=0)

    msk = np.rint(nib.load(sample_info["mask_path"]).get_fdata()).astype(np.int64)
    for src, dst in sample_label_map(sample_info).items():
        msk[msk == src] = dst

    if img.shape[1:] != msk.shape:
        raise ValueError(
            f"Shape mismatch {sample_info['sample_id']}: "
            f"image {tuple(img.shape[1:])} vs mask {tuple(msk.shape)} "
            f"({sample_info['mask_path']})"
        )
    return img, msk


class BrainTumor3DDataset(Dataset):
    """
    3D multi-modal MRI dataset with a segmentation mask.
    If the cache exists, only the required patch is read from disk (much faster).
    """
    def __init__(
        self,
        samples: List[Dict],
        patch_size: Tuple[int, int, int] = PATCH_SIZE,
        is_train: bool = True,
        tumor_crop_prob: float = 0.5,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self.samples = samples
        self.patch_size = patch_size
        self.is_train = is_train
        self.tumor_crop_prob = tumor_crop_prob
        self.cache_dir = Path(cache_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_arrays(self, sample_info: Dict):
        img_path, msk_path = cache_paths(sample_info["sample_id"], self.cache_dir)
        if img_path.exists() and msk_path.exists():
            try:
                img = np.load(img_path, mmap_mode="r")
                msk = np.load(msk_path, mmap_mode="r")
                coords, center = load_fg_info(sample_info["sample_id"], msk, self.cache_dir)
                return img, msk, coords, center
            except Exception:
                pass  # corrupted cache; fall back to the NIfTI files
        img, msk = load_nifti_sample(sample_info)
        return img, msk, None, None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        img_arr, msk_arr, coords, center = self._load_arrays(sample_info)

        img_crop, msk_crop = crop_patch(
            img_arr, msk_arr, self.patch_size, self.is_train, self.tumor_crop_prob,
            tumor_coords=coords, tumor_center=center,
        )
        # Explicit label mapping (idempotent; also safe for older caches) + range validation.
        # Instead of a silent clamp during training, an invalid label raises an error naming the sample.
        for src, dst in sample_label_map(sample_info).items():
            msk_crop[msk_crop == src] = dst
        lo, hi = int(msk_crop.min()), int(msk_crop.max())
        if lo < 0 or hi > NUM_CLASSES - 1:
            raise ValueError(
                f"Invalid label range [{lo}, {hi}] in sample '{sample_info['sample_id']}' "
                f"({sample_info['mask_path']}); expected 0..{NUM_CLASSES - 1}."
            )

        image_tensor = torch.from_numpy(img_crop)
        mask_tensor = torch.from_numpy(msk_crop)

        if self.is_train:
            # random flip along the three axes (image and mask together)
            for axis in range(3):
                if random.random() < 0.5:
                    image_tensor = torch.flip(image_tensor, dims=[axis + 1])
                    mask_tensor = torch.flip(mask_tensor, dims=[axis])
            # mild per-channel intensity scaling/shift (non-zero voxels only)
            for c in range(image_tensor.shape[0]):
                scale = random.uniform(0.9, 1.1)
                shift = random.uniform(-0.1, 0.1)
                ch_img = image_tensor[c]
                nz = ch_img != 0
                image_tensor[c] = torch.where(nz, ch_img * scale + shift, ch_img)

        return {
            "image": image_tensor.contiguous(),
            "segmentation": mask_tensor.contiguous(),
        }


def load_samples(samples_json: str = SAMPLES_JSON, split: Optional[str] = None) -> List[Dict]:
    """Read samples_clean.json (written by glioma3d.prep.build_samples). split: train|val|test|None"""
    path = Path(samples_json)
    if not path.exists():
        raise FileNotFoundError(
            f"'{samples_json}' not found. Run `python -m glioma3d.prep.build_samples` first."
        )
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    if split is not None:
        samples = [s for s in samples if s["split"] == split]
    return samples


def _summarize(name: str, samples: List[Dict]) -> None:
    by_ds: Dict[str, int] = {}
    for s in samples:
        by_ds[s["dataset"]] = by_ds.get(s["dataset"], 0) + 1
    patients = len({s["patient_id"] for s in samples})
    print(f"[{name}] {len(samples)} samples / {patients} patients | " + ", ".join(f"{k}={v}" for k, v in sorted(by_ds.items())))


def create_dataloaders(
    batch_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    samples_json: str = SAMPLES_JSON,
    cache_dir: Path = CACHE_DIR,
) -> Tuple[DataLoader, DataLoader]:
    """
    train/val loaders built from the `split` field of samples_clean.json (patient-level split).
    The test split is deliberately never loaded here, so it can never be used for model selection.
    """
    random.seed(seed)
    np.random.seed(seed)

    train_samples = load_samples(samples_json, "train")
    val_samples = load_samples(samples_json, "val")
    if not train_samples or not val_samples:
        raise RuntimeError("The train or val split is empty; check the output of build_samples.py.")

    cached = sum(
        1 for s in train_samples + val_samples if all(p.exists() for p in cache_paths(s["sample_id"], cache_dir))
    )
    total = len(train_samples) + len(val_samples)
    print(f"[cache] {cached}/{total} samples are read from the cache"
          + ("" if cached == total else " (the rest from NIfTI, slower; run glioma3d.prep.preprocess_cache)"))
    _summarize("train", train_samples)
    _summarize("val", val_samples)

    train_dataset = BrainTumor3DDataset(train_samples, patch_size=PATCH_SIZE, is_train=True, cache_dir=cache_dir)
    val_dataset = BrainTumor3DDataset(val_samples, patch_size=PATCH_SIZE, is_train=False, cache_dir=cache_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader