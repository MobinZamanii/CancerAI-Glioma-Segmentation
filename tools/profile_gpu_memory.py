"""Peak GPU memory of one forward/backward pass for several widths and patch sizes.

Used to choose base_channels=24 / 96^3 patches for a 4 GB GPU.

Usage:
    python tools/profile_gpu_memory.py
"""
import torch
import torch.nn.functional as F

from glioma3d.model import MultiTaskUNet3D


def test(base, patch):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = MultiTaskUNet3D(in_channels=4, num_classes=4, base_channels=base, dropout=0.05).cuda()
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(1, 4, *patch, device="cuda")
    y = torch.randint(0, 4, (1, *patch), device="cuda")
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            out = m(x)
            loss = F.cross_entropy(out["segmentation"].float(), y) + out["localization"].float().mean()
        scaler.scale(loss).backward()
        print(base, patch, "peak GB:", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
    except torch.OutOfMemoryError:
        print(base, patch, "OOM")
    del m
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for base in (8, 12, 16):
        for patch in ((64, 128, 128), (96, 96, 96)):
            test(base, patch)
