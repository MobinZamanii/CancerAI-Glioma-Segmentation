"""
MultiTaskUNet3D
================
Model architecture only (3D MRI):
- the segmentation output is returned as logits;
- the localisation output is returned as logits at a lower resolution;
- no Dataset, DataLoader, loss, optimiser or training code lives here.

Input:
    x: Tensor of shape [B, in_channels, D, H, W]

forward() returns:
    {
        "segmentation": Tensor [B, num_classes, D, H, W],
        "localization": Tensor [B, 1, D/loc_stride, H/loc_stride, W/loc_stride]
    }

Note:
The spatial sizes D/H/W may be arbitrary: the model pads the input to a multiple
of 16 and crops the segmentation logits back to the original input size.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def get_num_groups(channels: int, max_groups: int = 8) -> int:
    """Largest number of GroupNorm groups (<= max_groups) that divides ``channels``."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def pad_to_multiple(
    x: Tensor,
    multiple: int = 16,
) -> Tuple[Tensor, Tuple[int, int, int]]:
    """Pad the end of each spatial dimension up to a multiple of ``multiple``; also returns the original size."""
    if x.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got shape {tuple(x.shape)}")
    original_size = tuple(int(v) for v in x.shape[-3:])
    d, h, w = original_size
    pd = (multiple - d % multiple) % multiple
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if pd or ph or pw:
        # F.pad expects the order (W, H, D).
        x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="replicate")
    return x, original_size


class ResidualBlock3D(nn.Module):
    """3D residual block with Conv3d, GroupNorm and LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            padding=padding, bias=False
        )
        self.norm1 = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size,
            padding=padding, bias=False
        )
        self.norm2 = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(get_num_groups(out_channels), out_channels),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.act2(x + residual)


class LocalGraphMessagePassing3D(nn.Module):
    """
    Local message passing over the six axial neighbours (±D, ±H, ±W).

    Note: positions outside the volume are invalid and are masked before the softmax,
    so border padding never takes part in the attention as if it were a real neighbour.
    """

    def __init__(self, channels: int, attention_channels: Optional[int] = None) -> None:
        super().__init__()
        hidden = attention_channels or max(channels // 4, 8)
        self.query = nn.Conv3d(channels, hidden, kernel_size=1, bias=False)
        self.key = nn.Conv3d(channels, hidden, kernel_size=1, bias=False)
        self.value = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.project = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = hidden ** -0.5

        # The six neighbour offsets, in [D, H, W] order
        self.offsets = (
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        )

    @staticmethod
    def _shift_with_validity(x: Tensor, offset: Tuple[int, int, int]) -> Tuple[Tensor, Tensor]:
        """At every voxel, return the feature of its neighbour at ``offset``, plus a validity mask."""
        od, oh, ow = offset
        b, c, d, h, w = x.shape
        shifted = torch.zeros_like(x)
        valid = torch.zeros((b, 1, d, h, w), device=x.device, dtype=torch.bool)

        dst_d = slice(max(0, -od), min(d, d - od))
        dst_h = slice(max(0, -oh), min(h, h - oh))
        dst_w = slice(max(0, -ow), min(w, w - ow))
        src_d = slice(max(0, od), min(d, d + od))
        src_h = slice(max(0, oh), min(h, h + oh))
        src_w = slice(max(0, ow), min(w, w + ow))

        shifted[:, :, dst_d, dst_h, dst_w] = x[:, :, src_d, src_h, src_w]
        valid[:, :, dst_d, dst_h, dst_w] = True
        return shifted, valid

    def forward(self, x: Tensor) -> Tensor:
        # The attention (q·k summed over channels, then softmax) can overflow in fp16. This module
        # only runs at 1/8 and 1/16 resolution and is cheap, so it always runs in fp32
        # (the weights themselves are unchanged).
        out_dtype = x.dtype
        if x.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                return self._forward_fp32(x.float()).to(out_dtype)
        return self._forward_fp32(x.float()).to(out_dtype)

    def _forward_fp32(self, x: Tensor) -> Tensor:
        q = self.query(x)
        neighbor_keys = []
        neighbor_values = []
        neighbor_masks = []

        k_map = self.key(x)
        v_map = self.value(x)
        for offset in self.offsets:
            shifted_k, valid = self._shift_with_validity(k_map, offset)
            shifted_v, _ = self._shift_with_validity(v_map, offset)
            neighbor_keys.append(shifted_k)
            neighbor_values.append(shifted_v)
            neighbor_masks.append(valid)

        # [B, 6, hidden, D, H, W] and [B, 6, C, D, H, W]
        keys = torch.stack(neighbor_keys, dim=1)
        values = torch.stack(neighbor_values, dim=1)
        masks = torch.stack(neighbor_masks, dim=1)  # [B,6,1,D,H,W]

        scores = (q.unsqueeze(1) * keys).sum(dim=2) * self.scale
        scores = scores.masked_fill(~masks.squeeze(2), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)

        message = (values * weights.unsqueeze(2)).sum(dim=1)
        return x + self.gamma * self.project(message)


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_graph: bool = False,
    ) -> None:
        super().__init__()
        self.pool = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.block = ResidualBlock3D(out_channels, out_channels, dropout=dropout)
        self.graph = LocalGraphMessagePassing3D(out_channels) if use_graph else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.graph(self.block(self.pool(x)))


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )
        self.block = ResidualBlock3D(
            out_channels + skip_channels, out_channels, dropout=dropout
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat((skip, x), dim=1)
        return self.block(x)


class MultiTaskUNet3D(nn.Module):
    """
    Multi-task 3D U-Net.

    Args:
        in_channels: number of input MRI channels (e.g. 4 for T1, T1ce, T2, FLAIR).
        num_classes: number of segmentation classes, including background.
        base_channels: base width of the network.
        dropout: Dropout3d rate used in the blocks.
        use_graph: enable local message passing at the two deepest encoder levels.
        localization_stride: down-sampling factor of the localisation output (4 or 8).
        return_features: if True, the bottleneck features are returned as well.

    Returns:
        dict with the keys "segmentation" and "localization" (plus "bottleneck" on request).
        The outputs are raw logits; apply softmax/sigmoid outside the model.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 24,
        dropout: float = 0.1,
        use_graph: bool = True,
        localization_stride: int = 4,
        return_features: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if base_channels < 4:
            raise ValueError("base_channels must be >= 4")
        if localization_stride not in (4, 8):
            raise ValueError("localization_stride must be either 4 or 8")

        self.localization_stride = localization_stride
        self.return_features = return_features
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )

        self.stem = ResidualBlock3D(in_channels, c1, dropout=dropout * 0.25)
        self.down1 = DownBlock3D(c1, c2, dropout=dropout * 0.25)
        self.down2 = DownBlock3D(c2, c3, dropout=dropout * 0.5)
        self.down3 = DownBlock3D(c3, c4, dropout=dropout, use_graph=use_graph)
        self.down4 = DownBlock3D(c4, c5, dropout=dropout, use_graph=use_graph)

        self.up3 = UpBlock3D(c5, c4, c4, dropout=dropout)
        self.up2 = UpBlock3D(c4, c3, c3, dropout=dropout * 0.75)
        self.up1 = UpBlock3D(c3, c2, c2, dropout=dropout * 0.5)
        self.up0 = UpBlock3D(c2, c1, c1, dropout=dropout * 0.25)

        self.segmentation_head = nn.Conv3d(c1, num_classes, kernel_size=1)

        # The localisation head reads encoder features at 1/4 or 1/8 resolution.
        loc_channels = c3 if localization_stride == 4 else c4
        self.localization_head = nn.Sequential(
            ResidualBlock3D(loc_channels, loc_channels, dropout=dropout * 0.5),
            nn.Conv3d(loc_channels, max(loc_channels // 2, 8), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(max(loc_channels // 2, 8)), max(loc_channels // 2, 8)),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(max(loc_channels // 2, 8), 1, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected input [B,C,D,H,W], got {tuple(x.shape)}")
        x, original_size = pad_to_multiple(x, multiple=16)

        e0 = self.stem(x)       # 1x
        e1 = self.down1(e0)     # 1/2
        e2 = self.down2(e1)     # 1/4
        e3 = self.down3(e2)     # 1/8
        bottleneck = self.down4(e3)  # 1/16

        d3 = self.up3(bottleneck, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, e0)

        seg_logits = self.segmentation_head(d0)
        d, h, w = original_size
        seg_logits = seg_logits[..., :d, :h, :w]

        loc_features = e2 if self.localization_stride == 4 else e3
        loc_logits = self.localization_head(loc_features)

        result = {
            "segmentation": seg_logits,
            "localization": loc_logits,
        }
        if self.return_features:
            result["bottleneck"] = bottleneck
        return result


if __name__ == "__main__":
    # Light smoke test that only checks the output shapes (real training uses glioma3d.train).
    model = MultiTaskUNet3D(
        in_channels=1,
        num_classes=4,
        base_channels=8,
        dropout=0.0,
        use_graph=True,
        localization_stride=4,
    )
    model.eval()
    with torch.no_grad():
        sample = torch.randn(1, 1, 33, 48, 51)
        output = model(sample)
    print("segmentation:", tuple(output["segmentation"].shape))
    print("localization:", tuple(output["localization"].shape))