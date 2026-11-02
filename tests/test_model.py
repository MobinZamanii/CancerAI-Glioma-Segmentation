import pytest
import torch

from glioma3d.model import MultiTaskUNet3D


def test_output_shapes_small():
    model = MultiTaskUNet3D(in_channels=4, num_classes=4, base_channels=4).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 32, 32, 32))
    assert tuple(out["segmentation"].shape) == (1, 4, 32, 32, 32)
    assert out["localization"].shape[0] == 1 and out["localization"].shape[1] == 1


def test_reported_parameter_count():
    model = MultiTaskUNet3D(in_channels=4, num_classes=4, base_channels=24, dropout=0.05, use_graph=True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 17_312_695


@pytest.mark.parametrize("use_graph", [True, False])
def test_backward(use_graph):
    model = MultiTaskUNet3D(in_channels=4, num_classes=4, base_channels=4, use_graph=use_graph)
    out = model(torch.randn(1, 4, 32, 32, 32))
    (out["segmentation"].mean() + out["localization"].mean()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
