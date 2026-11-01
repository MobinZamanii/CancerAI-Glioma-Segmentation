import torch

from glioma3d.train import DiceCrossEntropyLoss, LocalizationBCELoss, make_localization_target


def test_dice_ce_perfect_prediction_is_near_zero():
    target = torch.randint(0, 4, (1, 8, 8, 8))
    logits = torch.nn.functional.one_hot(target, 4).permute(0, 4, 1, 2, 3).float() * 50
    total, ce, dice = DiceCrossEntropyLoss()(logits, target)
    assert total.item() < 1e-3 and dice.item() < 1e-3


def test_dice_ce_worse_for_wrong_prediction():
    target = torch.randint(1, 4, (1, 8, 8, 8))
    good = torch.nn.functional.one_hot(target, 4).permute(0, 4, 1, 2, 3).float() * 10
    bad = torch.nn.functional.one_hot((target % 3) + 1, 4).permute(0, 4, 1, 2, 3).float() * 10
    loss = DiceCrossEntropyLoss()
    assert loss(bad, target)[0] > loss(good, target)[0]


def test_localization_target_keeps_small_focus():
    seg = torch.zeros(1, 32, 32, 32, dtype=torch.long)
    seg[0, 5, 5, 5] = 3
    t = make_localization_target(seg, (8, 8, 8))
    assert t.shape == (1, 1, 8, 8, 8) and t.max().item() == 1.0 and t.sum().item() == 1.0


def test_localization_loss_finite():
    logits = torch.randn(1, 1, 8, 8, 8)
    target = (torch.rand(1, 1, 8, 8, 8) > 0.9).float()
    assert torch.isfinite(LocalizationBCELoss()(logits, target))
