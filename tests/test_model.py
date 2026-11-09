"""模型前向形状与参数量测试。"""

import torch

from piano2.model import PianoModelV2


def test_model_shapes():
    model = PianoModelV2()
    o, f, v, p, po, off, onr, offr = model(torch.randn(2, 313, 264))
    assert o.shape == (2, 313, 88)
    assert f.shape == (2, 313, 88)
    assert v.shape == (2, 313, 88)
    assert p.shape == (2, 313, 1)
    assert po.shape == (2, 313, 1)
    assert off.shape == (2, 313, 88)
    assert onr.shape == (2, 313, 88)
    assert offr.shape == (2, 313, 88)


def test_param_count_reasonable():
    model = PianoModelV2()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 5_000_000 < n < 50_000_000
