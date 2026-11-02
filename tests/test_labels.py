"""数据 v2 标签测试：onset 宽度必须是 2 帧，踏板通道必须是 0/1。

本地没有数据 v2 时自动跳过（CI 或别的机器上也能跑其余测试）。
"""

import os

import numpy as np
import pytest

from piano2.constants import DATA_ROOT


def _first_lab():
    d = DATA_ROOT / "train"
    if not d.exists():
        pytest.skip("本地没有数据 v2")
    names = sorted(f for f in os.listdir(d) if f.endswith(".lab.npy"))
    if not names:
        pytest.skip("数据 v2 目录为空")
    return np.load(d / names[0])


def test_onset_width_is_two_frames():
    lab = _first_lab()
    onset2 = lab[:, 2, :]
    widths = []
    for k in range(onset2.shape[1]):
        col = onset2[:, k]
        if not col.any():
            continue
        idx = np.flatnonzero(np.diff(np.concatenate(([0], col, [0]))) != 0)
        widths.extend(int(b - a) for a, b in zip(idx[::2], idx[1::2]))
    widths = np.array(widths)
    assert len(widths) > 0
    assert widths.mean() > 1.8
    assert (widths == 2).mean() > 0.95


def test_pedal_channel_is_binary():
    lab = _first_lab()
    pedal = lab[:, 5, :]
    assert set(np.unique(pedal)).issubset({0, 1})
