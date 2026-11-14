"""onset_reg 亚帧修正测试。"""

import numpy as np

from piano2.core import NoteEvent
from piano2.decode import refine_onsets


def test_refine_onsets_recovers_subframe_position():
    """三角目标非对称时，起点应被修到 10.344 帧附近。"""
    onset_reg = np.zeros((40, 88), dtype=np.float32)
    key = 60 - 21
    onset_reg[9, key] = 0.731
    onset_reg[10, key] = 0.931
    onset_reg[11, key] = 0.869
    events = [NoteEvent(60, 10 / 31.25, 20 / 31.25, 80)]
    refined = refine_onsets(events, onset_reg)
    assert len(refined) == 1
    assert abs(refined[0].start - (10 + 0.344) / 31.25) < 1e-3
    assert refined[0].end == events[0].end
    assert refined[0].velocity == events[0].velocity


def test_refine_onsets_keeps_flat_curve_start():
    """没有明显 onset_reg 峰时，不应改动起点。"""
    onset_reg = np.zeros((40, 88), dtype=np.float32)
    events = [NoteEvent(60, 10 / 31.25, 20 / 31.25, 80)]
    refined = refine_onsets(events, onset_reg)
    assert refined[0].start == events[0].start
