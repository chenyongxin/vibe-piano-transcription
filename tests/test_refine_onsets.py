"""onset_reg 亚帧修正与踏板平滑测试。"""

import numpy as np

from piano2.core import NoteEvent
from piano2.decode import PedalEvent, refine_onsets, smooth_pedal_events


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
def test_smooth_pedal_bridges_small_gap():
    """相邻踏板间隔小于阈值时应合并，避免硬切断。"""
    events = [PedalEvent(0.0, 1.0), PedalEvent(1.1, 2.0), PedalEvent(3.0, 4.0)]
    out = smooth_pedal_events(events, max_gap_s=0.3, release_tail_s=0.05)
    assert len(out) == 2
    assert out[0].start == 0.0
    assert out[0].end >= 2.05
    assert out[1].start == 3.0
    assert out[1].end >= 4.05


def test_smooth_pedal_keeps_large_gap():
    """大间隔不合并，但前一个区间仍延长尾巴。"""
    events = [PedalEvent(0.0, 1.0), PedalEvent(2.0, 3.0)]
    out = smooth_pedal_events(events, max_gap_s=0.3, release_tail_s=0.1)
    assert len(out) == 2
    assert out[0].end >= 1.1
    assert out[1].start == 2.0
