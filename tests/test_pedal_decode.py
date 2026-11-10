"""踏板概率 -> 踩下/松开区间的解码测试。"""

import numpy as np

from piano2.decode import PedalEvent, decode_pedal


def test_decode_pedal_basic():
    # 1 帧 32 ms；前 3 帧松开，接着 5 帧踩下，再松开
    pedal = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    events = decode_pedal(pedal, on_threshold=0.5, off_threshold=0.35,
                          release_frames=2)
    assert len(events) == 1
    assert isinstance(events[0], PedalEvent)
    assert abs(events[0].start - 3 * 512 / 16000) < 1e-6
    assert events[0].end > events[0].start


def test_decode_pedal_merges_short_gap():
    pedal = np.array([1, 0, 1], dtype=np.float32)
    events = decode_pedal(pedal, on_threshold=0.5, off_threshold=0.3,
                          release_frames=1, min_gap_s=1.0)
    assert len(events) == 1