"""验证音频→谱面锚点插值和 verovio 时间轴换算。"""

from app.render_service import _clean_anchors, audio_to_quarter, quarter_to_ms


def test_audio_to_quarter_interpolates_between_anchors():
    """两个锚点之间应按局部速度线性插值。"""
    anchors = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
    assert audio_to_quarter(0.5, anchors) == 1.0
    assert audio_to_quarter(1.5, anchors) == 3.0


def test_audio_to_quarter_clamps_and_extrapolates():
    """锚点范围外应夹到 0 或按最后一段速度外推。"""
    anchors = [(1.0, 1.0), (2.0, 3.0)]
    assert audio_to_quarter(0.5, anchors) == 0.5
    assert audio_to_quarter(3.0, anchors) == 5.0


def test_clean_anchors_drops_invalid_and_sorts():
    """负时间、负位置和倒序位置都应被过滤。"""
    anchors = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (-1.0, 0.0)]
    assert _clean_anchors(anchors) == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]


def test_quarter_to_ms_interpolates_and_clamps():
    """四分音符位置应按 timemap 线性换算成毫秒。"""
    q_values = [0.0, 1.0, 2.0]
    ms_values = [0.0, 500.0, 1000.0]
    assert quarter_to_ms(0.5, q_values, ms_values) == 250
    assert quarter_to_ms(-1.0, q_values, ms_values) == 0
    assert quarter_to_ms(5.0, q_values, ms_values) == 1000
