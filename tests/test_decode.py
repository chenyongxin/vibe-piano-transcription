"""解码器测试：峰值+帧解码、帧门控、条件式抑制。"""

import numpy as np

from piano2.decode import DecodeConfig, conditional_suppress, decode


def make_probs(n_frames=200):
    """构造一份合成概率：键 60 在帧 50 起音、持续 20 帧；键 72 只有起音没有帧。"""
    onset = np.zeros((n_frames, 88), dtype=np.float32)
    frame = np.zeros((n_frames, 88), dtype=np.float32)
    vel = np.full((n_frames, 88), 0.5, dtype=np.float32)
    onset[50, 60 - 21] = 0.95
    frame[50:70, 60 - 21] = 0.9
    onset[50, 72 - 21] = 0.95  # 八度泛音：有 onset 但没有 frame
    return onset, frame, vel


def test_peak_frame_finds_note():
    onset, frame, vel = make_probs()
    cfg = DecodeConfig(method="peak_frame", onset_threshold=0.6, onset_prominence=0.005,
                       frame_off_threshold=0.18, release_frames=3, frame_gate=0.0,
                       harmonic_mode="off")
    events = decode(onset, frame, vel, cfg)
    pitches = sorted({e.pitch for e in events})
    assert pitches == [60, 72]
    main = next(e for e in events if e.pitch == 60)
    assert abs(main.start - 50 / 31.25) < 0.05


def test_frame_gate_drops_harmonic():
    onset, frame, vel = make_probs()
    cfg = DecodeConfig(method="peak_frame", onset_threshold=0.6, onset_prominence=0.005,
                       frame_off_threshold=0.18, release_frames=3, frame_gate=0.15,
                       harmonic_mode="off")
    events = decode(onset, frame, vel, cfg)
    assert all(e.pitch == 60 for e in events)


def test_conditional_suppress_lowers_harmonic():
    onset, frame, vel = make_probs()
    cfg = DecodeConfig(harmonic_mode="conditional", harmonic_strength=0.5,
                       harmonic_window_frames=5, harmonic_frame_max=0.6)
    suppressed = conditional_suppress(onset, frame, cfg)
    # 键 60 的锚点应该压低它的八度键 72
    assert suppressed[50, 72 - 21] < onset[50, 72 - 21]
    # 其他位置不应被压（局部性）
    assert np.isclose(suppressed[0, 72 - 21], onset[0, 72 - 21])
