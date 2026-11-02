"""评测指标测试：oracle 自检必须接近满分，并对错误敏感。"""

from piano2.core import NoteEvent
from piano2.eval.metrics import note_metrics, oracle_check, to_arrays


def make_gt():
    return [NoteEvent(60, 0.0, 0.5, 80), NoteEvent(64, 0.5, 1.0, 80),
            NoteEvent(67, 1.0, 1.5, 70), NoteEvent(60, 1.5, 2.5, 90)]


def test_oracle_exact():
    gt = make_gt()
    scores = oracle_check(gt)
    assert scores["exact"] > 0.999
    assert scores["jitter32ms"] > 0.999
    assert scores["jitter64ms"] < 0.2
    assert scores["shuffle_pitch"] < 0.35


def test_partial_recall():
    gt = make_gt()
    ref_iv, ref_p = to_arrays(gt)
    est = gt[:2]
    m = note_metrics(ref_iv, ref_p, *to_arrays(est))
    assert 0.4 < m["onset_R"] < 0.6
