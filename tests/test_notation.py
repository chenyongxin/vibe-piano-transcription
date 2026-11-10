"""记谱模块测试：音符事件 → MusicXML → 解析回来。"""

import os

from piano2.eval import mxlib
from piano2.notation import write_musicxml

AUDIO = "/Docker/piano2_data/raw/maestro-v3.0.0/2008/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--2.wav"
MIDI = "/Docker/piano2_data/raw/maestro-v3.0.0/2008/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--2.midi"


def test_write_musicxml(tmp_path):
    if not (os.path.exists(AUDIO) and os.path.exists(MIDI)):
        import pytest
        pytest.skip("本地没有 MAESTRO 测试曲目")
    events = [(60.0, 0.0, 0.5, 80), (64.0, 0.0, 0.5, 80), (67.0, 0.5, 1.0, 70),
              (60.0, 1.0, 2.5, 90)]
    xml = str(tmp_path / "out.musicxml")
    write_musicxml(events, AUDIO, xml, title="test")
    assert os.path.exists(xml) and os.path.getsize(xml) > 0
    parsed = mxlib.parse_musicxml(xml)
    assert parsed["n_parts"] == 2
    for part in parsed["parts"]:
        assert len(part["measures"]) == len(parsed["parts"][0]["measures"])
    notes = [e for p in parsed["parts"] for e in p["events"] if e["pitch"] is not None]
    assert len(notes) >= 4
