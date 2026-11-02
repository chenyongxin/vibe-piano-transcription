# 05 记谱与评测

## 1. 记谱：音符事件 → MusicXML（`src/piano2/notation.py`）

输入是解码器输出的音符事件（音高、起点、终点、力度），输出是两行钢琴谱（MusicXML）。
流程：

```text
① 拍点估计：librosa.beat_track 给拍点；bpm = 60 / 拍点间隔中位数；拍点回填到 t=0
② 时间量化：把每个音的起点/终点吸到最近的拍网格上（含三连音格子）
③ 同刻和弦化：同一只手、同一量化时刻的音符合成一个和弦
④ 时值重估：按「下一个同声部音」的空隙补齐时值，避免重叠
⑤ 时值分解：把一个时值拆成标准音符值 + 连音线（附点、二分、全音符等）
⑥ 建小节：按拍号手动分小节，左右手小节数强制一致
⑦ 输出 MusicXML（两行谱，不写 partName，避免出现三个「Piano」标签）
```

旧版（M0 之前）的配方是「固定 120 BPM + 按键秒数当时值 + 每个音单独塞进 music21」，
结果是 86–98% 的黑音符、小节错乱。改进后的关键指标：

| 指标 | 旧版 | v1.1 |
|---|---|---|
| 黑音符占比（目标 < 0.75） | 0.86–0.98 | 0.724（nezha2） |
| 左右手小节数一致 | 否 | 是（61 / 61） |
| 总时长误差（目标 ≤ 2%） | -18.8% | -2.83% |

## 2. 渲染：MusicXML → PNG（`src/piano2/render.py`）

verovio 把 MusicXML 渲染成 SVG，再用 `rsvg-convert`（librsvg）转 PNG。
踩过的坑（见 `history/07_debug_log.md`）：

- PyMuPDF 渲染 verovio SVG 会得到空白页 → 改用 rsvg-convert；
- 乐谱字体的私用区字符（PUA，例如 `U+ECA5`）会被渲染成巨大的「A5」字形 → 用正则剥掉对应 tspan；
- 加了 `svgViewBox` 选项再配 `adjustPageHeight` 会让文字巨大 → 去掉该选项。

## 3. 评测（`src/piano2/eval/metrics.py`）

口径（与 M0 报告一致）：

- **onset-only F1**：音高 + 起点（容差 50 ms），不看时值；
- **note-with-offset F1**：起点、音高、时值都匹配（offset 容差 20%，最小 50 ms）；
- 底层用 `mir_eval.transcription.precision_recall_f1_overlap`。

**oracle 自检**（评测链路本身是否正确）：把 GT 当预测 → 必须 1.0；
用抖动/打乱验证敏感度：

| 用例 | onset F1 |
|---|---|
| GT 当预测 | 1.000 |
| 起点 +32 ms | 1.000 |
| 起点 +64 ms | 0.082 |
| 打乱音高 | 0.193 |

这条自检是 M0 的纪律：**先证明评测链路对**，再看模型分数。

## 4. 记谱层的上限（GT 直通）

把标准答案 MIDI 直接输入记谱层（跳过模型和解码），得到的就是记谱层的上限：

![GT 直通](figs/30_gt_direct.png)

M0 的结论：GT 直通时黑音符占比仍有 86.3%，说明「黑音符太多」首先是记谱问题，不是模型问题。
修复后的对比（nezha2：旧产物 vs 最终版）：

![旧产物](figs/20_nezha2_old.png)

![最终版](figs/22_nezha2_step3.png)

## 5. 用法

```python
from piano2.notation import write_musicxml, events_from_midi
from piano2.render import render_png

events = [(60, 0.0, 0.5, 80), (64, 0.0, 0.5, 80)]   # (pitch, start, end, velocity)
write_musicxml(events, "audio.wav", "out.musicxml", title="demo")
render_png("out.musicxml", "out_p1.png")
```

命令行：

```bash
python scripts/transcribe.py --wav audio.wav --out-dir outputs/transcribe
python scripts/evaluate.py --model /path/to/checkpoint.pt
```

## 6. 已知问题

- 总时长误差 -2.83%，比 2% 的目标略超；根因是结尾休止符与最后一拍的估值，M3 之后再微调；
- 复杂节奏（切分、多声部交叉）的和弦化与声部划分仍是启发式，复杂曲目会出现声部串线；
- 弱起小节、装饰音（grace note）没有特别处理；
- 节拍记号（四分音符 = BPM）里的音符符号属于 PUA 字符，渲染时被剥掉，只剩 `=144.23` 的字样（M2 冒烟测试发现），M3/M4 打磨时处理。
