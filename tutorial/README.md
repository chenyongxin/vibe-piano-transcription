# tutorial：从声音到乐谱的每一步

这个目录是**教学/入门**内容：按「声音 → 特征 → 标签 → 数据 → 模型 → 解码 → 乐谱」的顺序，
讲清每一层的数据长什么样、为什么这么设计。适合第一次读这个项目的人，也适合复习。

和 `docs/` 的分工：

| 目录 | 定位 | 给谁看 |
|---|---|---|
| `docs/` | 产品与技术参考：设计、参数、指标、决策记录 | 开发者、维护者 |
| `tutorial/` | 教学与入门：概念解释、图解、常见问题 | 学习者、第一次接触项目的人 |
| `docs/history/` | M0–M3 的历史报告与调试日志 | 想追溯结论来源的人 |

## 目录

| 篇目 | 内容 | 关键图 |
|---|---|---|
| [`01_music_data_formats.md`](01_music_data_formats.md) | 音乐的四种数据形态：wav / MIDI / MusicXML / 数组 | `../docs/figs/00_pipeline.png` |
| [`02_features_cqt_hop_frames.md`](02_features_cqt_hop_frames.md) | 波形怎么变成 CQT；hop、帧、T 到底是什么 | `../docs/figs/16_data_types.png` |
| [`03_labels_from_midi.md`](03_labels_from_midi.md) | 6 个标签通道从 MIDI 怎么来；踏板；onset 的 2 帧 vs 3 帧 | `../docs/figs/01_onset_label.png` |
| [`04_dataset_slicing_batching.md`](04_dataset_slicing_batching.md) | 整首歌怎么切成 313 帧的段、怎么拼成 batch | `../docs/figs/16_data_types.png` |
| [`05_pipeline_end_to_end.md`](05_pipeline_end_to_end.md) | 端到端四段流水线，每段的输入/输出/意图 | `../docs/figs/00_pipeline.png` |
| [`06_faq.md`](06_faq.md) | 常见问题：黑音符、holdout、PUA、训练成本…… | — |
| [`07_running_app.md`](07_running_app.md) | 怎么运行浏览器应用（M4 完成后补全） | — |

## 建议阅读顺序

```text
01 → 02 → 03 → 04 → 05 → 06 → 07
```

每篇都尽量做到：**先给结论，再给例子或图，最后给「这个设计解决了什么问题」**。
基础概念（CQT、hop、帧、标签）弄懂之后，`docs/` 里的设计取舍会容易读得多。
