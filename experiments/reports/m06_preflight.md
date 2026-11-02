# 训练前门禁检查结果（自动生成）

日期：2026-11-02 12:52:32

| 检查项 | 状态 | 说明 |
|---|---|---|
| G1 数据完整性与标签统计 | FAIL | npz 数量 {'train': 962, 'validation': 137, 'test': 177}；抽样 21 个文件；onset 正样本率 0.010、frame 0.053；onset 目标宽度均值 3.04 帧（标准 2 帧，长度 3 帧占比 98.6%）；问题：onset 标签宽 3 帧（应为 2 帧） |
| G2 划分无泄漏 | WARN | 文件级重复 0 条；跨 split 的同一作品 5 个（占作品数 0.6%，MAESTRO 划分固有现象）；需在报告里注明，看 test 分数时留意 |
| G3 训练/推理特征一致性 | PASS | MIDI-Unprocessed_09_R3_2008_ 帧数 2099/2099，相关 1.000000；MIDI-Unprocessed_10_R2_2008_ 帧数 3036/3036，相关 1.000000 |
| G4 评测链路 oracle 自检 | PASS | GT 当预测 F1=1.000；+32ms=1.000；+64ms=0.082；打乱音高=0.193 |
| G5 解码器达到天花板 | PASS | haydn: F1=0.952（天花板 0.950）；mozart: F1=0.943（天花板 0.949）；beethoven: F1=0.798（天花板 0.797）；holdout2: F1=0.821（天花板 0.825） |
| G6 基线指标登记 | PASS | 写入 /Docker/git/piano_practice/outputs/M0/registry/EXP-000_baseline.json；模型 sha256=4f43ca733f6e…；曲目 ['haydn', 'mozart', 'beethoven', 'holdout2'] |
| G7 资源与预算 | PASS | 可行配置 batch=16、seq=313 帧，每步 0.082s、峰值显存 1.63 GB；按 159.24 小时音频、10 秒片段、20 轮估算 71,658 步 ≈ 4060 上 1.6 小时 / 4090 上约 0.7 小时（4090 按 2.5 倍加速估算，未实测） |
| G8 可复现性 | PASS | 重复解码一致=True；DataLoader 顺序固定种子可复现=True；注意：多进程 worker 内的 numpy 随机性仍需在训练脚本里显式设种子（当前未验证） |

## 数字明细

```json
{
  "G1": {
    "counts": {
      "train": 962,
      "validation": 137,
      "test": 177
    },
    "n_sampled": 21,
    "onset_pos_rate": 0.010297167070564769,
    "frame_pos_rate": 0.053472136546458514,
    "onset_width_mean_frames": 3.044318067713991,
    "onset_width3_fraction": 0.9856502569626377,
    "cqt_min": -80.0,
    "cqt_max": 0.0,
    "known_issue_frame_contains_pedal": true,
    "known_issue_short_file_pad_zero": true
  },
  "G2": {
    "duplicate_audio": 0,
    "cross_split_works": 5,
    "work_fraction": 0.005841121495327103
  },
  "G3": {
    "rows": [
      {
        "track": "MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--2.wav",
        "stored_frames": 2099,
        "recomputed_frames": 2099,
        "corr": 1.0,
        "mean_abs_diff": 0.0
      },
      {
        "track": "MIDI-Unprocessed_10_R2_2008_01-05_ORIG_MID--AUDIO_10_R2_2008_wav--1.wav",
        "stored_frames": 3036,
        "recomputed_frames": 3036,
        "corr": 1.0,
        "mean_abs_diff": 0.0
      }
    ],
    "min_corr": 1.0,
    "frames_match": true
  },
  "G4": {
    "exact": 1.0,
    "jitter32ms": 1.0,
    "jitter64ms": 0.08182386008744534,
    "shuffle_pitch": 0.19300437226733294
  },
  "G5": {
    "haydn": {
      "P": 0.951310861423221,
      "R": 0.9519050593379138,
      "F1": 0.9516078676241024,
      "ceiling": 0.95
    },
    "mozart": {
      "P": 0.9586806814063066,
      "R": 0.9280701754385965,
      "F1": 0.9431271171331788,
      "ceiling": 0.949
    },
    "beethoven": {
      "P": 0.8577260665441927,
      "R": 0.7455642299503193,
      "F1": 0.7977218794494543,
      "ceiling": 0.797
    },
    "holdout2": {
      "P": 0.8352023403217943,
      "R": 0.806497175141243,
      "F1": 0.8205988023952098,
      "ceiling": 0.825
    }
  },
  "G6": {
    "registry_path": "/Docker/git/piano_practice/outputs/M0/registry/EXP-000_baseline.json",
    "tracks": [
      "haydn",
      "mozart",
      "beethoven",
      "holdout2"
    ]
  },
  "G7": {
    "results": [
      {
        "batch": 8,
        "seq_frames": 313,
        "step_s": 0.04515254497528076,
        "peak_vram_gb": 0.9455509185791016
      },
      {
        "batch": 16,
        "seq_frames": 313,
        "step_s": 0.08216559886932373,
        "peak_vram_gb": 1.6260743141174316
      },
      {
        "batch": 8,
        "seq_frames": 625,
        "step_s": 0.0838690996170044,
        "peak_vram_gb": 1.62687349319458
      },
      {
        "batch": 16,
        "seq_frames": 625,
        "step_s": 0.1704646348953247,
        "peak_vram_gb": 2.98382568359375
      }
    ],
    "feasible_best": {
      "batch": 16,
      "seq_frames": 313,
      "step_s": 0.08216559886932373,
      "peak_vram_gb": 1.6260743141174316
    },
    "estimate": {
      "n_segments": 57326.4,
      "steps_per_epoch": 3582.9,
      "epochs": 20,
      "total_steps": 71658.0,
      "hours_4060": 1.635506245493889,
      "hours_4090_assumed": 0.6542024981975556
    }
  },
  "G8": {
    "decode_identical": true,
    "dataloader_order_reproducible": true
  }
}
```
