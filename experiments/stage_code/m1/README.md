# m1：数据 v2、模型 v2、训练脚本与 pilot

对应文档：`../doc/09_m1_data_v2_and_pilot.md`

## 目录内容

| 文件 | 作用 |
|---|---|
| `build_data_v2.py` | 构建数据 v2：memmap CQT + 标签 v2（3 种 onset 宽度 + velocity + pedal）+ 段索引 + manifest |
| `dataset_v2.py` | 按 10 秒段索引读取数据；memmap 切片；输入归一化；dB 增益增强 |
| `model_v2.py` | 五头模型：CNN(MaxPool 1×3) + 4 层 Transformer + onset/frame/velocity/pedal/pedal_on |
| `train_v2.py` | 训练脚本：固定验证集、按 F1 选模型、AdamW+warmup+cosine、梯度裁剪、可续训、`--max-steps`/`--dry-run` |
| `check_labels.py` | 标签宽度自检（onset2 是否 2 帧、onset3 是否多一帧） |
| `eval_old_model.py` | 旧模型在同样验证段上的基线指标 |
| `make_m1_figures.py` | 生成 `doc/figs/13_m1_pilot.png` |

## 数据位置

- 数据 v2：`$DATA_ROOT/processed/maestro_processed_v2`（约 33 GB，主机 `$DATA_ROOT\processed\maestro_processed_v2`）
- manifest：`outputs/M1/data_manifest.json`
- 训练产物：`outputs/M1/pilot_onset{1,2,3}/`

## 环境

沿用 `piano` 容器（pytorch 2.9 + CUDA 12.6）；画图需要 `fonts-noto-cjk` / `fonts-wqy-zenhei`（已装）。

## 常用命令

```powershell
# 构建数据 v2（约 2 分钟）
docker exec -w $REPO_ROOT/m1 `
  -e PYTHONPATH=$REPO_ROOT/m1 piano python build_data_v2.py --workers 8

# 标签自检
docker exec -w $REPO_ROOT/m1 piano python check_labels.py

# pilot（1000 步，约 90 秒）
docker exec -w $REPO_ROOT/m1 `
  -e PYTHONPATH=$REPO_ROOT/m0:$REPO_ROOT/m1 `
  piano python train_v2.py --max-steps 1000 --batch 16 --val-every 250 --val-segments 120

# 断点续训
docker exec -w $REPO_ROOT/m1 `
  -e PYTHONPATH=$REPO_ROOT/m0:$REPO_ROOT/m1 `
  piano python train_v2.py --resume outputs/M1/pilot_onset2/checkpoints/best.pt
```
