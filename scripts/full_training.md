# 训练断点续算说明（全量 71,160 步）

## 一、开始训练

```powershell
# 确认容器在跑（关机后先执行这行）
docker start piano

# 从头开始训练（后台运行）
docker exec -d -w /Docker/git/piano2 piano `
  bash -c "nohup bash scripts/full_train.sh >/tmp/train.log 2>&1 &"
```

## 二、查看进度

```powershell
# 进度文件（步数、最佳 F1、更新时间）
docker exec piano cat /Docker/git/piano2/outputs/M1/m3_full_4060/progress.json

# 训练日志末尾
docker exec piano tail -20 /Docker/git/piano2/outputs/M1/m3_full_4060_train.log
```

每 200 步保存一次检查点：
`outputs/M1/m3_full_4060/checkpoints/latest.pt`（含模型、优化器状态、当前步数、best F1）。

## 三、关机 / 中断后第二天续算

```powershell
docker start piano
docker exec -d -w /Docker/git/piano2 piano `
  bash -c "nohup bash -c 'RESUME=1 bash scripts/full_train.sh' >/tmp/train_resume.log 2>&1 &"
```

续训会：
1. 自动读 `checkpoints/latest.pt`；
2. 恢复模型权重、AdamW 优化器状态、已完成步数、best F1；
3. 按同一份 cosine 学习率计划从当前步数继续（所以 `--max-steps` 必须保持 71,160）；
4. 训练日志追加到原 CSV（不覆盖历史）。

最多丢失 `save-every=200` 步（约 16 秒）的进度。

## 四、结束判断

- `progress.json` 的 `step` 达到 71160；
- 日志出现 `完成: {...}`；
- 评测：`python scripts/evaluate.py --model outputs/M1/m3_full_4060/checkpoints/best.pt`
