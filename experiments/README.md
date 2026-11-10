# experiments：阶段程序与实验产物

这里保存 M0–M1 的实验过程，**不是产品代码**。

| 目录 | 内容 |
|---|---|
| `stage_code/m0/` | M0（诊断/评测链路/探针）、M0.5（解码器实验）、M0.6（训练前门禁）的程序 |
| `stage_code/m1/` | M1（数据 v2、模型 v2、训练、pilot）的程序 |
| `reports/` | 各阶段报告 |
| `artifacts/` | 关键图表、训练日志、summary、数据 manifest、标签检查结果 |
| `registry/` | 实验登记表（每个实验一条 JSON，用于训练前后对比） |

说明见 `../docs/06_stage_programs.md`。

这些脚本依赖旧仓库的路径与缓存（旧数据 `/Docker/piano2_data/processed/maestro_processed`、旧输出
`piano_practice/outputs/`、旧检查点 `trained_data/`），在新仓里**不保证直接可跑**；
它们的作用是追溯与重跑关键实验。日常使用请走 `scripts/`。
