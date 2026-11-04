# 08 M4 浏览器应用：Docker 本地交付方案

> 状态：设计阶段（M4 未开始）。本文是产品与技术设计文档；面向使用者的傻瓜教程见
> `../tutorial/07_running_app.md`（M4 完成后补全）。

## 1. 目标与形态

把训练好的模型封装成一个**本地 Web 应用**，以 Docker 镜像交付：

```text
接收方：装 Docker → docker load 导入镜像 → docker run 启动 → 浏览器打开 localhost:8000
交互：拖入一段钢琴录音 → 等待 → 看乐谱 → 下载 MusicXML / MIDI / PDF → 播放
```

为什么选 Docker + Web：

| 方案 | 优点 | 缺点 |
|---|---|---|
| **Docker + 本地 Web（本项目）** | 一套镜像覆盖 Win/mac/Linux；数据不出本机；无服务器成本；浏览器体验 | 接收方要装 Docker；mac 无 NVIDIA 时慢 |
| 云端托管 | 接收方零安装 | 要付 GPU 服务器钱、要运维、音频要上传 |
| 各平台原生打包 | 体验最好 | 每个平台单独构建，mac 签名麻烦 |

## 2. 架构

```text
浏览器（单页应用）
   │ 上传音频（multipart）
   ▼
FastAPI 后端（容器内）
   ├─ 任务队列（单 worker 串行，避免抢显存）
   ├─ 推理管线：转 16 kHz → CQT → 模型 → 解码 → 记谱
   ├─ 产物：MusicXML / PNG 分页 / PDF / MIDI / 合成音频（可选）
   └─ 清理：任务完成后保留 N 分钟，定时删
   │
   ▼
浏览器展示：乐谱分页 + 下载按钮 + 回放（前端 SoundFont 播放 MIDI）
```

## 3. API 草案

| 方法 | 路径 | 作用 |
|---|---|---|
| `POST` | `/api/transcribe` | 上传音频，返回 `job_id` |
| `GET` | `/api/jobs/{job_id}` | 查询状态与进度（queued/running/done/failed） |
| `GET` | `/api/jobs/{job_id}/score` | 乐谱 PNG 分页（或 SVG） |
| `GET` | `/api/jobs/{job_id}/files/{kind}` | 下载 `musicxml` / `midi` / `pdf` |
| `GET` | `/api/health` | 健康检查（含 GPU 是否可用） |

## 4. 技术选型

| 部分 | 选择 | 理由 |
|---|---|---|
| 后端 | FastAPI + uvicorn | Python 生态一致，异步够用 |
| 推理 | 复用 `piano2.inference` + `decode` + `notation` | 不重写已验证代码 |
| 渲染 | verovio → SVG → rsvg-convert → PNG/PDF | 已有，成熟 |
| MIDI 导出 | pretty_midi 从解码事件写文件 | 简单可靠 |
| 回放 | 前端 Web Audio + 钢琴 SoundFont（播放 MIDI） | 无额外服务端依赖 |
| 前端 | 单页 HTML + 原生 JS | 需求简单，不上框架 |
| 打包 | Dockerfile（pytorch runtime 基底）+ docker save | 离线交付 |

## 5. 镜像与分发

```powershell
# 在开发机（有 GPU）构建
docker build -t piano2-app:0.1 .

# 导出成 tar（约数 GB，因为含 PyTorch CUDA 运行时）
docker save -o piano2-app-0.1.tar piano2-app:0.1

# 接收方导入并运行
docker load -i piano2-app-0.1.tar
docker run --gpus all -p 8000:8000 piano2-app:0.1
# 浏览器打开 http://localhost:8000
```

待定问题：

- 模型权重（约 135 MB）打进镜像还是作为卷挂载？**倾向打进镜像**，接收方零配置；
- 镜像体积大（PyTorch CUDA runtime 数 GB），考虑 CPU 版镜像作为轻量替代；
- macOS 没有 NVIDIA：镜像自动降级为 CPU 模式（慢但可用）。

## 6. 范围与限制

| 项目 | MVP 范围 | 暂不做 |
|---|---|---|
| 输入 | 独奏钢琴录音，wav/mp3/m4a，≤5 分钟 | 多乐器、人声、超长音频 |
| 输出 | MusicXML、MIDI、PDF、PNG 分页、回放 | 交互编辑、排版美化 |
| 并发 | 单任务串行 | 多用户队列、鉴权 |
| 质量 | 结构正确（onset 达标）；时值有误差 | 时值修复（M3.5） |

## 7. 验收标准（MVP）

1. 在开发机用 `docker run` 启动，浏览器能打开页面；
2. 上传 3 段测试音频（短/中/长各一），全部得到乐谱与 PDF/MIDI 下载，无报错；
3. 65 秒音频的总耗时（上传→出谱）≤ 30 秒（GPU）；
4. 刷新/重启后任务状态不丢（或给出明确失败提示）；
5. 同一输入重复运行结果一致（可复现）。

## 8. 里程碑

| 阶段 | 内容 | 预计 |
|---|---|---|
| M4.1 | 后端 API + 单页前端，本机跑通上传→出谱→下载 | 1 天 |
| M4.2 | 回放（前端 SoundFont）+ 进度显示 | 0.5 天 |
| M4.3 | Docker 镜像 + save/load + 接收方验证 | 0.5–1 天 |
| M4.4 | 文档：tutorial/07 使用教程 + 本设计文档更新 | 0.5 天 |

## 9. 与质量改进的关系

M4 把推理管线封装好之后，M3.5（时值修复）只需要替换「解码/记谱」这一段，
应用层不用改。所以先做 M4 不会白费工。
