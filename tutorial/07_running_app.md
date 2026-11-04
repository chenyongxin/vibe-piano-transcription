# 07 运行浏览器应用

> 状态：**M4.1 MVP 可用**（本机运行）。Docker 镜像分发（M4.2）完成后，本文会补上「接收方」
> 的完整傻瓜步骤；当前内容适用于开发者/本机试用。

## 结论

三步就能用：

```bash
# ① 安装依赖（首次）
pip install -e .

# ② 启动后端（在仓库根目录）
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# ③ 浏览器打开
# http://127.0.0.1:8000
```

然后把一段钢琴录音拖进页面，等几秒到几十秒，就能看谱、下载、回放。

## 前置条件

| 项目 | 要求 |
|---|---|
| Python 环境 | 已安装本项目依赖（`pip install -e .`）；开发用的是 `piano` 容器 |
| 模型权重 | `models/best.pt`（约 135 MB）；没有的话用 `PIANO2_CKPT=/path/to/best.pt` 指定 |
| 浏览器 | Edge / Chrome / Firefox 任意现代浏览器 |
| 显卡 | 可选；有 NVIDIA GPU 更快，没有 GPU 走 CPU（5 分钟音频约 9 秒） |

## 使用步骤

1. **打开页面**：浏览器访问 `http://127.0.0.1:8000`，右上角会显示推理设备；
2. **上传**：把音频文件拖进虚线框，或点「选择文件」；
3. **开始转录**：点「开始转录」，页面会显示阶段（提取特征 → 模型推理 → 解码 → 生成乐谱 …）；
4. **看结果**：
   - 乐谱按页显示（PNG）；
   - 上方播放器可试听（合成钢琴音色）；
   - 三个下载按钮：MusicXML / PDF / MIDI；
5. **刷新/分享**：地址栏会带 `?job=xxxx`，在同一台机器上刷新可恢复结果（产物保留 2 小时）。

## 输出文件说明

| 文件 | 用途 |
|---|---|
| `score.musicxml` | 乐谱源文件，可用 MuseScore 打开编辑 |
| `score.pdf` | 直接打印/查看 |
| `score.mid` | 标准 MIDI，可导入 DAW 换更好的钢琴音色 |
| `preview.wav` | 网页内回放用的合成音频（方便快速听节奏） |

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 页面显示「后端未连接」 | 服务器没启动或端口不对，检查 `uvicorn` 是否在跑 |
| 右上角显示「⚠ 模型缺失」 | `models/best.pt` 不存在，用 `PIANO2_CKPT` 指定权重 |
| 端口 8000 被占用 | 换端口：`--port 8001`，浏览器相应改地址 |
| 上传后失败「不支持的格式」 | 只接受 wav / mp3 / m4a / flac / ogg / aac / webm |
| 谱面时值不准 | 已知问题（M3.5 修复），onset 已较准，节奏细节还有误差 |
| 音频很长 | 建议 ≤ 5 分钟；更长音频会变慢并占更多内存 |

## Docker 版（M4.2 计划）

```text
接收方：docker load -i piano2-app.tar
        docker run --gpus all -p 8000:8000 piano2-app
        浏览器打开 http://localhost:8000
```

注意：macOS 上 Docker 无法使用 Apple GPU，会走 CPU；Windows 需要 WSL2 + NVIDIA 驱动。
