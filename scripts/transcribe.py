"""音频 → 乐谱（端到端转录脚本）。

输入：一段钢琴音频 wav（任意采样率）。
输出：MusicXML（以及可选的 PNG 页预览）。
流程：CQT → 模型滑窗推理 → 解码（概率→音符事件）→ 记谱（事件→MusicXML）。

用法（容器内，已 pip install -e . 后）：
    python scripts/transcribe.py --wav /path/to/audio.wav --out-dir /tmp/piano2_out
    python scripts/transcribe.py --wav a.wav --model outputs/M1/pilot_onset2/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import os
import time

from piano2.constants import CKPT, DECODE_DEFAULTS, OUTPUT_ROOT
from piano2.decode import DecodeConfig, decode
from piano2.inference import compute_cqt, infer_sliding_window, load_model
from piano2.notation import write_musicxml


def main():
    ap = argparse.ArgumentParser(description="音频 → MusicXML（端到端）")
    ap.add_argument("--wav", required=True, help="输入音频路径")
    ap.add_argument("--model", default=str(CKPT), help="模型检查点路径")
    ap.add_argument("--out-dir", default=str(OUTPUT_ROOT / "transcribe"), help="输出目录")
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    ap.add_argument("--title", default="piano2 transcription", help="乐谱标题")
    ap.add_argument("--no-png", action="store_true", help="不渲染 PNG 预览")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.wav))[0]
    t0 = time.time()

    print("[-] 计算 CQT ...")
    cqt = compute_cqt(args.wav)
    print("[-] 加载模型 ...")
    model = load_model(args.model, args.device)
    from piano2.inference import pick_device
    print(f"[+] 推理设备: {pick_device(args.device)}")
    print("[-] 滑窗推理 ...")
    onset, frame, vel = infer_sliding_window(model, cqt, args.device)
    print("[-] 解码 ...")
    cfg = DecodeConfig(**DECODE_DEFAULTS)
    events = decode(onset, frame, vel, cfg)
    print(f"    得到 {len(events)} 个音符事件")

    xml_path = os.path.join(args.out_dir, f"{stem}.musicxml")
    write_musicxml([(e.pitch, e.start, e.end, e.velocity) for e in events],
                   args.wav, xml_path, title=args.title)
    print(f"[+] 乐谱: {xml_path}")

    if not args.no_png:
        try:
            from piano2.render import render_png
            png_path = os.path.join(args.out_dir, f"{stem}_p1.png")
            render_png(xml_path, png_path)
            print("[+] 预览:", png_path)
        except Exception as exc:  # 渲染失败不影响乐谱产出
            print(f"[!] PNG 渲染失败（可忽略）: {exc}")
    print(f"完成，用时 {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
