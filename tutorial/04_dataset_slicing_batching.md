# 04 数据集：切片、样本与 batch

## 摘要

本文解释整首歌的数据怎么切成训练样本与 batch（批）：切片规则、segments.json 的作用、单个样本包含哪些数组、内存映射（memory mapping）怎么加速读取。读者不需要了解 PyTorch 的 DataLoader；读完后能看懂 x、onset、frame 等张量的形状和 batch 的维度。

## 从整首歌到 batch

![数据形态](../docs/figs/16_data_types.png)

图中 ⑦ 示意了这个过程：

```text
一首曲目 T 帧（例子 27,307 帧 ≈ 874 秒）
   │  按 313 帧切段
   ▼
一个样本：CQT (313, 264) + 标签 (313, 6, 88) + 踏板 (313, 2)
   │  batch 16
   ▼
x        (16, 313, 264) float32
onset    (16, 313, 88)  float32
frame    (16, 313, 88)  float32
velocity (16, 313, 88)  float32
pedal    (16, 313, 1)   float32
```

## 为什么要切成 10 秒

1. **显存和速度**：Transformer 的注意力计算量随序列长度**平方**增长，65 秒（2048 帧）比 10 秒（313 帧）
   贵约 40 倍（注意力部分）；
2. **训练效率**：切段后每轮能见到约 57,000 段，而不是每首歌只截一个 65 秒窗口；
3. **事件完整**：10 秒足够包含若干完整的「起音-释放」，也不会跨太多乐句。

不足 313 帧的段会补零（Dataset 里处理），batch 里形状永远一致。

## memmap：为什么读得快

旧版每取一个样本要**解压整首压缩 npz**（平均 155 ms），但只用其中 10 秒。
新版把 CQT 存成无压缩 `.npy`，用内存映射（memory-mapped file）打开：

```python
arr = np.load(path, mmap_mode="r")     # 文件当数组，不整体读入
seg = np.asarray(arr[start:start + 313])  # 只把 10 秒搬进内存
```

实测：**155 ms → 1.5 ms/样本**（约 100 倍）。代价是磁盘占用变大（17 GB → 33 GB），
用空间换速度，很划算。

## 验证数据（可选）

```bash
python scripts/check_labels.py    # onset2 是否 2 帧、踏板通道是否 0/1
```

## 深入阅读

- 数据 v2 的完整设计：`../docs/02_data.md`
- 训练脚本怎么用这些样本：`../docs/03_model_training.md`

## 小结

- 整首歌的数组按 **313 帧（10 秒）** 切成段，`segments.json` 记住每段的（文件、起始帧、长度）；
- 一个**样本** = 313 帧的所有通道：`x (313, 264)`、`onset (313, 88)`、`frame`、`velocity`、`pedal`；
- 一个 **batch** = 16 个样本堆叠：`x (16, 313, 264)` 等；
- 数据用**内存映射**（memmap）读取，只读需要的 10 秒，不整首解压。
