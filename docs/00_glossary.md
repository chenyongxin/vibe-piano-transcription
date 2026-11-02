# 00 术语与缩写表（中英对照）

日期：2026-11-01
用途：本项目文档和对话里出现的英文术语、缩写，在这里统一给出中文名和一句话解释。
规则：**同一份文档或同一次对话里，英文术语首次出现时用「English（中文名）」的形式**；之后再出现直接用英文。

---

## 1. 任务与音乐概念

| 英文 | 中文名 | 一句话解释 |
|---|---|---|
| ground truth (GT) | 标准答案 | 数据集给的、与音频对齐的正确音符记录 |
| transcription | 转谱 | 把音频变成音符或乐谱的任务 |
| onset | 起音（击弦时刻） | 一个音开始发声的瞬间 |
| frame | 逐帧状态 | 每 32 ms 一帧，判断某个键此刻是否在响 |
| offset | 止音时刻 | 一个音停止发声的时刻 |
| velocity | 力度 | 按键的轻重，对应谱面上的强弱记号 |
| sustain pedal | 延音踏板 | 踩下后琴弦继续振动、声音拖长 |
| note event | 音符事件 | 一条记录：音高、开始时间、结束时间、力度 |
| pitch | 音高 | 88 键中的哪一个（MIDI 编号 21–108） |
| polyphony | 复音数 | 同一时刻有几个音在响 |
| chord | 和弦 | 同一时刻一起响的几个音 |
| voice | 声部 | 乐谱里独立的一条旋律线 |
| tie | 连音线 | 同一个音跨时值或跨小节时用弧线连起来 |
| rest | 休止符 | 表示不发声的时间 |
| measure / bar | 小节 | 两个小节线之间的时间单位 |
| time signature | 拍号 | 每小节几拍、以什么音符为一拍 |
| key signature | 调号 | 谱号旁表示调性的升降号 |
| clef | 谱号 | 高音谱号、低音谱号等 |
| tempo | 速度 | 每分钟多少拍（BPM） |
| beat tracking | 拍点跟踪 | 从音频里估计拍子的位置 |
| quantization | 量化 | 把连续的时间吸附到节拍网格上 |
| harmonic | 泛音 | 一个音带出的八度、五度等高次振动 |
| rubato | 弹性速度 | 演奏时速度的自由伸缩 |

## 2. 信号与特征

| 英文 | 中文名 | 一句话解释 |
|---|---|---|
| waveform | 波形 | 音频采样点序列 |
| spectrogram | 频谱图 | 时间 × 频率的能量图 |
| CQT (constant-Q transform) | 常 Q 变换 | 频率轴按半音均匀分布的时频变换 |
| log-mel | 对数梅尔频谱 | 按人耳感知划分频带的时频特征 |
| dB (decibel) | 分贝 | 能量的对数单位 |
| sampling rate | 采样率 | 每秒采样点数，本项目 16000 Hz |
| hop length | 帧移 | 相邻两帧相隔的采样点数 |

## 3. 模型与训练

| 英文 | 中文名 | 一句话解释 |
|---|---|---|
| CNN (Convolutional Neural Network) | 卷积神经网络 | 擅长找局部模式的网络 |
| Transformer | 变换器（自注意力网络） | 靠注意力机制建模长距离依赖的网络 |
| self-attention | 自注意力 | 每一帧和所有帧互相计算关联 |
| encoder | 编码器 | 把输入压成向量的部分；本项目只有编码器 |
| decoder | 解码器 | 机器学习里指逐步生成输出的网络；本项目没有，别和「解码」混淆 |
| positional encoding | 位置编码 | 告诉模型每一帧在时间轴上的位置 |
| residual connection | 残差连接 | 让梯度更容易传回浅层 |
| layer normalization | 层归一化 | 稳定训练的归一化方式 |
| epoch | 轮次 | 把训练集完整过一遍（本项目实际每轮只采样一小部分） |
| batch | 批 | 一次前向/反向传播用到的样本数 |
| step | 参数更新步 | 一次梯度更新；训练量真正该看的指标 |
| learning rate (LR) | 学习率 | 每次更新的步长 |
| warmup | 预热 | 训练初期把学习率从小逐渐升上去 |
| scheduler | 学习率调度器 | 控制学习率随训练变化的规则 |
| loss | 损失函数 | 训练时最小化的目标 |
| BCE (binary cross entropy) | 二元交叉熵 | 二分类任务的损失函数 |
| MSE (mean squared error) | 均方误差 | 回归任务的损失函数 |
| pos_weight | 正样本权重 | 正样本太少时给它加的权重 |
| underfitting | 欠拟合 | 训练量或容量不够，没学到 |
| overfitting | 过拟合 | 记住了训练集但不会泛化 |
| augmentation | 数据增强 | 用变换造出更多训练样本 |
| checkpoint | 模型检查点 | 保存的模型权重文件 |
| inference | 推理 | 用训练好的模型做预测 |
| fine-tune | 微调 | 在已有权重上继续训练 |
| EMA (exponential moving average) | 指数滑动平均 | 对权重做平均，让模型更稳 |
| gradient clipping | 梯度裁剪 | 防止梯度爆炸 |

## 4. 解码与评测

| 英文 | 中文名 | 一句话解释 |
|---|---|---|
| probability map | 概率图 | 模型输出的 (时间帧 × 88 键) 概率矩阵 |
| decode / decoding | 解码 | 本项目指「概率图 → 音符事件」的规则层，不是神经网络解码器 |
| peak detection | 峰值检测 | 在概率曲线上找尖峰 |
| hysteresis | 迟滞 | 用不同阈值控制开始和结束，避免抖动 |
| state machine | 状态机 | 用一组规则在状态之间切换的程序结构 |
| harmonic suppression | 谐波抑制 | 人为压低泛音位置的概率，减少假音 |
| anchor | 锚点 | 高置信度的真音，被当作抑制泛音的依据 |
| precision | 精确率 | 报出来的音里有多少是对的 |
| recall | 召回率 | 标准答案里有多少被找到了 |
| F1 | F1 分数 | 精确率与召回率的调和平均，0 到 1 |
| onset-only F1 | 只看起音的 F1 | 只要求音高和起点对，不管时值时 |
| note-with-offset F1 | 含时值的 F1 | 起点和时值都要对才算匹配 |
| frame F1 | 逐帧 F1 | 每一帧每个键的「是否在响」判对率 |
| mir_eval | 音乐信息检索评测库 | 音乐转录领域的标准评测工具 |
| oracle check | 标准答案自检 | 把 GT 当预测喂进评分函数，必须得满分 |
| ablation | 消融实验 | 去掉或改一个因素，看指标怎么变 |
| baseline | 基线 | 用来对比的参照方案 |
| grid search | 网格搜索 | 把参数组合逐个试一遍 |
| holdout | 留出集 | 调参时不碰、最后才用的数据 |
| pre-registration | 预注册 | 动手前先写下预期与判据，防止事后解释 |

## 5. 训练前门禁与工程

| 英文 | 中文名 | 一句话解释 |
|---|---|---|
| pre-flight gate | 训练前门禁 | 上云训练前必须通过的自动检查 |
| dry-run | 试运行 | 小步数跑通完整流程，不追求效果 |
| pilot run | 小规模试跑 | 先跑一小段训练，用来外推预算与验证链路 |
| manifest | 清单（含文件指纹） | 记录每个数据文件的哈希，保证版本可追溯 |
| hash | 哈希（指纹） | 文件内容的唯一摘要，用来判断是否一致 |
| reproducibility | 可复现性 | 同样命令跑出同样结果 |
| rollback | 回滚 | 退回上一个可用状态 |
| budget guard | 成本护栏 | 限制最大步数、时长、费用 |
| early stopping | 早停 | 指标不再提升时提前结束训练 |
| registry | 实验登记表 | 记录每次实验的假设、配置、数字、结论 |

## 6. 文件格式与工具

| 英文 | 中文名 | 一句话解释 |
|---|---|---|
| MIDI (Musical Instrument Digital Interface) | 乐器数字接口 | 记录演奏动作（音高、时间、力度）的文件格式 |
| MusicXML | 乐谱交换格式 | 记录乐谱符号（小节、声部、时值）的格式 |
| SMuFL (Standard Music Font Layout) | 音乐符号字体标准 | 规定音乐符号在字体中编码的标准 |
| SVG (Scalable Vector Graphics) | 可缩放矢量图形 | 矢量图格式，放大不失真 |
| PNG (Portable Network Graphics) | 便携式网络图形 | 位图格式，本项目用来贴谱面预览 |
| API (Application Programming Interface) | 应用程序接口 | 程序之间约定的调用方式 |
| GPU (Graphics Processing Unit) | 图形处理器 | 用来做神经网络计算加速的硬件 |
| VRAM (Video Random Access Memory) | 显存 | GPU 上的内存，决定能跑多大模型 |
| Docker | 容器工具 | 把运行环境打包成镜像的工具 |
| WSL (Windows Subsystem for Linux) | Windows 的 Linux 子系统 | 在 Windows 上跑 Linux 环境的机制 |
