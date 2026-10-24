# Next-ssdinfer

**MiniMax H3 在 Apple Silicon（M1 Pro / 32 GB）上的本地视频生成流水线，以及为把它推到画质上限所做的工程记录。**

上游引擎是 [antirez/h3.c](https://github.com/antirez/h3.c)（C + Metal，无 Python 运行时依赖），配 MiniMax 官方 FL2VA 权重。本仓库包含：打过补丁的引擎源码、批量生成与合成脚本、镜头脚本，以及**为什么这台机器的画质上限是这样一组数字**的完整实验记录。

> 结论先行：在这台机器上，画质的真实瓶颈**不是显存，而是 DiT 的 token 总数**。
> 可用的最高配置是 **512×896 渲染 / 37 帧 / 20–30 步 / 50 层 / BF16 流式**，
> 原生输出 768×1344。详见 [docs/PERFORMANCE.md](docs/PERFORMANCE.md)。

---

## 这个仓库解决什么问题

大多数人在这类任务上会先怀疑"内存不够"，然后把配置一降再降。本仓库的经验正相反：

| 常见假设 | 实际情况 |
|---|---|
| 512×896 画布在 32 GB 上跑不动 | 跑得动，22 帧完全正常 |
| 帧数不够是因为显存溢出 | rc=0 正常退出，无 abort、无 OOM |
| 满配 72 帧只是慢一点 | 成片是 16px 马赛克噪声，扩散未生成内容 |
| 降配置就能救回画质 | 降配置正是画质差的原因 |

满配 72 帧那次**成片了**，所以它特别危险：不看画面只会看到 rc=0 和一个看似正常的 mp4。

---

## 快速开始

```bash
# 1. 构建引擎（需要 macOS + Xcode Metal 工具链）
cd engine && make -j

# 2. 获取权重（约 134 GB，不在本仓库内）
export H3_MODEL=/path/to/MiniMax-H3/snapshots/master
# 或使用 scripts/fetch_model.sh

# 3. 跑单镜样板验证画质
cd ../pipeline
python3 h3_batch.py --script ../shots/h3_shots_b37.jsonl \
    --preset max --out-dir outputs/probe --only 01_hairpin

# 4. 质检
python3 qc_shot.py outputs/probe/01_hairpin_37f.mp4
```

引擎必须从其所在目录运行（`h3_gpu.m` 从当前工作目录解析 `h3_shaders.metal`），`h3_batch.py` 已自动处理 `cwd`。

---

## 目录结构

```
engine/          打过补丁的 h3.c 源码（47 个文件，1.6 MB）
  h3_dit.c          DiT 主体、int8 开关、token 削减
  h3_gpu.m          Metal/MPSGraph 后端，含注意力实现
  h3_shaders.metal  着色器
  tests/            上游测试（make test）
pipeline/        批量生成与合成
  h3_batch.py       把 JSONL 镜头脚本喂给 h3 管道模式
  qc_shot.py        单镜质检（区分马赛克/暗糊/正常）
  make_h3_shots.py  从故事稿生成镜头脚本
  assemble_*.py     成片合成
shots/           已验证的镜头脚本（b37 = 生产配置）
story/            《嫁金钗》12 镜故事稿与字幕
docs/
  PERFORMANCE.md       性能与画质上限实验记录（核心）
  ENGINEERING.md       工程总纲、M1 Pro 补丁说明
  UPSTREAM-H3-METAL.md antirez 上游 README 原文
```

权重（约 134 GB）、模型缓存、生成的视频**刻意不入库**，见 `.gitignore`。

---

## 核心方法论

### 1. 变量分离，而不是堆配置

把"画质差"拆成四个独立因素，逐个确认：4 步去噪、45 层、token 削减开启、384×672 输出被放大 2.8×。确认问题在生成侧而非合成侧后，所有努力才集中在生成配置上。

### 2. 从失败模式反推瓶颈

满配 72 帧产出 16px 马赛克。**rc=0、无 abort、无 OOM** 这一条直接排除了内存假设——内存不足会崩溃，不会安静地产出垃圾。

同画布只降帧数到 22 → 完全正常。变量只剩一个。换算 token 数后得到阈值表，瓶颈定位为 **token 总数**（画布 × 帧数可互换，乘积受限）。

### 3. 判据必须先用已知样本校准

用 FFT 周期伪影比当画质指标时，马赛克样本得分 0.039% 比正常样本还低，差点把噪声判成"最干净"。

改用 `block16`（相隔 16px 的像素差，即 1 个 latent 像素经 VAE 16× 解码后的尺度）后，又发现 `block_ratio` 同样不可靠：**好的 37 帧样本 7.66 反而高于坏的 72 帧样本 7.26**——马赛克是噪声块不是平块，block16 升高的同时相邻梯度也被抬高。

最终判据是双条件，并用三个已知样本回验：

```python
MOSAIC  iff  block16 > 0.105  AND  std < 0.14    # 短边 ≥1200px
MUSH    iff  std < 0.09
DARK    iff  mean < 0.12
```

### 4. 流式换精度，不换帧数

`--ssd-streaming` 不只省内存，它把三条 int8 通路整体关掉（`h3_dit.c:1641/1644/1647`），MLP / QKV / attention-output 全部走 BF16。这是实打实的精度收益。

但它**换不来帧数**：72 帧失败是 bf16 注意力在长序列下的数值问题，与权重存放位置无关。

---

## 已确认的生产配置

```
render  512×896   →  output 768×1344
frames  37        (latent_t=10, tokens≈4480)
steps   20 或 30   layers 50   reuse 1
token-reduction off    ssd-streaming on
```

| token 数 | 配置 | 结果 |
|---:|---|---|
| 2688 | 512×896 / 22f | ✅ |
| **4480** | **512×896 / 37f** | **✅ 生产配置** |
| 3360 | 256×448 / 120f | ✅（旧交付档，画质最低） |
| 7168 | 512×896 / 61f | ❌ 推测失败（未测完） |
| 7560 | 384×672 / 120f | ❌ 暗糊 |
| 8064 | 512×896 / 72f | ❌ 16px 马赛克 |

单镜耗时：20 步 ≈44.5 min，30 步 ≈66.7 min（实测 4004s）。

**时长的结构性上限**：4480 tokens 锁死单镜 37 帧（1.54s），3× 慢放后 4.875s/镜，12 镜成片约 66.5s。想要更长只能加镜数，或迁到无此上限的云 GPU 跑真满配（768×1344 + 362 帧）。

---

## 给 M1 Pro 打的补丁

上游所有调优都以 M5 Max（128 GB）为基准，M1 Pro 落在它的保守分支，三个高价值开关默认关闭：

1. **`h3_gpu.m` 零拷贝权重扩展到非 M5** — 32 GB 机器上匿名拷贝会让 66 GB transformer 峰值翻倍，文件-backed 映射可被内核回收。`H3_ZERO_COPY_WEIGHTS=0` 可关回。
2. **`h3_dit.c` 命令缓冲切分扩展到所有层数** — M1 Pro 只有 10 核，CPU 命令编码相对更贵，与 GPU 执行重叠在任何层数都划算。只移动命令缓冲边界，**输出字节不变**。
3. **`h3_text_encoder.c` Qwen 预取环深度 2 → 4** — text encoder 每 clip 要从 SSD 读 66 GB，深度 2 会让 I/O 路径在 Metal 执行时空闲。

补丁后上游 `./h3_tests` **1768 项检查全部通过**，无回归。详见 [docs/ENGINEERING.md](docs/ENGINEERING.md)。

---

## 已知限制与未验证项

诚实列出，避免误用：

- **token 阈值的根因是推测**：指向 `h3_gpu.m` 的 MPSGraph 原生 SDPA 在 bf16 长序列下数值失稳，但**没有验证手段**，未做逐层数值审计。
- 30 步 vs 20 步**不是像素级受控对比**：步数改变 sigma 轨迹，同 seed 两版同帧 SSIM 仅 0.83。局部高频提升（+22%~36%）不能理解为"整体更清"。
- 未跑 int8 vs BF16 的同种子 A/B，BF16 的收益只证明了"配置处于 BF16"，未量化幅度。
- 7168 tokens 未测完即放弃（7560 与 8064 均失败，概率低且需 2.5h）。
- 仅在单台 M1 Pro / 32 GB / macOS 15.3 上验证，其他 Apple Silicon 芯片的阈值可能不同。

---

## 许可

引擎源码沿用上游 antirez/h3.c 的许可，见 `engine/LICENSE` 与 `engine/THIRD_PARTY_NOTICES.md`。流水线脚本与文档以 MIT 发布。
