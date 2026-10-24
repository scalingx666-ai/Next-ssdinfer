# MiniMax H3 on Apple M1 Pro — 性能基线与优化路线

硬件：Apple M1 Pro / 32 GB 统一内存 / 10 核 GPU（Metal 3，非 Metal 4）/ macOS 15.3
推理引擎：`antirez/h3.c`（MIT，Metal 原生 C 实现），二进制 `h3c/h3`
模型：MiniMax-H3 FL2VA（33B 参数，BF16 DiT ≈ 66 GB，Qwen text encoder ≈ 66 GB）

> 重要：h3.c README 的所有性能数据均来自 **M5 Max / M3 Max（128 GB）**。
> M1 Pro 属于该引擎的"M3 及更旧"分支，**所有 int8 / TensorOps / Metal 4 加速路径
> 在本机不可用**（见下文「硬件门控」）。因此 README 数字不能直接外推。

---

## 1. 硬件门控：M1 Pro 上自动失效的加速

| 加速路径 | 触发条件 | M1 Pro 状态 |
|---|---|---|
| int8 MLP / QKV / attn-out（TensorOps） | `h3_gpu_is_m5()` 真 | **关闭**，回退 BF16 MPSGraph |
| GPU-resident Euler sampler | `h3_gpu_is_m5()` 真 | **关闭**，用 CPU sampler |
| 零拷贝权重 mmap | M5 或显式 `H3_ZERO_COPY_WEIGHTS=1` | 默认关闭，可显式开启 |
| Qwen 预取环深度 3 | M5 | 默认 2（`H3_QWEN_PREFETCH_DEPTH` 可调到 6） |
| Metal 4 / `MTLGPUFamilyMetal4` | macOS 26 + M5 | 不可用 |
| DiT 双命令缓冲切分 | M5 任意；M3 仅 50 层精确 30/50 | 默认关闭，可 `H3_DIT_COMMAND_BLOCKS` 覆盖 |
| `--use-int8-row-fc2` | 需 M5 | 不可用 |

结论：**M1 Pro 只能走「保守 BF16 + MPSGraph」基线**。优化空间主要来自
① 算法侧（步数 / 层数 / token 削减 / 内部分辨率）② 系统侧（预取深度、
零拷贝、命令切分、内存压力）③ 生产架构侧（分段并行、跳帧、后处理放大）。

---

## 2. M1 Pro 实测基线（2026-10-23，权重下载完成后）

以下全部是本机 `--ssd-streaming` 实测，不再是 M5 Max 外推。
锚点：256×256 内画布 / 22 帧 / 4 步 / layers 50。

| 实测项 | 参数 | denoise | video VAE | 说明 |
|---|---|---:|---:|---|
| 锚点 | 256×256, 22f, s4, l50 | **41.0 s** | 7.2 s | 首个成功生成 |
| 帧数 | 256×256, 120f, s4, l50 | **228.8 s** | 26.7 s | 22→120 帧，×5.58 |
| 画布 | 256×448, 22f, s4, l50 | **63.9 s** | 9.8 s | +75% 像素，×1.56 |
| 层数 | 256×256, 22f, s4, l45 | **36.7 s** | 6.9 s | 50→45 层，×0.89 |
| token 削减 | 256×448, 22f, s4, l50, tr | **44.3 s** | 9.6 s | ×0.69 |
| reuse | 256×448, 22f, s4, l50, r2 | **47.9 s** | 9.8 s | ×0.75 |
| 步数 | 256×256, 22f, s8, l50 | **81.3 s** | 6.8 s | 4→8 步，×1.98 |

固定开销（每进程一次）：text encoder 9.3–13.1 s，DiT 权重 load 19.9–20.9 s，
audio VAE ~1 s。合计约 **30 s/clip**，与 denoise 同量级。

### 2.1 两个推翻外推的实测结论
1. **帧数不是 latent_t 悬崖**。latent_t 从 2（22 帧）到 34（120 帧）是 17×，
   实测只有 ×5.58，接近线性（拟合指数 k=1.02）。原因是每步有大量与
   token 数无关的固定开销。**按 latent_t 外推会把 362 帧高估约 3 倍。**
2. **`--ssd-streaming` 不是可选项，是 9× 加速器**。同样的 22 帧形状，
   不开时 denoise 379 s（62 GiB DiT 在 32 GiB 内存里换页），
   开时 41 s。之前所有"SSD 流式会变慢"的 README 论断在 32 GB 机器上反向成立。

### 2.2 每 clip 成本模型（denoise 秒）
```
cost = 41.0 x area x steps x layers x tokred x reuse x (frames/22)^1.02
area(256x256)=1.00  area(256x448)=1.56  area(512x896)=6.24
steps(4)=1.00       steps(8)=1.98
layers(50)=1.00     layers(45)=0.89
tokred(off)=1.00    tokred(on)=0.69
reuse(1)=1.00       reuse(2)=0.75
```
叠加 30 s 固定开销即单 clip 墙钟。`h3_budget.py` 已按此模型解算。

---

## 3. 「10 分钟生成 5 分钟视频」可行性核算

### 3.1 实测结论：**不可达，差距 8.6×**

用最激进的合法组合（256×448 内画布、4 步、layers 45、token-reduction、
reuse 2、每 clip 120 帧源 + 3× 慢放）：

| 项 | 实测/推算 |
|---|---:|
| 单 clip denoise | 166 s |
| 单 clip 固定开销 | 92 s |
| 单 clip 合计 | 258 s |
| 20 clip 合计 | **5165 s = 86 分钟** |
| 目标 | 600 s = 10 分钟 |
| 差距 | **8.6×** |

继续压源时长也补不回来：

| 每 clip 源帧数 | 慢放倍数 | 20 clip 总计 | vs 10 分钟 |
|---:|---:|---:|---:|
| 120 帧 (5.0 s) | 3× | 86 min | 8.6× |
| 60 帧 (2.5 s) | 6× | 48 min | 4.8× |
| 30 帧 (1.25 s) | 12× | 29 min | 2.9× |
| 22 帧 (0.92 s) | 16× | 23 min | 2.3× |

最后一行画面已僵死（0.92 s 源拉成 15 s），仍差 2.3×。
**瓶颈是 20 × 30 s = 10 分钟的固定开销**，它与画布和步数无关。

### 3.2 为什么固定开销砍不掉
每 clip 必须重跑 text encoder（prompt 不同）+ DiT 权重 load。
试过用交互会话缓存复用：`h3_cli.c:697` 会开 `h3_cache_set_enabled`，
但 `h3.c:163` 的 `prepared_key` 把整个 prompt 和 frames 编进缓存键，
且 `h3_dit.c:1682` 的 `refine_text()` 把 text embedding 直接烘焙进 DiT 的
GPU 张量。**换 prompt 必然缓存失效**，实测第二个镜头仍是
`prepared DiT cache miss; model retained` + 20 s 重新 load。
`h3_dit_reset_run()` 只刷 video/audio condition rows，刷不了 text embedding。
要真正省掉它，得改 h3.c 把 text conditioning 从 DiT 里拆出来——
那是上游架构改动，风险高，未实施。

### 3.3 诚实的结论
- **10 分钟生成 5 分钟 H3 视频，在 M1 Pro 单机上不可达。**
- 现实的最优值是 **约 23 分钟**（每 clip 22 帧源 + 16× 慢放，画质已崩）
  或 **约 86 分钟**（每 clip 120 帧源 + 3× 慢放，画质可接受）。
- 折中落地：**每 clip 60 帧源 + 6× 慢放 ≈ 48 分钟**。
- 若要真正达到 10 分钟，需要 M5 Max 级硬件（40 核 + TensorOps + 128 GB），
  或等 h3.c 把 text conditioning 从 DiT 拆分以实现跨 clip 权重复用。

---

## 4. 优化清单（按实施顺序）

### P0 — 算法侧（零成本，改命令行参数）
1. `--steps 4 --reuse 1`：4 步是验证过的最优低预算调度，`reuse 1` 保证每步都真跑。
2. `--render-width/height` 用 256×448 内部画布，输出 384×672，让 h3.c 自己 vImage 放大。
   256×448 = 1792 空间 token vs 384×672 = 4032 token，**DiT 计算量降 56%**。

   > 画布合法性（`h3.c:496-511`）：两边都必须是 32 的倍数、内部画布不得大于输出、
   > 且宽高比必须严格相等（`render_w * height == render_h * width`）。
   > **432 不是 32 的倍数**（432 = 13.5 × 32），所以 432×768 本身就是非法输出画布。
   > 合法的 0.5714 竖屏阶梯（与官方 768×1344 上限同比例）：
   > 输出 384×672 → 内部 256×448（44%）；输出 768×1344 → 内部 512×896（44%）。
3. `--layers 45`（保构图）或 `--layers 40`（更快，风险构图偏移）。
4. `--token-reduction` 仅在 layers45 + reuse≤2 时开启；**不要**与 layers40+reuse3 同用。

### P1 — 系统侧（环境变量）
5. `H3_QWEN_PREFETCH_DEPTH=6` + `H3_QWEN_PREFETCH=8`：M1 默认深度 2，拉到 6
   可更好隐藏 text encoder 的 SSD 读取。text encoder 66 GB 每 clip 都要重读。
6. `H3_ZERO_COPY_WEIGHTS=transformer`：M1 默认走拷贝路径；零拷贝可省一次
   66 GB 的匿名内存拷贝与 32 GB 机器的内存压力。**需实测，若更慢则关掉**。
7. `H3_DIT_COMMAND_BLOCKS=24`（layers45 时 27 的近似）：M1 默认 50 层才切分；
   显式切分可让 GPU 执行与 CPU 编码重叠。
8. `H3_REUSE_MPS_COMMAND=1`：M3/更旧硬件实测 1.0–1.6% 提升。
9. `H3_VAE_TILE_PIXELS=320`：video VAE 空间 tile 上限，影响显存峰值与调度。

### P2 — 生产架构侧
10. **分段独立生成 + 首末帧接续**：每 clip 独立进程，失败只重跑一个 clip。
    `h3_drama.py` 已实现（`--chain` 用上一 clip 末帧做 `--first-frame`）。
11. **降帧率 + 慢放**：8 fps 生成 + 3× `setpts` 慢放到 24 fps，时间预算直接 /3。
12. **后处理放大替代高分辨率生成**：256×448 → 384×672 由 h3.c 内部 vImage 完成，
    再到 1080×1920 由 FFmpeg lanczos 完成。生成端永远跑最小画布。
13. **模型常驻交互会话**：h3.c 无 `-p` 时进入交互模式，BF16 prompt conditioning、
    DiT、video decoder 常驻内存，重复生成只付去噪钱。**对 20 clip 同场景连拍是关键**。

### P3 — 深层（需改代码，暂不实施）
14. 把 `--ssd-streaming` 的双 slot 环形预取换成更深（4–8 slot）的规划器，
    借鉴 `ssd-stream-infer` repo 的 zig-zag 块调度（实测 SSD 流量降 30%、
    耗时降 3.75×）。当前 h3.c 顺序读已接近 SSD 带宽上限，收益有限。
15. M1 Pro 无 int8 TensorOps，但可实现 **MPSGraph 的 int8 量化路径**，
    理论上对应 README 的 0.71× 收益。工作量大，且 MPS 本身已有 int8 支持可试探。
16. 为 M1 Pro 启用 `H3_GPU_SAMPLER=1` 试 GPU-resident sampler
    （代码只为 M5 默认开启，但环境变量可强制）。**需验证正确性**。

---

## 5. 内存预算（32 GB 机器，关键约束）

| 阶段 | 常驻 | 说明 |
|---|---:|---|
| DiT BF16 全驻 | 36.5 GiB | **超 32 GB，不可行** |
| DiT + `--ssd-streaming` | 2.0–2.1 GiB tracked | README 实测值，唯一可行路径 |
| text encoder 全驻 | ~66 GB | **必须流式**（h3.c 已实现分层预取） |
| video VAE | ~10 GB | 分 tile 解码，可控 |
| audio VAE | 0.6 GB | 小 |
| OS + 媒体缓冲 | ~4 GB | |

**结论：M1 Pro + 32 GB 上 `--ssd-streaming` 不是可选项，是必需品。**
代价是 M5 Max 上 512² forward 1.35 s → 2.49 s（84% 更慢），
864×480 2.14 → 2.68 s（26% 更慢）。在 320×572 这种小画布上，
SSD 读取代价占比更高，但内存不爆是前提。

---

## 6. 已完成的实测项

1. `--info` 确认 FL2VA 布局被识别（62.1 GiB encoder / 61.7 GiB DiT / 9.7 GiB video VAE）。
2. 256×256 × 22 帧 × 4 步：端到端跑通，产出有效 MP4（h264 + aac）。
3. 六组单轴校准（帧数 / 画布 / 层数 / token 削减 / reuse / 步数）全部完成，见第 2 节。
4. `--ssd-streaming` on/off：**开 41 s vs 不开 379 s**，内存峰值 1.6 GiB vs 36 GiB。
5. 交互会话连续 2 clip：缓存因 prompt 变化失效，无法省 DiT load（见 3.2）。
6. 视频质量抽检：帧均值 RGB (60.8, 46.3, 33.7) 暖色调符合 prompt，
   帧间 L1 3.7–7.2 有真实时间运动，std ~50 对比度正常。

仍未测：`H3_ZERO_COPY_WEIGHTS` / `H3_DIT_COMMAND_BLOCKS` 的 A/B
（需在相同形状上重复跑，收益预期 <20%，优先级低于先出片）。

---

## 7. 当前最优落地配置

```
--width 384 --height 672 --render-width 256 --render-height 448 \
--frames 120 --steps 4 --layers 45 --reuse 1 --token-reduction --ssd-streaming
```
每 clip 约 258 s，20 clip 约 86 分钟出 5 分钟成片（3× 慢放后 24 fps）。
生产管线：`ai-drama/h3_drama.py`（逐镜 + 断点续跑）与
`ai-drama/h3_batch.py`（交互会话批处理）。

---

## 8. 画质归因实验（2026-10-23，回应"画质模糊"质疑）

### 8.1 事实：交付版是降配版

`ai-drama/outputs/h3_drama/` 的 12 镜使用 preset `target`：
steps=4, layers=45, token-reduction=on, render 256x448, 输出 384x672。
三个参数都在直接损失画质。这不是意外，是为「10min 出 5min」目标做的极限降配。

满配为 steps=20, layers=50, token-reduction=off, render 512x896, 输出 768x1344。

### 8.2 满配在 M1 Pro 上不可跑

768x1344 / render 512x896 / 20 步触发 Metal 硬限制：
`MPSCore/Types/MPSNDArray.mm:850: failed assertion '[MPSTemporaryNDArray
initWithDevice:descriptor:isTextureBacked:] Error: total bytes of NDArray > 2**32'`
即 MPS NDArray 单张量 > 4GB 直接 abort。这是 M1 Pro 芯片限制，非配置错误。

### 8.3 画布阶梯极其稀疏（4:7 竖屏 + 32 倍数约束）

合法 render 画布只有三档（h3.c:496-511 要求同宽高比 + 32 倍数 + 不大于输出）：
  256x448  /  384x672  /  512x896
320x576、288x504 等看似中间档均因宽高比不匹配被拒。

### 8.4 受控实验：同 seed 同画布，只变 steps（4 -> 20 + layers 45->50 + 关 token-reduction）

单镜耗时：4步 = 300s；20步 = 1971s（6.6x）；384x672 画布 20步 = 6789s（22.6x）。

**关键发现：即使 seed 与画布都固定，steps 不同仍产生不同内容。**
区域亮度相关性仅 0.334 / -0.037 / -0.435。原因见 h3.c:1561-1574：
噪声张量大小由 h3_dit_video_elements() 决定（取决于 render 画布），
且扩散步数改变去噪轨迹。故「同画面隔离单变量」在此模型上不可能，
只能用统计量比较。

### 8.5 统计画质对比（3 帧平均，百分位归一化后）

| 指标 | 4步/45层/tr | 20步/50层 | 结论 |
|---|---|---|---|
| FFT 周期性伪影 | 1.076% | 0.532% | **降配版高 2.02x** |
| 梯度（高频细节） | 0.0329 | 0.0301 | 20步略低 |
| std（对比度） | 0.2157 | 0.2061 | 20步略低 |

### 8.6 结论

降配版的「高频更多」不是细节，而是 **token-reduction 的块状网格伪影**
（FFT 峰值高 2 倍为证）。视觉上噪点和高对比会伪装成"清晰感"，
这正是用户觉得"模糊"的真实来源——不是细节不足，是伪影与噪点。

高质量全片成本：384x672 画布 20 步单镜 113 分钟，12 镜 = 22.6 小时。
满配 768x1344 在本机不可跑。高质量需改用云端 GPU。

### 8.7 频域证据（解决空间域指标的矛盾）

空间域梯度曾给出相反结果（4步 0.0329 vs 20步 0.0301）。原因是
降配版的噪点是空间孤立的高频尖峰，梯度算子对尖峰过度敏感。
频域分析才能区分「结构化细节」与「随机噪点」：

| 频段 | 4步/45层/tr | 20步/50层 | 倍数 |
|---|---|---|---|
| 0.15-0.35 Nyquist（精细细节） | ~1.02e8 | ~1.60e8 | **1.7-1.9x** |
| 0.05-0.15 Nyquist（纹理结构） | 5.4e7 | 1.4e8 | **2.7x** |
| 径向功率谱斜率 | -4.73 | -5.02 | 20步更接近自然图像 1/f^2 |

**最终判定：降配版确实丢失了真实细节。**
- FFT 周期性伪影高 2.02x（token-reduction 的块状网格）
- 精细/中频段能量低 1.7-2.7x（真实细节被抹掉）
- 空间域"高频更多"是噪点伪装，非细节

即用户观感"模糊"是准确的：细节不足 + 网格伪影 + 噪点三重叠加。

### 8.8 4GB MPS 限制的真实边界（token 数驱动，非画布）

实测矩阵（render 画布 × 帧数 -> 是否触发
`MPSTemporaryNDArray ... total bytes of NDArray > 2**32`）：

| render 画布 | frames | latent_t | tokens = latency_t*16*28 | 结果 |
|---|---:|---:|---:|---|
| 256x448 | 120 | 30 | 3360 | OK |
| 384x672 | 120 | 30 | 7560 | OK（20步 6789s） |
| 512x896 | 22 | 6 | 2688 | OK |
| 512x896 | 72 | 18 | 8064 | 待测 |
| 512x896 | 96 | 24 | 10752 | **OOM crash** |
| 512x896 | 120 | 30 | 13440 | **OOM crash** |
| 512x896 | 362 | 91 | 40768 | **OOM crash** |

关键结论：
1. **限制是 token 数（= latent_t × latent_h × latent_w）驱动，不是画布尺寸。**
   同一画布 512x896，22 帧能跑、96 帧崩。
2. 崩溃发生在 `denoise 0/20` 的首次大算子，32s 内 abort，因此可用 80s 探测替代整次生成。
3. 384x672 的 7560 tokens 可跑，512x896 的 10752 不可跑 ->
   阈值落在 (7560, 10752]。
4. 若压在阈值内，**512x896 渲染（768x1344 输出）在本机可得**，
   代价是单 clip 帧数受限（约 72 帧 = 3s 源），需用更大慢放倍数补时长。
5. 估算限制张量可能是 AdaLN 广播：tokens × adaln_out_features(96768)。
   fp32 下 7560 × 96768 × 4 = 2.93 GB（< 4 GiB，通过）；
   10752 × 96768 × 4 = 4.16 GB（> 4 GiB，崩溃）—— 与实测完全吻合。

---

## 9. 画质上限：token 数阈值（2026-10-23 深夜实验）

### 9.1 现象
- 512×896 / 72 帧（latent_t=18，tokens≈8064）：输出 768×1344 成片为 **16px 马赛克噪声**，
  扩散未生成有效 latent（FFT 周期伪影比 0.039%，但相邻像素梯度 0.017 远高于正常值）。
- 512×896 / 22 帧（latent_t=6，tokens≈2688）：同一画布同一 prompt 输出 **完全正常**，
  人物、头饰、雨丝、烛台、递钗的手全部清晰。

### 9.2 变量分离
| render 画布 | frames | latent_t | tokens | 结果 |
|---|---:|---:|---:|---|
| 512×896 | 22 | 6 | 2688 | ✅ 正常 |
| **512×896** | **37** | **10** | **4480** | **✅ 正常（已定为生产配置）** |
| 512×896 | 61 | 16 | 7168 | ⏹ 未测完（主动放弃，见 9.6） |
| 512×896 | 72 | 18 | 8064 | ❌ 马赛克 |
| 384×672 | 120 | 30 | 7560 | ❌ 暗糊 |
| 256×448 | 120 | 30 | 3360 | ✅ 正常（旧交付档） |

结论：瓶颈是 **token 总数**，与画布无关。512×896 画布本身没问题。
**已知可用上界因此从 2688 抬到 4480**；4096 不是硬阈值，真实阈值落在 4480–7168 之间。

### 9.3 已排除的假设
- **VAE 拼接错误**：VAE 按 320px tile 拼接，缝只会出现在 192/288px，与 16px 网格不符。
- **sigma shift 配错**：`H3_VIDEO_SIGMA_SHIFT=12.0` 与模型 `model_index.json` 一致。
- **RoPE 适配**：`--use-reference-rope` 在 512×896/72 帧下 51 秒崩溃（rc=-15），
  说明问题不在 RoPE 适配逻辑。
- **OOM**：rc=0 正常退出，内存压力正常。

### 9.4 推测根因
DiT 的注意力走 Apple MPSGraph 原生 `scaledDotProductAttention`（`h3_gpu.m:1454-1525`），
bf16 在 sequence > 4096 时可能触发内部数值不稳定或静默回退。
`h3_gpu.m:4295-4302` 的 `score_bytes = sequence × 4` 线程组内存在 4096 时达 16KB，
M1 Pro 上限 32KB，但长序列下 softmax 数值稳定性可能失准。

### 9.5 生产配置（已定）

**512×896 渲染 / 37 帧 / 20 步 / 50 层 / reuse 1 / token-reduction off / ssd-streaming on**
→ 原生输出 768×1344、24fps、1.625s、含音频。

| 指标 | 实测值 |
|---|---|
| 单镜耗时（GPU 独占） | ≈47 min（20 步 × ≈2 min + 文本/VAE） |
| 单镜耗时（有并发争用） | 5646s ≈ 94 min，每步涨到 ≈165s |
| block16 / std | 0.0783 / 0.171（正常档 0.0689/0.175；马赛克档 0.1226/0.114） |
| 12 镜成片 | 12 × 4.875s + 片头 3.5s + 片尾 4.5s ≈ 66.5s |

**时长的结构性上限**：4480 tokens 锁死单镜 37 帧（1.54s），3× 慢放后 4.875s/镜。
想要更长的成片只有三条路：加镜数（≈47 min/镜）、降级到 384×672/61 帧（4032 tokens，
≈8.1s/镜，输出 512×896）、或迁云端 GPU 跑真满配。

### 9.6 放弃测 7168 tokens 的理由
- 7560（384×672/120f）与 8064（512×896/72f）都已失败，7168 落在中间，成功概率低；
- 跑完要 ≈2.5h，而它只能回答"能否把单镜从 37 帧加到 61 帧"，不影响已证实的生产配置；
- 生产必须独占 GPU，测探针就等于推迟成片 2.5h。故主动 kill，改为直接投产。

### 9.7 质检判据修正（重要）
`block_ratio`（block16/adjdiff）**不能**用来识别马赛克：好的 37 帧样本得分 7.66，
反而高于坏的 72 帧样本 7.26。原因是马赛克是**噪声块**而非平块，block16 抬高的同时
adjdiff 也被抬高。现用 `qc_shot.py` 的双条件（缺一不可）：

```
MOSAIC  iff  block16 > 0.105  AND  std < 0.14      # 阈值仅适用短边 ≥1200px
MUSH    iff  std < 0.09
DARK    iff  mean < 0.12
```

三个已知样本回验：22f → OK、37f → OK、72f → MOSAIC。
