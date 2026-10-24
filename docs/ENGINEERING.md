# MiniMax H3 本地视频生成 — 工程总纲

硬件：Apple M1 Pro / 32 GB / macOS 15.3 / 10 核 GPU（Metal 3）
状态：权重下载中（FL2VA 子树约 143 GB，来自 ModelScope `MiniMax/MiniMax-H3`）

---

## 1. 组件与位置

| 组件 | 路径 | 说明 |
|---|---|---|
| 推理引擎 | `h3c/` | `antirez/h3.c` 源码副本 + 已编译 `h3` 二进制 |
| 引擎补丁 | `h3c/h3_gpu.m` `h3c/h3_dit.c` `h3c/h3_text_encoder.c` | 3 处 M1 Pro 优化，见 §4 |
| 性能分析 | `h3c/PERFORMANCE.md` | 硬件门控、外推核算、优化清单 |
| 权重 | `ai-drama/models/ms-h3-cache/.../snapshots/master/` | FL2VA 子树 |
| 下载脚本 | `/tmp/dl_h3_fl2va.py` → 日志 `ai-drama/outputs/dl_h3_fl2va.log` | ModelScope，需清空代理变量 |
| 生成驱动 | `ai-drama/h3_drama.py` | 逐镜调用 h3，支持首末帧接续 |
| 基线测试 | `ai-drama/h3_bench.sh` | 4 组计时 A/B，产出 `outputs/h3_bench/` |
| 预算规划 | `ai-drama/h3_budget.py` | 解算「10min→5min」可行配置 |
| 后期合成 | `ai-drama/make_h3_drama.py` | 24fps 竖屏放大 + 字幕 + BGM 混音 |
| 剧本 | `ai-drama/drama/script.json` | 《嫁金钗》4 集 16 镜 |

## 2. 为什么选 h3.c 而不是 ComfyUI / GGUF / MLX

| 路线 | 阻塞原因 |
|---|---|
| ComfyUI + GGUF | 需要 NVIDIA 或大量 Metal 转换工作；最小公开 DiT 量化是 pruned IQ1_S 3.78 GiB，非官方 |
| FastVideo MLX INT8/INT4 | 构建未发布或需 CUDA |
| 官方 PyTorch/SGLang | 需 NVIDIA |
| **antirez/h3.c** | MIT、Metal 原生、T2V/A 与首末帧都通、自带 `--ssd-streaming`，**唯一现成可跑** |

h3.c 的 `--ssd-streaming` 正是「模型参数放 SSD 流式送入」的现成实现：
只留 2 个 DiT block 在内存，GPU 跑当前块时从 SSD 读下一块，
DiT 存储 36.5 GiB → 2.0 GiB，输出字节级一致。

## 3. 关键约束（已从源码与 README 交叉验证）

- **单 clip 上限 362 帧 = 15.08 s**（`h3_host.c` 的 `5 + 17*n` 对齐，
  `h3.c:514` 硬检查）。5 分钟视频**必须**切成 ≥ 20 个 clip。
- 宽高须各为 32 的倍数且乘积 ≤ 768×1344；H3 是 768p 级模型。
- 输出固定 24 fps；原生 32 kHz 立体声 AAC，无需再配音轨。
- `--render-width/height` 必须与输出同宽高比且不超过输出，内部跑小画布后
  由 vImage 放大。
- `--token-reduction` **不能**与 `--layers 40 --reuse 3` 同用（色晕 + 残肢）。
- `--reuse` 与 `--core-reuse` 互斥。
- `--ssd-streaming` 与 `--use-int8-row-fc2` 互斥，且 M1 Pro 上后者本就不可用。

## 4. 对 h3.c 打的 3 个 M1 Pro 补丁

上游所有性能调优都以 M5 Max（128 GB）为基准，M1 Pro 落在它的「M3 及更旧」
保守分支，三个高价值开关默认关闭。补丁让它们在本机生效：

1. **`h3_gpu.m:615` 零拷贝权重扩展到非 M5**
   上游仅 M5 默认 mmap safetensor。32 GB 机器上匿名拷贝会让 66 GB
   transformer 的峰值翻倍，而文件-backed 映射可被内核回收。
   `H3_ZERO_COPY_WEIGHTS=0` 仍可显式关回。

2. **`h3_dit.c:208` 命令缓冲切分扩展到所有层数**
   上游仅在 50 层精确时切 30/50（M3 上 24/40 回退）。M1 Pro 只有 10 核，
   CPU 命令编码相对更贵，与 GPU 执行重叠在任何层数都划算。按 M5 的 60%
   深度切分，只移动命令缓冲边界，**输出字节不变**。

3. **`h3_text_encoder.c:136` Qwen 预取环深度 2 → 4**
   上游 M5 给 3、更旧给 2。text encoder 每 clip 要从 SSD 读 66 GB，
   深度 2 会让 I/O 路径在 Metal 执行时空闲。4 层只多几百 MiB 环形缓冲。

补丁后 `./h3_tests` **1768 项检查全部通过**，无回归。

## 5. 「10 分钟生成 5 分钟」结论

### 不可达的部分
5 分钟 = 7200 帧 @ 24 fps。即便用 4 步 + 320×572 内部画布 + 层削减，
M1 Pro 上每 clip（360 帧）仍需数十秒，20 个 clip 合计远超 600 s。
**全帧率 24 fps 的 5 分钟 H3 视频在 M1 Pro 上做不出来。**

### 可达的落地路线
降帧率 + 慢放是唯一能进预算的杠杆，且对古装宅斗文戏成立
（大量固定机位、推轨、特写，本身就是慢速运动）：

| 交付规格 | 生成规格 | clip 数 | 每 clip 源时长 | 慢放 | DiT 工作量 |
|---|---|---:|---:|---:|---:|
| 300 s @ 24 fps | 362 帧 × 20 clip，256×448，s4 l45 | 20 | 15.08 s | 1× | 1.00× |
| 300 s @ 12 fps | 180 帧 × 10 clip，256×448，s4 l45 | 10 | 7.50 s | 2× | 0.50× |
| 300 s @ 8 fps | 120 帧 × 7 clip，256×448，s4 l45 tr | 7 | 5.00 s | 3× | 0.33× |

关键建模点：降交付帧率**不会**缩短每个 clip，只减少 clip 数；
每个 clip 仍要填满自己的时长份额。真正省工作量的是
「短源时长 + 大慢放倍数」组合。第三行是预算内的现实选择。

`h3_budget.py` 会用 `h3_bench.sh` 的实测数字重算这张表，
不再依赖 M5 Max 的外推。

## 6. 执行顺序

1. 等 FL2VA 权重下完（`outputs/dl_h3_fl2va.log`）。
2. 软链模型目录到 h3.c 期望的布局，`./h3 --info -d ...` 确认识别。
3. 跑 `h3_bench.sh` 拿 M1 Pro 实测基线（4 组计时）。
4. 用实测数字跑 `h3_budget.py`，锁定最终配置。
5. `h3_drama.py --preset <锁定配置> --chain` 逐镜生成（16 镜）。
6. `make_h3_drama.py` 合成竖屏成片。

## 7. 已知风险

- **内存**：32 GB 上 `--ssd-streaming` 是必需品而非可选项（DiT 全驻 36.5 GiB）。
  代价是 forward 变慢，但内存不爆是前提。
- **质量**：4 步对 29 步的 SSIM 约 0.55；320×572 内部画布需放大到 1080×1920，
  细节弱于原生 768p。这是「10 分钟」预算的必然代价。
- **接续**：`--first-frame` 用上一 clip 末帧做条件，是最廉价的身份保持手段，
  但跨 16 个镜头的角色一致性仍需实测验证。
- **网络**：HuggingFace / github.com 直连不可达，代理已失效；
  ModelScope 可用但必须清空代理环境变量。
