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
python3 render.py --script ../shots/h3_shots_b37.jsonl \
    --preset max --out-dir outputs/probe --only 01_hairpin

# 4. 质检
python3 qc_shot.py outputs/probe/01_hairpin_37f.mp4
```

引擎必须从其所在目录运行（`h3_gpu.m` 从当前工作目录解析 `h3_shaders.metal`），`render.py` 已自动处理 `cwd`。

先看配置是否可行，不必启动渲染：

```bash
# 只报告 capacity，不渲染（默认 preset 是 smoke，最便宜的档）
python3 render.py --script ../shots/h3_shots_b37.jsonl --check-only

# 打印将要下发的命令流
python3 render.py --script ../shots/h3_shots_b37.jsonl --dry-run
```

```bash
$ python3 render.py --script ../shots/h3_shots_b37.jsonl --preset max --check-only
capacity: 4480/4480 tokens (100% of the measured budget) — within 10% of the measured budget
```

预设阶梯与各自的 token 占用：

| preset | 画布 / 帧数 | tokens | 说明 |
|---|---|---:|---|
| `smoke` | 256×448 / 22f | 672 | 最便宜的端到端验证 |
| `max` | 512×896 / 37f | 4480 | **生产配置** |
| `max30` | 512×896 / 37f | 4480 | 同 `max` 但 30 步（A/B 用） |
| `mid` | 384×672 / 61f | 4032 | 时长优先 |
| `target` | 256×448 / 120f | 3360 | 旧低质档（保留作对比） |
| `steps20_same` | 256×448 / 120f | 3360 | 同画布换 20 步 |

### 为什么 CapacityModel 是一等对象

因为真实上限不是你以为的那个。在 M1 Pro 上它不是显存：

> 超预算的任务**照样 rc=0 退出，并写出一个能播放的 16px 马赛克 mp4**。

所以"检查是否跑完"毫无意义，必须"检查是否可能跑成"。`CapacityModel` 把请求形状换算成 token 数，对照实测预算给出**拒绝理由**，而不是静默降级：

```bash
$ python3 render.py --preset max --only 02_mirror --check-only   # 该镜 frames=72
capacity: 02_mirror: 8064 tokens exceeds the 4480 budget measured on this
hardware. The renderer would still exit rc=0 and write a 16px-mosaic clip,
so this is refused rather than degraded. Reduce frames to 37 or drop to a
smaller canvas.
```

跑测试（命令流 golden + capacity 锚点）：

```bash
cd pipeline && python3 -m unittest discover -s tests -v
```

---

## 目录结构

```
engine/          打过补丁的 h3.c 源码（47 个文件，1.6 MB）
  h3_dit.c          DiT 主体、int8 开关、token 削减
  h3_gpu.m          Metal/MPSGraph 后端，含注意力实现
  h3_shaders.metal  着色器
  tests/            上游测试（make test）
pipeline/
  render.py         CLI 入口：参数解析 + resume + 会话（薄封装）
  core/             引擎无关机制，不含任何模型阈值
    shots.py          Shot 记录与脚本加载
    session.py        命令流拼装、渲染器子进程、产物回映射
    resume.py         崩溃后的幂等重入
    qc.py             ffprobe 探测与帧统计（只测数，不下判语）
    adapter.py        EngineAdapter / CapacityModel / Preset 协议
  models/
    h3/               H3 专属知识：token 算术、preset 阶梯、QC 阈值
  tests/             命令流 golden 测试与 capacity 锚点
  qc_shot.py         单镜质检（区分马赛克/暗糊/正常）
  make_h3_shots.py   从故事稿生成镜头脚本
  assemble_*.py      成片合成
shots/           已验证的镜头脚本（b37 = 生产配置）
story/            《嫁金钗》12 镜故事稿与字幕
docs/
  PERFORMANCE.md       性能与画质上限实验记录（核心）
  ENGINEERING.md       工程总纲、M1 Pro 补丁说明
  UPSTREAM-H3-METAL.md antirez 上游 README 原文
```

权重（约 134 GB）、模型缓存、生成的视频**刻意不入库**，见 `.gitignore`。

---

## 框架结构：机制与知识分离

这个流水线被刻意切成两层，理由是**本仓库最值钱的资产是一组实测数字，把它们稀释成通用框架里的脚注是负优化**。

```
core/                           models/h3/
─────────────────────────       ─────────────────────────────
Shot                            TOKEN_BUDGET = 4480
Session / resume                PRESETS（smoke/max/max30/mid/...）
命令流拼装与回映射              H3Capacity：token 算术
帧统计（probe、std、block16）    QC_BLOCK16_MOSAIC = 0.105
EngineAdapter 协议              命令语法、!cache 语义
```

**`core/` 里没有任何模型阈值。** 它回答的是"对任何本地生成器都成立"的问题：一个请求长什么样、怎么把一条命令流喂给长驻进程、崩溃后如何不重做、以及怎么量一帧的统计量。

**`models/<name>/` 里没有任何通用机制。** 它回答的是"只有这个模型知道"的问题：这台硬件的 token 预算是多少、什么统计量组合意味着 latent 没去噪、预设阶梯怎么排。

### 为什么 CapacityModel 是一等对象

因为真实上限不是你以为的那个。在 M1 Pro 上它不是显存：

> 超预算的任务**照样 rc=0 退出，并写出一个能播放的 16px 马赛克 mp4**。

所以"检查是否跑完"毫无意义，必须"检查是否可能跑成"。`CapacityModel` 把请求形状换算成 token 数，对照实测预算给出**拒绝理由**，而不是静默降级：

```bash
$ python3 render.py --preset max --only 01_hairpin --check-only
capacity: 4480/4480 tokens (100% of the measured budget) — within 10% of the measured budget
```

```bash
$ python3 render.py --preset max --only 02_mirror --check-only   # 该镜 frames=72
capacity: 02_mirror: 8064 tokens exceeds the 4480 budget measured on this
hardware. The renderer would still exit rc=0 and write a 16px-mosaic clip,
so this is refused rather than degraded. Reduce frames to 37 or drop to a
smaller canvas.
```

### 命令流为什么必须逐字节稳定

批量驱动不是"每镜起一个进程"。长驻渲染器把权重和 conditioning 留在内存里，所以驱动下发**一条命令流**，渲染器按序执行。这带来一个必须守住的约束：

> **逐镜覆盖如果不复位，会泄漏到后面每一镜。**

它不报错，只是让后续镜头的形状悄悄变化——缓存失效、成本模型失真，而你在画面上要等到第 5 镜才看出来。因此每个覆盖都配一个复位（`tests/test_h3_stream.py` 对此有 golden 断言）。

同理，`!frames` 必须显式下发：h3 默认 56 帧，漏掉这一行的镜头会安静地变成另一个形状。

### 扩展到新模型

新增一个 adapter，不改 `core/`：

```python
# pipeline/models/<name>/__init__.py
class MyAdapter:
    name = "<name>"
    presets = {...}                 # Preset 阶梯
    capacity = MyCapacity()         # 实测预算，不是 datasheet 数字

    def header(self, preset, out_dir): ...      # 全局设置行
    def shot_lines(self, shot, i, preset): ...  # 单镜，覆盖要复位
    def footer(self): ...                       # 结束会话
    def is_complete(self, path, shot): ...      # 只有模型层知道"完整"的含义
    def judge(self, path, shot): ...            # 对照自己校准过的阈值
```

然后在 `models/registry.py` 的 `load()` 里加一个分支。`core/` 不需要动——它从头到尾不知道 H3 的存在。

`capacity` 的数字必须是**实测**的。datasheet 上的显存、上下文长度、推荐批量，在这个问题上全部误导过人；这个仓库的核心结论就是"标称能力与本地可用能力不是一回事"。

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

## 附带评估：本机能跑哪个本地 LLM

同一台机器上顺带核实了四个候选模型，结论只有 Flint 可行，且**已经实测跑通**：

| 模型 | 权重体积 | 32 GB 可行性 |
|---|---|---|
| **Recursive Flint**（40B-A2.5B） | 16.61 GiB（3-bit，实测 17,837,530,688 B） | ✅ **已跑通** |
| Edge0（35B MoE） | ~17 GB | ⚠️ 需额外运行时 |
| MiniMax M3.1 Flash（428B） | ~150–215 GB | ❌ 差 4 倍 |
| DeepSeek V4.1 Flash 1017（552B） | ~510 GB | ❌ 差 16 倍 |

**M1 Pro 实测（2026-10-25）**：prefill 82.7 tok/s，**decode 47.7 tok/s**，
峰值内存 **17.953 GB** —— 对比 21.33 GiB 工作集上限，占用 79%，余 4.6 GiB。
官方 125 tok/s 是 M4 Pro 的数，不可当本机预期。

Flint 已接入为 `pipeline/models/flint/`（框架第二个 adapter，第一个非视频模型）：

```bash
cd pipeline
python3 render.py --script ../shots/flint_smoke.jsonl --model flint \
  --preset plan --qc
```

坑点：Flint 官方引擎 `rec` **明确要求 M4 Pro / M5 Pro / M5 Max + 48GB**，`install.sh` 会跑
`rec doctor` 直接拒绝本机。但 Flint 架构已被 llama.cpp（[PR #30928](https://github.com/ggml-org/llama.cpp/pull/30928)）
和 mlx-vlm ≥ 0.7.10 原生支持，社区版 3-bit 量化就是为 24/32GB Mac 做的。

注意 GPU 的可用显存不是 32GB，而是 macOS 默认工作集上限
**≈ 21.3 GB**（`max_recommended_working_set_size`）。
另一个必踩的坑：Flint **关不掉思考**（`--thinking-mode disabled` 无效），
`max_tokens` 给小了会产出纯思考、正文为空、返回码 0 的文件。
详见 [docs/LOCAL-LLM-EVALUATION.md](docs/LOCAL-LLM-EVALUATION.md)。

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
