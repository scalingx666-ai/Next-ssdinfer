# 本地 LLM 候选评估：在 M1 Pro / 32 GB 上跑哪个

> 结论先行：**只有 Recursive Flint 值得投入，且可行。**
> DeepSeek V4.1 Flash 1017 和 MiniMax M3.1 Flash 差 4–16 倍内存，不是"优化一下"能弥补的数量级差距。
> Edge0 的 SSD 流式思想与本仓库同构，但它的 35B 模型同样需要显存承载。

本文记录 2026-10-24 对四个候选模型的核实过程与结论，避免以后重复调研。

---

## 硬件与运行时现状

| 项 | 值 |
|---|---|
| 机器 | Apple M1 Pro，10 核 CPU/GPU |
| 统一内存 | 32 GB |
| **GPU 推荐工作集** | **≈ 21.3 GB**（`max_recommended_working_set_size = 22906503168`） |
| Metal | `applegpu_g13s`，family 7，**无 Metal 4** |
| 可用磁盘 | 460 GB（`models/` 已占 207 GB） |

**注意那个 21.3 GB**：它不是统一内存，而是 macOS 默认允许 GPU 的有线工作集上限。
任何模型的实际可用显存按这个数算，不是 32 GB。

`iogpu.wired_limit_mb` 默认为 0（未手动提升）。24 GB Mac 上才需要
`sudo sysctl iogpu.wired_limit_mb=20480`；本机不需要。

---

## 四个候选的结论

| 模型 | 架构 | 权重体积 | 32 GB 可行性 |
|---|---|---|---|
| **Recursive Flint** | 40B MoE / 2.5B 激活 | 17.8 GB（3-bit 社区量化） | ✅ **可行** |
| Edge0 | 35B MoE / SSD 流式 | ~17 GB BF16 | ⚠️ 权重装得下，但需额外运行时 |
| MiniMax M3.1 Flash | 428B MoE / 23B 激活 | ~150–215 GB | ❌ 差 4 倍 |
| DeepSeek V4.1 Flash 1017 | 552B MoE / 8B prefill·16B decode | **~510 GB FP8** | ❌ 差 16 倍，磁盘也装不下 |

被排除的两个不是"调参能救"的量级问题：

- **MiniMax M3.1 Flash**：1–2bit GGUF 在 24GB GPU 上跑的前提是**还有 128GB+ 系统 RAM 把专家层 offload 到 CPU**。
- **DeepSeek V4.1 Flash 1017**：552B 总参，FP8 checkpoint 约 510GB。本机 460GB 可用磁盘连权重都装不下。它是文本/多模态 LLM，不是视频模型。

---

## 为什么 Flint 能跑，而官方路径跑不了

这里有一个容易踩的坑：**Flint 官方本地引擎 `rec` 明确拒绝了本机。**

### 官方 `rec` 的门禁（不可绕过）

`rec` 的 README 与 `install.sh` 都写明了要求：

```
- Apple Silicon Mac with an M4 Pro, M5 Pro, or M5 Max
- At least 48 GB unified memory
```

`install.sh` 会先执行 `rec doctor`，不达标直接中止：

```bash
doctor=$("$tmp/$artifact" doctor 2>&1) || { ...; fail 'this Mac does not meet the requirements above'; }
```

本机是 **M1 Pro / 32GB**，两条都不满足。Flint 官方基准的 125+ tok/s 是在 M4 Pro 上测的，
**不要把这个数字当成本机的预期**。

### 可行的替代路径

关键事实：**Flint 架构已被两个主流运行时原生支持**，不依赖 `rec`。

**1. llama.cpp**（PR [ggml-org/llama.cpp#30928](https://github.com/ggml-org/llama.cpp/pull/30928)，已合并）
新增 `src/models/flint.cpp`（194 行独立实现）、`gguf-py/constants.py` 的架构常量，
以及 `tests/fusion/MTL.csv` 的 Metal 融合条目。官方量化仓：

- `RecursiveSI/flint-int4-gguf`（4.1 GB 起）
- `RecursiveSI/flint-nvfp4-gguf`
- `ggml-org/Flint-GGUF`

**2. mlx-vlm ≥ 0.7.10**（本文采用）

`mlx_vlm/models/flint/` 提供完整的 `Model` / `LanguageModel` / `config` / `processing_flint` 四件套，
架构特征全部对上：

| Flint 架构特性 | mlx-vlm 0.7.10 实现 |
|---|---|
| 35 层滑窗 + 14 层全注意力 | `layer_types` → `window_size` 逐层判定 |
| 滑窗 513 token | `sliding_window: 513` |
| 每头 QK RMSNorm | `q_norm` / `k_norm` = `RMSNorm(head_dim)` |
| 可学习 attention sinks | `softmax_type: "learnable"`，`sinks=softmax_offset` |
| sigmoid 路由 + top-6 | `router_score_function: "sigmoid"` |
| 滑窗层 KV cache | `RotatingKVCache(max_size=window_size)` |

### 选用的量化版本

社区版 [`mlx-community/flint-mixed-3-4bit`](https://huggingface.co/mlx-community/flint-mixed-3-4bit)，
标签明确写着 `24gb` / `32gb`：

| | Recursive INT4 | 本方案 |
|---|---|---|
| 磁盘体积 | 22.6 GB | **17.8 GB** |
| 最小内存 | 48 GB（`rec`） | **24 GB** |
| 峰值内存 | — | ≈ 19.5 GB |
| 6 项编码任务 | 6/6 | 6/6 |
| 与 INT4 的 top-1 一致率 | — | 86.2% |

量化方法：**只把 routed experts（占权重 95%）从 4bit 降到 3bit**，attention / embedding / 输出层
保留 Recursive 训练过的值。perplexity 4.42，低于 INT4 的 4.62。

> ⚠️ 作者注明：**未在真实 24GB Mac 上实测**，19.5GB 的拟合度来自 64GB Mac 上的峰值测量。
> 本机 21.3GB 的 GPU 工作集上限装 19.5GB 是紧的，需关闭重应用。

---

## 与 H3 管线的同构性

Edge0 的思路值得记一笔，因为它和本仓库攻的是同一堵墙。

Edge0 让 35B MoE 只常驻约 2.5GB RAM：权重留在 SSD 上流式读入，
配一个**训练过的 router** 提前预测下一步需要哪些专家。H3 那边的瓶颈不是显存而是
DiT token 总数（4480 上限），但它同样面对"装不下就流着读"——`!ssd-streaming on`
让 BF16 权重走 SSD 流式而非整体载入。

`pipeline/core/` 的 `EngineAdapter` 协议（`cwd` / `binary` / `extra_args` /
`is_complete` / `judge`）正是为这种异构引擎留的扩展点。
接入 Flint 作为 `models/flint/` adapter 是设计目标，不是改造。

---

## 实操记录

### 1. 恢复 venv 的 pip

`ai-drama/.venv` 原先没有 pip 模块：

```bash
./.venv/bin/python -m ensurepip --upgrade
```

### 2. 升级 mlx-vlm

```bash
./.venv/bin/python -m pip install -U "mlx-vlm==0.7.10"
```

只额外拉了一个 `fsspec`，`mlx` / `transformers` / `numpy` 均未变动。
**H3 管线与 mlx-vlm 零耦合**（H3 是独立编译的 C 二进制 `engine/h3`，542KB），
升级后 dry-run 命令流与 18 个单测全部照旧通过。

### 3. 下载权重

```bash
./.venv/bin/hf download mlx-community/flint-mixed-3-4bit \
  --local-dir models/flint-mixed-3-4bit
```

约 17.8 GB，4 个 safetensors 分片。

### 4. 首次推理

```bash
./.venv/bin/mlx_vlm.generate --model models/flint-mixed-3-4bit \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --max-tokens 32768 \
  --prompt "Write a Python function that merges overlapping intervals."
```

采样参数用 Recursive 的推荐值（temp 1.0 / top-p 0.95 / top-k 20），
难题给足 32768 输出预算——Flint 会先思考 10–30K token 再作答。

---

## 已知风险

1. **M1 Pro 无 Metal 4**：Flint 的 Metal megakernel 是否完全兼容未知，官方基准是 M4/M5。
2. **官方 125 tok/s 不可直接预期**：那是 M4 Pro 的数。本机需实测。
3. **19.5GB vs 21.3GB 工作集**：余量约 1.8GB，跑前关掉浏览器等重应用。
4. **tool call 参数名错误**：mlx-vlm 0.7.10 已知问题，作者测得 OpenCode 测试中
   85 次调用有 41 次参数名错误（如 `path` 而非 `filePath`），但任务最终仍全部通过。
   修复在 [mlx-vlm#2571](https://github.com/Blaizzy/mlx-vlm/pull/2571)，发布后用 `--constrain-tool-calls`。
5. **上下文控制在 32K 左右**：Flint 思考链很长，长上下文会挤占内存。

---

## 来源

- Flint 模型卡：https://huggingface.co/RecursiveSI/flint
- Flint 发布文：https://recursive.com/updates/introducing-recursive-flint
- `rec` 引擎：https://github.com/recursive-oss/rec
- llama.cpp 支持 PR：https://github.com/ggml-org/llama.cpp/pull/30928
- 社区 24/32GB 量化版：https://huggingface.co/mlx-community/flint-mixed-3-4bit
- Edge0：https://github.com/Edge0-AI/Edge0 ，arXiv https://arxiv.org/abs/2609.18063
- MiniMax M3.1 Flash GGUF：https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF
- DeepSeek V4.1 Flash 1017：https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
