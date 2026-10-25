# 本地 LLM 候选评估：在 M1 Pro / 32 GB 上跑哪个

> 结论先行：**只有 Recursive Flint 值得投入，且可行。**
> DeepSeek V4.1 Flash 1017 和 MiniMax M3.1 Flash 差 4–16 倍内存，不是"优化一下"能弥补的数量级差距。
> Edge0 的 SSD 流式思想与本仓库同构，但它的 35B 模型同样需要显存承载。

本文记录 2026-10-24/25 对四个候选模型的核实过程与结论，以及把跑通的 Flint
做成 OpenAI 兼容服务的实测记录，避免以后重复调研。

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

| | Recursive INT4 | 本方案（社区 3-bit） |
|---|---|---|
| 磁盘体积 | 22.6 GB | **16.61 GiB**（17,837,530,688 B，实测） |
| 最小内存 | 48 GB（`rec`） | **24 GB** |
| 峰值内存 | — | **17.953 GB**（MLX 自报，实测） |
| 6 项编码任务 | 6/6 | 6/6 |
| 与 INT4 的 top-1 一致率 | — | 86.2% |

量化方法：**只把 routed experts（占权重 95%）从 4bit 降到 3bit**，attention / embedding / 输出层
保留 Recursive 训练过的值。perplexity 4.42，低于 INT4 的 4.62。

---

## 本机实测结果（2026-10-25）

**结论：Flint 在 M1 Pro / 32 GB 上跑得动，而且速度可用。**

```
Prompt: 11–83 tokens, 82.7–197 tok/s（prefill）
Generation: 512 tokens, 47.737 tok/s
Peak memory: 17.953 GB
```

| 指标 | 实测 | 预算 | 余量 |
|---|---|---|---|
| 峰值内存 | 17.953 GB（16.72 GiB） | 21.33 GiB 工作集 | **+4.6 GiB（79% 占用）** |
| decode | 47.7 tok/s | — | — |
| prefill | 82.7 tok/s（11 tok）/ 197 tok/s（83 tok） | — | — |
| 单次请求墙钟 | ~27 s（含 16.6 GiB 权重加载 + 思考 + 作答） | — | — |

三条被实测推翻的原先估计：

1. **峰值不是 19.5 GB，是 17.95 GB。** 作者的 19.5 GB 是从 64 GB Mac 外推的，
   而 MLX 惰性分页只把实际 touch 的专家读进内存——39.8B 里每 token 只激活 2.5B。
2. **余量不是"紧的 1.8 GB"，是 4.6 GiB。** 不需要关浏览器。
3. **官方 125 tok/s 与 76–119 tok/s 都不能当本机预期**；本机 decode 是 47.7 tok/s，
   即 M4 Pro 的约 38%。这个数字只能实测，推不出来。

### 容量模型：为什么装得下

先前的"19.5 GB 装进 21.3 GB"是拍脑袋。实际的账可以算清楚：

```
峰值 = 权重 16.61 GiB
     + KV 缓存（8-bit K/V，仅 14/49 层走全注意力，其余滑窗 513）
     + 激活与 Metal scratch 预留 1.4 GiB
```

KV 随上下文增长极慢，这是长上下文能装下的关键：

| 上下文 | KV | 峰值 | 结论 |
|---|---|---|---|
| 8,192 | 0.126 GiB | 18.14 GiB | ✅ 85% |
| 32,768 | 0.455 GiB | 18.46 GiB | ✅ |
| 65,536 | 0.892 GiB | 18.90 GiB | ✅ |
| 131,072 | 1.767 GiB | 19.78 GiB | ✅ 93%（推荐上限） |
| 262,144（满） | 3.517 GiB | **21.53 GiB** | ❌ 超限，拒绝 |

也就是说：**显存由权重决定，不由上下文决定**；满 262144 上下文反而装不下。
按同一公式二分，能压线通过的最大上下文是 **247,669** tokens——但那正好占满 100% 工作集、
零余量，运行期还需要 scratch 之外的分配，实际不可用。所以 **128K 才是有余量的实用上限**。

模型侧的原生窗口是 262,144（`max_position_embeddings`），**输入与输出共享这一个窗口**，
不是各 256K。服务端在请求未给 `max_tokens` 时用 `DEFAULT_MAX_TOKENS = 2048`
（可用 `MLX_VLM_MAX_TOKENS` 或 `--max-tokens` 改），且思考计入 completion。
**HTTP 路径没有容量门禁**（门禁只存在于 `render.py` 的 `shot.meta.context_tokens`），
所以客户端要自己守住 128K；要硬保险可给服务设 `MAX_KV_SIZE`。
这个账现在写在代码里（`pipeline/models/flint/FlintCapacity`），不是文档里的估算。

### 一个必须知道的坑：Flint 关不掉思考

`--thinking-mode disabled` 会被 CLI 接受，但**对本模型无效**——
`chat_template.jinja` 在每一次 assistant 轮次无条件输出 `<think>`，不查任何开关。

后果是实测到的：

- `--max-tokens 512` 时，模型产出 1921 字节**纯思考**、正文一个字没有，**返回码 0**。
- 所以"文件有内容"完全不能当作成功；必须剥掉思考块、再检查正文非空。

这是文本模型版的"未去噪 latent"，已写进 `FlintAdapter.answer()` / `judge()`：
未闭合的思考块返回空串，判定 `THINK_ONLY`。默认预算因此提到 1536 token。

---

## 接入框架（`pipeline/models/flint/`）

Flint 是 `EngineAdapter` 的第二个实现，也是第一个非视频模型。它验证了抽象，
但**不是直接套进去就成立**的——写第二个 adapter 才暴露了两处必须改的地方：

| 改动 | 原因 |
|---|---|
| `Preset` 拆成 `Preset` + `MediaPreset` | 帧数/画布/层数是扩散模型专属；LLM 用 `max_tokens`/`temperature`。硬塞会让"通用接口"变成谎言 |
| `judge` 改为可选（`Judge \| None`） | 原本以为文本失败一眼可见。实测证明不成立（见上），所以 Flint 反而实现了它；**可选是对的，预设理由错了** |
| `Capacity` 的 `cost`/`budget` 带 `unit` | H3 花的是 DiT token，Flint 花的是字节。同一个"跑之前先算"的问题，不同量纲 |
| `mode = "oneshot"` + `stdout_is_output` | `mlx_vlm.generate` 没有交互模式，每次请求一个完整 argv；正文走 stdout，所以输出文件名要在进程启动前定好 |

两边的容量检查共享同一堵墙（`core/hardware.py` 探测的 21.33 GiB），
但成本函数各自派生——H3 是实测的 4480 token 上限，Flint 是从 config 算出的字节数。

跑法：

```bash
cd pipeline
python3 render.py --script ../shots/flint_smoke.jsonl --model flint \
  --preset plan --out-dir ../ai-drama/outputs/flint_plans --qc
```

---

## OpenAI 兼容服务：给 api key、url、模型名就能被调用

CLI 跑通之后，下一步是让它变成一个服务。`mlx-vlm` 自带 `mlx_vlm.server`，
直接暴露 `/v1/models` 与 `/v1/chat/completions`——不用再自己写一层 HTTP。

### 启动

框架入口只决定"监听在哪、要不要 key"，argv 由 adapter 提供（哪个二进制、
哪份权重、自称什么 id 属于模型知识）：

```bash
cd pipeline
python3 serve.py --model flint              # 默认 127.0.0.1:8080
python3 serve.py --model flint --dry-run    # 只看会下发什么
```

两个默认是有意写死的：

- **只绑 `127.0.0.1`**。本地模型服务监听 `0.0.0.0`，等于把一台无人监管的 LLM
  挂到笔记本加入的每一个网络上。
- **key 永不为空**。不传 `--api-key` 就生成一把并打印一次（想固定就设
  `FLINT_API_KEY`）。空 key 会静默降级成无鉴权，而"没看到提示"不会被任何人读成决定。

它用 `os.execv` 而不是 `subprocess`：Ctrl-C 与退出码归属于服务端本身，
套一层会吞掉信号的壳只会让卡在加载阶段的进程杀不掉。

`ai-drama/serve_flint.sh` 是等价的持久启动脚本（key 存 `.flint_api_key`，chmod 600，不入库）：

```bash
cd ai-drama
./serve_flint.sh start      # 后台启动（默认动作），就绪后打印客户端三元组
./serve_flint.sh status     # 是否在跑 + 三元组
./serve_flint.sh stop
./serve_flint.sh fg         # 前台运行，盯日志用
```

它的子进程被 launchd 收养（PPID 1），关掉终端不会带走服务；`start` 是幂等的，
已在运行时不会再起第二个（第二个会撞破 21.33 GiB 工作集）。

### 客户端三元组

| 项 | 值 |
|---|---|
| `base_url` | `http://127.0.0.1:8080/v1` |
| `api_key` | 启动横幅打印的那把 / `.flint_api_key` |
| `model` | **权重绝对路径**（`/v1/models` 原样返回的 id），**无短别名** |

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1",
           api_key=open(".../ai-drama/.flint_api_key").read())
r = c.chat.completions.create(
    model="/Users/max/codebase/rsi/ai-drama/models/flint-mixed-3-4bit",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=1536, temperature=1.0, top_p=0.95)
print(r.choices[0].message.content, r.choices[0].finish_reason)
```

### 服务端比 CLI 多出来的三条（实测，不是读 `--help` 得来的）

1. **思考块单独分离**。服务端把 `<think>…</think>` 放进 `reasoning_content`，
   `content` 是干净正文——CLI 模式必须自己剥。这一条上服务端严格更好。
2. **`max_tokens` 把思考也计入**。预算 300 实测得到 `content: None` +
   `finish_reason: length`；1536 才能稳定出正文。客户端必须知道给足。
3. **鉴权是真的**：错误 key 与缺 key 都是 401，本机实测。

服务端 decode **50.7 tok/s**（CLI 47.7），峰值内存 `timings.peak_memory`
17.9157 GB，与 CLI 的 17.953 GB 一致。流式 `stream=True` 可用，
`finish_reason` 与 `usage` 计数精确，客户端能真正校验完成。

### decode 为什么慢：MoE kernel，不是带宽

先量这台机器的真实可达带宽：MLX 对 3 GiB fp32 求和，实测 **165.2 GB/s**
（M1 Pro 官方规格 200 GB/s）。

再算每 token 的访存（专家权重按真实 shape，`[256, 512, 192]` U32 packed，
而不是按 4-bit 全量估）：

| 组成 | 每 token |
|---|---|
| attention（49 层，q 4-bit / kv 8-bit / o 4-bit） | 490 MiB |
| embed + lm_head（vocab 150144 × 2048，8-bit，未 tie） | 587 MiB |
| shared expert（49 × 3.0 MiB，无条件） | 147 MiB |
| router | 12 MiB |
| MoE 激活专家（6/256，3-bit） | 110 MiB |
| **合计** | **1.35 GiB** |

按 165 GB/s 推算上限 **123 tok/s**，而实测 50.7 tok/s——**只用上 41% 的带宽**。
所以慢不在带宽，在 kernel。

逐层 profile 证实了这一点（`/tmp/prof2.py`，1 token decode）：

```
full decode:    28.83 ms/token -> 35 tok/s
MoE block:       0.50 ms -> x49 = 24.6 ms  (85% of decode)
lm_head:         1.98 ms  (7%)
```

**MoE 吃掉 85% 的时间**。而它的访存只占 8%，说明 MLX 的 `gather_mm`
在这个 shape（256 专家 / 3-bit / in 2048）上是**调度受限而非带宽受限**。
决定性证据是缩放行为：

| 选中专家数 | SwitchGLU 耗时 |
|---|---|
| 1 | 1.853 ms |
| 2 | 1.704 ms |
| 6 | 1.930 ms |
| 16 | 1.929 ms |
| 64 | 2.532 ms |
| 256 | 4.614 ms |

**top-1 和 top-6 几乎一样快**。带宽受限的 kernel 应该按选中专家数线性增长；
实际只涨 4%，说明它读了整个 stacked tensor 或走了低效分发路径。

Batch 曲线证实了"固定开销"这个判断（同一 MoE block，x49 层折算）：

| batch | ms/token | 49 层合计 | 聚合吞吐 |
|---|---|---|---|
| 1 | 1.508 | 73.9 ms | 14 tok/s |
| 2 | 0.702 | 34.4 ms | 58 tok/s |
| 4 | 0.393 | 19.3 ms | 208 tok/s |
| 8 | 0.239 | 11.7 ms | 683 tok/s |
| 16 | 0.146 | 7.2 ms | 2229 tok/s |

batch 从 1 到 32，kernel 总时间只从 1.5 ms 涨到 3.4 ms。
**单请求慢是因为付了全额固定开销；并发请求能摊薄它。**
服务端默认 continuous batching（`--max-num-seqs` 无上限），
所以这一点对多客户端已经生效，对单请求无能为力。

对照 llama.cpp 的 Metal 后端：`MUL_MAT_ID` 有专用的 `kernel_mul_mm_id_map0`，
按 (expert, token) 对生成 tile 描述符、只发射实际需要的 tile
（`ggml_metal_op_mul_mat_id_extra_tpe` / `_extra_ids` 就是为它准备的），
这正是 MoE decode 该走的路径。

### 换引擎的容量账：为什么选 unsloth UD-Q3_K_XL

M1 Pro 的 `max_recommended_working_set_size` 是 **22906503168 B = 21.33 GiB**，
不是统一内存的 32 GB。任何方案都必须塞进这个数：

| 权重 | 大小 | 扣权重后余量 | 128K 上下文可行？ |
|---|---|---|---|
| RecursiveSI Q4_1 | 24.2 GiB | **-2.87 GiB** | 否，装不进 |
| unsloth UD-Q4_K_XL | 23.33 GiB | -2.00 GiB | 否 |
| **unsloth UD-Q3_K_XL** | **17.22 GiB** | **+4.11 GiB** | **是**（KV 3.06 GiB） |
| unsloth UD-IQ2_XXS | 11.04 GiB | +10.29 GiB | 是（262K 全开） |

KV 每 token = 49 层 × 2(K,V) × 4 KV 头 × 128 head_dim × 0.5 B（8-bit KV）
= **24.50 KiB/token**，llama.cpp 按 `-c` 全量预分配。

Recursive Q4_1 超出上限 2.87 GiB，只能用 `--cpu-moe` 把 MoE 专家放 CPU——
那正好是占 85% 时间的部分，等于把最热的代码挪到最冷的设备。
所以选 UD-Q3_K_XL：整模进 GPU，余 4.11 GiB 给 KV，且 UD 动态量化质量优于同尺寸 Q4。

**unsloth 的 GGUF 能用上游 llama.cpp 加载**（这里更正一个常见误解）。
unsloth 的 `iq1-narrow` 分支确实没有 Flint，但它另有 `model-arch/flint` 分支
（`08f367a2`），其中 `src/models/flint.cpp` 与上游 master **逐字节相同**，
`LLM_ARCH_FLINT` 注册齐全。它相对上游只多一个 vocab 提交，把
`LLAMA_VOCAB_PRE_TYPE_FLINT` 从 62 改成 63；而该枚举值不写进 GGUF，
加载时从 `tokenizer.ggml.pre = "flint"` 字符串解析，因此上游加载无碍。

实测确认 unsloth UD-Q3_K_XL 的 metadata 与本地 safetensors 架构完全一致：
`block_count=49`、`embedding_length=2048`、`expert_count=256`、
`expert_used_count=6`、`head_count=32`、`head_count_kv=4`、
`key/value_length=128`、`sliding_window=513`、`leading_dense_block_count=2`、
`expert_shared_count=2`、`context_length=262144`。

HF 的 Xet 后端对单连接限速到 10–48 KB/s（重定向到
`us.aws.cdn.hf.co/xet-bridge-us`），此时必须分段并行下载。
32 段并行实测约 187 MB/s，比单连接快三个数量级；
`hf-mirror.com` 在该文件上直接连接超时，不要用。

### 慢的另一半原因：前缀缓存默认是关的

日志里 29,390 token 的 Agent 上下文 prefill 108.2s、首字 108.8s，`cached_tokens=0`
——每轮都在重算整个 system prompt。根因是 **mlx-vlm 的 `APC_ENABLED` 默认 `"0"`**，
自动前缀缓存不开。打开后同一前缀二轮实测：

| | prompt tokens | prefill | 首字 | 总耗时 |
|---|---|---|---|---|
| 关 APC | 12,389 | 37.95s（326.5 tok/s） | 41.0s | 42.88s |
| 开 APC | 12,389（cached 12,388） | **0.15s**（82,623 tok/s） | 0.45s | **2.38s** |

**18.1×**。所以 agent 类客户端必须确认前缀缓存是开的——
`serve_flint.sh` 已默认 `APC_ENABLED=1`（关掉对照：`APC_ENABLED=0 ./serve_flint.sh restart`）。
内存块池默认 2048×16 = 32K token，超出部分落磁盘层
（`~/.cache/mlx-vlm/apc`，默认上限 20GB）——**这才是 SSD 在生成阶段真正吃到优势的地方**：
读回 KV 远比重算便宜，且权重仍整体常驻（实测进程 phys_footprint 19GB），
不存在每 token 从盘上读权重那种会瞬间被 I/O 拖死的路径。

### 两个必踩的坑

- **`model` 必须是长绝对路径**。写 `"flint"` 之类别名会 400——服务端把未知名字
  当 Hugging Face repo id 去下载。这是服务端行为，仓库内不可控，照抄 `/v1/models`
  的 id 即可。
- **一个服务实例占 17.9 GB**。服务运行期间不可再起第二个模型进程：
  两个就是 35.8 GB，超过 21.33 GiB 工作集，直接 OOM。

---

## 与 H3 管线的同构性

Edge0 的思路值得记一笔，因为它和本仓库攻的是同一堵墙。

Edge0 让 35B MoE 只常驻约 2.5GB RAM：权重留在 SSD 上流式读入，
配一个**训练过的 router** 提前预测下一步需要哪些专家。H3 那边的瓶颈不是显存而是
DiT token 总数（4480 上限），但它同样面对"装不下就流着读"——`!ssd-streaming on`
让 BF16 权重走 SSD 流式而非整体载入。

`pipeline/core/` 的 `EngineAdapter` 协议（`cwd` / `binary` / `extra_args` /
`is_complete` / `judge`）正是为这种异构引擎留的扩展点。
Flint 现在已经是 `pipeline/models/flint/` 的正式 adapter——**接入过程中改了四处协议**，
见上面「接入框架」一节。抽象不是写完就成立的，是第二个实现逼出来的。

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
./.venv/bin/python -m mlx_vlm.generate \
  --model models/flint-mixed-3-4bit \
  --prompt "Write a Python function that merges overlapping intervals." \
  --max-tokens 512 --temperature 1.0 --top-p 0.95 --top-k 20 --verbose
```

采样参数用 Recursive 的推荐值（temp 1.0 / top-p 0.95 / top-k 20）。
`--verbose` 是唯一能拿到 MLX 内部 tok/s 与峰值内存的方式：

```
Prompt: 11 tokens, 54.365 tokens-per-sec
Generation: 512 tokens, 47.737 tokens-per-sec
Peak memory: 17.953 GB
```

注意 CLI 的几个真实参数名（猜错会直接报错）：没有 `--extra-eos-token`
（是 `--eos-tokens`），`--verbose` 是开关对（关闭用 `--no-verbose` 而非
`--verbose False`），`--output` 只对图像/视频生效、文本走 stdout。

---

## 已知风险

1. ~~M1 Pro 无 Metal 4 会不兼容~~ —— **已证伪**：跑通了，decode 47.7 tok/s。没有 megakernel 相关的报错。
2. ~~官方 125 tok/s 不可直接预期~~ —— **已实测**：47.7 tok/s，约 M4 Pro 的 38%。
3. ~~19.5GB vs 21.3GB，余量 1.8GB~~ —— **已实测**：17.95 GB，余 4.6 GiB，不必关重应用。
4. **思考链吃预算**（仍然有效）：默认就要 1500 token 起步，`max_tokens` 给小了会
   得到一份纯思考、正文为空、**返回码 0** 的输出。
5. **满上下文装不下**：262144 上下文峰值 21.53 GiB，超过 21.33 GiB 工作集。
   实用上限约 131K（19.78 GiB）。
6. **`APC_ENABLED` 默认关闭**（mlx-vlm 上游默认值）：不开的话每个请求都全量
   prefill，长 system prompt 首字延迟按分钟计。`serve_flint.sh` 已默认打开。
6. **tool call 参数名错误**：mlx-vlm 0.7.10 已知问题，作者测得 OpenCode 测试中
   85 次调用有 41 次参数名错误（如 `path` 而非 `filePath`），但任务最终仍全部通过。
   修复在 [mlx-vlm#2571](https://github.com/Blaizzy/mlx-vlm/pull/2571)。
   **本仓库的 adapter 不做 tool call**，所以不受影响。
7. **`required_sections` 只查存在性，不查结构**：实测有一例把五个标签挤在一行、
   正文不分区，仍然通过。要更严需要按 shot 写正则，目前没做。

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
