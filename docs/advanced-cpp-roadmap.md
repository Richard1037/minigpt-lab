# 本地小规模 Transformer 大模型：从零实现指导手册

> 目标：在一台笔记本 CPU 上，用 **C++（主实现）+ Python（参考实现与交叉验证）** 手写一个能自回归生成文本的小型语言模型，
> 并真正跑通「分词 → 预训练 → 检查点 → 采样生成」全链路。
>
> 适用机器（本手册按此配置校准）：Intel Core Ultra X7 358H / 16 逻辑核 / 32GB 内存 / 无独立 GPU / 支持 AVX2+FMA，**不支持 AVX-512**。
> 工具链：MSYS2 UCRT64 (g++ 16.2)、CMake 4.4.2、Ninja 1.13.2、Python 3.14.7 + numpy 2.5.2。

---

## 0. 先把预期对准（重要）

「初步实现大模型功能」在这台机器上能达到的真实位置（下表时长按**实测 120~180 GFLOPS 有效吞吐**算，
用 `FLOPs ≈ 6·N·tokens` 推得；先按 §2.4 量出你自己的 GFLOPS 再套）：

| 参数量 | 训练量 | 训练时长 | 能做到的事 |
|---|---|---|---|
| 3M | 50M tok（字符级） | 2~4 小时 | 莎士比亚台词片段：语法大致正确，词经常不存在 |
| 17M | 0.4B tok ≈ 1 epoch | 2.6~3.9 天 | 连贯的短故事（3~5 句），人物偶尔错乱 —— **本手册的推荐主线** |
| 29M | 0.4B tok ≈ 1 epoch | 4.4~6.6 天 | 更稳的 TinyStories 水平，但已接近"一次实验等一周"的上限 |
| 37M | 1.2B tok ≈ 3 epoch | 17~25 天 | 不推荐：多 epoch 的边际收益远小于这段时间成本 |

> 时长用 `FLOPs ≈ 6·N·tokens` 反推，`tokens` 上限受语料决定：完整 TinyStories 用 8192 BPE 编码后约 **0.4B token，即 1 个 epoch**。
> 想跑第 2、3 个 epoch，时间就线性翻倍——这是本项目最硬的一道墙，且与代码质量无关。

**不能做到的**：知识问答、指令跟随、长上下文、中文流畅输出。这些需要 10B+ token 和几十倍的算力/内存，
不是代码质量问题，是预算问题。把这当作"理解大模型 internals"的项目，而不是"造一个能用的助手"。

推荐主线目标：**16.8M 参数 + TinyStories + 1 个完整 epoch（0.4B token）**，约 3 天挂机换一次真正跑通的完整训练。
29M 那一档作为"如果第一次很顺利"的备选；先不要想多 epoch。

> 注意：0.4B token 对 17M 参数远低于 Chinchilla 最优（336M），TinyStories 原论文更是跑了约 10 个 epoch。
> 也就是说你**注定欠训练**——这是 CPU 预算下的正常状态，不是 bug。val loss 到平台就停，接受它，
> 把成果当作"理解了机制"的证据，而不是"一个可用的模型"。

---

## 1. 能力路线图：M0 → M7

每个里程碑都有可运行的程序和明确的验收标准。不要跳级。

| 里程碑 | 交付物 | 验收标准 | 预计耗时 |
|---|---|---|---|
| **M0** | 环境 + 构建骨架 + BLAS 基准 | `cmake --build` 出可执行；测得本机 GEMM 吞吐 | 半天 |
| **M1** | Tensor 库（形状/广播/归约/softmax/AdamW） | 与 numpy 逐元素误差 < 1e-6 | 2~4 天 |
| **M2** | 分词器（字符级 → BPE） | 编码→解码 100% 还原语料样本 | 1~3 天 |
| **M3** | 注意力 + 因果掩码 + FFN + Block | 与 numpy 参考实现前向误差 < 1e-5 | 3~5 天 |
| **M4** | 完整模型 + 检查点格式 | 同一权重 C++/Python 推理结果逐 token 一致 | 2~4 天 |
| **M5** | 训练循环 + 优化器 + 调度 | TinyShakespeare loss 从 4.5 → **1.85** 以下 | 1~3 天（含挂机） |
| **M6** | 推理 + 采样（温度/top-k/top-p）+ KV cache | 生成 200 token 不重复、无乱码、速度 ≥150 tok/s | 2~3 天 |
| **M7** | 性能优化（OpenMP/SIMD/内存/三角掩码） | 有效 GFLOPS 相对 naive 循环基线 ≥3×（实测方法见 §2.4） | 5~10 天（可持续） |

---

## 2. M0：环境准备

### 2.1 装 OpenBLAS（这一步能给你 5~20 倍的矩阵乘速度）

在 **"MSYS2 UCRT64"** 终端里执行（不是 Git Bash）：

```bash
pacman -S --noconfirm mingw-w64-ucrt-x86_64-openblas mingw-w64-ucrt-x86_64-dlfcn
ls /ucrt64/lib/libopenblas*                      # 应看到 libopenblas.a 与 bin/libopenblas.dll
# 链接自测：能编出 a.exe 就说明 header/库/链接路径都通了
printf '#include <cblas.h>\nint main(){float a[4]={1,2,3,4},b[4]={5,6,7,8};return (int)cblas_sdot(4,a,b)-30;}\n' > /tmp/t.c
gcc /tmp/t.c -o /tmp/t.exe -I/ucrt64/include -L/ucrt64/lib -lopenblas && echo LINK_OK
```

**为什么必须做**：自己写的嵌套循环 GEMM 在这台机器上大概只有 15~40 GFLOPS，OpenBLAS 的 AVX2 kernel 能到 150~250 GFLOPS 峰值。
训练时长直接从"两周"变成"两天"。

### 2.2 目录结构（C++/Python 双实现共仓）

```
D:/Transformer/
├── CMakeLists.txt
├── src/
│   ├── core/    tensor.cpp tensor.hpp dtype.hpp shape.hpp arena.hpp
│   ├── ops/     matmul.cpp softmax.cpp layernorm.cpp rope.cpp adamw.cpp
│   ├── model/   config.hpp attention.cpp ffn.cpp block.cpp gpt.cpp checkpoint.cpp
│   ├── token/   char_tokenizer.cpp bpe.cpp vocab.cpp
│   ├── train/   dataset.cpp trainer.cpp lr_schedule.cpp
│   ├── infer/   generate.cpp sampler.cpp kv_cache.cpp
│   └── tools/   bench_gemm.cpp train_main.cpp infer_main.cpp convert_ckpt.cpp
├── py/                       # 参考实现，永远比 C++ 先写
│   ├── ref_tensor.py  ref_model.py  ref_tokenizer.py
│   ├── make_golden.py     # 生成"金标准"张量与权重
│   ├── verify.py          # 读 C++ 输出与 numpy 对比，报最大绝对误差
│   └── train_toy.py       # 用 numpy 复现 M5 的小实验（可选）
├── data/          tinystories_min.txt  shakespeare.txt
├── ckpt/          step_000500.bin  vocab.json
├── scripts/       estimate.py  measure_tokens_per_sec.sh
└── tools/         tiny_shakespeare_char.pt （karpathy nanoGPT 数据集，可选）
```

### 2.3 CMakeLists.txt（UCRT64 起点配置）

```cmake
cmake_minimum_required(VERSION 3.25)
project(mini_gpt CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)          # 必须在 link OpenMP::OpenMP_CXX 之前
add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native -funroll-loops)

find_library(OPENBLAS_LIB NAMES openblas libopenblas HINTS C:/msys64/ucrt64/lib)
if(NOT OPENBLAS_LIB)
  message(FATAL_ERROR "OpenBLAS not found. Run: pacman -S mingw-w64-ucrt-x86_64-openblas")
endif()

file(GLOB_RECURSE CORE_SRC CONFIGURE_DEPENDS src/core/*.cpp src/ops/*.cpp src/model/*.cpp
                                        src/token/*.cpp src/train/*.cpp src/infer/*.cpp)

add_library(minigpt STATIC ${CORE_SRC})
target_include_directories(minigpt PUBLIC src)
target_link_libraries(minigpt PUBLIC ${OPENBLAS_LIB} OpenMP::OpenMP_CXX)

add_executable(bench_gemm  src/tools/bench_gemm.cpp)
add_executable(train       src/tools/train_main.cpp)
add_executable(infer       src/tools/infer_main.cpp)
target_link_libraries(train  PRIVATE minigpt)
target_link_libraries(infer  PRIVATE minigpt)
target_link_libraries(bench_gemm PRIVATE minigpt)
```

配置与构建（在 UCRT64 终端，路径用 `/d/Transformer`）：

```bash
cd /d/Transformer
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bench_gemm
```

> ⚠️ 编译参数注意：这台 CPU 没有 AVX-512，`-march=native` 会落到 AVX2+FMA，是正确的。
> 若某天迁移到支持 AVX-512 的机器，不要用 `-march=knl` 之类的强制 flag，仍用 native。

### 2.4 先测出你的真实算力（决定了后面所有决策）

`src/tools/bench_gemm.cpp` 里对 `4096×4096×4096` 做 `cblas_sgemm`，然后：

```bash
OMP_NUM_THREADS=16 ./build/bench_gemm
```

把结果（GFLOPS、tokens/sec）填进 `scripts/estimate.py`：

```python
# scripts/estimate.py —— 训练预算估算器（Python 3.14，仅标准库）
# 用法：把 bench_gemm 实测的有效 GFLOPS 填到 GFLOPS，直接读出"要挂几天"。
def n_params(L, d, V):
    return 12 * L * d * d + V * d          # 4d² attention + 8d² SwiGLU FFN，输出 embedding tied

def days(tokens, N, gflops):
    return tokens * 1e6 * 6 * N / (gflops * 1e9) / 86400

# 主线配置（见 §5）：6 层 / d=576 / V=8192 ≈ 28.6M 参数
MODEL = dict(L=6, d=576, heads=9, ctx=512, V=8192)
N = n_params(MODEL["L"], MODEL["d"], MODEL["V"])
GFLOPS = 150    # ← 换成你 bench_gemm 在真实训练 batch 形状下测到的数字

print(f"params = {N/1e6:.1f}M   chinchilla = {20*N/1e6:.0f}M tokens")
print(f"weights+grads+Adam = {N*16/1e6:.0f} MB   (param+grad+2 moment × 4B)")
for tok_m in (50, 200, 400, 600):
    print(f"{tok_m:>4}M tokens -> {days(tok_m, N, GFLOPS):5.1f} days @ {GFLOPS} GFLOPS")
    # 反向核对：想要 3 天出结果，能跑多少 token？
print(f"3 天能跑 {3*86400*GFLOPS*1e9/(6*N)/1e6:.0f}M tokens")
```

**如何标定 GFLOPS**（比任何理论估算都可靠）：

1. `bench_gemm` 的大方阵 GEMM 给你的是**峰值**，通常比训练时有效吞吐高 1.5~3×（训练的 M 维小得多）。
2. 真实数字这样拿：用 `tiny` 配置跑 200 步，记录 `steps/s`，则
   `GFLOPS_effective = steps/s × batch_tokens × 6 × N / 1e9`。再换 `mid` 配置测一次，看吞吐随 d 的衰减趋势。
3. 把这个数填进 `estimate.py`，它决定你选 `quick`(16.8M) 还是 `main`(28.6M)，以及敢不敢碰第 2 个 epoch。**先量再训**，
   否则你会在第 5 天发现原计划"2~3 天"其实是 9 天。

---

## 3. M1：Tensor 库（地基，别急着跳过）

### 3.1 设计约束（这些选择是为 CPU 服务的）

1. **统一 fp32 存储**，不用 fp16 训练。CPU 上没有 tensor core，fp16 只省内存不省时间，反而增加转换开销。
2. **`std::pmr::monotonic_buffer_resource` 做前向 arena**：每个 training step 复用同一块内存，避免百万次 malloc。
3. **反向用显式反向函数**（手动写 grad），不做 tape 自动微分。层数固定、结构已知，手写反向更短更快，而且逼你真正理解梯度流。
4. **行主序 (row-major) 连续存储**。所有 reshape 尽量做成 no-op view；转置只在 GEMM 里用 `cblas_strans` 表达，不落内存。
5. **张量只是 view + 指针**，所有权在 arena/上层。拷贝只在跨步/跨检查点时发生。

### 3.2 核心类型（C++）

```cpp
// src/core/tensor.hpp
#pragma once
#include <cstdint>
#include <span>
#include <vector>
#include <memory_resource>

enum class DType { F32 };

struct Tensor {
  float*            data = nullptr;      // 非拥有；拥有权见 Storage
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;           // 元素个数为单位
  DType             dtype = DType::F32;

  int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
  std::span<float> flat() { return {data, (size_t)numel()}; }

  Tensor view(std::vector<int64_t> new_shape) const;      // 只改 shape/stride
  Tensor transpose(int64_t a, int64_t b) const;           // 只交换 stride
  Tensor slice(int64_t dim, int64_t from, int64_t to) const;
};

struct Storage {                                         // 拥有内存
  std::pmr::vector<float> buf;
  explicit Storage(int64_t n, std::pmr::memory_resource* r) : buf(n, 0.f, r) {}
  Tensor tensor(std::vector<int64_t> shape) { return {buf.data(), std::move(shape), {}, DType::F32}; }
};

// 逐元素/归约算子（每个都在 src/ops/ 下，配套一个 .cpp 的 naive 版本供对拍）
void add(Tensor out, Tensor a, Tensor b);
void mul_elementwise(Tensor out, Tensor a, Tensor b);
void softmax_inplace_rows(Tensor t, float scale);        // 每行独立 softmax
void layernorm_fwd(Tensor out, Tensor in, Tensor w, Tensor b, float eps);
void rmsnorm_fwd(Tensor out, Tensor in, Tensor w, float eps);
void adamw_step(Tensor param, Tensor grad, Tensor m, Tensor v,
               float lr, float b1, float b2, float wd, float eps, int64_t step);
```

`stride` 为空表示按 shape 紧凑行主序；`view`/`transpose` 只动元数据，这是 CPU 性能的关键。

### 3.3 对拍策略（M1 的真正验收）

`py/make_golden.py` 生成随机张量与算子输入输出，C++ 读进去跑 naive 版本，再和 numpy 比：

```python
# py/make_golden.py
import numpy as np, json, pathlib
rng = np.random.default_rng(1234)
out = pathlib.Path("tools/golden"); out.mkdir(parents=True, exist_ok=True)

cases = []
for op, shape in [("softmax_rows", (16, 32)), ("layernorm", (64, 128)),
                  ("rmsnorm", (64, 128)), ("gelu", (32, 256))]:
    x = rng.standard_normal(shape, dtype=np.float32)
    y = softmax_rows(x) if op == "softmax_rows" else eval(op)(x)   # numpy 版算子
    np.save(out / f"{op}_x.npy", x); np.save(out / f"{op}_y.npy", y)
    cases.append({"op": op, "shape": list(shape)})
json.dump(cases, open(out / "cases.json", "w"), indent=2)
```

```cpp
// src/tools/check_golden.cpp —— 断言最大绝对误差
double max_abs_err(const float* a, const float* b, int64_t n) {
  double e = 0; for (int64_t i = 0; i < n; ++i) e = std::max(e, (double)std::abs(a[i]-b[i]));
  return e;
}
// 期望：softmax/layernorm/rmsnorm < 1e-6，gelu(tanh 近似) < 5e-7
```

> **纪律**：任何一个算子，先用 numpy 写一遍（10 分钟），再生成 golden，再写 C++，再对拍。
> 顺序颠倒你会在三天后对着一个未知 bug 无还手之力。

---

## 4. M2：分词器

### 4.1 分两步走，不要一上来就 BPE

| 阶段 | 词表 | 用途 | 备注 |
|---|---|---|---|
| **字符级 (char)** | 64~260 | M5 的 TinyShakespeare 实验 | 训练/推理都简单，no vocab file，先用它把全链路打通 |
| **BPE** | 8192（主线）/ 4096（省时间） | 正式训练 TinyStories | vocab 越大，embedding 与输出头越大；20~30M 模型在 4k~8k 之间最划算 |

字符级是"低风险跑通训练管线"的最短路径。**必须**先跑通它，你会在几小时内看到 loss 下降和真实文本输出，
这对坚持下去极其重要。

### 4.2 词表大小对参数量的影响（决定 BPE 尺寸）

用 weight tying（输入 embedding 和输出投影共享权重，强烈建议）：
`embedding 参数 = vocab × d_model`。主线配置 8192 × 576 = **4.7M**，占 28.6M 的 16%，划算。
真实 LLM 的词表 128256 × 576 = 74M，会超过你整个模型的预算——**这就是为什么小模型必须用小词表**。

### 4.3 BPE 实现要点（C++，约 300 行）

```text
算法（标准 byte-pair-encoding，GPT-2 简化版）：
1. 以字节/Unicode 单字符为初始符号，统计语料中每个词（按 \s+ 和标点切）的字符序列频次。
2. 重复 vocab_target - 256 次：找出频次最高的相邻符号对 (a,b)，合并为新符号，更新计数。
3. 保存 merges 表（有序对列表）；编码时按 merges 顺序依次应用。
4. 特殊 token：<|eos|> = 0。不做 byte fallback，遇未知字符替换成 '<unk>'（语料是 ASCII，无影响）。
```

- 训练语料只喂 **前 100MB** 给 BPE 学习即可，20 分钟以内完成（用 `unordered_map<uint64_t,int>` 存 pair 计数，key = `(a<<32)|b`）。
- **必须验证**：随机取 10000 条样本，`decode(encode(text)) == text` 要 100% 成立，否则有 off-by-one。
- Python 侧用 `tiktoken`（可 `pip install tiktoken`）或对同一 merges 表写一份 numpy-free 实现，做交叉校验。

### 4.4 数据集打包（M5 之前）

预训练时**不要**逐条 story 喂，会把短故事边界浪费掉。做法：整个语料一次性编码成 `uint32` 大数组
（TinyStories ~1G token，4GB 内存文件，32GB 机器完全放得下，直接 mmap）：

```cpp
// 每个 batch 从大数组里随机取一段 [start, start+ctx+1]
// target = input 右移一位（next-token prediction）
// start 用 xorshift RNG 保证可复现；跨 story 拼接是可接受的，TinyStories 官方就这么做
```

---

## 5. 模型配置：直接抄这张表

参数量按 `params = 12·L·d² + V·d`（每层 4d² 的 qkvo + 8d² 的 SwiGLU FFN；输出头与输入 embedding tied，
故 `V·d` 只计一次）。下表与 §0 的时长都由这条式子算出，`FLOPs ≈ 6·N·tokens`，时长列取 120~180 GFLOPS 有效吞吐。

| 配置 | L | d | heads | head_dim | ctx | vocab | 参数量 | 优化器内存 | 跑 1 epoch 时长 |
|---|---|---|---|---|---|---|---|---|---|
| `tiny`（调试） | 2 | 128 | 4 | 32 | 128 | 260 | 0.4M | 7MB | 按 30M tok：<1 小时 |
| `small`（M5 字符级） | 4 | 256 | 4 | 64 | 256 | 65 | 3.2M | 51MB | 按 50M tok：2~4 小时 |
| **`quick`（主线）** | 4 | 512 | 8 | 64 | 512 | 8192 | 16.8M | 268MB | 0.4B tok：**2.6~3.9 天** |
| `main`（备选） | 6 | 576 | 9 | 64 | 512 | 8192 | 28.6M | 458MB | 0.4B tok：4.4~6.6 天 |
| `stretch`（不建议） | 8 | 576 | 9 | 64 | 512 | 8192 | 36.6M | 585MB | 0.4B tok：5.6~8.5 天 |

1 epoch = 完整 TinyStories 编码后的约 0.4B token；时长列按实测 120~180 GFLOPS 有效吞吐给出。

`heads` 必须整除 `d`。SwiGLU 中间层取 `round_up(8/3·d, 64)`：384 / 704 / 1408 / 1536 / 1536。
优化器内存 = `16 字节 × 参数量`（param + grad + Adam 两个动量，fp32）；激活另占约 1~3×，32GB 机器上 `main` 总占用约 1~2GB，很宽松——
**内存不是你的瓶颈，时间才是**，所以选配置时优先看时长列。

超参（`quick` 与 `main` 共用；`quick` 的 lr 可提到 4e-4，d_model 更小可以更快收敛）：

```
optimizer   AdamW, β1=0.9, β2=0.95, ε=1e-8, weight_decay=0.1（只对 2D 权重，bias/norm 不加）
lr          peak 3e-4；d_model ≤256 时用 6e-4
schedule    linear warmup 400 步 → cosine 衰减到 0.1×peak
batch       tokens_per_step = 512×64 = 32768（micro-batch 64×512，累积 1 次；内存吃紧就 32×512 累积 2 次）
steps       1 epoch = 0.4B token / 32768 ≈ 12000 步；lr 调度按这个总步数设，中途提前停就按实际步数重算 cosine
grad_clip   global L2 norm 1.0
init        N(0, 0.02)；残差出口投影额外乘 1/sqrt(2·L)（GPT-2 做法）
dtype       fp32；loss 用 sum/count 平均，忽略 padding（本设计无 padding）
```

---

## 6. M3–M4：模型结构

### 6.1 架构决策表（每个决定都有理由，别盲从）

| 选择 | 推荐 | 理由（本地 CPU / 小规模语境） |
|---|---|---|
| LayerNorm 位置 | **Pre-norm** | 训练稳定，小模型无损 |
| Norm 类型 | **RMSNorm** | 少一次均值归约，CPU 上快 ~15%，效果等价 |
| 位置编码 | **RoPE** | 无额外参数，长序列外推好；比正弦/learned PE 都稳 |
| FFN 激活 | **SwiGLU**（hidden = 8/3·d，对齐到 64） | 同样参数下质量略优；若嫌复杂先用 GELU+tanh 近似 |
| Attention bias | **无 bias**（仅 qkv 投影） | 现代配置，省参数 |
| 输出 embedding | **与输入 tied** | 小词表下省 4%，且小模型收敛更快 |
| Dropout | **0.0** | 数据量足够时不需要；CPU 上省一半内存带宽 |
| 残差缩放 | 每层加法即 `x = x + attn(x)`，无额外系数 | 简单；靠 init 缩放保稳定 |
| Tie + 缩放 | 输出头不额外缩放 | 保持与 embedding 同尺度 |

### 6.2 前向（伪代码 = 你的 C++ 类结构）

```
h = embed[input_ids] * 1.0
for l in 0..L-1:
    r = h
    x = rmsnorm(h, wln1[l], eps=1e-5)
    q = x @ Wq[l]        # (B,T) -> (B,T,d)
    k = x @ Wk[l]
    v = x @ Wv[l]
    split_heads → (B, H, T, hd)
    rope(q), rope(k)     # 只作用 q,k，不作用 v
    att = softmax( q @ kᵀ / sqrt(hd)  +  causal_mask )     # ← 掩码必须在 softmax 前
    o = att @ v                                                 # (B,H,T,hd)
    merge_heads → (B,T,d)
    o = o @ Wo[l]
    h = r + o
    r = h
    x = rmsnorm(h, wln2[l], eps=1e-5)
    h = r + ( swiglu(x, Wu[l], Wg[l], Wd[l]) )
h = rmsnorm(h, wln_f, eps=1e-5)
logits = h @ embedᵀ                                             # tied
loss   = cross_entropy(logits[:, :-1], input_ids[:, 1:])
```

### 6.3 RoPE 的实现细节（最容易写错的一处）

```cpp
// 半切（GPT-NeoX 风格），half = head_dim/2
// 每个 2D 旋转对 (x0, x1)：
//   out0 = x0*cosθ - x1*sinθ
//   out1 = x0*sinθ + x1*cosθ
// θ_j = pos * inv_freq[j],  inv_freq[j] = 1 / base^(2j/head_dim),  base = 10000
// 关键：q 和 k 必须用同一套 inv_freq；cos/sin 表预计算到 (max_seq, half)，别每次调用现算。
```

**Python 等价（用于对拍）**：

```python
def apply_rope(x, cos, sin):          # x: (B,H,T,hd)
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c, s = cos[None, None], sin[None, None]      # (1,1,T,h)
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)
```

### 6.4 检查点格式（自定义二进制，跨语言零依赖）

```
[  0, 4)  magic "MGPT"
[  4, 8)  uint32 version = 1
[  8,12)  uint32 n_config_json_bytes
[ 12, ...) config_json (UTF-8)  —— {"d_model":512,"n_layer":8,"n_head":8,"ctx":512,"vocab":4096,"swiglu_hidden":1408}
之后按张量列表重复：
   uint32 name_len | char[] name | uint32 ndim | int64[ndim] shape | fp32 data (little-endian, 连续行主序)
命名固定：wln1.{l}  wq.{l}  wk.{l}  wv.{l}  wo.{l}  wln2.{l}  wu.{l}  wg.{l}  wd.{l}  wln_f  embed
（embed 只存一份，tied 输出头在加载时复用）
```

**必须做**：Python 和 C++ 各写一个读写器，`py/verify.py` 读 C++ 存的 ckpt，跑前向，比 logits。
这是 M4 的验收标准，也是之后所有 bug 的定位基准。

```python
def read_ckpt(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"MGPT"
        ver = np.frombuffer(f.read(4), np.uint32)[0]
        n = np.frombuffer(f.read(4), np.uint32)[0]
        cfg = json.loads(f.read(n))
        tensors = {}
        while True:
            head = f.read(4)
            if not head: break
            name = f.read(np.frombuffer(head, np.uint32)[0]).decode()
            nd = np.frombuffer(f.read(4), np.uint32)[0]
            shape = list(np.frombuffer(f.read(8*nd), np.int64))
            cnt = int(np.prod(shape))
            tensors[name] = np.frombuffer(f.read(4*cnt), np.float32).reshape(shape).copy()
    return cfg, tensors
```

---

## 7. M5：训练循环

### 7.1 顺序（严格照做，每步都能验证）

```
1. 在 tiny(0.4M) + 随机数据上，loss 应在 50 步内从 ln(V) 显著下降 → 证明反向与优化器正确。
2. 【过拟合测试】固定一个 batch（64 条），训 2000 步，loss 必须 → 0.02 以下。
   过不到拟合 = 反向有 bug，不要继续。这是 karpathy 强调的最重要单步诊断。
3. 换 TinyShakespeare 字符级 (V≈65) + small 配置，loss 目标 1.85，约 3~8 小时。
4. 检查生成文本：应出现类似莎士比亚的行句式、正确的换行与大写分布。
5. 再上 BPE + TinyStories + main 配置。
```

**第 2 步（loss=0 测试）能抓住 90% 的实现 bug**：掩码方向反了、梯度少乘了转置、embedding 索引 off-by-one、
AdamW 的 bias correction 漏了 —— 全都会在这里暴露。

### 7.2 训练主循环骨架

```cpp
// src/train/trainer.cpp
for (int64_t step = 0; step < max_steps; ++step) {
  float lr = lr_schedule(step, warmup, max_steps, peak_lr);
  arena.reset();                                   // 单调内存池，每步回收
  double loss_sum = 0; int64_t tokens = 0;
  for (int mb = 0; mb < accum_steps; ++mb) {
    auto xb = dataset.next_batch(batch, ctx, rng); // (B,T) uint32
    model.forward(xb, /*train=*/true);
    loss_sum += model.loss; tokens += (int64_t)batch * ctx;
    model.backward();                              // 梯度累积进 .grad
  }
  double gn = clip_grad_norm(grads_view(), 1.0);   // 全局 L2，包含所有张量
  adamw_step_all(lr, weight_decay, step + 1);
  zero_grad(grads_view());
  if (step % eval_every == 0) evaluate_val_subset();   // 至少 200 batch，别用 1 batch 的噪声
  if (step % save_every == 0) save_ckpt(path, step, loss, lr);
}
```

### 7.3 训练监控与日志

日志文件 `ckpt/run_YYYYMMDD.jsonl`，每 eval 写一行：`{"step":2000,"loss":3.11,"val":3.08,"tok_s":2480,"lr":2.9e-4,"gnorm":0.71}`。
**必须记录** `tokens/sec` 和 `grad_norm`，前者是你优化 M7 的唯一反馈，后者是发散的先行指标。

发散信号与处理：

| 现象 | 原因（按概率排序） | 处理 |
|---|---|---|
| loss 变 NaN | 掩码漏了 / RMSNorm eps=0 / lr 太大 | 检查 mask 加在 softmax **前**；eps=1e-5；lr 砍半 |
| loss 平台期不降（char 级 >2.2） | vocab 太大 / lr 太小 / 容量不足 | 确认已用 weight tying；lr×1.5；加深而非加宽 |
| 早期 spike 后恢复 | warmup 太短 | warmup 400→800 步 |
| 输出全是重复同一 token | 训练步数不够 / temperature 太低 / label shift | 先做 loss=0 测试；采样加 `repetition_penalty` 或更多步数 |
| 生成乱码但 loss 正常 | ckpt 读写张量顺序错 / head split 内存布局错 | 用 §6.4 的 C++↔Python 对拍定位 |

### 7.4 数据

按顺序用这三档，不要跳过中间档：

1. **TinyShakespeare**（~1MB，`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`）— 分钟级验证管线。
2. **TinyStories 子集 100MB**（HuggingFace `roneneldan/TinyStories`，取前 20 万条）— 半天验证 BPE + `quick` 配置能不能收敛。
3. **完整 TinyStories**（约 2.1GB 文本 / 2.1M 条故事）— 正式训练。

**先量 token 再定步数**：编码完整语料后统计总 token 数（8192 词表的 BPE 大约得到 0.4B token = 1 个 epoch）。
主线就按 **1 个 epoch（约 12000 步）** 规划，因为它已经要挂 3 天。TinyStories 论文跑了约 10 个 epoch，
你跑不起——第 2 个 epoch 会让总时长翻倍。**这不是缺陷，是这台机器的边界，写进你的结论里。**
唯一例外：如果你只训练 `small` 那级的 3M 模型，10 个 epoch 也才 0.7 天，这时尽情重复。
**用 val loss 判断停点，别看 epoch 数。**

> ⚠️ 语料必须是**英文**。TinyStories 是为"极小模型也能学会连贯语言"专门设计的；古登堡/维基类语料在 50M 参数量下
> 只能学会拼词，学不会讲故事。中文语料同理，且需要 8192+ 词表与更大模型，先不要碰。

---

## 8. M6：推理与生成

### 8.1 KV cache（不实现它，你的生成速度是 1/50）

```
每个 layer 预分配 k_cache/v_cache: (B, H, max_seq, head_dim)，fp32。
prefill：一次处理 T 个 token，写入 cache[0..T)。
step  ：只算 1 个 token 的 q,k,v；把 (B,H,1,hd) 与 cache 的 (B,H,t,hd) 做 attention。
        → 每步的 QK^T 只有 1×t×hd FLOPs，而不是 T×T×hd。
RoPE   ：cache 存"已旋转后"的 k，或存原始 k 并按 pos 旋转 —— 二选一，别混用（混用会重复旋转）。
                                ↑ 这是本里程碑最常见的 bug
```

### 8.2 采样

```cpp
// temperature → top-k → top-p → multinomial（顺序固定）
float inv_T = 1.0f / temperature;              // 推荐 0.8
for (auto& x : logits) x = x * inv_T;
top_k_filter(logits, 40);
top_p_filter(logits, 0.9);                     // 累积概率截断
int next = sample_from(logits, rng);           // 用 alias/线性扫描即可，vocab 小
```

推荐默认值：`temperature=0.8, top_k=40, top_p=0.9`。加 `repetition_penalty=1.1`（对乘法作用于已出现 token 的 logits）。
`argmax`（temperature→0）在小模型上会循环，不要用。

### 8.3 CLI

```bash
./build/infer --ckpt ckpt/main_step_20000.bin --prompt "Once upon a time" \
              --max-tokens 300 --temp 0.8 --seed 7
```

---

## 9. M7：性能优化（按收益排序，逐条上，每条单独测吞吐）

| # | 手段 | 收益 | 难度 | 备注 |
|---|---|---|---|---|
| 1 | **OpenBLAS GEMM** 替换自写循环 | 5~20× | 低 | M0 已装，确保所有 matmul 走 `cblas_sgemm` |
| 2 | **OpenMP 并行非-GEMM 算子**（norm/rope/activation/add） | 4~8× | 低 | `#pragma omp parallel for`；注意 `OMP_PROC_BIND=spread` |
| 3 | **因果三角只算下三角**（QK^T 与 AV） | ~2× | 中 | 用 `cblas_sgemm` 按行分块只算 `j<=i`；小 ctx 时收益被 block 开销吃掉，ctx≥256 才划算 |
| 4 | **内存布局重排：(B,H,T,hd) → (B,T,H,hd)** | 1.2~1.8× | 中 | GEMM 直接输出连续 head 拼接，省 transpose |
| 5 | **batch 尽量大** | 1.3× | 低 | 提高 GEMM 的 M 维，摊薄访存 |
| 6 | **AVX2 手写 softmax/rmsnorm** | 1.2~1.5× | 中高 | `-mavx2 -mfma`；先测过再留 |
| 7 | **线程数调优**：`OMP_NUM_THREADS` = P 核数（本机 4~6）而非 16 | 1.1~1.4× | 低 | BLAS + OpenMP 超线程会互相抢；实测决定 |
| 8 | **arena 复用 & 避免临时分配** | 1.1~1.3× | 低 | M1 已设计好，检查热路径无 `new`/`push_back` |
| 9 | **fp16 存储 + fp32 累加**（仅推理） | 1.3~1.8× | 中 | `f16c` 有硬件转换；训练不要动 |
| 10 | 用 `sqrt_fast` / `rsqrt`（`_mm_rsqrt14_ps`）做 norm | 1.05× | 中 | 只在 profile 显示热点时用 |

> **NPU 提示**：这台机器的 NPU 是 Windows Copilot+ 栈（MLIR/DirectML），拿来做**从零反向传播训练**
> 在 2026 年仍然非常折腾。不要在这上面花时间；把 NPU 留给你自己实现完成后的**推理**实验。
> 真正想吃 CPU 硬件红利的话，路线是 §9 的 1~7 条。

---

## 10. 双语言分工（这才是双语的价值）

| 语言 | 负责 | 说明 |
|---|---|---|
| **Python 3.14 + numpy** | 参考实现、实验设计、golden 数据生成、可视化、分词器快速原型 | 不做性能优化；代码越"笨"越好，它是**规格说明书** |
| **C++23** | 正式实现、训练、推理 | 性能路径全部在这里 |

具体协作协议：

1. 每个新算子：**Python(numpy) 先写 → 生成 golden → C++ 实现 → 对拍 → 通过才进下一个**。
2. 每个新层（attention / block）：先用 numpy 跑通前向，把随机权重和中间激活全部存成 `.npy`；
   C++ 逐层加载比对，**报告第一个不匹配的张量名**（不要只比最终 logits）。
3. 完整模型对拍：同一 ckpt，同一输入，比 logits；`max_abs_err < 1e-4` 才算通过。
4. `main` 配置训练**只在 C++ 跑**。Python 只在两个场景用：`tiny` 配置快速实验、以及验证 C++ 的 loss 曲线形状一致。

---

## 11. 执行计划：4 周实现 + 挂机训练窗口

```
第 1 周
  D1  M0：装 OpenBLAS、搭 CMake、bench_gemm 出数字、跑 estimate.py 得到自己的有效 GFLOPS
  D2-3 M1：Tensor + 12 个算子，全部与 numpy 对拍通过
  D4  M2：字符级分词器 + 数据集打包 + mmap
  D5  M3：RMSNorm + RoPE + 因果 attention 前向（numpy 先行）
  D6-7 反向全部写完，通过 §7.1 步骤 2 的 loss=0 过拟合测试

第 2 周
  D8  M4：完整 GPT 模型 + 自定义 ckpt 格式 + C++↔Python logits 对拍
  D9-10 训练循环 + cosine/warmup + AdamW + grad clip；TinyShakespeare(small) 挂机
  D11-12 调 loss 到 1.85；实测有效 GFLOPS，回填 estimate.py，据此定主线预算
  D13-14 M6：采样生成 + 首次"看到模型讲故事"的文本，存档

第 3 周
  D15-16 BPE (8192) + TinyStories 100MB 子集 + tiny 配置，确认 loss 曲线形状正常
  D17-19 M7 优化 #1-#3（BLAS 全覆盖 / OpenMP / 三角掩码），每上一条都重测吞吐
  D20   先跑 100 步 smoke test 确认 quick 配置能跑、loss 在降，然后启动它挂机
        （16.8M × 1 epoch = 12000 步 → 2.6~3.9 天，横跨到第 4 周）
  D21-22 挂机期间：写 §8.1 KV cache + 推理 CLI（只读 ckpt，不碰训练进程）
  D23-24 挂机期间：M7 #4-#7 优化写代码 + 单测，但**不要重启训练去应用**

第 4 周末 ~ 第 5 周
  D25-26 quick 训完：评估 val loss，用固定 20 条 prompt 记样例与生成质量
  D27-28 采样调参（temperature / top-p / repetition_penalty），挑出最好的 10 段文本
  D29-30 写结论：把你实测的 GFLOPS、天数、loss 曲线、能力边界写清楚
  顺延   只有当 quick 明显"还想要一档质量"时，才启动 main(28.6M) → 再挂 4.4~6.6 天
```

**关键排期原则**：写代码的 2~3 周和挂机的 3 天是**串联**的，别按"四周搞定"规划；预留第 5 周。
真正省时间的做法是：用 `tiny`/`small` 把 90% 的管线 bug 在**第 1~2 周**暴露掉，
`quick` 那次 3 天挂机只负责"真正学语言"，不负责 debug。挂机前务必用 100 步 smoke test 确认配置能跑。

挂机训练的正确姿势（32GB 笔记本）：

```bash
# 先 smoke test：100 步确认配置/数据/内存没问题（几分钟）
OMP_NUM_THREADS=12 ./build/train --config configs/quick.json --steps 100 --save-every 100

# 再正式启动：12000 步 × 32768 token/步 ≈ 0.4B token = 1 个完整 epoch
OMP_NUM_THREADS=12 OMP_PROC_BIND=spread OMP_WAIT_POLICY=active \
  ./build/train --config configs/quick.json --data data/tiny_bpe.bin \
    --steps 12000 --save-every 500 2>&1 | tee logs/train_quick.txt
# Windows：电源计划设"高性能"、关闭睡眠与硬盘休眠；用 start /cmd /K 或 nohup 保活
# 每 500 步存 ckpt ≈ 30~50 分钟一存，写盘几百 MB，对训练速度的干扰可以忽略
```

---

## 12. 参考项目（读代码，不 fork）

| 项目 | 读它学什么 | 地址 |
|---|---|---|
| **nanoGPT** | 最干净的"能跑"训练循环与 ckpt 约定；§7.1 的 loss=0 测试来自 karpathy | github.com/karpathy/nanoGPT |
| **picoGPT** (keras-team) | ~60 行的纯 numpy 前向实现，§6.2 的最佳对照物 | github.com/karpathia/picoGPT |
| **llama2.c** | 纯 C 训练 + 推理，内存布局与 fused GEMM 的做法直接可抄 | github.com/karpathy/llama2.c |
| **nanoGPT.rs / minGPT** | 结构分层参考 | github.com/karpathy/minGPT |
| **TinyStories 论文** | 为什么这个规模能做到什么，能力边界的第一手证据 | arxiv.org/abs/2305.07759 |
| **llama.cpp** | M7 之后：SIMD kernel、KV cache 量化、CPU 调度 | github.com/ggml-org/llama.cpp |

---

## 13. 一页纸检查清单

开工前
- [ ] OpenBLAS 装上且 bench 出数字（<50 GFLOPS 峰值说明 flag 或链接错了）
- [ ] `estimate.py` 里填的是**实测有效 GFLOPS**（不是理论峰值），并算出主线要挂几天
- [ ] 那个天数你能连续接受 —— 不能就立刻缩小配置，别开训后再改
- [ ] 语料已下载（TinyShakespeare 1MB + TinyStories 子集）

实现纪律
- [ ] 每个算子都有 numpy golden 对拍，误差阈值写在代码注释里
- [ ] 每层加完就做逐层比对，不攒到最后比 logits
- [ ] **loss=0 过拟合测试通过**才允许开始正式训练
- [ ] ckpt 格式有版本号，读写器跨语言测试

训练
- [ ] 记录 tokens/sec、val loss、grad_norm 三条曲线
- [ ] `main` 每 500 步存 ckpt，且**验证加载后 loss 一致**（别等训完才发现存坏了）
- [ ] 采样脚本每次改完都跑一遍固定 prompt（seed 固定），方便对比退化

完成标准（M6/M7）
- [ ] `quick` 跑满 1 epoch 后 val loss ≤ **1.8**（对应困惑度 ~40，是能讲故事的门槛）
- [ ] 给定 "Once upon a time" 能生成语法正确、前后连贯的 150 词小故事
- [ ] 推理速度（带 KV cache）≥ 150 tok/s；不带 cache 的朴素实现 ≥ 20 tok/s
- [ ] `docs/notes.md` 记满你踩过的坑 —— 这份文档比代码更有价值

---

## 14. 常见心态陷阱

1. **"再优化一下就开训"** — 错。loss=0 测试通过后立刻开训，性能优化放到第一次训练的挂机时间里做。
2. **一开始就上 BPE + 完整数据集** — 你会在两天后仍然看不到 loss 下降，且不知道是模型还是数据还是分词器的问题。
3. **想加 MoE / 长上下文 / 多轮对话** — 这些在 17M~30M 规模下不会带来任何可感知收益，只会带来 bug。全部砍掉。
4. **对比 HF 上同参数模型的 benchmark 分数** — 那些模型是在 10~100 倍于你的 token 量上训出来的，比了只会难受。
   你的对手是"自己的 val loss 曲线有没有真的在降"和"生成的故事有没有前后一致"。

---

### 附：本手册所有代码片段的语言标注

- 无标注块 = C++23
- 标 `python` 或 `scripts/estimate.py` = Python 3.14（仅 numpy + 标准库，不依赖 torch）
- 训练/推理正式实现必须 C++；Python 只做参考与验证。若你希望某一阶段反过来（先用 Python 打通概念再移植），
  在 M1~M4 阶段这样做完全合理，M5 之后请全部落到 C++。
