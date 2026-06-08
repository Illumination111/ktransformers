# KTransformers 论文深度解析：释放 CPU/GPU 异构推理的全部潜力

> **论文全称**：KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models  
> **发表会议**：ACM SIGOPS 31st Symposium on Operating Systems Principles (SOSP '25)  
> **发表时间**：2025 年 10 月  
> **作者机构**：清华大学 & Approaching.AI 等  
> **代码仓库**：https://github.com/kvcache-ai/ktransformers

---

## 目录

1. [背景与动机：为什么需要 KTransformers？](#1-背景与动机为什么需要-ktransformers)
2. [核心创新点详解](#2-核心创新点详解)
   - 2.1 [AMX 感知的高性能 CPU 内核](#21-amx-感知的高性能-cpu-内核)
   - 2.2 [异步 CPU-GPU 任务调度机制](#22-异步-cpu-gpu-任务调度机制)
   - 2.3 [NUMA 感知的张量并行策略](#23-numa-感知的张量并行策略)
   - 2.4 [专家延迟（Expert Deferral）机制](#24-专家延迟expert-deferral机制)
   - 2.5 [灵活模块注入框架](#25-灵活模块注入框架)
3. [系统整体框架结构](#3-系统整体框架结构)
   - 3.1 [系统架构总览](#31-系统架构总览)
   - 3.2 [各模块协调运行流程](#32-各模块协调运行流程)
   - 3.3 [预填充阶段（Prefill）执行流水线](#33-预填充阶段prefill执行流水线)
   - 3.4 [解码阶段（Decode）执行流水线](#34-解码阶段decode执行流水线)
4. [KTransformers 与 SGLang 的联合工作方式](#4-ktransformers-与-sglang-的联合工作方式)
   - 4.1 [SGLang 简介](#41-sglang-简介)
   - 4.2 [联合架构设计](#42-联合架构设计)
   - 4.3 [KT-Kernel 集成到 SGLang 的关键参数](#43-kt-kernel-集成到-sglang-的关键参数)
   - 4.4 [实际部署流程](#44-实际部署流程)
5. [实验结果分析](#5-实验结果分析)
   - 5.1 [实验环境配置](#51-实验环境配置)
   - 5.2 [端到端性能对比](#52-端到端性能对比)
   - 5.3 [Expert Deferral 对精度的影响](#53-expert-deferral-对精度的影响)
   - 5.4 [各优化模块的性能拆解](#54-各优化模块的性能拆解)
6. [总结与贡献](#6-总结与贡献)

---

## 1. 背景与动机：为什么需要 KTransformers？

### 1.1 MoE 模型是什么？

**混合专家模型（Mixture-of-Experts，MoE）** 是一种特殊的 Transformer 架构，其核心思想是：**不是所有计算模块都需要对每个 token 都参与计算，而是通过一个"门控网络"（Gating Network）动态地为每个 token 选择少数几个"专家"（Expert）子网络来处理**。

在普通的密集 Transformer 模型（如 GPT）中，每一层的前馈网络（FFN）对每个 token 都要完整计算。而在 MoE 模型中：

```
普通 Dense FFN：每个 token → 完整的 FFN 权重（全部激活）
MoE 稀疏 FFN：每个 token → 门控网络 → 选 top-k 个专家 → 仅激活 k 个专家
```

以 DeepSeek-V3 为例，它拥有 256 个路由专家，但每个 token 只会激活其中 8 个。这种稀疏性带来了显著的计算效率提升：
- **总参数量**：671B（6710 亿个参数）
- **每次推理激活的参数量**：约 37B（仅约 5.5%）

这意味着 MoE 模型在推理时实际计算量远小于其参数规模所暗示的量，但**存储所有专家权重仍然需要巨大的内存**。

### 1.2 本地部署 MoE 模型的核心难题

部署如 DeepSeek-V3（671B 参数）这样的超大 MoE 模型面临的最核心挑战是：**单台普通服务器的 GPU 显存根本装不下**。

- NVIDIA A100（40GB VRAM）装 671B BF16 模型需要 **约 1342GB VRAM** → 需要约 34 块 A100
- 消费级 GPU（如 RTX 4080 16GB）则更无能为力

**CPU/GPU 混合推理** 是一种解决方案：将大部分专家权重存在 CPU 的 DRAM 中（容量大、成本低），而将计算密集的注意力层（Attention Layer）和少量共享专家（Shared Experts）放在 GPU 上执行。

然而，现有的混合推理系统（如 Fiddler、Llama.cpp）在实践中存在严重瓶颈：

| 问题 | 具体表现 |
|------|---------|
| **CPU 计算能力未充分利用** | 使用 AVX-512 而非更强的 AMX 指令集；仅达到 AMX 理论峰值算力的 7% |
| **CPU-GPU 同步开销过高** | Fiddler 每解码一个 token 需要 7000+ 次 CUDA kernel 启动，占 GPU 执行时间的 73% |
| **NUMA 感知缺失** | 跨 socket 的内存访问速度远低于本地访问，但现有框架忽略这一区别 |
| **CPU 与 GPU 无法并行** | 注意力层（GPU）和 MoE 专家层（CPU）串行执行，导致硬件资源大量闲置 |

这些问题的综合结果是：在一台配备 NVIDIA A100 + 2 块 Intel Xeon CPU 的服务器上，使用现有系统推理 DeepSeek-V3 时：
- 预填充吞吐量：仅 70.02 tokens/s
- 解码吞吐量：仅 4.68 tokens/s
- GPU 利用率：低于 30%

**KTransformers 正是为了从根本上解决这些问题而提出的。**

---

## 2. 核心创新点详解

KTransformers 的创新点可以用一句话概括：**通过精细的系统工程，让 CPU 和 GPU 都能在混合推理中发挥出接近理论峰值的性能，同时尽量保持并发执行以消除闲置等待**。

具体来说，有以下五个核心创新：

### 2.1 AMX 感知的高性能 CPU 内核

#### 2.1.1 什么是 AMX？

Intel AMX（Advanced Matrix Extensions）是从 Intel Sapphire Rapids（2023 年）系列处理器开始引入的矩阵运算加速指令集。与 AVX-512（向量指令集，一次处理 512 位向量）不同，AMX 是**矩阵级别**的加速：

- 每个 AMX 核心有 8 个 Tile 寄存器
- 每个 Tile 存储一个 **16 行 × 64 字节** 的子矩阵
- AMX 指令可以一次性完成两个 Tile 的矩阵乘法并将结果存入第三个 Tile
- 在大矩阵乘法（高算术强度场景）下，AMX 理论峰值可达 **73.7 TFLOPS**（而 AVX-512 仅 ~27 TFLOPS）

但 PyTorch 通过 oneDNN 使用 AMX 时，实测峰值仅有 **5.4 TFLOPS**，约为理论峰值的 7%！原因在于内存布局不对齐，导致大量 cache miss。

#### 2.1.2 AMX 感知的内存布局设计

KTransformers 的关键洞察是：**AMX 的效率瓶颈根源在于内存布局与 AMX Tile 不匹配，导致内存访问效率极低**。

为此，KTransformers 在**模型加载时**就将专家权重矩阵重排成 AMX 友好的布局：

```
传统布局（行优先）：                AMX Tile 感知布局：
┌─────────────────────┐          ┌──────┬──────┬──────┐
│  行0: w00 w01 w02...│          │Tile00│Tile01│Tile02│
│  行1: w10 w11 w12...│   →      ├──────┼──────┼──────┤
│  行2: w20 w21 w22...│          │Tile10│Tile11│Tile12│
│  ...                │          └──────┴──────┴──────┘
└─────────────────────┘          (每个 Tile 64 字节对齐)
```

具体设计要点：
1. **64 字节对齐**：Tile 的起始地址与 CPU cache line 对齐，避免跨 cache line 读取
2. **预先分块**：推理时无需做矩阵转置或重排，直接用于 AMX 计算
3. **量化集成**：支持 Int4/Int8 量化，Int4 打包为 Int8 块，使用 SIMD 指令解包

#### 2.1.3 缓存友好的 AMX 内核执行流程

基于优化的内存布局，KTransformers 设计了一个充分利用 CPU 多级缓存的计算流程：

```
① 专家权重矩阵按列切分成多个任务 → 动态调度给各 CPU 线程
         ↓
② 每个线程的任务：将专家权重按行切分成适合 L2 Cache 的 Block
         ↓
③ 每个 Block 由多个 AMX Tile 大小的子矩阵组成
         ↓
④ 输入激活（通常在 L3 Cache 中）和权重块一次性装入 L2 Cache
         ↓
⑤ 使用 AMX 指令执行 Tile 级别的矩阵乘法，中间结果暂存在 L1 Cache 中
```

这种设计的优势：
- **DRAM 访问最小化**：权重只从 DRAM 读入 L2 Cache 一次，避免重复读取
- **输入激活复用**：输入激活常驻 L3 Cache，所有线程共享读取
- **动态调度**：任务队列允许空闲线程动态领取新任务，解决负载不均衡问题（预填充阶段提升最多 1.83×）

实测结果：KTransformers 的 AMX 内核在单 socket CPU 上实现 **21.3 TFLOPS**，比 PyTorch+oneDNN 基线快 **3.98×**。

#### 2.1.4 自适应 AVX-512 内核（低算术强度场景）

尽管 AMX 在大批量（高算术强度）场景下优异，但在解码阶段（每步只处理 1 个 token），计算量极小，AMX 反而会因为**装填整个 Tile 的固定开销**而效率低下。

KTransformers 实现了双内核自适应切换策略：

```
算术强度判断（每专家分配 token 数）：
  ≤ 4 tokens/expert → 使用 AVX-512 内核（轻量、低延迟）
  > 4 tokens/expert → 使用 AMX 内核（高吞吐）
```

实测效果：
- 解码阶段 AVX-512 vs 纯 AMX：最多快 **1.20×**
- 预填充阶段 AMX vs 纯 AVX-512：最多快 **10.81×**

#### 2.1.5 融合 MoE 算子（Fused MoE Operator）

MoE 层包含三种线性投影：Gate、Up、Down。传统实现是三次独立的矩阵乘法，每次都需要线程同步。KTransformers 将其融合：

- 将所有专家的 Gate 投影合并为**一个大任务**
- 将所有专家的 Up 投影合并为**一个大任务**
- Gate 和 Up 之间无数据依赖，可以进一步合并
- Down 投影独立执行

这样 MoE 执行只需要**两轮**线程同步（原本需要每个专家独立同步），大幅降低线程调度开销。

---

### 2.2 异步 CPU-GPU 任务调度机制

#### 2.2.1 问题根源：CUDA kernel 启动开销

在混合推理中，GPU 执行注意力层和共享专家，CPU 执行路由专家。传统的 GPU kernel 调用流程是：

```
CPU                    GPU
 │                      │
 ├─ 准备数据 ────────────►│
 │                      ├─ 执行 kernel（共享专家）
 ├─ 等待 GPU 完成 ◄───────┤
 │                      │
 ├─ 触发路由专家计算      │
 │  (CPU 执行)           │
 ├─ 等待 CPU 完成        │
 │                      │
 ├─ 发起下一个 kernel ───►│
 │                      ├─ 执行 kernel（Attention）
...                    ...
```

每次 GPU kernel 启动都有 **~5-16 μs 的固定延迟**。解码一个 token 需要触发 **3000-7000 次** kernel 启动，这些开销积累起来占到 GPU 总执行时间的 **21-73%**。

#### 2.2.2 CUDA Graph：将所有 kernel 捕获为单一图

CUDA Graph 是 NVIDIA 提供的机制：**将多次 kernel 启动的序列录制成一张"图"，之后整张图只需一次启动**，消除了中间的所有 kernel 启动开销。

但问题是：混合推理中 CPU 的同步点（发送激活给 CPU、等待 CPU 完成）会**打断 CUDA Graph 的录制**。传统做法要么不用 CUDA Graph，要么每层单独录一张图（内存开销大）。

#### 2.2.3 KTransformers 的核心突破：单图封装全部解码流程

KTransformers 的突破在于：**使用 `cudaLaunchHostFunc` 将 CPU 同步点包装成 GPU stream 中的回调函数**，使其对 CUDA Graph 透明。

具体机制：

```
原始流程（同步）：
GPU Stream: [kernel1] → [等待CPU] → [kernel2] → [等待CPU] → [kernel3]
                           ↑ 打断 CUDA Graph ↑

KTransformers 方案（异步）：
GPU Stream: [kernel1] → [HostFunc: 发任务给CPU] → [CUDA自旋等待] → [kernel2]
                              ↑ CUDA Graph 可以录制整个流程 ↑
```

**关键细节**：

1. **无锁任务队列**：GPU 的门控网络完成后，控制线程将路由专家任务推入无锁队列，无需阻塞
2. **CUDA 自旋等待（Spinning）**：GPU 在等待 CPU 时，通过 CUDA kernel 在 GPU 上自旋检查标志位，而不是真正暂停 stream
3. **单图方案**：整个 decode 阶段（一个 token 的所有层）被封装进**一个 CUDA Graph 实例**，kernel 启动开销从数千次降至**近乎零**

效果：解码速度提升最多 **1.23×**。

---

### 2.3 NUMA 感知的张量并行策略

#### 2.3.1 NUMA 架构是什么？

现代服务器通常有多个 CPU socket（插槽），每个 socket 有自己的内存控制器，连接本地内存（本地 DRAM）。访问**本地 DRAM** 很快，但访问**另一个 socket 的远端 DRAM** 要经过 CPU 间的 QPI/UPI 总线，速度慢得多：

- 同 socket 内存带宽：220 GB/s
- 跨 socket 内存带宽：125 GB/s（仅为本地的 57%）

现有框架（Fiddler、Llama.cpp）把双 socket 机器当作单一节点，数据分配不考虑本地性，导致大量跨 socket 访问。实测：单 socket 跑一个 MoE 层需要 6.9ms，双 socket 仅降至 5.8ms（仅提升 16%），远低于预期的近 2× 加速。

#### 2.3.2 两种并行策略的对比

**专家并行（Expert Parallel）**：每个 socket 存一批完整的专家，各自处理

```
问题：某些专家被高频激活，导致对应 socket 过载；另一些专家冷门，socket 闲置
```

**KTransformers 的张量并行（Tensor Parallel）**：将每个专家的权重矩阵**按列/行切分**，每个 socket 存储并计算自己的那一份：

```
Socket 0 存 Expert_a 的前半部分权重 → 计算得到部分输出
Socket 1 存 Expert_a 的后半部分权重 → 计算得到部分输出
两个 socket 各自本地计算完成后 → 轻量级 reduce-scatter 合并结果
```

这种方式的优势：
- **接近 100% 本地内存访问**：每个线程只读取本 socket 上的权重
- **天然负载均衡**：无论激活哪个专家，两个 socket 都参与计算
- **可扩展性**：随 socket 数量线性扩展

实测：双 socket 服务器上解码吞吐比 NUMA 无感知基线提升 **1.63×**。

---

### 2.4 专家延迟（Expert Deferral）机制

这是论文中最具原创性和工程价值的创新，也是理解起来最需要细想的部分。

#### 2.4.1 根本问题：硬件资源大量闲置

即使有了上述所有优化，在解码阶段对 DeepSeek-V3 的分析发现：
- CPU 利用率：74%（并非满载）
- GPU 利用率：仅 28%（严重空闲）
- CPU-GPU 并发时间：仅占总执行时间的 5%

原因是 Transformer 的标准执行顺序造成的严格串行依赖：

```
层 k 的 MoE 执行（CPU）
    ↓  必须等待完成
层 k+1 的 Attention 计算（GPU）
    ↓  必须等待完成
层 k+1 的 MoE 执行（CPU）
    ↓  ...
```

GPU 在等 CPU 计算 MoE，CPU 在等 GPU 计算 Attention，双方都大量闲置。

#### 2.4.2 关键洞察：残差连接带来的鲁棒性

KTransformers 的作者发现了一个**关键事实**：现代 Transformer 使用残差连接（Residual Connection），即每一层的输出是 `输入 + 本层计算结果`。这意味着即使某些专家的输出被"延迟"一层才加入，残差连接保证了信息不会丢失，模型对此有一定的天然鲁棒性。

#### 2.4.3 Expert Deferral 的核心机制

**核心思路**：将 MoE 层中的路由专家分为两类：

- **立即专家（Immediate Experts）**：计算结果立即被下一层 Attention 使用（遵循标准流程）
- **延迟专家（Deferred Experts）**：计算被推迟，结果在下下一层才被合入输出

数学表达：

**标准 MoE 输出**（第 k 层）：
$$O_k = I_k + S_k(I_k) + R^{all}_k(I_k)$$

其中 $S_k$ 是共享专家，$R^{all}_k$ 是所有路由专家，$I_k$ 是层输入。

**Expert Deferral 后的 MoE 输出**：
$$O_k = \begin{cases} I_k + S_k(I_k) + R^{imm}_k(I_k) & k = 1 \\ I_k + S_k(I_k) + R^{def}_{k-1}(I_{k-1}) + R^{imm}_k(I_k) & 1 < k < L \\ I_k + S_k(I_k) + R^{def}_{k-1}(I_{k-1}) + R^{all}_k(I_k) & k = L \end{cases}$$

解释：第 k 层的输出 = 输入 + 共享专家（当前层，GPU 执行）+ **上一层的延迟专家**（CPU 已经算完了）+ 当前层立即专家（CPU 执行）。

#### 2.4.4 执行时序对比

**标准执行**：
```
时间线 →
CPU: [MoE Layer k] → [等待GPU] → [MoE Layer k+1] → ...
GPU: [等待CPU]     → [Attn k+1] → [等待CPU]       → ...
```

**Expert Deferral 执行**：
```
时间线 →
CPU: [立即专家(k)] → [延迟专家(k)并发执行中...] → [立即专家(k+1)]...
GPU:               → [Attn(k+1)并发执行中...]   → [Attn(k+2)]...
```

关键变化：**CPU 上的延迟专家与 GPU 上的 Attention 层同时执行**，CPU 不再等待 GPU，GPU 也不再等待 CPU（大部分时间）。

#### 2.4.5 如何确定延迟专家的数量？

论文对 DeepSeek-V3 进行了系统性分析，测试了延迟 2、3、4 个专家的效果：

| 配置 | CPU 利用率 | GPU 利用率 | 执行时间减少 |
|------|-----------|-----------|------------|
| 0 延迟（基线） | 74% | 28% | 0% |
| 2 延迟 | ~90% | 37% | 19% |
| **3 延迟（最优）** | **100%** | **37%** | **26%** |
| 4 延迟 | 100% | 37% | 26%（无额外提升） |

最优方案：**5 立即专家 + 3 延迟专家**（DeepSeek-V3 的 BF16 版本）。

**通用启发式规则**：延迟尽量少的专家，达到 CPU 满载即可，同时保留至少 2 个立即专家以维持模型稳定性。

#### 2.4.6 为什么只在解码阶段使用 Expert Deferral？

在预填充阶段，每个 batch 中的 token 会激活非常多样化的专家（几乎覆盖所有专家），这意味着延迟专家和立即专家几乎涵盖了全部专家，导致**内存访问量翻倍**，反而成为新的瓶颈。而解码阶段每步只处理少数 token，Expert Deferral 的延迟不会带来显著的内存访问放大。

---

### 2.5 灵活模块注入框架

#### 2.5.1 框架设计动机

上述所有优化需要被整合到现有的模型推理框架中。如果每个模型都要手写一遍集成代码，开发代价极高。KTransformers 设计了一个声明式的**模块注入框架**，允许通过**一个 YAML 配置文件**完成所有集成。

#### 2.5.2 工作原理

系统启动时，框架遍历 HuggingFace 的模型树，根据 YAML 配置中的规则，将标准 PyTorch 模块替换为 KTransformers 的高性能实现：

```yaml
# 将所有 DeepseekV3MoE 模块替换为高性能 FusedMoE（CPU 执行）
- match:
    class: modeling_deepseek_v3.DeepseekV3MoE
  replace:
    class: operators.experts.FusedMoE
    device: "cpu"
    kwargs:
      backend: "hybrid_AMX_AVX512"  # 使用 AMX+AVX512 混合内核
      data_type: "Int4"              # Int4 量化
      n_deferred_experts: 6          # 延迟 6 个专家

# 将注意力层替换为基于 FlashInfer 的高效实现（GPU 执行）
- match:
    name: "^model\\.layers\\..*\\.self_attn$"
  replace:
    class: operators.attention.FlashInferMLA
    device: "cuda:0"

# 将所有线性层（除 lm_head）替换为 Marlin 量化内核
- match:
    name: "^(?!lm_head$).*"
    class: torch.nn.Linear
  replace:
    class: operators.linear.MarlinLinear
    device: "cuda:0"
    kwargs:
      data_type: "Int4"
```

这个框架的优势：
- **对 HuggingFace 接口零侵入**：用户代码无需修改
- **支持正则匹配**：按模块名或类名灵活定位替换目标
- **支持多 GPU pipeline**、**混合精度**、**KV Cache offloading** 等高级特性
- **可移植性强**：DeepSeek-V2 只需修改一行 class 名称就能复用

---

## 3. 系统整体框架结构

### 3.1 系统架构总览

KTransformers 的整体架构可以分为四个层次：

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户接口层                               │
│  HuggingFace Transformers API / OpenAI 兼容 API                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │ 模块注入（YAML 配置）
┌─────────────────────────────▼───────────────────────────────────┐
│                    KTransformers 核心层                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ 模块注入框架  │  │异步调度引擎  │  │ Expert Deferral 调度  │  │
│  │ (YAML驱动)   │  │(CUDA Graph)  │  │ (延迟/立即专家管理)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└───────────┬────────────────────┬────────────────────────────────┘
            │                   │
┌───────────▼──────┐  ┌─────────▼──────────────────────────────┐
│   CPU 计算层      │  │              GPU 计算层                  │
│                  │  │                                          │
│ ┌──────────────┐ │  │  ┌─────────────┐  ┌──────────────────┐  │
│ │ AMX 内核     │ │  │  │FlashInfer   │  │ Marlin 量化内核  │  │
│ │(高算术强度)  │ │  │  │Attention 层  │  │(GPU 线性投影)   │  │
│ └──────────────┘ │  │  └─────────────┘  └──────────────────┘  │
│ ┌──────────────┐ │  │  ┌─────────────────────────────────────┐ │
│ │ AVX-512 内核 │ │  │  │     共享专家（Shared Experts）        │ │
│ │(低算术强度)  │ │  │  │     (GPU VRAM 中常驻)               │ │
│ └──────────────┘ │  │  └─────────────────────────────────────┘ │
│ ┌──────────────┐ │  └──────────────────────────────────────────┘
│ │NUMA 感知     │ │
│ │张量并行      │ │
│ └──────────────┘ │
└──────────────────┘

底层异构硬件：
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Intel Xeon CPU   │    │   NVIDIA A100    │    │  NVIDIA RTX 4080 │
│ (654B 专家权重)  │    │   (17B GPU参数)  │    │  (量化版本)      │
│ Socket 0+1 DRAM  │    │   40GB VRAM      │    │  16GB VRAM       │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

### 3.2 各模块协调运行流程

KTransformers 的各个组件如何协同工作，可以通过一个完整的推理请求的生命周期来理解：

**初始化阶段**：

```
1. 读取 YAML 配置文件
2. 加载模型权重
   - 共享专家权重 → GPU VRAM
   - 注意力层权重 → GPU VRAM（可选量化）
   - 路由专家权重 → CPU DRAM（AMX 感知布局，预量化）
3. 模块注入：替换 PyTorch 标准模块为高性能实现
4. 录制 CUDA Graph（将整个 decode 流程录制成单一图）
```

**推理阶段**：见下方详细流程。

### 3.3 预填充阶段（Prefill）执行流水线

预填充处理**整个输入序列**（可能有数千个 token），计算量大，瓶颈在 CPU 的矩阵计算能力。

```
输入：长序列（如 8192 tokens）

对每一个 Transformer 层：
┌────────────────────────────────────────────────────────────┐
│ 1. LayerNorm → GPU 执行                                     │
│ 2. Multi-head Attention（FlashInfer MLA）→ GPU 执行         │
│ 3. LayerNorm → GPU 执行                                     │
│ 4. 门控网络（Gate）→ GPU 执行，得到每个 token 的专家分配     │
│ 5. 将激活值传输给 CPU                                       │
│ 6. CPU 执行路由专家（AMX 内核，所有被激活的专家）            │
│    - 动态任务调度解决专家负载不均衡                          │
│    - 并行执行 Gate+Up 融合和 Down 投影                      │
│ 7. GPU 同时执行共享专家                                     │
│ 8. 合并结果 → 进入下一层                                    │
└────────────────────────────────────────────────────────────┘

注：预填充不使用 Expert Deferral（会倍增内存访问）
```

### 3.4 解码阶段（Decode）执行流水线

解码每次只生成**一个 token**，计算量小但对延迟敏感，瓶颈在 CPU-GPU 协调开销。

```
CUDA Graph 录制的单次 token 解码完整流程：

GPU Stream:
┌──────────────────────────────────────────────────────────────────────┐
│ [Attention Layer k]                                                    │
│    ↓                                                                  │
│ [共享专家（GPU）+ HostFunc: 提交路由专家任务给 CPU]                    │
│    │                         ↑ CPU 开始计算（后台异步）               │
│    ↓                                                                  │
│ [CUDA 自旋等待：检查 CPU 是否完成]───────────────────────────────────┐│
│    ↓（CPU 完成后）                                                    ││
│ [合并共享专家+路由专家的结果]                                          ││
│    ↓                                                                  ││
│ [Attention Layer k+1]                                                 ││
│    ↓                                                                  ││
│ [共享专家（GPU）+ HostFunc: 提交下一层路由专家任务]                    ││
│    │                                                                  ││
│    ...（重复 58 层）...                                               ││
│    ↓                                                                  ││
│ [语言模型头（LM Head）→ 采样输出 token]                               ││
└───────────────────────────────────────────────────────────────────────┘│
                                                                         │
CPU:  [延迟专家计算 ←─────────────────────────────────────────────────────┘
       (NUMA 感知张量并行，双 socket 均分计算)]
```

**Expert Deferral 的时序效果**（以 DeepSeek-V3 BF16 为例）：

```
无 Expert Deferral：
时间线: ════════════════════════════════════════════════════
CPU:    ████████████████░░░░░░░░░░████████████████░░░░░░░░
        |←—MoE Layer k—→|←—等待—→|←—MoE Layer k+1—→|←等→|
GPU:    ░░░░░░░░░░░░░░░░████████░░░░░░░░░░░░░░░░░░████████
                        |←Attn→|                  |←Attn→|
CPU 利用率: 74%，GPU 利用率: 28%

有 Expert Deferral（5 立即 + 3 延迟）：
时间线: ══════════════════════════════════════════
CPU:    ████████████████████████████████████████████
        |←立即(k)→|←─────延迟(k)─────→|←立即(k+1)→|...
GPU:    ░░░░░░░████████████████░░░░░░████████████████
               |←───Attn(k+1)───→|  |←───Attn(k+2)──→|
CPU 利用率: 100%，GPU 利用率: 37%，总时间减少 26%
```

---

## 4. KTransformers 与 SGLang 的联合工作方式

### 4.1 SGLang 简介

SGLang（Structured Generation Language）是一个高效的大语言模型推理框架，由 UC Berkeley 等团队开发，专注于：
- **高吞吐量服务**：支持多请求并发处理
- **RadixAttention**：通过前缀共享优化 KV Cache 利用率
- **结构化生成**：原生支持 JSON/正则表达式等格式约束
- **OpenAI 兼容 API**：提供标准的 REST 接口

SGLang 原本面向**全 GPU 推理**设计，对 CPU/GPU 异构计算没有原生支持。

### 4.2 联合架构设计

KTransformers 团队维护了一个 SGLang 的 fork（`kvcache-ai/sglang`），为其注入了 KT-Kernel（KTransformers 的核心计算内核）支持。联合工作架构如下：

```
┌─────────────────────────────────────────────────────────────────────┐
│                      客户端层                                        │
│  OpenAI 兼容 API / curl / Python client                              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP 请求（port 30000）
┌────────────────────────────────▼────────────────────────────────────┐
│                  SGLang Server (kvcache-ai fork)                     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              请求调度层（Scheduler）                              │ │
│  │  - 批处理（Batching）                                           │ │
│  │  - 优先级管理                                                   │ │
│  │  - Continuous Batching（支持多并发请求）                         │ │
│  └─────────────────────────────┬───────────────────────────────────┘ │
│                                │                                      │
│  ┌─────────────────────────────▼───────────────────────────────────┐ │
│  │              模型执行层（Model Executor）                         │ │
│  │                                                                  │ │
│  │  GPU 部分（SGLang 原生）：                                       │ │
│  │  ┌──────────────────┐  ┌──────────────────┐                     │ │
│  │  │ Attention（Triton│  │ 共享专家（FP8/   │                     │ │
│  │  │ / FlashInfer）   │  │ BF16 CUDA）      │                     │ │
│  │  └──────────────────┘  └──────────────────┘                     │ │
│  │                                                                  │ │
│  │  CPU 部分（KT-Kernel 注入）：                                    │ │
│  │  ┌──────────────────────────────────────────────────────────┐   │ │
│  │  │        kt_kernel：路由专家 CPU 计算（AMX/AVX512）          │   │ │
│  │  │        - INT4/INT8 量化推理                               │   │ │
│  │  │        - NUMA 感知多 socket 并行                          │   │ │
│  │  │        - 异步调度（与 GPU 并发）                           │   │ │
│  │  └──────────────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              KV Cache 管理（PagedAttention）                     │ │
│  │  - GPU VRAM 中的 KV Cache                                       │ │
│  │  - RadixAttention 前缀共享                                       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

底层硬件：
┌────────────────────┐           ┌──────────────────────────┐
│  CPU（双 Socket）   │           │    NVIDIA GPU            │
│  INT4 路由专家权重  │◄─────────►│    FP8 GPU 层权重         │
│  ~350GB DRAM       │  PCIe 4.0 │    ~27GB VRAM            │
└────────────────────┘           └──────────────────────────┘
```

### 4.3 KT-Kernel 集成到 SGLang 的关键参数

启动 SGLang 服务时，通过以 `--kt-` 为前缀的参数启用 KT-Kernel：

| 参数 | 含义 | 示例值 |
|------|------|--------|
| `--kt-weight-path` | CPU 端量化权重文件路径 | `/path/to/model-INT4` |
| `--kt-cpuinfer` | CPU 推理线程数 | `60`（30 核 × 双 socket） |
| `--kt-threadpool-count` | NUMA 线程池数量 | `2`（对应 2 个 NUMA 节点） |
| `--kt-num-gpu-experts` | 常驻 GPU 的共享专家数量 | `1` |
| `--kt-method` | 计算内核类型 | `AMXINT4` |
| `--kt-gpu-prefill-token-threshold` | 触发 GPU prefill 的 token 阈值（kt-kernel 特有） | - |

SGLang 端的配套参数：

| 参数 | 含义 |
|------|------|
| `--attention-backend triton` | 使用 Triton 注意力内核 |
| `--mem-fraction-static 0.98` | 分配 98% 显存给模型 |
| `--chunked-prefill-size 4096` | 将长 prefill 切块，避免 OOM |
| `--enable-mixed-chunk` | 支持 prefill + decode 混合批次 |
| `--disable-shared-experts-fusion` | 禁用 SGLang 自身的共享专家融合（由 kt-kernel 接管） |

### 4.4 实际部署流程

以 DeepSeek-V3.2（671B）在单张 NVIDIA L20（48GB）+ 双 Intel Xeon CPU 上的部署为例：

**第一步：量化 CPU 权重**

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/deepseek-v3.2 \
  --input-type fp8 \
  --output /path/to/deepseek-v3.2-INT4 \
  --quant-method int4 \
  --cpuinfer-threads 60 \
  --threadpool-count 2
```

将 FP8 GPU 权重转换为 INT4 格式，专门为 AMX 优化的内存布局。

**第二步：启动 SGLang 服务**

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model /path/to/deepseek-v3.2 \
  --kt-weight-path /path/to/deepseek-v3.2-INT4 \
  --kt-cpuinfer 60 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 1 \
  --attention-backend triton \
  --mem-fraction-static 0.98 \
  --chunked-prefill-size 4096 \
  --kt-method AMXINT4
```

资源占用：
- GPU VRAM：~27GB（1 个 GPU 专家 + 注意力层）
- CPU DRAM：~350GB（INT4 量化的路由专家权重）

**第三步：发送推理请求**

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3.2",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

**KTransformers + SGLang 联合的技术优势总结**：

| 特性 | 单独使用 SGLang | KTransformers + SGLang |
|------|---------------|----------------------|
| 671B 模型部署 | 需要 8+ 张 A100（多机） | 单机单张 GPU（消费级可用） |
| 调度器 | SGLang 原生高性能调度 | 沿用 SGLang 调度 |
| CPU 计算 | 不支持 | AMX/AVX512 高性能内核 |
| API 兼容性 | OpenAI 兼容 | 完全兼容 |
| 高并发支持 | 好 | 好（SGLang 负责） |
| 专家并行 | 不支持 | CPU NUMA 感知张量并行 |

---

## 5. 实验结果分析

### 5.1 实验环境配置

**硬件**：

| 组件 | 规格 |
|------|------|
| CPU | Intel Xeon Platinum 8452Y × 2（双 socket，各 36 核） |
| CPU 内存 | 2TB DDR5（每 socket 1TB） |
| 内存带宽 | 同 socket 220 GB/s，跨 socket 125 GB/s |
| GPU（服务器级） | NVIDIA A100 40GB |
| GPU（消费级） | NVIDIA RTX 4080 16GB |
| PCIe | PCIe 4.0（峰值带宽 32 GB/s） |

**被评估的模型**：

| 模型 | 总参数 | GPU 上参数 | CPU 上参数 | MoE 层数 | 每层专家数 | 路由策略 |
|------|-------|-----------|-----------|---------|----------|---------|
| DeepSeek-V3（DS-3） | 671B | 17B | 654B | 58 | 256 | Top-8 |
| DeepSeek-V2.5（DS-2） | 236B | 13B | 223B | 59 | 160 | Top-6 |
| Qwen2-57B（QW-2） | 57B | 8B | 49B | 28 | 64 | Top-8 |

**对比基线**：
- **Fiddler**：PyTorch 实现的 CPU/GPU 混合系统
- **Llama.cpp**：C++ 高性能框架（扩展了专家级 offloading 支持以保证对比公平性）

### 5.2 端到端性能对比

#### 预填充阶段（Prefill）

| 模型 | 对比 Fiddler 加速比 | 对比 Llama.cpp 加速比 |
|------|-------------------|---------------------|
| DS-3（BF16，8192 tokens） | **19.74×** | ~5× |
| DS-2（BF16，8192 tokens） | **~10×** | ~4× |
| QW-2（BF16，8192 tokens） | **~6×** | ~3× |

**关键洞察**：
- 对于 Fiddler，KTransformers 的加速比随提示长度增大而增大（AMX 在高算术强度下优势更明显）
- 对于 Llama.cpp，短提示场景下 Llama.cpp 较好（因为其激进的算子融合），长提示下 KTransformers 优势明显
- KTransformers 的 AMX 内核（21.3 TFLOPS）比 PyTorch+oneDNN（5.4 TFLOPS）快约 4×

#### 解码阶段（Decode，无 Expert Deferral）

| 模型 | 对比 Fiddler 加速比 | 对比 Llama.cpp 加速比 |
|------|-------------------|---------------------|
| DS-3（BF16） | **4.09×** | **1.76×** |
| DS-2（BF16） | **2.42×** | **1.25×** |
| QW-2（BF16） | **~3×** | **~1.5×** |

**关键洞察**：
- 加速主要来自：AVX-512 内核（低 ARI 场景）+ CUDA Graph（消除 kernel 启动开销）+ NUMA 感知并行
- 量化模型（Int4/Int8）相比 Llama.cpp 有更高加速（1.77-1.93×），因为 kernel 时间更短，同步策略的相对收益更大

#### 解码阶段（Decode，加上 Expert Deferral）

| 模型 | 对比 Llama.cpp 总加速比 |
|------|----------------------|
| DS-3（BF16，3 延迟） | **2.22×** |
| DS-3（Int4，6 延迟） | **2.56×** |
| DS-2（BF16，4 延迟） | **2.10×** |
| QW-2（Int8，4 延迟） | **1.66×** |

**Expert Deferral 额外带来的加速**（在已有 KTransformers 优化基础上）：

| 模型 | Expert Deferral 额外加速 |
|------|------------------------|
| DS-3 BF16（3 延迟） | +33% |
| DS-3 Int4（6 延迟） | +45% |
| DS-2 BF16（4 延迟） | +~30% |
| QW-2 Int8（4 延迟） | +~25% |

### 5.3 Expert Deferral 对精度的影响

这是验证 Expert Deferral 可行性的关键实验。论文从两个维度进行了评估：

#### 标准基准测试（4 个任务）

| 模型（配置） | HumanEval | MBPP | GSM8K | StrategyQA |
|------------|-----------|------|-------|-----------|
| DS-3 (8+0，无延迟) | 83.0 | 71.2 | 94.8 | 83.0 |
| DS-3 (2+6，6个延迟) | 83.0 | 70.2（-1.0） | 95.2（+0.4） | 82.9（-0.1） |
| DS-2 (6+0，无延迟) | 80.5 | 67.6 | 93.3 | 79.7 |
| DS-2 (2+4，4个延迟) | 82.5（+2.0） | 66.8（-0.8） | 92.8（-0.5） | 80.4（+0.7） |
| QW-2 (8+0，无延迟) | 65.7 | 52.4 | 84.7 | 83.6 |
| QW-2 (4+4，4个延迟) | 67.4（+1.7） | 53.4（+1.0） | 83.4（-1.3） | 82.5（-1.1） |

**结论**：Expert Deferral 对准确率的影响极小，变化幅度在 ±2 分以内，很多情况下甚至略有提升（这可能与量化误差的随机性有关）。

#### 与 Expert Skipping 的对比（LiveBench 细粒度评估）

对 DS-3 在 LiveBench 上的评估揭示了 Expert Deferral 相比"直接跳过专家"的本质优势：

| 影响专家数 | Expert Skipping（平均准确率变化） | Expert Deferral（平均准确率变化） |
|---------|-------------------------------|-------------------------------|
| 1 | -0.2% | +0.2% |
| 2 | +0.1% | +0.1% |
| 3 | -0.6% | +0.2% |
| 4 | -2.3% | +0.2% |
| 5 | -5.9% | +0.1% |
| **6（默认配置）** | **-13.3%** | **-0.5%** |
| 7 | -28.6% | -1.9% |
| 8 | -88.7% | -6.7% |

**核心发现**：
- **Expert Skipping**：直接丢弃低权重专家，6 个专家时平均准确率下降 13.3%，严重影响数学、推理类任务
- **Expert Deferral**：延迟计算而非丢弃，6 个专家时平均准确率仅下降 0.5%，是 Expert Skipping 的 **26 分之一的损失**
- 两种方法在吞吐量上提升相当（因为延迟专家与 GPU 完全并行，overhead 可忽略）

这说明 Expert Deferral 的核心价值在于：**以"延迟"代替"删除"，利用残差连接保留专家的计算贡献，以最小的准确率代价换取最大的硬件利用率提升**。

### 5.4 各优化模块的性能拆解

论文提供了从 Fiddler 基线开始，逐步叠加各项优化的性能分解：

#### 预填充阶段（以 8192 tokens 为基准）

```
基线（Fiddler）  1.0×
  +AVX-512 内核  → 比 AMX 慢（高 ARI 场景 AMX 更优）
  +AMX 内核      → 1.0×~3.14×（DS-3 提升最大）
  +动态调度      → 最多额外 1.83×（均衡负载）
  +NUMA 张量并行 → 最多额外 1.22×（减少跨 socket 访问）
  +CUDA Graph   → 几乎无额外提升（prefill 的 kernel 启动开销被大量 token 摊薄）
```

#### 解码阶段

```
基线（Fiddler）  1.0×
  +AVX-512 内核  → 最多 2.22×（decode 低 ARI，AVX-512 优于 AMX）
  +动态调度      → 几乎无提升（decode 每专家 token 数均匀，无负载不均）
  +NUMA 张量并行 → 最多额外 1.63×（decode 内存带宽瓶颈，NUMA 效果显著）
  +CUDA Graph   → 额外最多 1.23×（消除 kernel 启动开销，decode 效果明显）
```

**各优化模块的适用场景总结**：

| 优化模块 | 预填充效果 | 解码效果 | 原因 |
|---------|----------|---------|------|
| AMX 内核 | ★★★★★ | ★★ | 高算术强度 AMX 才有优势 |
| AVX-512 内核 | ★ | ★★★★ | 低算术强度 AVX-512 更轻量 |
| 动态任务调度 | ★★★★ | ★ | 预填充时专家激活不均匀 |
| NUMA 张量并行 | ★★ | ★★★★ | 解码内存带宽瓶颈更突出 |
| CUDA Graph | ★ | ★★★ | kernel 启动开销只在 decode 时显著 |
| Expert Deferral | N/A | ★★★★ | 仅用于解码，CPU-GPU 并行 |

---

## 6. 总结与贡献

### 6.1 论文贡献总结

KTransformers 的学术贡献可以归纳为四个方面：

1. **系统贡献**：设计并实现了针对 MoE 模型异构推理的完整高性能系统，涵盖从硬件指令级优化到系统架构级设计的全栈优化

2. **算法贡献**：提出了 Expert Deferral 这一新型执行策略，通过计算重排序在精度损失极小（<0.5%）的情况下大幅提升 CPU-GPU 并发度（CPU 利用率 74%→100%，解码吞吐提升最多 45%）

3. **工程贡献**：实现了 11,000 行 C++ 高性能扩展 + 2,000 行 Python 集成代码；提供了即插即用的 HuggingFace 兼容接口；以声明式 YAML 配置驱动异构优化，大幅降低使用门槛

4. **实践贡献**：让 671B 参数的超大 MoE 模型可以在单台配备消费级 GPU 的普通服务器上高效运行，使本地隐私部署成为可能

### 6.2 整体性能提升总结

| 阶段 | 对比基线 | 加速比范围 |
|------|---------|----------|
| 预填充（无 Expert Deferral） | Fiddler / Llama.cpp | **4.62–19.74×** |
| 解码（无 Expert Deferral） | Fiddler / Llama.cpp | **1.25–4.09×** |
| 解码（有 Expert Deferral） | Fiddler / Llama.cpp | **1.66–4.90×** |

### 6.3 与相关工作的区别

| 系统 | 主要方法 | 与 KTransformers 的区别 |
|------|---------|----------------------|
| Fiddler | CPU 计算 offloading | 缺乏 AMX 优化和 NUMA 感知 |
| Llama.cpp | C++ 算子融合 + weight offloading | 无高效 CUDA Graph，无 NUMA 张量并行 |
| PowerInfer | 热门专家 GPU 缓存 | 面向密集模型，MoE 适配不足 |
| MoE-Infinity | 激活感知 expert offloading | 专注 weight offloading，PCIe 带宽瓶颈 |
| EdgeMoE | 静态专家重要性量化 | 量化策略与 KTransformers 正交（可集成） |

### 6.4 未来展望

论文中提到，KTransformers 的灵活模块注入框架为未来集成更多高级优化奠定了基础，包括：
- 多 GPU Pipeline 并行（已通过 YAML 配置初步支持）
- KV Cache offloading（减少 GPU 显存占用）
- 更细粒度的混合精度推理
- 与 SGLang 等高性能服务框架的更深度集成（已通过 KT-Kernel + `sglang-kt` 实现）

---

> **参考文献**：Hongtao Chen et al. "KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models." SOSP '25, Seoul, Republic of Korea, October 2025. DOI: https://doi.org/10.1145/3731569.3764843
