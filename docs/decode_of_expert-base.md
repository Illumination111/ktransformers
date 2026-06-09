# `experts_base.py` 代码设计深度解析

> **文件路径**：`ktransformers/kt-kernel/python/experts_base.py`  
> **模块定位**：kt-kernel CPU 侧 MoE 推理基础设施层  
> **核心作用**：为所有量化后端（AMX、MXFP4、Llamafile 等）提供统一的缓冲区管理、CPUInfer 单例调度和推理流水线接口

---

## 目录

1. [模块概览与分层架构](#1-模块概览与分层架构)
2. [核心组件详解](#2-核心组件详解)
   - 2.1 [`generate_gpu_experts_masks`：专家 GPU 选择策略](#21-generate_gpu_experts_masks专家-gpu-选择策略)
   - 2.2 [`KExpertsCPUBuffer`：Pinned Memory 双缓冲管理](#22-kexpertscpubufferpinned-memory-双缓冲管理)
   - 2.3 [`_MoEBase`：CPUInfer 单例管理基类](#23-_moebasecpuinfer-单例管理基类)
   - 2.4 [`BaseMoEWrapper`：推理流水线核心抽象](#24-basemoewrapper推理流水线核心抽象)
3. [关键设计模式分析](#3-关键设计模式分析)
   - 3.1 [异步提交 + 延迟同步（Submit/Sync 解耦）](#31-异步提交--延迟同步submitsync-解耦)
   - 3.2 [Expert Deferral（专家延迟调度）](#32-expert-deferral专家延迟调度)
   - 3.3 [双 Slot 循环缓冲（buffer_depth=2）](#33-双-slot-循环缓冲buffer_depth2)
   - 3.4 [GPU 专家掩码的 C/Python 共享内存](#34-gpu-专家掩码的-cpython-共享内存)
4. [继承体系与对外接口](#4-继承体系与对外接口)
5. [与 SGLang 的对接方式](#5-与-sglang-的对接方式)
   - 5.1 [集成架构总览](#51-集成架构总览)
   - 5.2 [关键耦合参数](#52-关键耦合参数)
   - 5.3 [实际调用链路](#53-实际调用链路)
   - 5.4 [完整集成代码示例](#54-完整集成代码示例)
6. [与 LlamaFactory 的对接方式（SFT 模式）](#6-与-llamafactory-的对接方式sft-模式)
   - 6.1 [SFT 模式与推理模式的差异](#61-sft-模式与推理模式的差异)
   - 6.2 [LlamaFactory 集成架构](#62-llamafactory-集成架构)
   - 6.3 [完整 SFT 集成代码示例](#63-完整-sft-集成代码示例)
7. [性能调优建议](#7-性能调优建议)
8. [常见问题排查](#8-常见问题排查)

---

## 1. 模块概览与分层架构

`experts_base.py` 是 kt-kernel 整个 Python 层的底层基础模块，处于如下分层结构中：

```
外部框架（SGLang / LlamaFactory）
         │
         ▼
  KTMoEWrapper（experts.py）   ← 工厂入口，自动路由 mode/method
         │
         ▼
  BaseMoEWrapper（experts_base.py）  ← 本文件：统一推理流水线
  ├── KExpertsCPUBuffer              ← CPU Pinned Memory 双缓冲
  ├── _MoEBase                       ← CPUInfer 单例 + 配置校验
  └── generate_gpu_experts_masks     ← GPU 专家分配工具函数
         │
         ▼
  后端实现（utils/amx.py, utils/llamafile.py, utils/moe_kernel.py）
         │
         ▼
  C++ 扩展（kt_kernel_ext）← AMX/AVX512 内核、NUMA 线程池
```

**核心职责划分：**

| 组件 | 职责 |
|------|------|
| `generate_gpu_experts_masks` | 根据激活频率统计选出最热门专家放置到 GPU |
| `KExpertsCPUBuffer` | 管理 Pinned Memory 双缓冲，避免每次推理重新分配内存 |
| `_MoEBase` | 持有 `CPUInfer` 单例，负责 NUMA 线程池初始化 |
| `BaseMoEWrapper` | 定义 `submit_forward`/`sync_forward`/`forward` 推理协议；管理 Expert Deferral 状态机 |

---

## 2. 核心组件详解

### 2.1 `generate_gpu_experts_masks`：专家 GPU 选择策略

```python
def generate_gpu_experts_masks(
    activation_freq: torch.Tensor,  # shape: (num_layers, num_experts)
    num_gpu_experts: int,
) -> torch.Tensor:                  # shape: (num_layers, num_experts), dtype=bool
```

**算法原理：**

该函数实现了一种**跨层全局贪心选择**策略——将所有层的所有专家展平成一个列表，用 `torch.topk` 选出激活频率最高的 `num_gpu_experts` 个专家放到 GPU 上。

```
activation_freq[layer_0] = [0.1, 0.5, 0.3, 0.8]
activation_freq[layer_1] = [0.2, 0.4, 0.9, 0.1]

展平 → [0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1]
topk(3) → 索引 [3, 6, 1]（即 layer0-e3=0.8, layer1-e2=0.9, layer0-e1=0.5）

GPU mask:
  layer0: [F, T, F, T]
  layer1: [F, F, T, F]
```

**设计考量：**
- 跨层分配比每层固定数量更灵活，某些层可以全部在 CPU 上运行（节省显存）
- 通常在模型预热阶段统计激活频率，之后固定 mask（冷启动后不变）
- 若 `num_gpu_experts=0`，返回全零 mask，退化为全 CPU 推理

**使用场景：** 在 SGLang 启动时，通过 `--num-gpu-experts` 参数指定，框架自动调用此函数生成 mask 后传入每层的 `BaseMoEWrapper`。

---

### 2.2 `KExpertsCPUBuffer`：Pinned Memory 双缓冲管理

```python
class KExpertsCPUBuffer:
    capture_bs: List = list()       # 需要预分配的 batch size 列表
    capture_buffers: Dict = dict()  # {batch_size → buffer_tuple} 持久缓存
    temp_bs: int = 0                # 上一次 temp buffer 的 batch size
    temp_buffer: tuple = tuple()    # 临时缓冲区（最多缓存一个非预设 batch）
    buffer_depth: int = 2           # 双缓冲槽数
```

**缓冲区结构（每个 batch_size 对应一套，每套有 `buffer_depth=2` 个槽）：**

| 张量 | 形状 | 设备 | 作用 |
|------|------|------|------|
| `input_tensor_cpu` | `[bs, hidden_size]` | CPU (pinned) | 存放从 GPU 异步拷贝过来的输入 hidden states |
| `immediate_experts_ids_cpu` | `[bs, num_experts_per_tok]` | CPU (pinned) | 本轮立即计算的专家 ID（`-1` 表示跳过） |
| `deferred_experts_ids_cpu` | `[bs, num_experts_per_tok]` | CPU (pinned) | 延迟到下一轮计算的专家 ID |
| `weights_cpu` | `[bs, num_experts_per_tok]` | CPU (pinned) | 每个专家的路由权重 |
| `output_cpu` | `[bs, hidden_size]` | CPU (pinned) | CPU 专家计算的输出结果 |
| `bsz_tensor_cpu` | `[1]` | CPU (pinned) | batch size 整型张量（C++ 接口需要指针） |
| `output_gpu` | `[bs, hidden_size]` | GPU | 最终输出，从 output_cpu 异步拷贝而来 |

**为什么用 Pinned Memory？**

Pinned Memory（页锁定内存）允许 CUDA 通过 DMA 直接读写，**绕过操作系统的页表映射**，实现真正的异步 GPU↔CPU 数据传输（`non_blocking=True`）。在 MoE 推理中，Attention 层在 GPU 上运行时，CPU 侧就可以异步拷贝数据并启动专家计算，两者**并行执行**，这是 ktransformers 性能提升的关键之一。

**缓冲区查找逻辑（`get_buffer` 方法）：**

```
batch_size → 查 capture_buffers（持久缓存）
          → 查 temp_bs（上次 temp buffer）
          → 新建 → 若在 capture_bs 中：写入 capture_buffers
                  → 否则：写入 temp_buffer（只保留一份）
```

**预分配用法：**

```python
# 在模型初始化时预分配所有可能的 batch size，避免推理时动态分配
BaseMoEWrapper.set_capture_batch_sizes([1, 2, 4, 8, 16, 32])
```

---

### 2.3 `_MoEBase`：CPUInfer 单例管理基类

```python
class _MoEBase:
    _cpu_infer_instance = None  # 进程级单例

    @classmethod
    def _get_cpu_infer(cls, cpuinfer_threads, threadpool_count, numa_nodes=None):
        if cls._cpu_infer_instance is None:
            worker_config = kt_kernel_ext.WorkerPoolConfig()
            # 配置 NUMA 子线程池
            worker_config.subpool_count = threadpool_count
            worker_config.subpool_numa_map = numa_nodes or list(range(threadpool_count))
            worker_config.subpool_thread_count = [...]  # 均匀分配线程数
            cls._cpu_infer_instance = kt_kernel_ext.CPUInfer(worker_config)
        return cls._cpu_infer_instance
```

**CPUInfer 是什么？**

`kt_kernel_ext.CPUInfer` 是 C++ 实现的异步任务调度引擎，核心能力：
- 管理多个 NUMA 感知的线程池（`subpool`），每个 subpool 绑定到指定 NUMA 节点
- 提供 `submit_with_cuda_stream(stream, task)` 接口：将计算任务提交到 CPU 线程池，与 GPU 计算异步并行
- 提供 `sync_with_cuda_stream(stream, allow_pending)` 接口：等待 CPU 任务完成，再触发 GPU→CPU 结果拷贝

**NUMA 感知的线程池分配：**

```
cpuinfer_threads=60, threadpool_count=4, numa_nodes=[0,1,2,3]
→ subpool_0: 15 线程, NUMA node 0
→ subpool_1: 15 线程, NUMA node 1
→ subpool_2: 15 线程, NUMA node 2
→ subpool_3: 15 线程, NUMA node 3
```

跨 NUMA 节点的内存访问延迟高达本地访问的 2-3 倍，NUMA 绑定可显著提升专家权重读取效率。

---

### 2.4 `BaseMoEWrapper`：推理流水线核心抽象

这是 `experts_base.py` 的核心类，定义了所有推理后端必须实现的协议：

**构造参数一览：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `layer_idx` | int | 当前 MoE 层索引（用于双缓冲槽选择） |
| `num_experts` | int | 本层专家总数（DeepSeek-V3 为 256）|
| `num_experts_per_tok` | int | 每 token 激活的专家数 top-k（DeepSeek-V3 为 8）|
| `hidden_size` | int | 隐状态维度（DeepSeek-V3 为 7168）|
| `moe_intermediate_size` | int | 专家 FFN 中间维度（DeepSeek-V3 为 2048）|
| `gpu_experts_mask` | `Optional[Tensor]` | shape `[num_experts]`，True=该专家在 GPU，None=全部 CPU |
| `cpuinfer_threads` | int | CPU 推理总线程数 |
| `threadpool_count` | int | NUMA 子线程池数量（通常等于 NUMA 节点数或 TP 数）|
| `weight_path` | str | 量化权重文件路径 |
| `chunked_prefill_size` | int | Prefill 阶段最大 chunk 大小（token 数）|
| `max_deferred_experts_per_token` | int | 每 token 最多延迟几个专家（0 = 禁用延迟）|
| `method` | str | 量化方式：`"AMXINT4"`, `"AMXINT8"`, `"MXFP4"`, `"LLAMAFILE"` 等 |
| `numa_nodes` | List[int] | 显式指定每个 subpool 绑定的 NUMA 节点 |
| `swiglu_limit` | float | DeepSeek-V4-Flash 2604B SwiGLU 激活截断值（通常 0.0=禁用，10.0=DS V4）|

**两个抽象方法（子类必须实现）：**

```python
@abstractmethod
def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
    """从磁盘文件加载已量化的权重（AMX/GGUF 格式）"""
    pass

@abstractmethod
def load_weights_from_tensors(
    self, gate_proj, up_proj, down_proj, physical_to_logical_map_cpu
):
    """从 BF16/FP16 张量在线量化并加载（SGLang 集成时使用）"""
    pass
```

---

## 3. 关键设计模式分析

### 3.1 异步提交 + 延迟同步（Submit/Sync 解耦）

`BaseMoEWrapper` 将 MoE 的前向计算分为两个独立阶段：

```python
# 阶段 1：提交（非阻塞）
def submit_forward(self, hidden_states, topk_ids, topk_weights, cuda_stream):
    # 1. 从 GPU 异步拷贝输入数据到 Pinned CPU Memory
    input_tensor_cpu[slot].copy_(flat_hidden_states, non_blocking=True)
    # 2. 提交 CPU 计算任务到线程池（立即返回，不等待结果）
    self.cpu_infer.submit_with_cuda_stream(cuda_stream, self.moe.forward_task(...))

# 阶段 2：同步（等待结果）
def sync_forward(self, hidden_states, cuda_stream) -> torch.Tensor:
    # 等待 CPU 计算完成
    self.cpu_infer.sync_with_cuda_stream(cuda_stream, allow_pending)
    # 从 Pinned CPU Memory 异步拷贝结果回 GPU
    output_gpu[slot].copy_(output_cpu[slot], non_blocking=True)
    return output_gpu[slot]
```

**为何要解耦 Submit 和 Sync？**

在 MoE 模型的一次 Decode 迭代中，GPU 需要执行 Attention 计算，CPU 需要执行专家 FFN 计算。正常串行执行时，GPU 和 CPU 各自有一半时间处于空闲状态。

通过 Submit/Sync 解耦：

```
时间轴：
Layer L-1: GPU Attention ─────┐
           CPU submit_forward ─┤ （上一层的 CPU 任务并行执行）
                               ↓
Layer L:   GPU Attention ←─── sync_forward（等待 CPU 结果）
           CPU submit_forward ─────────→（启动本层 CPU 任务）
```

这种**异步流水线**使 GPU 和 CPU 几乎完全并行，是 ktransformers 在 Decode 阶段实现接近 GPU 算力上限的核心机制。

---

### 3.2 Expert Deferral（专家延迟调度）

```python
def select_deferred_experts(
    self, expert_ids, expert_scores, protected_k
) -> Tuple[immediate_ids, deferred_ids]:
```

**机制说明：**

当 `max_deferred_experts_per_token > 0` 时，每 token 激活的 `top-k` 个专家被分成两组：

- **Immediate（立即处理组）**：得分最高的 `protected_k` 个专家，在 `submit_forward` 中立即提交 CPU 计算
- **Deferred（延迟处理组）**：剩余 `max_deferred_experts_per_token` 个专家，延迟到**下一层的提交阶段**一起计算

```
例：num_experts_per_tok=8, max_deferred_experts_per_token=2, protected_k=6
→ 前 6 个（得分高）专家：立即计算
→ 后 2 个（得分低）专家：写入 deferred_experts_ids_cpu，在下一层 submit 时追加
```

**延迟状态机：**

```python
# 类级别共享状态（所有层实例共用）
_layer_has_pending_deferred: Dict[int, bool] = {}

# submit_forward 中
if deferred_ids is not None:
    # 提交延迟任务，结果写入 next_slot（避免覆盖 current_slot 的输出）
    self.cpu_infer.submit_with_cuda_stream(cuda_stream, 
        self.moe.forward_task(..., output_cpu[next_slot].data_ptr(), ...))
    BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True

# sync_forward 中
allow_pending = 1 if _layer_has_pending_deferred[layer_idx] else 0
self.cpu_infer.sync_with_cuda_stream(cuda_stream, allow_pending)
```

**为什么可以延迟低分专家？**

在 DeepSeek 等模型中，路由权重分布极度不均匀——排名靠前的少数专家贡献了绝大部分输出。延迟低分专家几乎不影响精度，但可以将 CPU 时间平摊到相邻层，进一步提升流水线并行度。

---

### 3.3 双 Slot 循环缓冲（buffer_depth=2）

```python
current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth  # 0 或 1
next_slot = (current_slot + 1) % KExpertsCPUBuffer.buffer_depth
```

每层使用 `layer_idx % 2` 决定用哪个缓冲槽：

- **偶数层**（`slot=0`）：写入 `output_cpu[0]`，从 `output_gpu[0]` 读取
- **奇数层**（`slot=1`）：写入 `output_cpu[1]`，从 `output_gpu[1]` 读取

当 Expert Deferral 启用时，延迟任务的输出写入 `next_slot`，确保与当前层结果隔离，支持 `sync_with_cuda_stream(allow_pending=1)` 模式：CPU 可以在等待主任务结束的同时继续执行延迟任务。

---

### 3.4 GPU 专家掩码的 C/Python 共享内存

```python
# 使用 pin_memory=True 的 bool 张量
self.gpu_experts_mask = torch.empty(num_experts, dtype=torch.bool, device="cpu", pin_memory=True)

# C++ 侧通过 uint8_t* 指针读写相同内存区域
# （torch.bool 在内存中等价于 uint8_t，每个元素占 1 字节）
```

`gpu_experts_mask` 同时被 Python 层和 C++ 内核共享：
- Python 层通过 `.copy_()` 更新掩码（用于热迁移专家时动态调整 GPU 分配）
- C++ 内核通过原始指针访问掩码，决定哪些专家走 GPU 路径、哪些走 CPU 路径

---

## 4. 继承体系与对外接口

```
_MoEBase（CPUInfer 单例 + 配置验证）
    │
    ├── BaseMoEWrapper（推理流水线，本文件）
    │       │
    │       ├── AMXMoEWrapper（utils/amx.py，AMX INT4/INT8）
    │       ├── NativeMoEWrapper（utils/amx.py，FP8/MXFP4/GPTQ）
    │       ├── LlamafileMoEWrapper（utils/llamafile.py，GGUF）
    │       └── GeneralMoEWrapper（utils/moe_kernel.py，通用量化）
    │
    └── BaseSFTMoEWrapper（sft/base.py，SFT 训练流水线）
            │
            └── AMXSFTMoEWrapper（sft/amx.py，AMX SFT）
```

**对外统一入口（`experts.py`）：**

```python
from kt_kernel import KTMoEWrapper  # 工厂类，自动路由后端

wrapper = KTMoEWrapper(
    mode="inference",   # 或 "sft"
    method="AMXINT4",   # 自动选择 AMXMoEWrapper
    ...
)
# 返回 BaseMoEWrapper 实例
```

---

## 5. 与 SGLang 的对接方式

### 5.1 集成架构总览

kt-kernel 与 SGLang 的集成基于 kvcache-ai 维护的 SGLang fork（`sglang-kt`），该 fork 在原版 SGLang 基础上添加了 kt-kernel 专用参数和 MoE 层替换逻辑：

```
用户请求 (HTTP/WebSocket)
         │
         ▼
  SGLang Server（sglang-kt fork）
  ├── Attention Layer → GPU (Flash Attention / FlashInfer)
  ├── Shared Experts  → GPU
  └── Routed MoE Layer → kt-kernel CPU 侧
           ├── BaseMoEWrapper.submit_forward(...)   ← 提交 CPU 任务
           │   （GPU 同时执行其他层）
           └── BaseMoEWrapper.sync_forward(...)    ← 同步结果
```

### 5.2 关键耦合参数

SGLang 启动时通过以下参数控制 kt-kernel 行为：

| SGLang 参数 | 对应 BaseMoEWrapper 参数 | 说明 |
|-------------|--------------------------|------|
| `--kt-num-gpu-experts` | `gpu_experts_mask`（由 `generate_gpu_experts_masks` 生成） | GPU 上放置的专家总数 |
| `--kt-cpuinfer-threads` | `cpuinfer_threads` | CPU 推理线程总数 |
| `--kt-threadpool-count` | `threadpool_count` | NUMA 子线程池数 |
| `--kt-numa-nodes` | `numa_nodes` | 各子线程池绑定的 NUMA 节点 |
| `--kt-chunked-prefill-size` | `chunked_prefill_size` | Prefill chunk 大小 |
| `--kt-max-deferred-experts` | `max_deferred_experts_per_token` | 每 token 延迟专家数 |
| `--kt-method` | `method` | 量化方式（AMXINT4/MXFP4 等）|
| `--kt-swiglu-limit` | `swiglu_limit` | DS V4-Flash SwiGLU clamp（通常 10.0）|
| `--kt-gpu-prefill-token-threshold` | — | Prefill 超过此 token 数时转为纯 GPU 处理 |
| `--kt-weight-path` | `weight_path` | 预量化权重目录 |

### 5.3 实际调用链路

在 SGLang 的 MoE 推理路径中（`sglang/model_executor/models/deepseek_v2.py` 等），替换 MoE 层的核心逻辑为：

```python
# SGLang 内部（sglang-kt fork 中的 MoE 层实现）
class KTMoELayerForSGLang(nn.Module):
    def __init__(self, layer_idx, model_config, kt_config):
        self.wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=model_config.n_routed_experts,
            num_experts_per_tok=model_config.num_experts_per_tok,
            hidden_size=model_config.hidden_size,
            moe_intermediate_size=model_config.moe_intermediate_size,
            gpu_experts_mask=generate_gpu_experts_masks(
                activation_freq, kt_config.num_gpu_experts
            ),
            cpuinfer_threads=kt_config.cpuinfer_threads,
            threadpool_count=kt_config.threadpool_count,
            weight_path=kt_config.weight_path,
            chunked_prefill_size=kt_config.chunked_prefill_size,
            method=kt_config.method,
            swiglu_limit=kt_config.swiglu_limit,  # DS V4-Flash
        )
        self.wrapper.load_weights(physical_to_logical_map)

    def forward(self, hidden_states, topk_ids, topk_weights, cuda_stream):
        # 推理时使用异步 submit + sync 流水线
        self.wrapper.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
        # ... GPU 侧 Attention 计算 ...
        return self.wrapper.sync_forward(hidden_states, cuda_stream)
```

### 5.4 完整集成代码示例

**安装 sglang-kt：**

```bash
# 方式 A：一键安装（推荐，从 ktransformers 根目录）
./install.sh

# 方式 B：pip 安装
pip install sglang-kt

# 方式 C：从源码安装
git clone --recurse-submodules https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
pip install "third_party/sglang/python[all]"
```

**启动 SGLang 服务（DeepSeek-V3 为例）：**

```bash
python -m sglang.launch_server \
    --model-path /path/to/DeepSeek-V3 \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 1 \
    --trust-remote-code \
    --kt-weight-path /path/to/DeepSeek-V3-AMX \
    --kt-method AMXINT4 \
    --kt-num-gpu-experts 32 \
    --kt-cpuinfer-threads 60 \
    --kt-threadpool-count 2 \
    --kt-numa-nodes 0 1 \
    --kt-chunked-prefill-size 25600 \
    --kt-max-deferred-experts 2 \
    --kt-gpu-prefill-token-threshold 4096
```

**使用 kt CLI 交互式配置（推荐）：**

```bash
# 安装 kt-kernel 后
kt run  # 交互式引导配置，自动生成上述参数

# 或直接启动
kt run --model /path/to/DeepSeek-V3 --method AMXINT4 --num-gpu-experts 32
```

**验证 sglang-kt 支持 kt-kernel：**

```python
from kt_kernel.cli.utils.sglang_checker import check_sglang_kt_kernel_support

result = check_sglang_kt_kernel_support()
if result["supported"]:
    print("SGLang kt-kernel 支持已就绪")
else:
    print(f"不支持：{result['error']}")
    # 需要安装 kvcache-ai fork 版本
```

**客户端调用（与标准 OpenAI API 兼容）：**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")

response = client.chat.completions.create(
    model="DeepSeek-V3",
    messages=[{"role": "user", "content": "你好！"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## 6. 与 LlamaFactory 的对接方式（SFT 模式）

### 6.1 SFT 模式与推理模式的差异

`experts_base.py` 中的 `BaseMoEWrapper` 专为**推理**设计，SFT 训练使用平行的 `BaseSFTMoEWrapper`（`sft/base.py`），两者共享 `_MoEBase` 的 CPUInfer 单例。

| 特性 | 推理模式（BaseMoEWrapper）| SFT 模式（BaseSFTMoEWrapper）|
|------|--------------------------|------------------------------|
| 缓冲区 | 双缓冲（buffer_depth=2）| 单缓冲，grow-only |
| 梯度 | 无 | 支持 backward pass |
| LoRA | 无 | LoRA 适配器（rank, alpha）|
| 目标框架 | SGLang | LlamaFactory / HuggingFace Trainer |
| 核心类 | `BaseMoEWrapper` | `BaseSFTMoEWrapper` + `KTMoELayerWrapper` |

### 6.2 LlamaFactory 集成架构

```
LlamaFactory Trainer（标准 HuggingFace 流程）
         │
         ▼
  HuggingFace 模型（DeepSeek-V3 / Qwen-MoE 等）
  └── MoE Layer（原始 nn.Module）
          │  被替换为
          ▼
  KTMoELayerWrapper（sft/layer.py）
  ├── 路由器（gate，保留在 GPU）
  ├── PEFT LoRA 适配器（挂载在 experts 上，GPU）
  └── CPU 专家计算（委托给 BaseSFTMoEWrapper）
           └── C++ AMX 内核（BF16/INT4/INT8 前后向）
```

**替换 MoE 层的关键步骤（`sft/wrapper.py`）：**

```python
# 遍历模型所有 MoE 层并替换
for layer_idx, moe_layer in enumerate(model.model.layers):
    if hasattr(moe_layer, "mlp") and is_moe_layer(moe_layer.mlp):
        kt_wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            mode="sft",
            method="AMXBF16_SFT",  # SFT 专用方法
            lora_rank=16,
            lora_alpha=32.0,
            ...
        )
        kt_wrapper.load_weights(physical_to_logical_map)
        moe_layer.mlp = KTMoELayerWrapper(
            original_moe=moe_layer.mlp,
            wrapper=kt_wrapper,
            ...
        )
```

### 6.3 完整 SFT 集成代码示例

**环境准备：**

```bash
pip install llamafactory kt-kernel
# 或从源码安装
git clone --recurse-submodules https://github.com/kvcache-ai/ktransformers.git
cd ktransformers && pip install -e ".[sft]"
```

**使用 kt CLI 启动 SFT：**

```bash
kt sft \
    --model /path/to/DeepSeek-V3 \
    --dataset alpaca_data.json \
    --method AMXBF16_SFT \
    --lora-rank 16 \
    --lora-alpha 32 \
    --cpuinfer-threads 60 \
    --output-dir ./output
```

**Python 代码手动集成示例：**

```python
import torch
from kt_kernel import KTMoEWrapper
from kt_kernel.sft.layer import KTMoELayerWrapper
from kt_kernel.sft.arch import MOEArchConfig

# 1. 加载原始模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/DeepSeek-V3",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

# 2. 配置 MoE 架构信息
moe_config = MOEArchConfig(
    router_attr="gate",           # DeepSeek 的 gate 属性名
    experts_attr="experts",
    router_type="topk",
    has_shared_experts=True,
)

# 3. 替换每层的 MoE 模块
for layer_idx, layer in enumerate(model.model.layers):
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
        # 创建 SFT wrapper
        sft_wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=256,
            num_experts_per_tok=8,
            hidden_size=7168,
            moe_intermediate_size=2048,
            gpu_experts_mask=None,   # SFT 模式使用 num_gpu_experts
            num_gpu_experts=0,       # SFT 通常全部在 CPU
            cpuinfer_threads=60,
            threadpool_count=2,
            weight_path=f"/path/to/amx-weights/layer_{layer_idx}",
            chunked_prefill_size=25600,
            method="AMXBF16_SFT",
            mode="sft",
            lora_rank=16,
            lora_alpha=32.0,
        )
        sft_wrapper.load_weights(physical_to_logical_map)

        # 替换原始 MoE 层
        layer.mlp = KTMoELayerWrapper(
            original_moe=layer.mlp,
            wrapper=sft_wrapper,
            lora_params=None,
            moe_config=moe_config,
            hidden_size=7168,
            layer_idx=layer_idx,
        )

# 4. 挂载 PEFT LoRA（可选，挂在 experts 上）
from peft import get_peft_model, LoraConfig
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 注意力层 LoRA
)
model = get_peft_model(model, peft_config)

# 5. 用标准 HuggingFace Trainer 训练
from transformers import Trainer, TrainingArguments
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        bf16=True,
    ),
    train_dataset=train_dataset,
)
trainer.train()
```

---

## 7. 性能调优建议

### 7.1 线程数与 NUMA 配置

```bash
# 查看 NUMA 拓扑
numactl --hardware

# 双路服务器（2 NUMA 节点，每节点 32 核）推荐配置：
--kt-cpuinfer-threads 60    # 总线程数（留 4 个给系统）
--kt-threadpool-count 2     # 2 个 NUMA 子池
--kt-numa-nodes 0 1         # 各子池绑定到 NUMA 0 和 NUMA 1
```

### 7.2 GPU 专家数量的权衡

| num_gpu_experts | VRAM 消耗 | Decode 延迟 | 适用场景 |
|-----------------|-----------|-------------|----------|
| 0 | 最少 | 最高 | 显存极度不足 |
| 32 | 中等 | 中低 | 通常推荐的平衡点 |
| 64 | 较多 | 较低 | 显存充裕时优先 |
| 全部 GPU | 全量 | 最低 | 纯 GPU 推理（退化为标准 SGLang）|

### 7.3 Expert Deferral 配置

```python
# DeepSeek-V3（8 experts per token）推荐
max_deferred_experts_per_token = 2   # 延迟 2 个低分专家，保护 6 个高分专家

# 禁用（追求精度）
max_deferred_experts_per_token = 0
```

### 7.4 缓冲区预分配

```python
# 在模型初始化后、首次推理前调用
# 避免推理时动态分配 Pinned Memory（会短暂阻塞）
BaseMoEWrapper.set_capture_batch_sizes([1, 2, 4, 8])

# 切换模型或测试结束后清理
BaseMoEWrapper.clear_buffer_cache()
```

---

## 8. 常见问题排查

### Q1：`None of the required 'hwloc' found`

```bash
# Ubuntu/Debian
sudo apt-get install -y libhwloc-dev

# CentOS/RHEL
sudo yum install -y hwloc-devel
```

### Q2：SGLang 启动时报 `--kt-gpu-prefill-token-threshold` 参数不存在

当前安装的是官方 SGLang，需要换成 kvcache-ai fork：

```bash
pip uninstall sglang -y
pip install sglang-kt
```

验证：
```python
from kt_kernel.cli.utils.sglang_checker import check_sglang_kt_kernel_support
print(check_sglang_kt_kernel_support())
```

### Q3：子模块（third_party）为空目录，导致 pybind11 找不到

```bash
cd ktransformers
git submodule update --init --recursive
```

### Q4：NUMA 配置错误 `numa_nodes length must match threadpool_count`

```python
# 错误示例
KTMoEWrapper(..., threadpool_count=2, numa_nodes=[0, 1, 2])  # 长度不匹配

# 正确示例
KTMoEWrapper(..., threadpool_count=2, numa_nodes=[0, 1])
```

### Q5：SFT 训练时显存 OOM

SFT 模式下共享专家（`shared_experts`）和路由器（`gate`）仍在 GPU，可通过以下方式缓解：

```python
# 使用梯度检查点
model.gradient_checkpointing_enable()

# 减小 batch size，增大 gradient_accumulation_steps
TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)
```

---

## 附录：核心数据流示意图

```
                    ┌─────────────────────────────────────┐
                    │          GPU (CUDA Stream)           │
                    │  ┌─────────┐      ┌──────────────┐  │
Input tokens ──────►│  │ Routing │─────►│   Attention  │  │
                    │  │  Gate   │      │  + Shared FFN│  │
                    │  └────┬────┘      └──────┬───────┘  │
                    │       │topk_ids           │          │
                    └───────┼───────────────────┼──────────┘
                            │ non_blocking copy │
                    ┌───────▼───────────────────▼──────────┐
                    │     Pinned CPU Memory (双缓冲)         │
                    │  input_cpu  immediate_ids  weights   │
                    └───────────────┬──────────────────────┘
                                    │ submit_with_cuda_stream
                    ┌───────────────▼──────────────────────┐
                    │     CPUInfer 线程池 (NUMA 感知)         │
                    │  ┌──────────┐    ┌──────────────┐   │
                    │  │ NUMA 0   │    │ NUMA 1       │   │
                    │  │ 30 线程  │    │ 30 线程       │   │
                    │  │ AMX INT4 │    │ AMX INT4     │   │
                    │  └──────────┘    └──────────────┘   │
                    └───────────────┬──────────────────────┘
                                    │ output_cpu
                    ┌───────────────▼──────────────────────┐
                    │     Pinned CPU Memory (输出)           │
                    └───────────────┬──────────────────────┘
                                    │ non_blocking copy (sync后)
                    ┌───────────────▼──────────────────────┐
                    │          GPU output_gpu               │
                    └──────────────────────────────────────┘
```

---

*文档生成时间：2026-06-09*  
*对应代码版本：`ktransformers/kt-kernel` main 分支*
