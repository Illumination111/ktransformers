# decode_of_expert：`experts.py` 设计、功能与实现解析

> 文件路径：`ktransformers/kt-kernel/python/experts.py`
> 许可证：Apache-2.0

---

## 一、概述

`experts.py` 是 ktransformers 项目中 **MoE（Mixture of Experts，混合专家）CPU 推理与微调（SFT）** 的核心入口模块。它通过**工厂模式**对外暴露统一的 `KTMoEWrapper` 接口，屏蔽了底层多种量化后端的差异，使调用方无需关心具体的硬件指令集（AMX、AVX-512、AVX2）或量化格式（INT4、INT8、FP8、BF16、MXFP4、GGUF），只需声明所需的 `method` 与 `mode` 即可完成专家层的初始化、权重加载和前向推理。

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       外部调用方（sglang / training）             │
└───────────────────────────────┬─────────────────────────────────┘
                                │ new KTMoEWrapper(...)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KTMoEWrapper（工厂类）                         │
│  ┌──────────────────────┐    ┌────────────────────────────────┐  │
│  │  INFERENCE_METHODS   │    │        SFT_METHODS             │  │
│  │  (10 种推理方法)      │    │  (12 种 SFT 训练方法)           │  │
│  └──────────┬───────────┘    └──────────────┬─────────────────┘  │
└─────────────┼────────────────────────────────┼────────────────────┘
              │ _create_inference_wrapper()      │ _create_sft_wrapper()
              ▼                                 ▼
┌─────────────────────────────┐   ┌────────────────────────────────┐
│      推理后端（BaseMoEWrapper）│   │    SFT后端（AMXSFTMoEWrapper）   │
│  ┌────────────────────────┐ │   │    (来自 sft/amx.py)            │
│  │  AMXMoEWrapper         │ │   └────────────────────────────────┘
│  │  NativeMoEWrapper      │ │
│  │  LlamafileMoEWrapper   │ │
│  │  GeneralMoEWrapper     │ │
│  └────────────────────────┘ │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│              C++ 扩展内核（kt_kernel_ext）                        │
│   CPUInfer / AMXInt4_MOE / AMXFP8_MOE / AVX2BF16_MOE / ...      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、支持的方法集合

### 3.1 推理方法（`INFERENCE_METHODS`）

| 方法名称          | 后端类                | 说明                                            |
|-------------------|-----------------------|------------------------------------------------|
| `AMXINT4`         | `AMXMoEWrapper`       | AMX 加速 INT4 量化，需要 AVX-512 ISA            |
| `AMXINT8`         | `AMXMoEWrapper`       | AMX 加速 INT8 量化，需要 AVX-512 ISA            |
| `RAWINT4`         | `NativeMoEWrapper`    | 原始 INT4 K-Group 量化，支持 AMX/AVX-VNNI/AVX2  |
| `FP8`             | `NativeMoEWrapper`    | 按组 FP8 量化，支持 AMX 与 AVX2 回退            |
| `FP8_PERCHANNEL`  | `NativeMoEWrapper`    | 逐通道 FP8 量化，需要 AVX512_BF16+VBMI          |
| `BF16`            | `NativeMoEWrapper`    | BF16 原始精度，支持 AMX 与 AVX2 回退            |
| `GPTQ_INT4`       | `NativeMoEWrapper`    | GPTQ 对称 INT4，支持 AVX-VNNI-256 与 AVX2       |
| `MXFP4`           | `NativeMoEWrapper`    | MX 微格式 FP4（E2M1 + ue8m0 组缩放），专为 DeepSeek-V4-Flash 设计 |
| `LLAMAFILE`       | `LlamafileMoEWrapper` | GGUF 格式权重，通过 llamafile 后端运行           |
| `MOE_INT4`        | `GeneralMoEWrapper`   | 通用 INT4 内核（`Int4_KERNEL_MOE`）             |
| `MOE_INT8`        | `GeneralMoEWrapper`   | 通用 INT8 内核（`Int8_KERNEL_MOE`）             |

### 3.2 SFT（监督微调）方法（`SFT_METHODS`）

所有 SFT 方法均路由到 `AMXSFTMoEWrapper`，后缀 `_SkipLoRA` 的变体在反向传播时跳过所有 LoRA 计算，仅计算基础权重的梯度输入（`grad_input`），适合冻结 LoRA 适配器时的高效训练。

| 类别      | 方法名称                          |
|-----------|-----------------------------------|
| 标准 SFT  | `AMXBF16_SFT` / `AMXINT8_SFT` / `AMXINT4_SFT` / `AMXINT4_1_SFT` |
| K-Group   | `AMXINT4_KGroup_SFT` / `AMXINT4_1KGroup_SFT` |
| SkipLoRA  | 以上所有方法各自的 `_SkipLoRA` 变体（共 6 种）|

---

## 四、`KTMoEWrapper`：工厂类设计

### 4.1 工厂模式与 `__new__`

`KTMoEWrapper` 利用 Python 的 `__new__` 方法（而非 `__init__`）实现工厂模式。这意味着 `KTMoEWrapper(...)` 的实际返回值**并非** `KTMoEWrapper` 实例本身，而是根据 `mode` 和 `method` 分发出来的具体后端子类实例（如 `AMXMoEWrapper`、`AMXSFTMoEWrapper` 等）。

```python
# 调用方只需关心接口，不用关心底层实现
wrapper = KTMoEWrapper(
    layer_idx=0,
    num_experts=256,
    method="AMXINT4",
    mode="inference",
    ...
)
# wrapper 实际是 AMXMoEWrapper 的实例
```

### 4.2 参数分组

`KTMoEWrapper.__new__` 接受以下几组参数：

**通用参数**（推理与 SFT 均需要）：

| 参数                  | 含义                                        |
|-----------------------|---------------------------------------------|
| `layer_idx`           | 当前 Transformer 层的索引号                  |
| `num_experts`         | 专家总数（如 DeepSeek-V3 为 256）            |
| `num_experts_per_tok` | 每个 token 激活的专家数（top-k，如 8）       |
| `hidden_size`         | Attention 层隐藏维度（如 7168）              |
| `moe_intermediate_size` | MoE FFN 中间层大小（如 2048）             |
| `cpuinfer_threads`    | CPU 推理线程总数                            |
| `threadpool_count`    | NUMA 子池数量（通常等于 TP 并行度）          |
| `weight_path`         | 权重文件所在路径                            |
| `chunked_prefill_size` | 最大预填充块大小（token 数）               |
| `method`              | 量化/后端方法名称字符串                      |
| `mode`                | 运行模式：`"inference"` 或 `"sft"`          |

**仅推理模式参数**：

| 参数                          | 含义                                                |
|-------------------------------|-----------------------------------------------------|
| `gpu_experts_mask`            | `[num_experts]` bool 张量，`True` 表示该专家在 GPU 上 |
| `cpu_save`                    | 是否在 CPU 内存中保留权重（用于在线量化保存）        |
| `max_deferred_experts_per_token` | 每个 token 中可延迟执行的专家数量（流水线优化）   |
| `numa_nodes`                  | 显式指定各子线程池绑定的 NUMA 节点 ID 列表           |
| `swiglu_limit`                | SwiGLU 激活函数截断值（仅 `MXFP4` 方法有效，默认 0.0 禁用）|

**仅 SFT 模式参数**：

| 参数              | 含义                                        |
|-------------------|---------------------------------------------|
| `num_gpu_experts` | GPU 上的专家数量                            |
| `lora_rank`       | LoRA 秩（默认 16）                          |
| `lora_alpha`      | LoRA 缩放因子（默认 32.0）                  |
| `max_cache_depth` | 前向缓存最大深度（默认 1）                  |
| `group_size`      | 量化组大小（K-Group SFT 方法，默认 128）    |
| `zero_point`      | 是否使用零点量化（默认 True）               |

### 4.3 参数校验逻辑

`__new__` 中的校验流程如下：

```
1. 检查 mode 是否为 "inference" 或 "sft"
   └─ 非法 → 抛出 ValueError

2. 根据 mode 检查 method 是否在对应的合法集合中
   └─ 非法 → 抛出 ValueError，并打印全部合法方法

3. SFT 模式下检查 swiglu_limit 是否为 0.0
   └─ 非零 → 抛出 ValueError（SFT 后端不支持 V4-2604B 截断）
```

### 4.4 静态工具方法

| 方法                           | 作用                                         |
|--------------------------------|----------------------------------------------|
| `set_capture_batch_sizes(bs)`  | 预注册特定 batch size，触发提前分配 pinned 内存缓冲区 |
| `get_capture_batch_sizes()`    | 读取当前已注册的 capture batch sizes          |
| `clear_buffer_cache()`         | 清空推理用 CPU 缓冲区缓存，释放内存           |
| `clear_sft_buffer_cache()`     | 清空 SFT 用缓冲区缓存                        |

---

## 五、推理流水线：`BaseMoEWrapper` 与 `KExpertsCPUBuffer`

### 5.1 CPU-GPU 异构流水线

ktransformers 推理的核心思路是将**大量专家权重（可达数百GB）常驻 CPU 内存**，GPU 只保留少量高频激活专家，两者通过异步流水线并行执行，从而在消费级硬件上运行超大规模 MoE 模型。

```
GPU 侧                           CPU 侧
─────────────────────────────────────────────────────────────
token → router → topk_ids/weights
                    │
                    ├─ GPU 专家 → GPU kernel（正常 CUDA 计算）
                    │
                    └─ CPU 专家 → submit_forward()
                                  │
                           非阻塞拷贝至 pinned memory
                                  │
                           cpu_infer.submit_with_cuda_stream()
                                  │              （异步！）
GPU 继续执行其他计算 ─────────────┤
                                  │ C++ 线程池并行执行专家前向
                                  ▼
sync_forward() → cpu_infer.sync() → 结果 DMA 回 GPU
```

### 5.2 `KExpertsCPUBuffer`：双缓冲管理

`KExpertsCPUBuffer` 是一个类级别（全局共享）的缓冲区管理器，使用**双缓冲（`buffer_depth=2`）** 设计防止流水线中不同层之间的数据竞争。

每个缓冲区组包含 7 种张量，均分配在 `pin_memory=True` 的 CPU 内存上，以加速 DMA 拷贝：

| 张量名称                      | 形状                                   | 用途                         |
|-------------------------------|----------------------------------------|------------------------------|
| `input_tensor_cpu`            | `[batch_size, hidden_size]` (BF16)     | 从 GPU 拷贝的输入隐藏状态     |
| `immediate_experts_ids_cpu`   | `[batch_size, num_experts_per_tok]` (long) | 立即执行的专家 ID 列表    |
| `deferred_experts_ids_cpu`    | `[batch_size, num_experts_per_tok]` (long) | 延迟执行的专家 ID 列表（-1 为占位）|
| `weights_cpu`                 | `[batch_size, num_experts_per_tok]` (float32) | 专家权重系数              |
| `output_cpu`                  | `[batch_size, hidden_size]` (BF16)     | CPU 计算输出（写入位置）      |
| `bsz_tensor_cpu`              | `[1]` (int32)                          | 当前 batch size 标量          |
| `output_gpu`                  | `[batch_size, hidden_size]`            | 结果在 GPU 上的最终存放位置   |

缓冲区的分配策略分为两级：
- **Capture 缓冲区**：通过 `set_capture_batch_sizes()` 预注册的 batch size，常驻内存，避免重复分配。
- **临时缓冲区**：对未注册的 batch size 分配临时缓冲区（`temp_buffer`），只保留最近一次的。

### 5.3 延迟专家执行（`select_deferred_experts`）

当 `max_deferred_experts_per_token > 0` 时，启用延迟执行优化：

```
topk 专家 = protected（立即）+ deferred（延迟）
                 │                    │
         按专家得分 top-k 选出     剩余专家延迟
         (protected_k 个)          到下一个 slot
```

- **立即专家（immediate）**：得分最高的 `protected_k` 个专家立即提交 CPU 计算任务
- **延迟专家（deferred）**：其余专家写入下一个双缓冲 slot，在下一次 `sync_forward` 时一并合并

这种设计能将 CPU 专家计算与 GPU 其他计算（如 attention）**充分重叠**，减少端到端延迟。

### 5.4 前向函数接口

`BaseMoEWrapper` 提供三种前向接口：

```python
# 异步提交（不阻塞 GPU）
wrapper.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)

# 等待结果并拷回 GPU
output = wrapper.sync_forward(hidden_states, cuda_stream)

# 同步执行（submit + sync 合并）
output = wrapper.forward(hidden_states, topk_ids, topk_weights, cuda_stream)
```

推荐使用 `submit_forward` + `sync_forward` 分离的方式，以充分利用异步流水线。

### 5.5 CPUInfer 单例与 NUMA 感知线程池

通过 `_MoEBase._get_cpu_infer()` 方法维护全局唯一的 `CPUInfer` 实例（C++ 扩展对象）：

```python
worker_config.subpool_count = threadpool_count
worker_config.subpool_numa_map = [0, 1, 2, ...]  # 各子池绑定的 NUMA 节点
worker_config.subpool_thread_count = [...]        # 各子池线程数（均分 cpuinfer_threads）
```

线程被**均匀分配**到各 NUMA 子池（余数优先分配到前几个子池），结合 `numa_nodes` 参数可以精确控制线程对 NUMA 内存节点的亲和性，从而避免跨 NUMA 内存访问带来的性能损耗。

---

## 六、推理后端详解

### 6.1 `AMXMoEWrapper`（AMX INT4/INT8）

面向英特尔第四代 Xeon（Sapphire Rapids）及以上处理器，使用 AMX（Advanced Matrix Extensions）瓦片矩阵指令加速量化矩阵乘法。

**权重加载流程**：
1. 检测权重目录是否存在 `.safetensors` 合并权重文件
2. 若存在：通过单例 `SafeTensorLoader` 读取已量化（INT4/INT8 + 缩放因子）的权重，直接传入 C++ 内核指针列表
3. 若不存在（`cpu_save=True` 模式）：读取原始 BF16 权重，触发 C++ 侧在线量化并保存到磁盘
4. 通过 `cpu_infer.submit(moe.load_weights_task(...))` 将权重加载任务提交到异步线程池

**NUMA 感知内存布局**：权重张量按 NUMA 节点分片存储，`gate_ptrs`/`up_ptrs`/`down_ptrs` 的结构为 `[numa_id][expert_id] → pointer`，保证每个线程池子集访问本地内存。

### 6.2 `NativeMoEWrapper`（RAWINT4 / FP8 / BF16 / GPTQ\_INT4 / MXFP4）

通用量化后端，支持预量化的 SafeTensor 权重，并在运行时自动选择最优 CPU 内核：

| 方法            | 首选内核               | 回退内核              |
|-----------------|------------------------|-----------------------|
| `RAWINT4`       | AMX（`AMXInt4_KGroup_MOE`）| AVX-VNNI-256 → AVX2  |
| `FP8`           | AMX（`AMXFP8_MOE`）    | AVX2（`AVX2FP8_MOE`）|
| `FP8_PERCHANNEL`| AMX（`AMXFP8PerChannel_MOE`）| 无回退（仅 AMX）  |
| `BF16`          | AMX（`AMXBF16_MOE`）   | AVX2（`AVX2BF16_MOE`）|
| `GPTQ_INT4`     | AVX-VNNI-256           | AVX2                  |
| `MXFP4`         | AMX（`AMXFP4_KGroup_MOE`）| AVX2（`AVX2MXFP4_MOE`）|

**MXFP4 特别说明**：针对 DeepSeek-V4-Flash 2604B 路由专家设计，使用 E2M1 nibble 打包权重 + ue8m0 每组缩放因子（已转换为 bf16）。当 `swiglu_limit != 0.0` 时（通常为 10.0），C++ `act_fn` 会在 SwiGLU 激活前对 gate/up 值做截断，与 TensorRT-LLM 的 `gemm1_clamp_limit` 行为保持一致。

**内存优化**：权重加载完成后立即 `del` 临时张量，并调用 `NativeMoEWrapper._release_loader()` 关闭所有 mmap 句柄，避免长期占用文件描述符和内存映射。

### 6.3 `LlamafileMoEWrapper`（GGUF）

通过 llamafile 后端加载 GGUF 格式的量化权重，适合与 llama.cpp 生态兼容的量化模型（Q4\_K、Q8\_0 等）。

### 6.4 `GeneralMoEWrapper`（MOE\_INT4 / MOE\_INT8）

基于通用 `Int4_KERNEL_MOE` / `Int8_KERNEL_MOE` 内核的通用实现，可在不具备 AMX 的旧版 CPU 上运行。

---

## 七、SFT（监督微调）路径

SFT 模式通过 `_create_sft_wrapper()` 私有函数路由到 `AMXSFTMoEWrapper`（来自 `sft/amx.py`），该模块集成了：

- **LoRA 适配器**：秩为 `lora_rank`，缩放因子为 `lora_alpha`
- **前向缓存**：深度为 `max_cache_depth`，缓存前向激活用于反向传播
- **SkipLoRA 变体**：在反向传播时完全跳过 LoRA 路径，仅计算基础权重的 `grad_input`，适用于只需要训练非 MoE 部分的场景

SFT 模式不支持 `swiglu_limit`（在工厂方法中直接拦截），因为现有 SFT 后端不实现 V4-2604B 截断逻辑。

---

## 八、`swiglu_limit` 的多层防护设计

`swiglu_limit` 是一个专为 DeepSeek-V4-Flash 2604B 路由专家设计的特殊参数，代码中设置了**三层防护**，防止误用：

```
第一层（experts.py KTMoEWrapper.__new__）：
  - 仅 mode="inference" 时允许非零值
  - mode="sft" + swiglu_limit != 0.0 → ValueError

第二层（experts.py _create_inference_wrapper）：
  - 仅 method="MXFP4" 时将 swiglu_limit 传入 extra_kwargs
  - 其他方法 + swiglu_limit != 0.0 → ValueError（含诊断信息，提示 SGLANG_DSV4_2604_SUBMODE 环境变量）

第三层（utils/amx.py NativeMoEWrapper.__init__ 及 load_weights）：
  - 即使绕过前两层直接构造 NativeMoEWrapper，method != "MXFP4" + swiglu_limit != 0.0 → ValueError
```

这种"深度防御"设计确保 SwiGLU 截断绝不会静默地应用于 RAWINT4/FP8/BF16 等其他量化路径。

---

## 九、工具函数：`generate_gpu_experts_masks`

定义于 `experts_base.py`，根据专家激活频率表生成 GPU 专家掩码：

```python
activation_freq: Tensor[num_layers, num_experts]
→ gpu_experts_masks: BoolTensor[num_layers, num_experts]
```

算法：将全局所有层的专家频率展平后，取 top-k 激活频率最高的专家标记为 GPU 专家。调用方将每层对应的 mask 行传入 `KTMoEWrapper` 的 `gpu_experts_mask` 参数，实现"把热门专家放 GPU、冷门专家留 CPU"的混合卸载策略。

---

## 十、设计总结

| 设计目标               | 实现手段                                                    |
|------------------------|-------------------------------------------------------------|
| 统一对外接口           | `KTMoEWrapper` 工厂类，`__new__` 返回具体后端实例           |
| 多后端透明切换         | `INFERENCE_METHODS` / `SFT_METHODS` 集合 + 分发函数        |
| CPU-GPU 异步流水线     | `submit_forward` / `sync_forward` 分离 + CUDA stream 绑定   |
| 内存效率               | pinned memory 双缓冲 + 权重加载后及时释放                   |
| NUMA 亲和性            | `WorkerPoolConfig` 子池绑定 + 按层 slot 调度                |
| 硬件自适应             | 运行时检测 CPU flags，自动选择 AMX/AVX-512/AVX-VNNI/AVX2    |
| 安全防护               | 多层 `swiglu_limit` 校验 + 方法/模式合法性校验              |
| 训练支持               | SFT 模式 LoRA 集成，SkipLoRA 变体降低训练开销               |
