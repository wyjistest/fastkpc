# Phase 10 精确提速：发现与建议（评审后修订版）

> 执行跟踪：[2026-10-20 实验记录](PHASE10_EXACT_SPEEDUP_EXECUTION_20261020.md)。
> 下文保留评审时的假设与建议；实际 A/B、归因和 GO/STOP 以执行记录为准。
> 本轮执行已完成：connections=32 的完整 canonical 单次开发对照为
> 1159.691 → 880.659 s，所有 CI/pMax 逐位不变；pinned 无额外收益并撤回，
> 扩 eig lanes 与当前 structured GPU 实现不进入生产，R2/R3 仍为 NOT YET GO。
> 以下“待做”是评审时状态，不代表执行记录中的最新结果。

> 日期：2026-10-20
> 基线：`main` @ `6a6a4b3`（runtime producer `dcb1749`；`fastkpc/build/fastkpc_cuda.so` 的 SHA-256 为 `fe8286f7…`，与审计值一致）
> 性质：分析与验证。没有改动任何生产代码、契约或 artifact。实验脚本在 `fastkpc/tools/research/exact_speedup_20261020/`。
> 范围：strict full-CUDA `dcc.gamma` 路线，canonical 351×48，默认 `m.max = Inf`。

## 修订说明

初版的 eig 剖析用错了配置。canonical 证据使用 retained 配置：11 GiB component cache、residual bypass、8 条 eig lane。我的剖析没有设置这三个开关，跑的是默认路线。在默认路线下，每个 component batch 都会新建一个**单 lane** 的临时执行上下文（`fastkpc/src/cuda/full_cuda_ci_vertical.cu:3306`）。初版 §3 看到的完全串行、16 万次 `cudaFree`、大量 stream，都来自这个默认路线，不能代表 retained 路线。

本版在 retained 配置下重新剖析，改写了 §3，并收回了初版几处证据不足的说法。两版冲突时以本版为准。

第二轮评审后又收紧了三处：
- eig 阶段的拷贝调用时间几乎全部和 GPU 执行重叠，不能当作传输开销（§3.3）。
- 批宽只能解释本子集的 lane 使用率；旧收口文档在大批次上扩 lane 同样无收益，原因还要独立归因（§3.3）。
- 结构化 eig 的验收门槛补齐了执行语义、资源释放和缓存身份（§4.7）。

| 初版说法 | 更正 |
|---|---|
| full eig 被 cuSOLVER 内部 `cudaFree` 完全串行，并发度 1.00 | 这是默认路线的现象。retained 路线上 eig 内核并发 1.0–1.85，存在 GPU 未充分利用的时段；主要限制因素尚未确定（§3） |
| eig 内核分布在 12 个 stream 上 | 分析脚本只打印了最忙的 12 个。默认路线实际有 3,934 个 stream，retained 路线是 9 个 |
| 500×50 样例比 20 核 CPU 还慢（25.1 s 对 19.9 s） | retained 配置下只要 13.2 s，比 CPU 快 |
| 优化器只用到约 7% 的 GPU | 只能说"已发射 block 的并发数上界约 35，容量约 512"。这是占用率的说法，不是算力利用率 |
| 锁等待 273 s | 这是约 600 个线程的等待时间之和，对应的墙钟窗口只有约 10.8 s，不能当作可省下的墙钟时间。retained 剖析中为 248 s |
| 常驻优化器"逐位不变" | 结构上有逐位不变的基础，但还没有实现和验证 |
| 结构化 eig "保守估计 10–20 s"、"O(n²)" | 撤回耗时估计，GPU 原型出来之前不给阶段耗时。复杂度包含 O(m²k) 的向量恢复，只有 k 固定（envelope 规定 numCol = 35）时才是 n 的二次 |
| 优化器 15–40 s；单卡 100–150 s；双卡 60–90 s | 全部撤回。剖析只覆盖 conditioning depth ≤ 3，外推依据不足 |
| 结构化参考实现 | 原来不支持只有两个取值的输入。现已支持，并加入二值、三值用例 |

## 0. 摘要

当前开发运行 1140.851 s（retained 配置）。其中 selected-SP 优化器 596.810 s，dCov/full eig 437.082 s，两者合计占 91%。门槛是 Checkpoint B < 180 s，最终五次中位数 ≤ 120 s。

1. **优化器：主机缓冲是 pageable，导致驱动锁争用。** 两种配置下都能复现。
   - 优化器内核每次平均只发射 7.4 个 block，已发射 block 的并发数上界约 35，该内核在 4090 上的容量约 512。
   - 优化器阶段墙钟 10.8 s，约 600 个主机线程在 pageable `cudaMemcpyAsync` 内部等驱动锁，等待时间之和 248 s。
   - 合成复刻中改用 pinned 后快 3.3×，但这只是机制证据。**下一步是生产环境 A/B（改动不涉及数值）。**
2. **full eig：retained 路线上存在 GPU 未充分利用的时段，但主要限制因素尚未确定。** 初版的"cuSOLVER 完全串行"不成立。
   - eig 内核并发 1.0–1.85。在分析脚本划定的 eig 窗口内，16% 的时间没有任何内核在跑，另有约 31% 的时间只有小内核（block 数少于 128）在跑。
   - 这个子集每个 batch 平均只有 3.6 次 syevd，限制了 lane 的使用率。每次 syevd 内部还有一次阻塞主机的范数回读。
   - 主机在 `cudaMemcpyAsync` 里的 37.5 s 中，有 37.0 s 与 GPU 内核执行重叠。这些调用主要是在等已提交的计算完成，不能当作传输开销。
   - 旧收口文档在 9/16/48 个 component 的大批次上扩 lane 也没有收益，批宽解释不了，需要独立归因。
3. **结构化 eig：CPU 参考实现的数值已验证，GPU 实现还没有。**
   - 真实数据上和 `kpcalg::dcov.gamma` 的 p 值最大差 6.2e-13。
   - 在含二值、三值在内的 16 类极端输入上，和稠密解的 p 值最大差 1.2e-13。
   - 需要先做 GPU 隔离原型，并通过全量资格验证。
4. **契约：** 数值契约不要求和旧 GPU 输出逐位一致；架构契约明确没有冻结 eigensolver representation。因此，数学语义相同、通过全部门槛的新 eig 算法，不需要因为低位比特不同就升级数值契约大版本。但抽样得到的 1e-13 不能代替全量资格验证。
5. **目前没有证据表明已经有一条可靠路径能达到 120 s 门槛。**

## 1. 基线与环境

### 1.1 canonical 阶段分解

数据来自 `MGCV_GPU_SEMANTIC_CLOSURE.md`。该运行使用 retained 配置：component 请求/命中/缺失 228,033 / 117,350 / 110,683，eig lane 请求/峰值 8 / 8。

| 阶段 | host boundary (s) |
|---|---:|
| selected-SP 优化器 | 596.810 |
| dCov / full eig | 437.082 |
| exact-fixed 残差 | 34.420 |
| native setup | 22.814 |
| 控制/打包（未归因） | 49.725 |
| **合计** | **1140.851** |

`performance_budget_v2` 的门槛：fresh-data compute-warm 中位数 ≤ 120 s，且 ≤ 0.80 × 同批次正确基线中位数（v1 campaign 中为 601.431 s）；cold 中位数 ≤ 正确基线。

### 1.2 环境与配置

- 2× RTX 4090（FP64 吞吐是 FP32 的 1/64），i9-10900X（10 核 20 线程），CUDA 12.5，驱动 555.42.06。
- 流水线运行都用 `CUDA_VISIBLE_DEVICES=0` 和单线程 BLAS；微基准跑在 GPU1。
- retained 配置：

  ```text
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES=11811160064
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS=1
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES=8
  ```

  这三个开关默认都是 0。研究脚本默认设置它们；要跑默认路线需设 `FASTKPC_PROFILE_DEFAULT_ROUTE=1`。

### 1.3 本次额外跑的工作负载

| 工作负载 | 配置 | 总耗时 | full eig | 多惩罚优化器 |
|---|---|---:|---:|---:|
| 500×50 开发样例（默认 Inf，止于第 3 层） | retained | 13.222 s | 7.181 s | 只有 16 个 target |
| 同上 | 默认路线（仅作对照） | 25.075 s | 18.675 s | — |
| canonical 前 22 列，`max_conditioning_size = 3`（nsys） | retained | 65.644 s | 34.971 s | 10.876 s / 4,662 个 target |
| 同上 | 默认路线（仅作对照） | 191.133 s | 145.567 s | 10.799 s / 4,662 个 target |
| canonical 前 30 列，`max_conditioning_size = 3` | 默认路线 | 441.834 s | 346.018 s | 23.709 s / 11,225 个 target |

- 500×50 样例的 20 核 CPU 基线是 19.9 s。
- eig 配置不影响优化器：22 列子集上默认路线 10.80 s，retained 路线 10.88 s。所以 30 列运行只用来提供优化器计数。

## 2. 发现一：优化器的主机缓冲是 pageable，导致驱动锁争用（改动不涉及数值）

### 2.1 回执计数（粗略估计）

| | 8 月 8 日 canonical 回执（exact-primary 之前） | 30 列子集 |
|---|---:|---:|
| 多惩罚 target / setup | 80,751 / 7,472 | 11,225 / 1,254 |
| 优化器墙钟 | 132.5 s | 23.7 s |
| 各 setup 主机耗时之和 ÷ 墙钟 | 46.9 | 45.9 |
| 插桩 CTA 周期 ÷（墙钟 × 2.52 GHz） | ≈16 | ≈20 |
| 每个 setup 的平均主机耗时 | 831 ms | 868 ms |
| 每个 target 的平均插桩 CTA 时间 | ≈27 ms | ≈42 ms |

主机端平均约有 46 个 setup 在飞，插桩估算出同时在执行的 CTA 只有十几个。两点说明：插桩只覆盖内核的部分阶段；时钟频率是假设值。所以这张表只用来说明"主机侧耗时远大于 GPU 侧"，不作为精确的占用率。

### 2.2 nsys 实测（22 列子集，retained 配置）

| 指标 | 值 |
|---|---:|
| `optimize_multi_penalty_targets_kernel` 发射次数 | 628 |
| 每次平均 block 数 | 7.4 |
| 内核时长之和 / 内核时间并集 = 平均并发内核数 | 48.63 s / 10.62 s = 4.58 |
| 已发射 block 并发数（Σ 时长 × grid ÷ 跨度，是常驻 block 的上界） | 34.6 |
| 该内核的常驻容量（255 寄存器 × 64 线程 → 每 SM 4 个） | 约 512 |
| 内核时长 p50 / p90 / p99 / max | 56.1 / 125.7 / 465.1 / 1078.5 ms |
| 优化器阶段墙钟；发起 CUDA 调用的主机线程数 | 10.82 s；629 |
| `cudaMemcpyAsync` 耗时（各线程之和） | 1,993 次，258.52 s |
| 其中发生在拷贝内部的 `pthread_mutex_lock` | 619 次，247.87 s |
| 发生在拷贝外部的锁等待 | 3 次，≈0 s |
| 阶段内的拷贝 | pageable H2D 1,295 次，pageable D2H 592 次 |

注意两点：
- block 并发数不等于算力利用率。
- 248 s 是各线程等待时间之和，对应的墙钟窗口只有 10.8 s。

默认路线下的剖析给出几乎相同的数字（628 次发射，并发 4.55，拷贝内锁等待 272.7 s）。

### 2.3 机制

- 每个 setup 由一个主机线程处理：上传 Y → 发射几个内核 → 下载 `host_states` → `cudaStreamSynchronize`。一个窗口最多 64 个 setup、64 个线程，窗口之间要 join。相关代码：
  - 窗口循环：`fastkpc/src/full_cuda_ci_one_call.cpp:5236` 的 `run_window`
  - worker 线程：`fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu:6305`
  - 并发上限：`kMultiPenaltyGcvMaximumConcurrentSetups = 64`（`fastkpc/src/cuda/mgcv_multi_penalty_gcv.hpp:27`）
  - 各 bucket 上限 64/64/8/2/1：`fastkpc/src/cuda/mgcv_multi_penalty_gcv_capacity.cpp:58`
- 两块主机缓冲都是 pageable：
  - 上传源 Y 来自 `request.Y`（`std::vector<double>`），见 `fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu:5771`
  - 下载目标 `prepared->host_states` 是 `std::vector<DeviceOptimizerState>`，见 `fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu:5869`
- pageable 的 `cudaMemcpyAsync` 对主机实际上是同步的，要经过驱动内部带锁的暂存缓冲；D2H 还要等本 stream 上的内核跑完。实测锁等待全部发生在拷贝调用内部，与"其他线程的上传排队，导致它们的内核迟迟起不来"这个解释一致。

### 2.4 合成复刻（`pinned_concurrency_bench.cu`，GPU1）

设置：47 个主机线程，每个线程 6 个"setup"，每个内核 8 个 CTA、约 55 ms。

| 主机缓冲 | 墙钟 | 平均并发内核 |
|---|---:|---:|
| pageable | 6.68 s | 2.8 |
| pinned | 2.04 s | 34.0 |

这在本机和本驱动上复现了同一机制。它是机制证据，不代表生产收益。

### 2.5 建议

- **R1：改用 pinned 缓冲，先做生产 A/B。** Y 上传和 `host_states` 下载改用每个 prepared handle 持有的 pinned 缓冲（`cudaMallocHost`，或对复用的 vector 做 `cudaHostRegister`）。验收：canonical 逐位门槛不变，测量优化器 boundary 时间，用 `analyze_nsys_profile.py` 复测。之后再检查 single-penalty、exact-fixed、method 和 eig 路径中逐批的 pageable 传输。
- **R2：一层一次发射的常驻优化器内核。** `optimize_multi_penalty_targets_kernel`（`fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu:2362`）一个 block 对应一个 target，读共享的只读 setup 数据，写自己的 workspace 和 state，没有跨 target 的归约。这为逐位不变提供了结构基础，但实际重构还要处理 workspace 的初始化与复用、不同容量 bucket、结果生命周期和长尾，必须用逐位门槛验证。是否做，等 R1 之后的剖析再定。
- **R3：lane-per-target**（§6）。同样等 R1 之后的剖析再定。

本节不给优化器的阶段耗时估计。剖析只覆盖 conditioning depth ≤ 3，canonical 更深的层级（q 最大 73，单个 target 最多迭代 187 次）没有剖析。

## 3. 发现二（已更正）：retained 路线的 full eig 存在 GPU 未充分利用的时段，主要限制因素待归因

### 3.1 初版的错误

在默认路线下，`component_cache == nullptr` 时每个 batch 会新建 `FullCudaCiLegacyEigExecutionContext`，lane 数为 `miss_count > 0 ? 1 : 0`（`fastkpc/src/cuda/full_cuda_ci_vertical.cu:3306`），batch 结束时销毁。初版看到的 3,934 个 stream、138,593 次 `cudaMalloc` 和 161,566 次 `cudaFree`、并发度 1.00，都是这个路线的特征。"cuSOLVER 内部 `cudaFree` 让 8 条 lane 完全串行"这一结论撤回。

### 3.2 两种配置的对照（22 列子集，nsys）

| 指标 | 默认路线（对照） | retained 8 lane |
|---|---:|---:|
| eig 窗口（分析脚本把相隔不到 200 ms 的 `sytrd4_cta` 合并而成） | 174.0 s | 48.2 s |
| eig 内核涉及的 stream 数 | 3,934 | 9（8 条 lane + 主 stream） |
| syevd 次数 / distance-row batch 数 | 20,078 / 3,933（每批 5.11） | 9,040 / 2,488（每批 3.63） |
| 有任意内核在跑的时间占比 | 89% | 84% |
| 有 ≥ 128 block 的内核在跑的时间占比 | 35% | 53% |
| `lansy_M_stage2` 之后紧跟的 D2H（每次 syevd 一次） | 20,078 | 9,040 |
| `cudaMalloc` / `cudaFree` | 138,593 / 161,566 | 14,847 / 12,716 |
| 发起 CUDA 调用的主机线程 | 17 | 1 |
| 窗口内 `cudaMemcpyAsync` | 93,333 次，149.3 s（各线程之和） | 94,911 次，37.5 s（单线程） |
| 其中与 GPU 内核执行重叠 / 没有内核在跑 | — | 37.0 s / 0.5 s |

eig 窗口是分析脚本的合并规则给出的，不同于 receipt 里的 `cuda_dcov_full_eig_ms`（retained 34.971 s，默认路线 145.567 s）。

retained 路线上主要 eig 内核的并发：

| 内核 | 次数 | 时长之和 | 并集 | 并发度 | 平均 block |
|---|---:|---:|---:|---:|---:|
| `sytrd4_gpu` | 61,117 | 21.03 s | 20.19 s | 1.04 | 2576 |
| `sytrd4_cta` | 9,040 | 20.23 s | 11.24 s | 1.80 | 1.0 |
| `laed2_par` | 52,982 | 7.79 s | 5.38 s | 1.45 | 7.4 |
| `ampere_dgemm_64x64_lower_nt` | 122,234 | 7.35 s | 3.97 s | 1.85 | 14.9 |
| `laed4_par` | 52,982 | 6.34 s | 4.65 s | 1.36 | 412 |

8 条 lane 的负载很不均衡：各 lane stream 上的 eig 内核发射数从 98,712 降到 10,296。原因是 round-robin 按每批的 miss 数取 `min(component_count, 8)` 条 lane。

### 3.3 解读

- **存在 GPU 未充分利用的时段。** 在 eig 窗口内，16% 的时间 GPU 上没有任何内核，另有约 31% 的时间只有小内核在跑。只有 1 个 block 的 `sytrd4_cta` 累计 20.2 s。
- **小批宽限制了这个子集的 lane 使用率，但解释不了此前的实验。** 本子集每批平均只有 3.6 次 syevd。旧收口文档（`PHASE10_FULL_EIG_COMPONENT_CLOSURE.md:153`）在 9、16、48 个 component 的批次上把 lane 扩到 9 或 16，设备加速只有 1.0013×、1.0024×、0.9986×；那些批次的任务足够用满更多 lane。大批次扩 lane 无收益的原因需要独立归因。旧收口文档本身的结论是"剩余主要成本是 cuSOLVER 特征分解本身"，并没有说硬件已饱和。
- **拷贝调用时间不等于传输开销。** 唯一的发起线程在 `cudaMemcpyAsync` 里花了 37.5 s，其中 37.0 s GPU 正在执行内核，只有 0.5 s 没有内核在跑。这些调用主要是在等已提交的计算完成，不能当作 pinned 能消除的开销。0.5 s 也不是收益上限：改善提交并发可能缩短 GPU 的总执行时间，但要靠 A/B 证明。
- **已观察到、但没有定量的因素：**
  - cuSOLVER 每次 syevd 都在范数内核之后做一次阻塞主机的回读（9,040 次，正好等于 syevd 次数），用来决定缩放分支。
  - eig 窗口内仍有 12,716 次 `cudaFree` 和 14,847 次 `cudaMalloc`，来源（是否在 syevd 内部）未确认。
  - 所有 eig 调用由一个主机线程提交；每批结束时要 join 各 lane 并下载结果。
- **结论。** 现有证据支持"存在 GPU 未充分利用的时段"，还不足以确定主要限制因素。

### 3.4 可选方向（效果都未量化）

- 结构化 eig 的 GPU 原型（§4）：不再调用 cuSOLVER，从而避开每次调用的主机同步和 1-block 的三对角化内核；还可以把一个 batch 的全部分量融合进一次发射。
- eig 路径逐批的元数据和结果传输改用 pinned 缓冲。拷贝时间大多是在等计算，收益只能靠 A/B 判断。
- 先独立归因大批次扩 lane 无收益的原因，再决定在 cuSOLVER 路线上还值不值得投入。
- 升级到提供 `cusolverDnXsyevBatched` 的 CUDA（≥ 12.6 Update 2，本机 12.5 没有），需要重新做资格验证。

## 4. 发现三：一维距离矩阵的精确结构化特征分解（CPU 参考已验证，GPU 未实现）

### 4.1 legacy 实际需要什么

`kpcalg/R/dcovgamma.R:51` 对**原始**（未中心化）的距离矩阵 `D = as.matrix(dist(x))` 调用 `eigs(D, numCol)`（which = "LM"），取 |λ| 最大的 k = numCol 个特征对，然后计算：

```
W    = H U                       （先选特征对，再对列中心化）
nV2  = Σ_ij λ_i μ_j (w_i·w_j)² / n
mean = mean(Dx) · mean(Dy)       （用完整的 D）
var  = c(n) · [Σ λ_i λ_j (w_i·w_j)²]_x · [Σ μ_i μ_j (w_i·w_j)²]_y / n²
p    = 1 − pgamma(nV2, shape = mean²/var, rate = mean/var)
```

`numerical_contract_v1.json:93` 的 `component_eigen_selection` 只规定了语义（按 |λ| 从大到小的稳定顺序，相等时取原始下标较大者），没规定算法。所有 CI 检验里 x、y 都是一维向量：第 0 层是原始列，其余层是残差。

### 4.2 数学

记号：排序后的不同取值 v₁ < … < v_m，计数 c_j，间隔 h_j = v_{j+1} − v_j（j = 1..m−1），累计计数 t_j = c₁ + … + c_j。

1. **中心化部分是布朗桥。** 令 f_j(i) = 1{x_i ≤ v_j}，则 |x_i − x_l| = Σ_j h_j (f_j(i) − f_j(l))²。只依赖单个下标的项在双中心化时被消掉：

   ```
   HDH = −2 · H F G Fᵀ H,          G = diag(h)
   K = Fᵀ H F,   K_jl = min(t_j, t_l) − t_j t_l / n      （布朗桥协方差）
   Q = K⁻¹ 是三对角阵：Q_jj = 1/c_j + 1/c_{j+1},  Q_{j,j+1} = −1/c_{j+1}
   ```

2. **闭式双对角因子。** Q = L_Q L_Qᵀ，其中

   ```
   (L_Q)_jj      =  √( t_{j+1} / (t_j · c_{j+1}) )
   (L_Q)_{j+1,j} = −√( t_j / (t_{j+1} · c_{j+1}) )
   ```

   只依赖计数，全是正数运算，没有抵消。令 B = G^{−1/2} L_Q（下双对角），O = H F L_Q；OᵀO = L_Qᵀ K L_Q = I，所以 O 是精确正交阵，且与间隔无关。

3. **HDH 的特征对。** −HDH/2 = O B⁻¹ B⁻ᵀ Oᵀ。设 B = Q_B Σ Pᵀ，则 HDH 的非零特征值 a_j = −2 / σ_j(B)²，对应特征向量 U = O P。

4. **原始 D 是箭头形矩阵。** 令 e = 1/√n · 1，s = D e，α = eᵀ D e，则 D = HDH + e sᵀ + s eᵀ − α e eᵀ。在正交基 [e, U] 下：

   ```
   D ≅ [ α    b̃ᵀ     ]          b̃ = Uᵀ s = Pᵀ (Oᵀ s)
       [ b̃   diag(a) ]
   ```

   - 特征值满足久期方程 α − λ − Σ_j b̃_j² / (a_j − λ) = 0，由交错性每个区间一个根；|λ| 最大的 k 个是 1 个正根（Perron 根）加 k−1 个最负的根。
   - 特征向量：y_j = b̃_j / (λ − a_j)，v ∝ e + O P y。
   - b̃_j ≈ 0 的模态解耦（deflation），此时 (a_j, O p_j) 本身就是特征对。所有 b̃ 都解耦时（例如两个取值、计数相等），e 本身就是特征向量，特征值为 α。

5. **行和按间隔累加：** rs_i = Σ_{k<i} k·g_k + Σ_{k≥i} (n−k)·g_k（排序位置），平移不变，无抵消。α = Σ rs / n，mean(D) = Σ rs / n²。

6. **并列值** 合并成同一取值后精确处理；并列点只贡献零特征值，不影响 top-k。常数输入（m = 1）时 D = 0，按契约的非有限值策略 fail closed。

### 4.3 拟议的 GPU 算法（每个分量，尚未实现）

1. 排序，求不同取值、计数、间隔 —— O(n log n)
2. 按闭式写出 B —— O(m)
3. 对 B 做 Demmel–Kahan 双对角 QR（`dbdsqr` 那一套）：一趟同时得到 σ（高相对精度）和 Pᵀq，并记录右旋转序列 —— O(m²) 时间；记录的旋转约 2 MB/分量
4. 久期方程按距最近极点的偏移量求根（`dlaed4` 思路），配合 `dlaed2` 式的 deflation —— O(m·k·迭代次数)
5. 在 P 坐标下写出 k 个 y，重放旋转得到 P y（O(m²k)），再用 O = H F L_Q 的累加和映回原坐标（O(nk)）

总复杂度 O(n log n + m²k)；只有在 k 固定时才是 n 的二次。过程中不形成 n×n 或 m×m 矩阵；片上工作集是 k×m 块，m = 350、k = 35 时约 98 KB。

### 4.4 验证结果

**真实数据**（`verify_structured_dcov_real_data.R`）：canonical 的 48 个原始列（每列有 1–171 个并列值）加 60 个 mgcv 残差（|S| = 1, 2, 3），共 3000 对；前 300 对与 `kpcalg::dcov.gamma` 对比。

| 比较 | 最大 \|Δp\| |
|---|---:|
| 结构化 vs `kpcalg::dcov.gamma` | 6.242e-13 |
| 稠密 eigen vs `kpcalg::dcov.gamma` | 3.564e-14 |
| 结构化 vs 稠密 eigen（3000 对） | 6.372e-13 |

α = 0.1 下判定全部一致。这个脚本用 R 的稠密 `svd`/`eigen` 代替双对角 QR 和久期方程求解，只有绝对精度。

**极端输入**（`structured_dcov_reference.py`；完整输出见脚本运行结果）：

| 用例 | n = 351 算子误差 | n = 1000 算子误差 |
|---|---:|---:|
| normal | 6.3e-15 | 1.2e-14 |
| cauchy_heavy_tail | 1.1e-14 | 2.3e-14 |
| lognormal_skew | 2.5e-14 | 9.2e-14 |
| exact_ties_50pct | 4.2e-15 | 8.0e-15 |
| zero_inflated | 4.3e-15 | 9.3e-15 |
| near_ties_1e-12 | 2.0e-14 | 1.4e-14 |
| near_ties_1e-15rel | 7.9e-15 | 1.3e-14 |
| clustered_scales | 6.8e-15 | 7.1e-15 |
| uniform_grid（会触发 deflation） | 7.0e-15 | 8.6e-15 |
| outlier_1e8 | 7.7e-14 | 2.3e-13 |
| large_offset_1e6 | 3.2e-15 | 9.3e-15 |
| dependent_on_base / base | 3.9e-15 / 6.1e-15 | 7.8e-15 / 5.2e-15 |
| binary_unequal（两个取值） | 2.7e-15 | 8.8e-15 |
| binary_half_split（两个取值，n = 1000 时计数相等） | 6.2e-15 | 1.5e-14 |
| three_values | 2.3e-15 | 7.5e-15 |
| **所有配对的最大 \|Δp\|（120 对）** | **5.44e-14** | **1.23e-13** |

- 算子误差指中心化秩 k 算子 W Λ Wᵀ 的相对 Frobenius 差，不受特征向量符号和顺序影响。
- outlier_1e8 的特征值相对误差（4e-8 / 6e-8）来自稠密参考解：它对小特征值只有绝对精度。

### 4.5 踩过的坑（实现时必须遵守）

| 尝试 | 结果 | 结论 |
|---|---|---|
| 对三对角 M = BBᵀ 用 MRRR（`dstemr`） | 近乎并列（间隔约 1e-15，矩阵元素到 1e14–1e18）时不收敛，info = 22 | 改用 Demmel–Kahan 双对角 QR |
| 特征向量用 U = Z W √θ 构造（Z = H F G^{1/2}） | 误差被放大约 √κ 倍，近乎并列时 p 值差到 3.5e-10 | 改用 U = (H F L_Q) P |
| 行和用 x 的前缀和计算 | 大偏移（1e6）时灾难性抵消 | 按间隔累加 |
| 直接用 (λ BᵀB + 2I)⁻¹ 的三对角求解求特征向量 | 近乎并列和均匀网格（deflation）时误差到 O(1) | 在 P 坐标下构造，再重放旋转 |
| 在绝对坐标里二分求久期方程的根 | 根离极点不到 1 ulp 时得到 NaN | 用相对最近极点的偏移量表示根 |
| 两个取值的输入 | ±h√(c₁c₂) 两个特征值 \|λ\| 相等，顺序有歧义 | 第 k 个位置上出现 \|λ\| 相等时，必须按契约的 tie-break 规则选取 |

### 4.6 现状与限制

- 目前只是 CPU 参考实现：Python 版显式构造了完整的 P（数值上等价于重放旋转，但不是 GPU 设计）；R 版用稠密 SVD/eigen。
- 还没实现或验证：GPU 上的旋转记录与重放、整批分量融合进一次发射、工作区上限、吞吐量、deflation 阈值、第 k 个位置上 |λ| 相等时的 tie-break。
- 单线程 CPU 计时（`structured_dcov_reference.py --bench`）：

  | n | 稠密 `eigh(D)` | 双对角 QR 只求奇异值 | + Pᵀq | + 35 个向量（≈ 重放旋转） | 比值 |
  |---:|---:|---:|---:|---:|---:|
  | 351 | 9.24 ms | 1.69 ms | 3.81 ms | 7.13 ms | 1.3× |
  | 500 | 20.93 ms | 3.14 ms | 7.59 ms | 14.08 ms | 1.5× |
  | 1000 | 143.38 ms | 11.65 ms | 28.11 ms | 52.74 ms | 2.7× |

  按 flop 算只省 1.3–2.7 倍。GPU 上可能的收益来自：去掉 n×n 工作集、去掉 cuSOLVER 每次调用的主机同步和 1-block 内核、整批融合发射。**这些都是假设，GPU 原型出来之前不给阶段耗时估计。**

### 4.7 需要通过的资格验证

- **数值**：
  - 全部 240,498 个 canonical CI p 值和有限 pMax，对 legacy oracle 的绝对误差 ≤ 1e-10
  - dCov 统计量（1e-9 / 1e-8）和矩（1e-10 / 1e-8）的误差在容差内
  - 近阈值检验零翻转
  - 三组大型真实输入同样通过
- **执行语义**：任务顺序、deletion trace、邻接图、语义 sepset、`n.edgetests` 与 authority 完全一致。
- **执行方式与资源**：
  - 全程 GPU 执行，没有 CPU 数值 authority 或回退
  - 调用结束后 CUDA 资源计数器全部归零
  - 重复运行结果确定
- **缓存身份**：新 solver 的身份必须写进 component cache key，保证 cuSOLVER 路线和结构化路线的分量不会混用；cache 的 route/status 不一致计数为 0。
- **独立资格项**（各自配专门的测试样例）：
  - deflation 阈值的行为，包括对称数据、计数相等的两值输入
  - 第 k 个位置上 |λ| 相等时，按 `component_eigen_selection` 的 tie-break 规则选取

## 5. 发现四：契约对新 eig 算法的约束

- **数值契约**（`fastkpc/inst/contracts/full_cuda_ci/numerical_contract_v1.json`）：
  - 容差：p 值 absolute 1e-10（near-alpha 也是 1e-10，翻转数 0）；残差 max abs 1e-7、rel L2 1e-8；dCov 统计量 1e-9 / 1e-8；矩 1e-10 / 1e-8。
  - `selected_log_sp_bit_identity_required = false`（第 60 行）。
  - 特征对选择只规定语义（第 93 行）。
- **架构契约**（`fastkpc/inst/contracts/full_cuda_ci/architecture_contract_v1.json:155`）：`not_frozen` 列表明确包含 `eigensolver-representation`（第 160 行）。

因此：保持相同数学语义、通过全部契约门槛的新 eig 算法，不需要仅因为和旧 GPU 输出低位不同，就升级数值契约大版本。`PHASE10_FULL_EIG_COMPONENT_CLOSURE.md` 中"换特征分解算法需要新契约大版本"的 reopen 条件，比契约文本更严。

但抽样 1e-13 不能预先证明全量合格，必须完成 §4.7 的验证。对于本应逐位不变的改动（R1/R2/R3），逐位回归仍然是合适的默认验收方式。

## 6. 发现五：lane-per-target 微基准

`qr_layout_bench.cu` 对 32,768 个矩阵做同一份标量 Householder QR 加显式 Q（按 `dgeqr2` + `dorg2r` 的运算顺序），比较三种组织方式：

- A：每个 target 一个 64 线程 CTA，只有 thread 0 干活（相当于现在 f2c LAPACK 的串行段）
- C：每个 target 一个 64 线程 CTA，按列 warp 协作（相当于现在的协作段）
- B：每个 lane 一个 target，数据交错存储

| 形状 | A | C | B | B/C | B/A | 和 A 逐位一致 |
|---|---:|---:|---:|---:|---:|---|
| 52×28 | 129.9 ms | 35.7 ms | 20.0 ms | 1.79× | 6.50× | 是 |
| 88×37 | 383.7 ms | 115.1 ms | 68.9 ms | 1.67× | 5.57× | 是 |
| 120×64 | 1420.2 ms | 555.1 ms | 354.4 ms | 1.57× | 4.01× | 是 |

现在的优化器是每个 target 一个 64 线程 CTA（`kBlockSize = 64`）。LAPACK 3.12 的 f2c 移植（`fastkpc/src/cuda/lapack_312_small_dgesdd.cuh:1227` 的 `small_dgesdd_left`）里有 thread 0 串行段，内核用满 255 个寄存器。这个基准只覆盖 QR 加 Q，没有测 `dbdsdc`、Newton 逻辑和分支发散。是否做，等 R1 之后的剖析再定。

## 7. 证据状态与决策

| 项目 | 状态 | 证据 | 决策 |
|---|---|---|---|
| 在正确的 retained 配置下重新剖析 | 已完成（本版 §2.2、§3） | 22 列子集的 retained nsys 剖析 | — |
| pinned 缓冲生产 A/B | 待做 | 机制在生产剖析和合成复刻中都已复现 | 优先 GO |
| 结构化 eig 的 GPU 隔离原型 | 待做 | CPU 参考实现的数值验证（§4.4） | GO，须完成 §4.7 全量资格验证 |
| eig 限制因素的独立归因（含大批次扩 lane 无收益） | 待做 | §3.3：现有证据只支持"存在未充分利用的时段" | 建议做 |
| 常驻优化器、lane-per-target | 待定 | 结构分析 + QR 微基准 | 根据 pinned 之后的剖析决定 |
| 单卡 100–150 s 预算 | 撤回 | 外推超出证据 | 不采纳 |
| 同一方法内部拆到两张 GPU | 不纳入 | — | 沿用完整的任务级并行 |

目前没有证据表明 120 s 门槛已经有可靠的实现路径。

## 8. 下一步与验收

1. **pinned 缓冲 A/B**：先做优化器的 Y 上传和 `host_states` 下载，再推广到其他逐批传输。验收：canonical 的 task、p 值、pMax、邻接、sepset、`n.edgetests` 与当前 producer 逐位一致；记录优化器 boundary 时间和总时间；在 retained 配置下用 `analyze_nsys_profile.py` 复测。
2. **eig 限制因素的独立归因**：拆分批宽、提交方式、cuSOLVER 每次调用的主机同步、eig 窗口内 `cudaMalloc/cudaFree` 的来源，并解释旧收口文档中大批次扩 lane 无收益的原因。
3. **结构化 eig 的 GPU 隔离原型**：先用分量级 oracle 对比，再按 §4.7 做全量资格验证。
4. **R2 / R3**：根据 pinned 之后的剖析决定。

## 9. 复现

脚本都在 `fastkpc/tools/research/exact_speedup_20261020/`，在仓库根目录运行。剖析脚本默认使用 retained 配置并打印出来；要跑默认路线，设 `FASTKPC_PROFILE_DEFAULT_ROUTE=1`。

```bash
# 结构化特征分解：极端输入（约 2 分钟）；加 --bench 做 CPU 计时
python3 -I fastkpc/tools/research/exact_speedup_20261020/structured_dcov_reference.py --bench

# 结构化特征分解：真实数据 vs kpcalg::dcov.gamma（约 1 分钟）
Rscript fastkpc/tools/research/exact_speedup_20261020/verify_structured_dcov_real_data.R

# 500×50 开发样例的阶段计时（retained 配置）
CUDA_VISIBLE_DEVICES=0 Rscript fastkpc/tools/research/exact_speedup_20261020/run_development_500x50.R

# 子集剖析（retained 配置）：前 SUBSET_P 列，max_conditioning_size = 3
CUDA_VISIBLE_DEVICES=0 SUBSET_P=22 nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
  -o /tmp/prof22r Rscript fastkpc/tools/research/exact_speedup_20261020/run_subset_profile.R
nsys export --type=sqlite -o /tmp/prof22r.sqlite /tmp/prof22r.nsys-rep
python3 -I fastkpc/tools/research/exact_speedup_20261020/analyze_nsys_profile.py /tmp/prof22r.sqlite

# 微基准（跑在 GPU1）
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -std=c++17 -o /tmp/pinned_bench \
  fastkpc/tools/research/exact_speedup_20261020/pinned_concurrency_bench.cu -lpthread
CUDA_VISIBLE_DEVICES=1 /tmp/pinned_bench 47 6 3000000
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -o /tmp/qr_bench \
  fastkpc/tools/research/exact_speedup_20261020/qr_layout_bench.cu
CUDA_VISIBLE_DEVICES=1 /tmp/qr_bench 32768
```

本次的 nsys 报告在 `/tmp/claude-structured-dcov/`：`prof22.*` 是默认路线（仅作对照），`prof22r.*` 是 retained 配置。没有放进仓库，机器重启后会丢失。

## 10. 尚未验证的部分与风险

- R1 在真实流水线上的收益还没测，目前只有机制证据。
- eig 阶段的主要限制因素尚未确定，包括大批次扩 lane 无收益的原因（§3.3）。
- 结构化 eig 的 GPU 实现、吞吐和全量资格验证都还没做（§4.6、§4.7）。
- R2 要处理 workspace、容量 bucket、结果生命周期和长尾，逐位不变需要实测验证。
- 所有剖析都只覆盖 conditioning depth ≤ 3。
