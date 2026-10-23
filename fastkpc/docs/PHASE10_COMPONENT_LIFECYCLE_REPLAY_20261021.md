# 分量生命周期离线重放：整层预计算的驻留峰值与未消费比例

日期：2026-10-21（同日按评审修订，见文末）。回应
[structured eig 真实残差重放报告](PHASE10_STRUCTURED_EIG_REAL_REPLAY_20261020.md)
"下一门"第 1 项：用当前 exact producer 的逐层 target/component 身份，重放
预计算、消费与释放生命周期，量出需要驻留的峰值及未消费项比例。

## 决策

```text
逐层 plan 重建与回执身份核对                        PASS（canonical 351×48，默认 m.max=Inf）
整层预计算是否需要同时驻留全部 132,974 个分量         否：分量不跨层复用，单层最大 54,878 条（5.04 GiB，level 3）
现有 119,817 条 / 11 GiB 容量是否足够                 是：仅按层 reset（层内不回收）需容纳单层累计生产量，
                                                      整层一次性 54,878 条（5.04 GiB）、round-robin 640 批 43,322 条（3.98 GiB）
层内回收（引用计数 + 槽位复用）能把驻留降到           23,245 条（2.13 GiB，整轮反馈模型）；这是独立于首轮原型的后续项
整层一次性预计算的未消费比例                           22,291 / 132,974 = 16.8%（level 3 为 22.3%）
round-robin + 640 静态批的未消费分量                   逐检验反馈 933 条（0.8%）；整轮反馈 1,002 条（0.9%）
生产集成 / 耗时预测                                    NOT YET GO / 本文不做耗时预测
```

这是纯离线的结构重放：不运行 GPU，不改 native，不预测 canonical 总耗时。
它证明的是分量缓存的容量可行性，不包含 structured eig 自身约 2.54 GiB 的
workspace、prepared-S 上下文、optimizer 与其他运行资源。Phase 10、candidate、
recommended route 和 sealed holdout 均不改变。

## 输入与身份

输入是 32 connections 开发运行（880.659 s）的完整 canonical 回执：

```text
fastkpc/artifacts/full_cuda_ci/exact_speedup_20261020/canonical-queues32.rds
  SHA-256 58b13362bf52612f40c5ab9887548f2fd92e0c4a752a5b06c6f86a8cabbe6f3a
  native  c4f599ef5267c9ff1c7a5a81d7e550330b730d84679a94328bf690c0bb51ffc0
```

该运行的全部 240,498 个 CI p 值、task/deletion/graph 字段与 RNG 与同 native
默认配置逐位相同，其 task identity、adjacency、sepset、n.edgetests 门全部为
TRUE，对 kpcalg 权威的最大 CI/pMax 误差 3.0547508966805026e-12（见执行报告）。
重放脚本按 native `make_layer_plan`（`fastkpc/src/skeleton_task_scheduler.cpp`）
的枚举规则重建每层 plan：按无向边 (x<y) 主序，先 x→y 方向的全部 S，再 y→x
方向，S 按字典序组合枚举；然后按回执重放已执行检验与删边。脚本在以下任一
核对失败时直接中止：

| 核对项 | 结果 |
|---|---|
| 每层 tasks_planned / tests_replayed / ignored / deletions | 9 层全部相等 |
| 每条已执行检验的 level、task_index、方向、S_key、edge | 240,498 条全部相等 |
| 每层 unique 分量 miss 数对 wave trace | 9 层全部相等（合计 110,683） |
| level≥1 的 planned 分量数对 prefill_unique_target_key_count | 132,926 = 132,926 |
| level≥1 的已消费分量数对 prefill_consumed_unique_target_key_count | 110,635 = 110,635 |
| 全部已消费分量数对 legacy_component_cache_insert_count | 110,683 = 110,683 |

结果 JSON 绑定回执顶层的 `native_sha256`、回执 rds 文件的 SHA-256 与四个导出
CSV 的 SHA-256；导出缺少 native 身份时脚本拒绝运行。分量身份为 `(target, S)`；
x→y 与 y→x 两个方向的同一 S 检验使用同一对分量。每条缓存项为
`(n·numCol + numCol + 2)·8 = 98,576` 字节；容量 119,817 条来自 11 GiB 预算。

## 按层分量计数

| level | planned tasks | 已执行 | planned 分量 | 已消费 | 未消费 | 未消费比例 | planned 分量字节 | 回执 full eig | 回执 miss/批 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,256 | 2,213 | 48 | 48 | 0 | 0.0% | 4.7 MB | 0.16 s | 24.0 |
| 1 | 96,120 | 52,659 | 2,256 | 2,256 | 0 | 0.0% | 0.21 GiB | 6.47 s | 23.5 |
| 2 | 303,870 | 125,293 | 49,919 | 42,685 | 7,234 | 14.5% | 4.58 GiB | 158.95 s | 1.88 |
| 3 | 54,788 | 40,694 | 54,878 | 42,662 | 12,216 | 22.3% | 5.04 GiB | 156.68 s | 4.27 |
| 4 | 15,630 | 13,293 | 17,767 | 15,433 | 2,334 | 13.1% | 1.63 GiB | 59.31 s | 3.88 |
| 5 | 5,862 | 5,422 | 6,816 | 6,377 | 439 | 6.4% | 0.63 GiB | 22.81 s | 4.37 |
| 6 | 903 | 835 | 1,148 | 1,080 | 68 | 5.9% | 0.11 GiB | 4.24 s | 3.46 |
| 7 | 80 | 80 | 124 | 124 | 0 | 0.0% | 12 MB | 0.53 s | 2.82 |
| 8 | 9 | 9 | 18 | 18 | 0 | 0.0% | 1.8 MB | 0.09 s | 2.00 |
| 合计 | 479,518 | 240,498 | 132,974 | 110,683 | 22,291 | 16.8% | — | 409.24 s | 2.87 |

分量不跨层复用（S 的大小等于 level），因此 132,974 这个数从不需要同时驻留。
整层残差物化的字节也很小：每层最多 154 MB（level 3，54,878 × 351 × 8）。

## 消费顺序、生产策略与反馈边界

四种消费顺序都保持每条边内部的 plan 顺序（x→y 的 S 序列在前，y→x 在后），
只改变跨边交错，因此逻辑语义（哪些检验执行、删边、sepset、n.edgetests）相同：

- `canonical`：native plan 顺序本身，一次处理完一条边；
- `round_robin`：第 r 轮取每条仍存活边的第 r 个任务，是分量预计算后不投机的
  自然批量消费者（每轮每条边一个 pair）；
- `frontier_largest_bucket`：生产 frontier 调度的"最大就绪 S 桶"规则，不含
  prepared-S 缓存偏好；
- `smallest_ready_s`：总是取字典序最小的就绪 S。

生产策略：`demand`（首次需要时生产，oracle 下界）、`batch_static_B`（按该
消费顺序的 plan 首次出现序，一次生产 B 个尚未生产且尚未判死的分量，可实现）、
`batch_oracle_B`（按实际首次执行使用序，oracle）、`level`（全部在首个检验前
生产）。释放规则：引用该分量的全部 planned 任务都已执行或因删边不可达时
释放。

反馈边界有两种模型：

- **逐检验反馈**：每个 pair 的结果（引用消解、删边）在该检验后立即生效。
- **整轮反馈**（仅 round-robin）：批量消费者先准备整轮输入再统一取得结果。
  一轮需要的全部分量在该轮开始前生产完毕；该轮的引用消解、删边与释放全部在
  轮结束后才生效。正在被该轮使用的分量因此不会提前回收。

正式的生产设计必须声明自己的反馈边界。整轮反馈模型覆盖的是"准备一轮、
完成消费、再反馈"这一种形状；跨轮预取或多个 in-flight 轮次的驻留尚未建模，
若生产与消费重叠，正被 GPU 使用、暂不可回收的分量还要另行计入。

## 驻留峰值

单位为分量条数；乘以 98,576 字节得到设备字节。

| level | 整层一次性 | canonical demand | canonical static-640（峰值 / 浪费） | round-robin demand | round-robin static-640（峰值 / 浪费） | frontier demand | smallest-S demand |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,256 | 2,112 | 2,256 / 0 | 2,243 | 2,248 / 0 | 2,244 | 2,196 |
| 2 | 49,919 | 23,722 | 34,864 / 7,066 | 17,800 | 23,199 / 626 | 26,041 | 21,104 |
| 3 | 54,878 | 3,644 | 5,190 / 8,629 | 2,950 | 3,870 / 210 | 4,556 | 4,187 |
| 4 | 17,767 | 1,096 | 1,960 / 1,621 | 350 | 999 / 59 | 1,452 | 494 |
| 5 | 6,816 | 799 | 1,438 / 335 | 226 | 865 / 15 | 489 | 23 |
| 6 | 1,148 | 205 | 712 / 68 | 77 | 640 / 23 | 2 | 2 |

round-robin static-640 在两种反馈边界下的对比（累计生产即"仅按层 reset"
时缓存必须容纳的量；峰值存活即"层内回收"后的驻留）：

| level | 逐检验：累计生产 / 浪费 / 峰值存活 | 整轮反馈：累计生产 / 浪费 / 峰值存活 |
|---:|---:|---:|
| 1 | 2,256 / 0 / 2,248 | 2,256 / 0 / 2,248 |
| 2 | 43,311 / 626 / 23,199 | 43,322 / 637 / 23,245 |
| 3 | 42,872 / 210 / 3,870 | 42,890 / 228 / 3,889 |
| 4 | 15,492 / 59 / 999 | 15,532 / 99 / 1,031 |
| 5 | 6,392 / 15 / 865 | 6,392 / 15 / 867 |
| 6 | 1,103 / 23 / 640 | 1,103 / 23 / 640 |
| 合计 | 111,616 / 933 / — | 111,685 / 1,002 / — |

整轮反馈的三个合计（生产 111,685、浪费 1,002、level 2 峰值 23,245）与评审
方独立做的只读重放逐条相同。

要点：

- 任何策略下单层驻留峰值都不超过整层 planned 分量数，而整层最大值 54,878
  条（5.04 GiB）低于现有容量 119,817 条（11 GiB）。所以"按层回收"就足够，
  层内不需要淘汰策略；132,974 不是容量需求。
- "仅按层 reset"与"层内回收"是两个不同的容量需求。现有缓存的 `reserve_slot()`
  只递增槽位、从不回收，只加层间 reset 时必须容纳单层累计生产量：整层一次性
  54,878 条（5.04 GiB），round-robin 640 批 43,322 条（3.98 GiB）。要把驻留降
  到约 2.13 GiB，需要层内引用计数、失效 key 删除、空闲槽位复用，以及 GPU 消费
  完成后的安全回收。首轮原型应继续使用 11 GiB 预算，只验证调度与吞吐；缩小
  缓存容量是独立的后续项。
- canonical 顺序下分量生命周期长（level 2 中位数为整层的 39.7%），demand 峰值
  仍有 23,722；且 plan 序生产会在分量判死前生产它们，浪费 7,066 + 8,629 +
  1,621 + 335 + 68 = 17,719 条，与整层一次性的 22,291 条同量级。
- round-robin 顺序下生产序与消费序接近，static-640 的总浪费为 933 条（逐检验）
  或 1,002 条（整轮反馈），约 0.8–0.9%。四种批宽在逐检验反馈下的总浪费：
  256 → 786，640 → 933，1,280 → 1,385，2,560 → 2,090。
- level 2 的驻留曲线在 round-robin static-640 下约在整层 53% 处达峰，之后
  随释放下降；整层一次性则从 49,919 开始单调下降，到 50% 处仍有约 39,000 条。

## round-robin 的批次结构

| level | 轮数 | pairs/轮 均值 | p50 | p10 | 最大 | ≥64 pairs 的轮 | 640 批数 | 满批比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 92 | 572.4 | 540 | 526 | 1,085 | 92 | 4 | 75% |
| 2 | 1,126 | 111.3 | 103 | 5 | 539 | 688 | 68 | 99% |
| 3 | 1,529 | 26.6 | 17 | 1 | 193 | 175 | 67 | 99% |
| 4 | 750 | 17.7 | 12 | 12 | 127 | 16 | 25 | 96% |
| 5 | 468 | 11.6 | 11 | 10 | 79 | 1 | 10 | 90% |
| 6 | 85 | 9.8 | 9 | 9 | 23 | 0 | 2 | 50% |

与现有 frontier 调度的每批 1.88–4.37 个 miss 相比，分量生产批宽在 level 2–5
达到 90–99% 满批（640）。但生产批满不等于整条流水高效：pair/gamma 消费者在
level 3、4 平均每轮只有 26.6、17.7 个 pair，实际收益还取决于残差准备、分量收集、
缓存访问和轮间同步。level≥6 分量总数不足两批，批宽收益消失，但这些层的
回执 eig 合计不到 5 s。

## 投机 S-major 对照

如果每层所有 planned 任务都按 S 分组物理评估、再按 canonical 顺序逻辑消费，
驻留上界只有约 1,300 条（双缓冲 2×640 加单个 S 的最多 46 个分量），但物理
pair/gamma 工作量变为 479,518 对 240,498（level 2 浪费 58.8%，level 1 为
45.2%），分量也要全部生产（浪费 16.8%）。路线图 §4.6 允许投机评估，但其收益
取决于 pair/gamma 的真实成本，本文没有测量，不推荐作为首选。

## 对设计的含义

1. **缓存**：现有 `FullCudaCiLegacyEigComponentCache` 只追加（`next_slot_`
   单调、`keys_` 不删除），没有层边界回收。首轮原型只加按层 reset，保持
   11 GiB 预算（可容纳任一层的累计生产量，余量 2.2× 以上）。层内回收
   （引用计数、key 删除、槽位复用、GPU 消费完成后再回收）是独立的后续项，
   收益是把驻留从 3.98 GiB 降到 2.13 GiB。新 solver 的身份必须进入 cache key
   （前次评审已要求）。
2. **生产者**：现有 eig 执行上下文按 `p = 48` 个分量和 `p²` 个 pair 创建
   （`full_cuda_ci_one_call.cpp` 的缓存创建参数），640 宽的分量生产需要独立
   的生产上下文；生产序为 round-robin plan 首次出现序，跳过已判死分量；
   反馈边界按整轮建模。
3. **消费者与逻辑重放**：round-robin 只改变物理执行顺序。task、deletion、pMax
   与 sepset 必须继续由现有的 canonical 逻辑重放生成（`full_cuda_ci_one_call.cpp`
   中"Physical CUDA rounds preserve per-edge reachability. Replay their compact
   results in the canonical layer-plan order"一段）；保持每条边内部顺序保证
   可达任务一致，返回 trace 的顺序也要明确保持不变。即便如此，它仍是新的
   调度器，task order、deletion trace、graph、sepset、n.edgetests、RNG 与
   资源闭合全部需要重新资格验证。
4. **残差语义是当前最关键的开放问题**。现有消费者明确保留原 cohort 的整体
   GEMM/残差物化（同文件中"The consumer still uploads Y and reproduces the
   original cohort-wide GEMM/residual materialization"），cohort 由 frontier
   组的 target 构成，而组的构成又取决于检验结果。整层或跨 S 攒出 640 个分量，
   意味着残差要先于消费、以不同的 cohort 形状物化。残差比特是否随 cohort
   形状变化本文没有验证，也不能从真实重放报告的 1.1e-13 p 差推断（那一比较
   同时包含 CPU/GPU 的 eig 与 gamma 差异）。新设计必须说明如何在保留这些
   输入比特的同时攒出跨 S 的大批分量；最直接的办法是先做残差比特身份实验：
   同一组 `(target, S)` 用原 cohort 与新 cohort 形状各物化一次，比较残差字节。
   若逐位相同，整层残差物化可以直接进入原型；若不同，则残差阶段必须保持原
   cohort 形状，分量生产只能在残差已物化的范围内攒批，或者接受完整方法重新
   资格验证。
5. **dense 对照的作用**：保留 `cusolverDnDsyevd` 作为生产者不自动保证整条
   管线逐位不变，残差 cohort、GEMM 形状、分量构造与 pair/gamma 批处理都可能
   影响结果。dense 对照的用途是验证调度改动能否保持与当前路线逐位一致；
   逐位一致是验收门，不是前提。
6. **吞吐换算（不是流水线预测）**：按真实残差 640 批的 889–927 分量/s，
   round-robin static-640 的 111,685 条约 120–126 s，整层一次性 132,974 条约
   143–150 s；回执中 dense 全 eig 合计 409.24 s，dense 吞吐换算 110,683 / 355
   ≈ 312 s。这些数不含残差物化、调度、pair/gamma 与主机开销。

## 下一门

后续执行更新：
[实际 persistent 8-lane cache payload 字节门](PHASE10_COMPONENT_PAYLOAD_GATE_20261021.md)
已完成，结果为 **STOP_PAYLOAD_BITS**。6,632 个条目按原 cache-miss cohort
全部复现；22 宽重组改变了其中 1,002 个原 singleton-miss 条目的 self_moment。
这不否定下面已完成的残差身份实验，但阻断第 3–5 步的集成与性能验证；
需先重新设计如何保留原首次 miss 的分量语义，不能用 p 级实验代替 payload 门。

1. ~~补齐整轮反馈模型和回执身份绑定~~（本次修订已完成）。
2. ~~残差比特身份实验（第 4 点）~~（已完成）。
   [独立 dispatcher 实验](PHASE10_RESIDUAL_COHORT_IDENTITY_20261021.md)先观察到
   批宽变化的低位差异；随后完成
   [真实生产 cohort 捕获与固定系数重放](PHASE10_PRODUCTION_COHORT_FROZEN_REPLAY_20261021.md)：
   22 列 default-Inf 的 2,592 个原 cohort 全部逐位复现，17 列重组在 32 个
   target 上产生最大 4.44e-15 差异；每个 S 按不超过 16 列分块则覆盖全部
   6,632 个实际消费 target，两次重放均逐位一致。此结论仅覆盖本次实际形状，
   不构成任意 q/输入的宽度类定理，也不能直接推广到 canonical 原本的宽组。
3. 在 22 列隔离原型中，multi-penalty 残差可采用上述已验证分块，single-penalty
   保持现行执行；继续保留 canonical 逻辑重放，用现有 dense producer 验证
   跨 S 分量收集与 round-robin 消费，缓存先维持 11 GiB。残差分块宽度与 eig
   分量批宽分开设置；验收门仍是 p 值、pMax、task/deletion trace、graph、
   sepset、n.edgetests、RNG 与资源计数全部逐位或逐项一致。
4. 逐位门、资源门通过后，接入独立 identity 的 structured eig，测完整 outer
   elapsed。
5. 子集确有收益，再做 canonical 和三套真实数据资格与重复性能 campaign。

本文只覆盖 canonical 351×48 一个数据集；三组真实数据没有带 task trace 的
完整回执，未做重放。

## 评审修订（2026-10-21）

评审意见与本次改动：

1. "按层 reset"与"只需 2.2 GiB"分开：决策块与驻留一节现在分别给出"仅按层
   reset"（单层累计生产量）与"层内回收"（峰值存活）两个需求，并说明首轮原型
   保持 11 GiB。
2. 整轮反馈：脚本新增 `*_round_feedback` 模型，结果（111,685 / 1,002 / 23,245）
   与评审方的只读重放相同；文中明确反馈边界定义。
3. dense 对照措辞：删去"零数值风险"，改为"验证调度改动能否保持逐位一致，
   逐位一致是验收门"，并把残差 cohort 语义列为最关键的开放问题，增加残差
   比特身份实验作为前置步骤。
4. 机器可核验身份：导出补入回执顶层 `receipt_native_sha256`、`receipt_elapsed_sec`
   等行（加前缀避免与 summary 同名字段碰撞），JSON 记录 native 与 rds 文件
   SHA-256，缺失时脚本拒绝运行。

## 复现与证据

工具与说明：[component_lifecycle_20261021/README.md](../tools/research/component_lifecycle_20261021/README.md)。

```text
fastkpc/tools/research/component_lifecycle_20261021/replay_component_lifecycle.py
  SHA-256 51453fe26d12eec003353e690ba37aed3bbf8fc4fd9855c34c01601c4685fd0e
fastkpc/artifacts/full_cuda_ci/component_lifecycle_20261021/component_lifecycle_replay.json
  SHA-256 3267f9a65733f87c60f85d9e8343f00f8ae04e2941e277334d8a6f69450d79f4
canonical_queues32_tasks.csv        0cfccb0ca2290b18b8039b911d863a80783d10c89bbc27166178b30cb3a85999
canonical_queues32_levels.csv       32e0b2fbe334a630dfcec58fe0c6cfb78541d691ae9c42a3527c498819569008
canonical_queues32_wave_groups.csv  4c7d1acd1feef197cbdb193d0a9a0397eb943799cf6883b4a14e69f3bff2313d
canonical_queues32_summary.csv      47ac52b62d6de1bdcbccebaeff2d7f1b881d313ddf4e300614ceab3eabd013cf
```

证据等级为离线结构诊断：不是性能证据，不是数值资格，不是 candidate 冻结或
promotion 证据。
