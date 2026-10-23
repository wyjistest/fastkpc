# 22 列 dense 对照原型设计：level 起始分量预填充（路径 (i)：提前 eig、延后 Gram）

日期：2026-10-21（第三次修订，正文统一到路径 (i)；修订记录见文末）。承接
[真实生产 cohort 捕获与固定系数重放](PHASE10_PRODUCTION_COHORT_FROZEN_REPLAY_20261021.md)、
[分量生命周期重放](PHASE10_COMPONENT_LIFECYCLE_REPLAY_20261021.md)、
[残差 cohort 逐位身份实验](PHASE10_RESIDUAL_COHORT_IDENTITY_20261021.md)、
[payload 第 0 道门 STOP](PHASE10_COMPONENT_PAYLOAD_GATE_20261021.md)与
[singleton GPU 差分重放](PHASE10_SINGLETON_GPU_PVALUE_REPLAY_20261021.md)。
本文是设计提案；评审已推荐采纳路径 (i) 进入默认关闭的 22 列 dense 原型。
默认关闭的 22 列原型已实现并通过完整逐位与资源资格，见
[实现与资格回执](PHASE10_LEVEL_COMPONENT_PREFILL_RESULT_20261021.md)。
该轮单次 source-closure 对照为 110.949 s → 93.461 s。
后续已修正生命周期边界，并在新 native 上通过 canonical 351×48 全量资格与
差异归因，见[canonical 执行结果](PHASE10_LEVEL_COMPONENT_PREFILL_CANONICAL_20261021.md)：
同 binary 单次 892.468 s → 842.873 s。生产默认启用仍未放行；以下保留设计
推导，历史估算不替代新回执。

## 提案

```text
原型形状        level 起始分量预填充（LSCP，路径 (i)）：
                阶段 A：在现有 multi-penalty optimizer prefill 的每个窗口内、句柄释放之前，
                        按 S 以不超过 16 列的 cohort 用生产 finalizer 物化本层残差；
                阶段 B：prefill 结束后跨 S 以执行上下文容量为批宽（22 列即 22）做距离构造、
                        deterministic syevd 与分量选择，把 vectors / selected_values / total
                        三字段写入按层 reset 的 component cache，self_moment 标记 pending；
                消费者：frontier 调度与分组逻辑不变，新增"hit 且 Gram pending"状态，在分量
                        首次被引用的组里按原 miss 顺序 gather 后执行与现行相同的 Gram GEMM、
                        归约与 self_moment 提交，再进入 pair 阶段
范围            仅 level ≥ 3 的 multi-penalty 路线；level 1–2 的 single-penalty 路线完全不变
为什么延后 Gram   捕获样本观察到 singleton 与多分量 Gram 的 self_moment 比特差异；原 miss
                结构由首次引用时所在组决定。恢复完整原批形状，并通过逐位门验证
验收链          预填充后三字段逐位一致 → 按原 miss 分组完成 Gram → 消费前四字段逐位一致
                → 完整 p/pMax 与图语义通过；"逐位不变"是验收目标，不是前提
驻留            整层累计生产即"仅按层 reset"需求：canonical level 3 为 54,878 条 / 5.04 GiB，
                在现有 11 GiB 预算内；22 列子集远小于此
浪费            整层一次性预计算：canonical level ≥ 3 为 15,057 / 80,751 = 18.65%，
                22 列为 638 / 7,270 = 8.78%；未消费的 pending 条目是正常情况
目的            验证语义与成本：预填充、gather、提交与调度开销；不承诺加速
状态            第 0 步、22 列与 canonical 默认关闭 dense 原型资格 PASS；生产默认启用 NOT YET GO
```

eager 的"阶段 B 计算四字段"方案在真实 miss 形状上得到 STOP_PAYLOAD_BITS，
保留为历史结论；路径 (i) 通过自己的验收链，不继承该 STOP，也不继承任何
"必然逐位不变"的假设。

## 范围与账本

原型只覆盖 level ≥ 3。按生命周期重放的逐层账本和冻结重放的实际捕获：

| 范围 | planned 分量 | 实际消费 | 未消费比例 | 现有 eig 时间 |
|---|---:|---:|---:|---:|
| canonical，level ≥ 3 | 80,751 | 65,694 | 18.65% | 243.66 s（level 3–8 回执合计） |
| 22 列，multi-penalty（default-Inf 至 level 6） | 7,270 | 6,632 | 8.78% | 子集回执中按层分摊 |

canonical 全部 409.24 s 的 eig 中，level 2 的 158.95 s 属于 single-penalty
路线，不在本轮收益空间内。dense 对照的预期不是提速：dense eig 设备饱和时约
355 分量/s，level ≥ 3 含浪费的 80,751 个分量约 227 s，与现行 243.66 s 同
量级，而且本方案新增了 gather 与状态管理开销（见"测量"）。对照的价值是把
调度、接口、拷贝成本与 eig 成本分开量出来。

## 为什么先不换消费者

生命周期重放显示 round-robin 消费 + 有界 640 批生产能把浪费降到 0.9%、驻留降
到 2.13 GiB，但它是一个新的物理调度器。路径 (i) 要求的正是物理分组与现行
完全相同：只有这样，分量首次被引用的组才等于 baseline 的 miss 组。因此本
方案不改调度器的分组逻辑；round-robin 留到 structured 阶段证明浪费或显存
成为限制时再做，而且届时要重新处理 Gram 类别问题。

## 每层数据流（level ≥ 3）

1. `make_layer_plan`、`exact_primary_coefficient_arena.begin_level` 与
   `prefill_multi_penalty_level_target_states`（`full_cuda_ci_one_call.cpp`）
   保持不变。prefill 已对本层全部唯一 `(target, S)` 完成优化并把系数写入
   arena；该集合就是本层 planned 分量。
2. **阶段 A，窗口内残差物化**：在每个 prefill 窗口的 optimizer 完成之后、
   `release_multi_penalty_prefill_window_handles` 之前，对窗口内每个 S 的
   target 按列号排序、均衡切成不超过 16 列的 cohort（17 列分 9 + 8，即冻结
   重放已验证的方案），用窗口已持有的 prepared 句柄调用生产函数
   `multi_penalty_gcv_cuda_finalize_selected_coefficients`，把残差 D2D 拷入
   本层设备残差存储（planned 分量 × n × 8 字节，canonical 最大 154 MB），随即
   释放残差槽位。放在窗口内：不新增 prepared 句柄 build/release；不对
   `NativeSetupCache` 发出额外 `get`，不改变其 LRU 顺序、驻留集合与
   `live_conditioning_keys_`；残差 key 直接取自同一 producer 的 base target
   key 与 log-SP 比特，与 frontier 路径的 `exact_residual_keys` 同源。
3. **阶段 B，整层三字段预填充**：prefill 结束后，按残差存储顺序、不分 S，以
   执行上下文的分量容量为批宽（上下文由 `create_full_cuda_ci_legacy_eig_component_cache(n,
   num_col, budget, bypass, p, p², lanes)` 创建，上限为 `p`：22 列即 22）执行
   现行批构建中的距离构造、`syevd`、分量选择与 total 归约，把 vectors、
   selected_values、total 写入 cache 槽位，**不执行 Gram GEMM 与 self_moment
   归约**。key map 元数据新增 `gram_state ∈ {complete, pending}`；self_moment
   槽位同时写入哨兵值作为防御性检查，但状态以元数据为准。现有批构建入口要求
   `pair_count ≥ 1`，阶段 B 需要一个不带 pair 的 component-only 入口，复用同一
   套语句。阶段 B 在本层记录一个 completion event，消费者的首个批在读取任何
   槽位前等待它。
4. **按层 reset**：实现中在 level ≥ 3 的阶段 A 之前检查容量并执行：等待上一层最后一个读取 cache 槽位
   的消费批的 completion event；清空 key map；槽位游标归零；账本记录
   reset 次数、丢弃的 complete 条目数与丢弃的 pending 条目数。累计计数不变。
5. **消费者**：frontier 调度、桶选择、组构成、task/deletion 逻辑重放与 trace
   顺序不变。批构建中每个分量有三种状态：
   - `HIT_COMPLETE`：四字段有效，直接用 cache 槽位指针进入 pair 阶段（现行行为）；
   - `HIT_PENDING_GRAM`：三字段有效，self_moment pending；
   - `MISS`：level 1–2 按现行路径完整构建；原型开启的 level ≥ 3 立即 fail closed，
     不进入按需构建。层开始检查 planned 条目容量，不实现 mixed MISS/PENDING 回退。
   批内 `eig_miss_count` 只计 `MISS`，继续控制 residual bypass（现行
   `residual_cache_bypass = component_cache && miss_count == 0 && …`）、eig 执行、
   eig 账本；`gram_pending_count` 独立计数，并独立取得、检查 Gram scratch，不能
   沿用 `miss_count == 0` 时的空指针分支。正常状态包括
   包括"eig miss 为零、Gram pending 大于零"（level ≥ 3 的常态）。
   pending 分量的处理：按组内 `component_targets` 顺序（与 baseline 的 miss
   顺序相同），把 vectors 与 selected_values 从 cache 槽位 D2D gather 到执行
   上下文的 miss 缓冲（`vectors + i × vector_stride`、`selected_values + i × k`，
   与现行布局、lda、stride 相同），以 `batchCount = gram_pending_count` 执行与
   现行完全相同的 `cublasDgemmStridedBatched` 与 `reduce_legacy_self_moment_kernel`
   语句（同一 handle、同一 pedantic/no-atomics 配置），把 self_moment D2D 提交到
   各自槽位并把元数据置为 complete。全部在批 stream 上顺序执行，并在 pair
   metadata 上传之前排队；pair 阶段在同一 stream 上读取 complete 槽位。
   元数据 complete 表示已为该 stream 排队，不能作为其他 stream 无事件读取的依据。
   level ≥ 3 的真实 miss 在数值消费前失败，不把两类分量拆成不同 Gram 批继续执行。
6. level 1–2 完全不变，原型只在 level ≥ 3 reset；分量不跨层复用。

## 账本与审计

现有硬门（`full_cuda_ci_one_call.cpp` 的 cache 审计与 arena 审计）要求
`entries == insert_count`、`store_d2d_bytes == insert_count × bytes_per_entry`、
`cache_batch_count == cuda_legacy_compatible_residual_batch_count`、
`miss == insert + bypass`，以及 arena 的 `coefficient_gather_copy_count ==
reuse_hit_target_count`、`store_target_count == exact_primary_target_count`。
路径 (i) 下这些关系改写为分项账本（开关关闭时全部现有硬门原样成立）：

| 计数 | 含义 | 开关打开时的硬门 |
|---|---|---|
| `prefill_finalization_target_count` | 阶段 A 物化的 target 数 | == planned 分量数；cohort 宽 ≤ 16 |
| `prefill_three_field_store_count` / `_bytes` | 阶段 B 写入的条目数与字节 | 条目 == planned 分量数；字节 == 条目 × (n·k + k + 1) × 8 |
| `gram_pending_created` | 阶段 B 置为 pending 的条目 | == 阶段 B 写入条目数 |
| `gram_resolved_count` / `self_moment_store_bytes` | 消费阶段首次解析的 pending 条目 | == level ≥ 3 实际消费的唯一分量数；字节 == 条目 × 8 |
| `gram_resolve_batches` | 含 pending 的消费批数 | == 开关关闭运行中 level ≥ 3 含 miss 的批数 |
| `gram_resolution_sequence_sha256` | 按执行顺序对 level、完整 request identity、计数、有序 component keys 做滚动 SHA | == 开关关闭运行的原 miss 身份与顺序摘要 |
| `gram_gather_d2d_bytes` | gather 字节 | == 解析条目 × (n·k + k) × 8 |
| `consumer_request / hit / miss` | level ≥ 3 消费阶段的 cache 请求（pending 计为 hit） | miss == 0；request == hit |
| `pending_at_level_end` | 层末未解析的 pending 条目 | == planned − 实际消费（未消费是正常情况，不要求清零） |
| `cache_reset_count` / `reset_discarded_complete` / `reset_discarded_pending` | reset 次数与丢弃量 | 累计写入 == 当前驻留 + 累计丢弃 |
| `coefficient_gather_copy_count` | arena 系数读取 | == 阶段 A 物化 target 数 + 消费者 reuse hit 数（后者在 level ≥ 3 应为 0） |
| level 1–2 的现有计数 | 按需路径 | 维持现行关系 |

"insert × 完整条目字节"的等式不再成立，改为三字段写入与 self_moment 写入
两项分别记账。`gram_resolution_sequence_sha256` 是本方案的关键门：它直接审计
"pending 集合与 baseline 的 miss 集合逐批对应"，任何缓存容量不足、请求跳过
或分组变化都会使它失败。

## 验收链与数值身份

验收链按顺序：

1. **三字段预填充逐位**：阶段 B 产出的 vectors、selected_values、total 对
   开关关闭运行中对应分量的 cache 条目逐位一致。已有证据：22 宽重组下全部
   6,632 个捕获分量的三字段逐位一致；原型仍要在自己的数据流上验证。
2. **gather 后 Gram 逐位**：按原 miss 分组、原顺序、原 batchCount、原矩阵
   尺寸、原 stride 与 cuBLAS 配置执行 Gram 与归约后，self_moment 对开关关闭
   运行逐位一致。这是新的数据流，此前的 scalar-only 差分重放没有测试它；
   必须先在 2,592 个原 miss cohort 上用研究探针验证（见"实现步骤"第 0 步），
   再写进原型。
3. **消费前四字段逐位**：pair 阶段读取的每个槽位四字段逐位一致。
4. **完整方法**：p/pMax、task/deletion、图、sepset、n.edgetests、RNG 与资源
   门（见"验收门"）。

残差：不超过 16 列的 cohort 与现行 ≤16 宽的组同一比特类，22 列子集 level ≥ 3
的实际 cohort 最宽 10。canonical level 3 的 54 个 ≥17 宽组的 target 残差会
落入另一类。设计时样本中观察到 ≤ 6.2e-15 的差异，这不是 canonical 的
误差上界。后续 canonical 实测 951 个首次消费分量变化，最大 residual 差
5.329e-15；906 个 p 值变化全部通过完整 residual/component identity 归因，
最大 off/on p 差 1.638e-15，全部最终 p/pMax 对 authority 仍独立满足 1e-10。
不能仅按 target 列号认定例外。

self_moment：现有证据只覆盖捕获的 6,632 个分量，证明三字段在重组后逐位
一致、差异集中在 singleton 的 self_moment；它不能证明所有形状都只有 1 与
≥ 2 两个类别。路径 (i) 不依赖这一假设：它复现的是每个分量原来的 batchCount，
逐位一致由验收链第 2、3 步实证。

component key：dense 对照保持现有表示与 key 规则；structured 替换时再加入
独立 solver identity。

## 物理分组一致性

frontier 调度在选择就绪桶时优先选择 `setup_cache.contains_conditioning_key(S)`
为真的 S，该集合与 `NativeSetupCache` 的 LRU 顺序、容量淘汰和 host 缓存状态
一起决定物理分组。路径 (i) 的正确性依赖物理分组与 baseline 完全相同：

1. 阶段 A 在 prefill 窗口内复用已持有的上下文与句柄，不新增 `setup_cache.get`；
   阶段 B 与 gather 不触及 setup cache。
2. 新增回执字段：每层一个物理分组序列摘要，按执行顺序对
   `(level, S key, 有序 selected task index, executed pair count)` 做 SHA-256，
   同一新 binary 开关关闭与打开两次运行必须相同。
3. `gram_resolution_sequence_sha256` 与开关关闭运行的逐批 miss 身份和顺序摘要相同；
   只比较计数不能发现 key 替换或顺序交换。
4. canonical 逻辑重放（"Physical CUDA rounds preserve per-edge reachability…"
   一段）与其全部严格门保持不变。

## 验收门

先 22 列子集（default-Inf，与冻结重放的未插桩基线比较），后 canonical
351×48 default-Inf。所有比较都在同一个新 binary 上做开关关闭 / 打开两次运行：

| 门 | 子集 | canonical |
|---|---|---|
| 第 0 步：探针上三字段→gather→原 Gram 逐位 | PASS：2,592 个原 miss cohort、6,632 个分量全部一致 | — |
| 开关关闭 vs accepted 回执 | p/pMax、task/deletion、图、sepset、n.edgetests、RNG 逐位一致 | 对 accepted native 的 canonical 审计逐位一致 |
| 验收链 1–3（三字段、Gram、消费前四字段） | 逐位一致 | 逐位一致（残差类别不同的 target 除外，须逐条归因） |
| 开关打开 CI p 值 / pMax | 对关闭运行逐位一致 | 全部 ≤ 1e-10 对 kpcalg 权威（`run_phase10_pvalue_contract_audit.R`）；记录逐位不同的 p 值数量、最大漂移与归因 |
| task/deletion identity、adjacency、sepsets、n.edgetests、RNG endpoint | 一致 | 一致 |
| 物理分组序列摘要、`gram_resolution_sequence_sha256` | 关闭 == 打开 | 关闭 == 打开 |
| 分项账本硬门 | 全部成立 | 全部成立 |
| CPU authority / fallback、residual 与 component D2H | 0 | 0 |
| native 资源计数 | 闭合 | 闭合 |
| compute-sanitizer memcheck | 阶段 A/B 与 gather 冒烟 | — |

开关建议为 `FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL`（默认 0）。加入代码后
产生新的源码闭包与 native SHA，必须重新记录。
当前实现硬限制为 `dcc.gamma` 的 351×22 或 351×48、fresh caches、exact-primary
reuse、component cache + bypass + 8 lanes、原 single-group scheduler。
上表 canonical 列已在后续窄工作中完成；同形状其他数据不能继承该输入的资格。

## 测量

性能对照只在同一新 binary 上做开关关闭 / 打开比较；不使用 880.659 s 作为
基线（它属于实验 native `c4f599ef…`）。Gram 的算术工作量不变，但本方案新增：

| 开销 | 规模（n = 351，k = 35） |
|---|---|
| gather D2D | 每个解析分量 (n·k + k) × 8 = 98,560 字节；canonical 65,694 个分量约 6.47 GB |
| self_moment 提交 | 每个解析分量 8 字节与一次 D2D 调用 |
| 状态初始化与元数据 | 每个预填充条目 |
| 额外预填充分量 | canonical 15,057 个、22 列 638 个未消费分量的距离、eig 与选择 |
| 残差存储 D2D | 每层 planned 分量 × n × 8 |

每层记录阶段 A 时间与 cohort 数、残差 D2D 字节与时间、阶段 B 时间（距离
构造、eig、选择、三字段提交）、gather 字节与时间、Gram 解析批数与时间、
planned − consumed 浪费、cache 峰值条目与字节、frontier 阶段时间、outer
elapsed。dense 对照的定位是验证语义与成本；后续已记录单次完整 outer 收益，
仍不承诺重复性能。structured 替换须独立完成数值与完整 outer 验证。

## 实现步骤与规模估计

0. ~~窄验证（研究探针，不改生产）~~ **已完成并通过**（见
   [第 0 步报告](PHASE10_GATHER_GRAM_STEP0_20261021.md)）：22 宽预填充的三字段
   6,632/6,632 逐位一致；按原 miss 分组 gather 后的 self_moment 6,632/6,632
   逐位一致，重复与反序相同；22 宽 Gram 对照精确重现 1,002 个 singleton 差异；
   memcheck 冒烟 0 errors、0 leaks。
1. cache：`gram_state` 元数据、`reset_level()`（含 completion event 等待）、
   分项账本：`full_cuda_ci_vertical.cu` cache 类与 `full_cuda_ci_one_call.cpp`
   层循环与审计段，约 150 行。
2. 窗口内阶段 A 与本层残差存储：放在 `complete_multi_penalty_prefill_window` 内部，
   `finish_store_window`、原 residual 租约释放之后，prepared handles 释放之前。
   D2D detach 完成后才能复用源 residual 槽位与 arena coefficient staging；
   key 同源，不触碰 setup cache。
3. component-only 三字段预填充入口：从 `run_full_cuda_ci_phase35_legacy_eig_batch_impl`
   抽出距离、eig、选择与三字段提交，允许无 pair 调用，批宽等于上下文容量，
   约 250 行。
4. 消费者 `HIT_PENDING_GRAM` 状态：gather、原 Gram/归约、self_moment 提交、
   stream/event 依赖、`gram_resolution_sequence_sha256`，约 200 行。
5. 物理分组序列摘要、诊断与回执字段、开关、R 侧运行器，约 150 行。
6. 子集门 → canonical 门 → 文档与新源码闭包。

约 950 行与两轮资格运行，预计 4–6 个工作日。

## 风险

- 第 0 步失败（gather 后 Gram 与原 miss 形状不一致）意味着 self_moment 还
  依赖除 batchCount 之外的因素，路径 (i) 需要重新设计。
- 抽出分量构建代码可能扰动现行路径的比特：开关关闭时必须对 accepted 回执
  逐位一致。
- 残差 key 不同源、缓存容量不足、请求跳过或分组变化都会破坏"pending 集合 ==
  baseline miss 集合"：由 `gram_resolution_sequence_sha256` 与分项账本捕获。
- 阶段 A 若意外触碰 setup cache，会改变物理分组：由分组序列摘要捕获。
- 显存：残差存储 ≤ 154 MB 加 11 GiB cache 与 prefill 窗口可容纳；structured
  阶段的 640 宽上下文另需约 2.5 GiB，届时复核峰值。
- level 2 仍按需执行；其 158.95 s eig 在 single-penalty cohort 问题解决前
  不受益。

## 修订记录（2026-10-21）

第一次修订（评审六点）：账本改为 level ≥ 3；按层 reset 分项账本与
completion event 等待；批宽改为上下文容量、payload 比较覆盖四字段；新增
物理分组一致性；删去 canonical 误差上界；同一新 binary 开关对照并记录新
源码闭包。

第二次补充：payload 门 STOP 后的路径分析与路径 (i) 设计。

第三次修订（评审五点）：
1. `gram_pending_count` 与 `eig_miss_count` 分离，允许"eig miss 为零、Gram
   pending 大于零"，pending 分量以显式状态与 stream/event 依赖进入 pair
   阶段，不只靠哨兵值。
2. "必然逐位不变"改为验收目标：原型恢复原 miss 数、顺序、矩阵尺寸、stride
   与 cuBLAS 配置后验证 gather 后的 Gram/self_moment；新增第 0 步探针验证。
3. 正文统一到路径 (i)："消费者源码不变""阶段 B 计算 Gram""四字段批宽无关是
   前置门"等表述已删除；eager 方案的 STOP 保留为历史结论。
4. 未消费 pending 条目为正常情况，不要求层末清零；分别记录生产、首次解析、
   当前 pending 与 reset 丢弃；三字段写入与 self_moment 写入分别记账。
5. "不新增 GPU 工作"改为明确的开销表：gather 每分量 98,560 字节、canonical
   约 6.47 GB，另有提交、状态初始化与额外预填充分量；dense 对照不承诺加速。
