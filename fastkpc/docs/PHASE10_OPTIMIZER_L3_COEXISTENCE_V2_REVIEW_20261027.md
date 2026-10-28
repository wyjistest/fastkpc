# L3 实际替换共存（分点原始容量合同）：复核

日期：2026-10-27。复核 [L3 共存 v2](PHASE10_OPTIMIZER_L3_COEXISTENCE_V2_20261027.md)（证据
`fastkpc/artifacts/full_cuda_ci/optimizer_l3_coexistence_v2_20261027/attempt1/`）。只读复核，没有新的 GPU 运行。
`GO_L3_REPLACEMENT_COEXISTENCE_SPLIT_CONTRACT` 成立；旧 `STOP_L3_REPLACEMENT_POOL_INCREMENT` 与
`GO_CAPTURED_L3_SLOTS` 保留；L3 runtime 仍为 `NOT_YET_GO`；本轮不是性能证据，accepted development median
仍为 368.851 s。Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。
2026-10-28 按 Codex 核对意见修订第 4 节和结论中的时间归因，见“修订记录”。

## 修订记录（Codex 核对后）

Codex 重算了 55 个 manifest 条目、101 个生产输入和 W/S 两组 67 项硬门，均一致；下游显存表准确；共存 GO 与
“进入默认关闭的 L3 runtime 集成与资格”的结论成立。第 4 节有两处归因过强，我对照 overlay 源码
（`attempt1/one_call.cpp`）复核，全部接受：

1. **诊断 shadow 在两种模式中的位置不对称。** S 的逐 setup 输出 shadow 检查（`l3_probe::optimizer`，第 5717 行）
   在 worker 内，处于 optimizer 计时（`started` 到 worker join）之内；W 的同一检查（第 5417 行）在窗口 optimizer
   返回后的 completion 中，处于 optimizer 计时之外。因此 36.853 s 只是当前计时边界之差，不是 optimizer 的净收益；
   L3 层其余部分的 3.119 s 含 W 的 shadow 时间，也不能都归于创建与释放的节省。另外，S 的 geometry 检查（第 5676 行）
   在打包 geometry 时进行，早于 arena reserve、建池和 optimizer 计时；首版把它列为 S optimizer 偏慢的原因，作废。
2. **prepared build 计数器不含新建池成本。** `prepared_build_sec` 只累计旧路径 `ensure_multi_penalty_handle` →
   `create_multi_penalty_handle` 的逐请求创建；槽位池经 `OptimizerSlotPool` / `Policy::create` 另一入口创建，不在其中，
   本轮报告也没有单列建池耗时。所以 1.371 → 0.319 s 的 1.052 s 不是净构建收益。

这两点不影响共存 GO。outer 差 40.992 s、其中 L3 层 39.972 s 的账本成立，但两种模式都带诊断 shadow，仍需没有
shadow 的正式重复计时。下游继续以 free ≥2 GiB 为硬门；S 多出的 4 MiB 只作描述。

## 结论

```text
完整性    protocol/build/W stage-a.bin/closure/manifest 的 SHA-256 吻合；manifest 55 个条目按当前文件重算一致；
          生产源码与 native 未变。协议 17:43:42 声明，早于 W（17:46:18）和 S（17:55:01）。两次都 exit 0、
          GPU1 独占；GPU0 外部进程 1918904 在运行前已声明，本轮只作语义与容量诊断
合同      上一轮复核的四点都已落实：门在新协议中预先声明；W/S free 比较只作描述，不设 noninferiority；
          envelope 外 fail closed；建池（任何 launch 之前）、optimizer 完成、总增量和整个 Stage A 的
          free 分点测量。Stage A 不设总增量上限，但 free ≥2 GiB
数字      S：建池 1,218,445,312 B、optimizer 新增 1,205,862,400 B、总增量 2,424,307,712 B，均过门；整个
          L3 最小 free 3,411,869,696 B，出现在 optimizer 阶段，Stage A 观测点没有更低。采样 W 6,347 次、
          S 2,322 次，最大间隔 42.379/13.672 ms。W 的 L3 最小 free 3,273,457,664 B 只作描述
正确性    W/S 都 67/67；changed CI 0、pMax 不一致 0；S 的 5,573 个 Stage A cohort / 54,878 targets 与 W
          逐位一致；结束时应用资源全部归零
下游      W 的 L4/L5/L6 池检查点与 L6 隔离 campaign 的 ON2 逐字节相同（L6 最小 free 2,260,533,248 B）；
          S 在每个下游检查点都恰好多 4,194,304 B（4 MiB）。L3 槽位没有减少最紧的 L6 余量
时间      单次共享主机诊断：outer 差 40.992 s，其中 L3 层 39.972 s，其他各层与层外合计 1.020 s。L3 内
          optimizer（36.853 s）与其余部分（3.119 s）的划分受诊断 shadow 位置影响（S 在 optimizer 计时内，
          W 在计时外），不是净收益分解；不是性能证据
建议      按顺序：默认关闭集成 → 数值、容量、异常清理资格 → 冻结 native → 预声明的隔离 OFF/ON campaign
          （无诊断 shadow）；runtime 按分点合同实现检查点守卫；下游以 free ≥2 GiB 为硬门
```

## 1. 完整性与合同

- `protocol.json`、`build.json`、`W/stage-a.bin`、`closure.json`、`manifest.json` 的 SHA-256 与报告一致；
  manifest 的 55 个条目按当前文件重算一致。L6 isolated build 绑定的 101 个生产源文件与 native 未变。
- 协议在 17:43:42 声明，W 于 17:46:18、S 于 17:55:01 开始。门：建池增量 ≤1,610,612,736 B（1.5 GiB），
  建池完成到 optimizer 完成 ≤1,342,177,280 B（1.25 GiB），optimizer 完成时总增量 ≤2,952,790,016 B
  （2.75 GiB），整个 L3 optimizer + Stage A 的 free ≥2 GiB；Stage A 总增量只记录；W/S free 比较只作描述；
  不扣模型、不重置基线、不加预热、不改 stack limit。
- 我在 stack 归因复核中提出的四点都已落实：新协议预先声明；W 对照的用途写明为描述；envelope（n=351、q=28、
  三个 penalty、root rank 24、cohort 2–33）外 fail closed；建池在任何 L3 launch 之前单独测量。

## 2. 数字复现

由 `{W,S}/probe.json`、`{W,S}/cleanup.json`、`{W,S}/report.json` 和 `{W,S}/memory-samples.csv` 重算：

- S：基线 free 5,836,177,408 B，建池后 4,617,732,096 B，optimizer 后 3,411,869,696 B。三项增量与前两轮
  逐字节相同。最小 free 3,411,869,696 B 出现在 optimizer 阶段的采样点；Stage A 阶段的观测点都不低于它。
- 观测：W 共 11,986 次（检查点 5,639、采样 6,347，最大采样间隔 42.379 ms），S 共 7,903 次（检查点 5,581、
  采样 2,322，最大间隔 13.672 ms）。与报告一致，通过门只说明观测点满足合同。
- W/S 各 67/67 门，changed CI 与 pMax 不一致均为 0；S 检查了 4,065 个请求的 geometry；Stage A 5,573 个
  cohort、54,878 个 target；两者结束时应用 allocations/bytes/streams/events/BLAS handle 与 worker 资源建销
  都为 0。

## 3. 对下游层的影响

| 检查点 | L6 campaign ON2 | 本轮 W | 本轮 S |
|---|---:|---:|---:|
| L4 建池前 free | 3,502,047,232 | 3,502,047,232 | 3,506,241,536 |
| L4 最小 free | 2,692,546,560 | 2,692,546,560 | 2,696,740,864 |
| L5 建池后 free | 2,365,390,848 | 2,365,390,848 | 2,369,585,152 |
| L6 建池前 free | 2,734,489,600 | 2,734,489,600 | 2,738,683,904 |
| L6 最小 free | 2,260,533,248 | 2,260,533,248 | 2,264,727,552 |

L4/L6 池增量在三者中都相同（809,500,672 / 473,956,352 B）。S 在每个下游检查点都恰好多 4 MiB，原因未查明；
L6 最小 free 比 2 GiB 门高 111.8 MiB（W 为 107.8 MiB）。L3 槽位在 L4 之前关闭，没有减少下游余量。这是单次
共享主机运行的观察。

## 4. 时间分解（诊断，不是性能证据）

由两份 `report.json` 的 level trace 与 optimizer 分层重算：

| 边界 | W | S | 差 |
|---|---:|---:|---:|
| outer | 383.627 | 342.635 | 40.992 |
| L3 层 | 132.090 | 92.118 | 39.972 |
| L3 optimizer（当前计时边界） | 52.099 | 15.246 | 36.853 |
| L3 层中 optimizer 以外 | 79.991 | 76.872 | 3.119 |
| 其他各层与层外 | 251.537 | 250.517 | 1.020 |

- 两种模式的诊断 shadow 量相同（各 653,267,712 B），但位置不同：S 的逐 setup 检查在 worker 内、optimizer 计时之内；
  W 的在窗口 optimizer 返回后的 completion 中、计时之外。所以 L3 optimizer 一行与“其余”一行都受这一不对称
  影响：36.853 s 是当前计时边界之差，不是 optimizer 的净收益；3.119 s 含 W 的 shadow 时间，不能都归于创建与释放。
- `prepared_build_sec` 从 1.371 s 降到 0.319 s，但这个计数器只统计旧的逐请求 prepared 创建，新槽位池经另一
  入口创建，不在其中；本轮报告也没有单列建池耗时。这 1.052 s 不是净构建收益。
- S 的 L3 optimizer 为 15.246 s，比原型计时的 S 工作边界（14.291 s）多 0.955 s。本轮 S 的计时内包含首次扩栈、
  首次 GEMM 加载和逐 setup shadow 检查，原型计时在预热之后；geometry 检查在 S 的 optimizer 计时开始之前，
  不在其中。两者不可直接比较。
- outer 差 40.992 s、其中 L3 层 39.972 s 的账本成立，但两种模式都带诊断 shadow，各只有一次，主机共享。它不能
  扣减 accepted median，也不能代替没有 shadow 的隔离重复计时。

## 5. 对 runtime 与隔离 campaign 的建议

1. **顺序**（Codex 的建议，我同意）：默认关闭集成 → 数值、容量、异常清理资格 → 冻结 native → 预先声明的隔离
   OFF/ON campaign。
2. **runtime 守卫**：按分点合同实现：建池完成、任何 L3 launch 之前检查建池增量；optimizer 完成时检查新增与总增量；
   optimizer 与 Stage A 的各检查点检查 free ≥2 GiB。生产中没有 10 ms 采样线程，报告应像 L6 一样写明只覆盖检查点。
   L4/L6 的 1 GiB 守卫保持不变。
3. **隔离 campaign**：同一 native 的 OFF/ON，不带诊断 shadow，预先声明 outer 中位比、L3 optimizer 中位比和三对
   全胜，阈值在运行前固定。报告分列 L3 optimizer、L3 层其余部分、其他各层与层外；若要单独报告建池成本，需在
   runtime 诊断中另设计数器，不能用 `prepared_build_sec` 的变化代替。
4. **下游余量**：L4/L6 池的 free ≥2 GiB 继续是硬门，runtime 报告照常列出最小 free；本轮 S 多出的 4 MiB 只作描述。

## 证据

`attempt1/{protocol,build,closure,manifest}.json`、`attempt1/{W,S}/{probe,cleanup,report}.json`、
`attempt1/{W,S}/{checkpoints,memory-samples}.csv`、`attempt1/{W,S}-process.json`；下游对照：
`optimizer_l6_slot_runtime_20261026_isolated_completion/on2/report.json`。计算为只读一次性脚本；没有修改生产
源码、native 或默认配置，没有 commit 或 push。
