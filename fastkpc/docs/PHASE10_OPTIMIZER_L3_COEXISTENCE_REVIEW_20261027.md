# L3 实际 replacement coexistence：复核

日期：2026-10-27。复核 [L3 共存探针](PHASE10_OPTIMIZER_L3_COEXISTENCE_20261027.md)（证据
`fastkpc/artifacts/full_cuda_ci/optimizer_l3_coexistence_20261027/attempt2/`）。只读复核，没有新的 GPU 运行。
`STOP_L3_REPLACEMENT_POOL_INCREMENT` 按预声明合同成立；`GO_CAPTURED_L3_SLOTS` 保留；L3 runtime 仍为
`NOT_YET_GO`；canonical development 中位数仍为 368.851 s。Phase 10 ACTIVE、candidate NOT_FROZEN、
holdout SEALED、recommended=false 均不变。同日按 Codex 核对意见修订，见“修订记录”。

## 修订记录（Codex 核对后）

Codex 复核了 manifest 的 64 个条目、报告的 6 个 SHA、native 资源表和三份 CUPTI trace，主要数字吻合；原合同的
STOP 保留，stack 归因可作为公共首次执行成本的有力解释继续做窄诊断。另提出四点，我对照执行顺序和数字复核，
全部接受：

1. **不能把“首次 L3 launch 之后”设为池增量的零点。** S 先建好全部 32 槽，再启动 optimizer；首次 launch 时，
   约 1.135 GiB 的建池成本已经存在，重置基准会把它一起抹掉。首版第 5 节的这一选项作废。新合同保留建池前
   的原始账本，分别记录建池、首次执行、Stage A 和释放后的占用；公共成本可以另列，但不能靠重置基准消掉槽位成本。
2. **stack 归因不是精确的扣除公式。** limit 变化 × 196,608 是估算，不是已测得、可全部扣除的物理分配。按默认
   1,024 B 算出 1,143 MiB，S 实测 1,150 MiB，仍差 7 MiB。下一轮要同时核对 limit、物理占用和应用账本，未解释的
   部分继续计入容量。首版“从物理增量中减去按 limit 变化预先声明的 stack 项”作废。
3. **L6 的表述和单位。** 首版写“runtime 余量只有 107.8125 MiB”，不准确：L6 runtime 检查点的实际空闲约 2.105 GiB，
   比 2 GiB 门高 107.8125 MiB，两者的风险含义不同。另外我把 MiB 当成了 GiB：1,150 MiB 是 1.123 GiB（1,164 MiB
   是 1.137 GiB），不是 1.15 GiB；原门槛下留给槽位池的额度约为 1.5 − 1.123 = 0.377 GiB，不是 0.35 GiB。
4. **压缩 stack 不能计入收益预算。** “每线程少 1 KiB 约省 192 MiB”只是估算：它要求整个 context 的最高 stack
   需求确实下降，且没有把内存转移到别的 workspace。只降低一个 kernel，之后更大的 stack 需求仍会把预留扩大回去。

Codex 建议下一项保持很窄：在原 W/S 执行中加入首次 L3 launch 前后的 stack limit 与显存记录，不加预热、不改
槽位数，据此形成新的容量合同，再完成 S 的实际 Stage A 共存与完整语义资格；新增观测的运行只作诊断证据。
我同意。状态：原容量合同 STOP 保留；captured L3 slots GO 保留；首次执行 stack 归因诊断 GO；完整 L3 runtime
集成 NOT_YET_GO；accepted development median 368.851 s。

## 结论

```text
完整性    报告列出的 6 个 SHA-256 吻合；manifest 的 64 个条目（含 attempt1）按当前文件重算全部一致；
          生产源码与 native 未变。W exit 0；S 与仅加日志的重放都按预期在容量守卫 exit 1，均 isolated
数字      S 检查点与失败归因复现：建池增量 1,218,445,312 B（1.134766 GiB），optimizer 后
          2,424,307,712 B（2.257813 GiB），剩余 3.177551 GiB；W 每个窗口关闭后的非应用增长
          1,158.679–1,164.025 MiB、L3 最小 free 3.048645 GiB，与报告一致
判定      STOP 成立。按“arena reserve 后、建池前”的基准，L3 首次执行的公共成本必然落进 S 的增量；
          不能事后改基准或改门槛
归因      公共首次执行成本与 device stack（local memory）预留相符：L3 small64 持久 kernel 每线程
          stack 7,120 B，L3 之前最大的是单 penalty kernel 的 800 B。按 196,608 个常驻线程，预测增长
          1,143–1,184 MiB，实测保留 1,150–1,164 MiB（1.123–1.137 GiB）。W 与 S 都付这笔成本，它不是槽位成本
联合      S 最低观测 free 3.178 GiB（optimizer 后、Stage A 前），比 W 的 L3 最小值高 0.13 GiB；
          S 的 Stage A 未测，联合资格未完成
建议      先在原 W/S 执行中记录首次 L3 launch 前后的 stack limit 与显存（不加预热、不改槽位数）。新合同保留
          建池前的原始账本，分列建池、首次执行、Stage A、释放后的占用；stack 项只按实测核对另列，未解释部分
          计入容量；联合余量覆盖 optimizer + Stage A，并与同配置 W 比较
```

## 1. 完整性

- `protocol.json`、`build.json`、`W/stage-a.bin`、`failure-attribution/closure.json`、`closure.json`、
  `manifest.json` 的 SHA-256 与报告一致；manifest 的 64 个哈希条目（`attempt1/` 条目相对 coexistence 根目录）
  按当前文件重算全部一致。
- L6 isolated build 绑定的 101 个生产源文件与 native `313d9f3c…` 当前哈希不变。
- W 进程 exit 0；S 与 S-attribution 都以 `optimizer slots exceeded the qualified memory budget` exit 1，
  三者 `isolated=true`、没有外部 GPU 进程。
- 协议在 GPU 运行之前固定：池增量 ≤1,610,612,736 B（1.5 GiB），基准为“coefficient arena reserve 后、建池前”，
  free ≥2 GiB，10 ms 采样。仅加日志的重放另有声明，门槛与 overlay 不变，预期结果写明“同样拒绝，不进入
  Stage A”。

## 2. 数字复现

S 检查点（`S-attribution/checkpoints.csv`、`failure.json`），相对 L3 起点：

| 时点 | free | 应用可见增长 | 物理增长 | 非应用增长 |
|---|---:|---:|---:|---:|
| arena reserve 后、建池前 | 5.435364 GiB | 1,945.6 MiB | 1,948.0 MiB | 2.4 MiB |
| 32 槽创建完成 | 4.300598 GiB | 2,785.0 MiB | 3,110.0 MiB | 325.0 MiB |
| optimizer workers 全部结束 | 3.177551 GiB | 2,785.0 MiB | 4,260.0 MiB | 1,475.0 MiB |

- 建池：应用可见 +839.4 MiB（880,205,568 B），物理 +1,162.0 MiB；非应用 +322.6 MiB，约 10.1 MiB/槽，
  同时新增 32 个 cuBLAS handle、32 个 stream、96 个 event。
- optimizer：应用可见 0，物理 +1,150.0 MiB（1,205,862,400 B）；4,065 setups / 54,878 targets 逐位一致，
  worker 期间受审计资源建销为 0。
- 2.258 GiB = 池应用 839.4 + 池非应用 322.6 + 首次执行 1,150.0 MiB。

W（`W/checkpoints.csv`，5,639 个检查点）：64 个窗口关闭后的非应用增长为 1,158.679–1,164.025 MiB（首个窗口
1,161.067、最后一个 1,162.362），BLAS handle 回到 38；L3 最小 free 为 3.048645 GiB，出现在第 57 个窗口
附近的 Stage A cohort 检查点，此时应用可见 +2,633.1 MiB、非应用 +1,758.9 MiB（约 1,162 MiB 加 64 个窗口
对象约 597 MiB，约 9.3 MiB/个）。

S 建池前的应用可见增长（1,945.6 MiB）与 W 的 L3 结束时完全相同：S 一开始就预留整层存储，W 则随窗口逐步增长。

## 3. 公共首次执行成本：device stack 预留

报告把来源列为“allocator、device local memory、库或 driver，尚未分离”。已有证据足以缩小到 local memory
中的 per-thread stack：

- **kernel 的 stack 需求**：`cuobjdump -res-usage fastkpc/build/fastkpc_cuda.so` 显示，multi-penalty
  `optimize_multi_penalty_targets` kernel 的 per-thread STACK 为 small64 7,120 B、base80 7,136 B、ext192
  7,344 B、ext384 7,728 B、ext559 8,016 B，LOCAL 均为 0；L0–L2 的单 penalty `replay_fused_exact_optimizer`
  kernel 为 800 B。CUPTI 的 `localMemoryPerThread` 只报 LOCAL，所以 trace 中一直是 0。
- **CUPTI 的预留**：生产 cache-ON trace 中，`localMemoryTotal` 在大多数 kernel 上为 108 MiB，0.44 s 的单
  penalty kernel 为 260 MiB；从 99.56 s 的第一个 L3 small64 持久 kernel 起为 1,444 MiB，L8 base80 为
  1,448 MiB。L3 原型的 W/S 两份 trace 中，持久 kernel 也都是 1,444 MiB。这个字段按 launch 报告该 kernel
  所需的预留；是否保留，要由显存检查点判断。
- **数量吻合**：用“约 108 MiB 常数 + STACK × 196,608”（Ada 每 SM 最多 1,536 个常驻线程 × 128 个 SM）计算，
  800/7,120/7,136 B 分别给出 258/1,443/1,446 MiB，实测 260/1,444/1,448 MiB；108 MiB 常数项的来源未查明。
  context 在 L3 之前实际保留多少，有两种读法：若等于此前最大的 kernel 需求，L3 增长为 1,444 − 260 =
  1,184 MiB；若等于默认的 per-thread stack limit（通常为 1,024 B，需确认），增长为 (7,120 − 1,024) ×
  196,608 = 1,143 MiB。实测保留：W 每个窗口关闭后 1,158.7–1,164.0 MiB，S optimizer 阶段 1,150 MiB，
  captured 原型一次原生预热后 1,152 MiB，落在两种预测之间（比 1,143 MiB 高 0.6–1.8%，比 1,184 MiB 低
  1.7–2.9%）。第 5 节的只读检查可以区分两种读法。
- **为何保留**：CUDA 文档说明，`cudaDeviceLmemResizeToMax` 已弃用，它的行为（launch 扩大 local memory 后
  不再缩小）现为默认；per-thread stack 由 `cudaLimitStackSize` 控制，driver 按 launch 需要自动增大，launch
  后不复原。

结论：公共首次执行成本与 context 的 stack 预留相符：第一次 launch L3 multi-penalty kernel 时一次性增长约
1,150–1,164 MiB（1.123–1.137 GiB），之后一直保留。W 与 S 都付这笔成本，它不是槽位池的成本。证据来自 kernel 资源表、CUPTI 字段和
显存检查点的吻合，没有直接观察 driver 内部分配；它解释了这笔成本的主体，但不是可以精确扣除的公式。

## 4. STOP 的含义与联合对照

- 合同的基准在第一次 L3 launch 之前，所以 S 的首次执行增长（1,150 MiB，1.123 GiB）必然计入 S 的增量。在这一
  基准下，槽位池本身要小于约 0.377 GiB（1.5 − 1.123）才能过 1.5 GiB 门。这一门实际上主要量到了公共成本。我同意 Codex 的处理：保留 STOP，
  不事后扣除，也不改门槛复用同一 receipt。
- 联合余量：S 的最低观测 free 为 3.177551 GiB（optimizer 后），比 W 的 L3 最小值（3.048645 GiB，Stage A
  期间）高 0.129 GiB。但 S 没有进入 Stage A，两者的时点也不同，不能据此说 S 的联合峰值更低。S 已预留整层
  存储，Stage A 期间增加的可能比 W 少，这需要实测。

## 5. 对新合同的建议

1. **在原 W/S 执行中直接记录 stack 预留，不加预热、不改槽位数**：于第一次 L3 multi-penalty launch 前后各读一次
   `cudaDeviceGetLimit(cudaLimitStackSize)`、`cudaMemGetInfo` 和应用账本。预期 limit 从启动时的值（通常为
   1,024 B）增至 ≥7,120 B，free 同时减少约 1,150 MiB、应用账本不变；启动时的 limit 也决定第 3 节两种读法中
   哪一种成立。这些运行只作诊断证据。
2. **保留建池前的原始账本**：基准仍取“arena reserve 后、建池前”，分别记录建池完成、首次 L3 launch 前后、
   optimizer 完成、Stage A 期间最小值和释放后的 free 与应用账本。不重置基准：S 的建池在首次 launch 之前，
   以 launch 之后为零点会把建池成本一起抹掉。
3. **公共成本只按实测另列**：stack 项只在 limit 变化、free 变化和应用账本三者一致的范围内另列；limit 变化 ×
   196,608 只是预期值。未解释的部分（例如默认 1,024 B 时 1,150 − 1,143 = 7 MiB）继续计入容量。
4. **门针对哪一项要预先写明**：建池增量（本次 1.134766 GiB）、首次执行增长（本次 1,150 MiB）、二者之和（本次
   2.257813 GiB），以及整个 L3 optimizer + Stage A 的联合余量（free ≥2 GiB，检查点和 10 ms 采样，并与同配置 W
   的 L3 最小 free 对照，本次 3.048645 GiB）。若池增量要沿用 L4/L6 的 ≤1 GiB，就需缩小池，这是设计变更，要重测
   时间与输出。
5. 同意 Codex 的其他要求：S 必须完成实际 Stage A 共存与完整语义门；不隐藏预热；若改为在 L3 之前设定 stack limit，
   它与预热一样，会把约 1,150 MiB 提前到 L0–L2，成本与显存都要计入完整生命周期与 outer。

## 6. 一项更大的观察

multi-penalty kernel 的 stack 预留约 1.30 GiB（7,120 B × 196,608 = 1,335 MiB），从 L3 一直保留到运行结束，包括
L6：L6 runtime 检查点的最小空闲约 2.105 GiB，只比 2 GiB 门高 107.8125 MiB。各 bucket 的 per-thread stack 为
7,120–8,016 B。按估算，每少 1 KiB/线程约省 192 MiB，但前提是整个 context 的最高 stack 需求确实下降，且没有把
内存转移到别的 workspace。canonical 运行中 L3 起由 small64（7,120 B）决定预留，L8 的 base80（7,136 B）会再
扩大到同一水平；只改一个 kernel，之后更大的需求仍会把预留扩大回去。这可以作为以后的方向，不能计入收益或
显存预算；改 kernel 也需要逐位资格，不在当前范围。

## 证据

`attempt2/S-attribution/{checkpoints.csv,failure.json,failure-points.csv}`、`attempt2/S/checkpoints.csv`、
`attempt2/W/{checkpoints.csv,probe.json}`、`attempt2/protocol.json`、`attempt2/failure-attribution/protocol.json`；
CUPTI：`optimizer_slot_line_review_20261025/ktrace-run-cache-on/`、`optimizer_l3_slots_20261027/{trace-W,trace-S}/`
（`kernels.bin` 的 `localMemoryTotal_v2` 字段）；kernel 资源表：`cuobjdump -res-usage fastkpc/build/fastkpc_cuda.so`。
计算为只读一次性脚本；没有修改生产源码、native 或默认配置，没有 commit 或 push。
