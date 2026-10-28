# L2/L3 Stage B 活动归因：复核

日期：2026-10-28。复核 [L2/L3 Stage B 活动归因](PHASE10_STAGE_B_ATTRIBUTION_20261028.md)（证据
`fastkpc/artifacts/full_cuda_ci/stage_b_attribution_20261028/attempt3/`）。只读复核，没有新的 GPU 运行。
`AUTHENTICATED_STAGE_B_ACTIVITY_DIAGNOSTIC_COMPLETE` 与 `UNAVAILABLE_PRIVILEGE_NOT_MEASURED` 成立；accepted
opt-in development median 仍为 329.078 s；L8 STOP、默认关闭、Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、
recommended=false 均不变。同日按 Codex 核对意见修订，见“修订记录”。

## 修订记录（Codex 核对后）

Codex 重新核对了 manifest 的 275 个条目，并确认空洞拆分准确。另提出四点和一项补充，我对照源码与 trace 复核，全部接受：

1. **核心 QR 只有一个线程执行。** `bidiagonal`（`full_cuda_ci_structured_eig.cu:188` 一带）用全部线程把 d/e/q 装入
   shared 后，`if (threadIdx.x != 0) return;`，只有 thread 0 执行 `recorded_dbdsqr_` 及其后的循环。首版说每 SM 4 个 warp，
   实际每 SM 至多 4 个线程在计算。后续审计必须同时看串行依赖和有效执行线程。
2. **资源表的两处误差。** 按寄存器分配粒度和 SM 子分区计算，bidiagonal 的寄存器上限是每 SM 16 个 block，不是 19；
   recover 的 shared 按分配粒度取整后剩余约 256 B，不是 288 B。bidiagonal ≤4、recover ≤1 的结论不变。
3. **status 读取来自同一个 host 线程。** 生产循环（`full_cuda_ci_vertical.cu:2251` 一带）先提交一个 wave 的全部 lane，
   再由同一线程逐 lane 调用 `read_status()`；trace 中 L2 的 780 次、L3 的 858 次都来自同一 thread id。首版写“lane 线程
   等待”，作废。
4. **依赖链估算只能提出假设。** 并发下单次 kernel 的时长含调度和资源竞争；静态步数乘估计延迟，不能判断瓶颈属于哪一种。

补充：recover 的 101,088 B 正好是 36 列 × 351 行 × 8 B，后续重点是各缓冲的同时活跃范围与复用条件。“降到约 10 KB、
每 SM 9 个”只作假设。

核对时我又看了 wave 内各 lane 的时序，结果见第 2 节末尾；它同样只是观察。

**补充更正（2026-10-28，依据 Codex 的 DBDSQR work 审计报告）**：首版说 bidiagonal “分两轮”，并给出晚启动 223–540 ms。
重算后：每个普通 wave 中 10 个 bidiagonal launch 区间都有同时重叠的时刻（L2 77/77、L3 85/85），不是互不重叠的两轮；
普通 wave 的晚启动为 223–347 ms（L2 223–344、L3 237–347），540 ms 只出现在 L2 的第一个 wave。正文已改。

**再次更正（2026-10-28，Codex 核对 DBDSQR 审计复核时指出）**：正文残留的“第一轮/第二轮”“全部同时开始”统一改为早启动、
晚启动和近乎同时启动；第 3 节“需要逐位资格的不进入原型”与本项目的做法矛盾，改为“未通过逐位资格的改动不得进入生产
集成”，隔离原型正是完成逐位资格的地方。

## 结论

```text
完整性    closure、manifest、analysis、protocol、capture report 的 SHA-256 吻合；manifest 275 个条目按当前文件
          重算一致；native e8c39703… 与 103 个生产输入未变
复现      我用原始 events.bin 与 markers.tsv 独立重算，L2/L3 的阶段边界、kernel 并集、memcpy+memset 并集、
          全部活动并集、无 kernel 与无活动区间、三类 kernel 的并集与“仅该类活动”、status D2H 的次数/字节/
          API 累计/重叠/实际拷贝时长、launch 资源，全部与报告一致。status 读取全部来自同一个 host 线程
空洞      无活动区间合计 0.397890/0.444116 s。报告中“单段超过 1 ms”的 0.006806/0.007702 s 包含阶段边界：
          L2 只有结束前 6.806 ms 一段，L3 是结束前 6.654 ms 加内部 1.048 ms 一段。阶段内部没有值得一提的提交空洞
静态上限  按 launch 资源推算：bidiagonal 每 SM 最多 4 个 block（受 shared 限制），全卡 512 个，少于每个 wave 的
          640 个 component；每个 block 只有 thread 0 计算，所以每 SM 至多 4 个计算线程。recover 每 SM 最多 1 个
          block，全卡 128 个，且几乎占满该 SM 的 shared，使同一 SM 上无法再驻留 bidiagonal。这些是上界，不是实测
          占用率。除 L2 的第一个 wave 外，所有 10-lane wave 中都是 8 个 lane 的 bidiagonal 先开始，另 2 个在部分
          block 结束后才开始；10 个 launch 区间有重叠，不是互不重叠的两轮
建议      同意下一项：只做 bidiagonal/DBDSQR 的依赖链与 scratch 生命周期审计，列清每个候选改动是否保持运算顺序，
          recover 作为共存约束。硬件计数器需要管理员权限，是否开放由用户决定
```

## 1. 复现

由 `attempt3/trace/events.bin`（100 字节定长记录，1,000,301 条）、`names.tsv` 和 `markers.tsv` 重算，单位秒：

| 指标 | L2 | L3 |
|---|---:|---:|
| CUPTI Stage B 边界 | 55.341451 | 61.249252 |
| kernel 并集 | 54.802182 | 60.649809 |
| memcpy + memset 并集 | 0.167039 | 0.187613 |
| 全部已记录 device activity 并集 | 54.943562 | 60.805136 |
| 无已记录 activity | 0.397890 | 0.444116 |
| bidiagonal 并集 / 仅该类活动 | 51.156629 / 29.164904 | 56.588267 / 32.417541 |
| recover 并集 / 仅该类活动 | 23.885557 / 2.632593 | 26.400796 / 2.949349 |
| roots 并集 / 仅该类活动 | 4.287537 / 0.151492 | 4.644960 / 0.131563 |
| status D2H：次数 / 字节 | 780 / 2,396,112 | 858 / 2,634,144 |
| status D2H：`cudaMemcpyAsync` 累计 / 与活动重叠 | 54.649306 / 54.646179 | 60.483250 / 60.470386 |
| status D2H：实际拷贝时长合计 | 0.001001 | 0.001104 |

- 每个 component 的 status 为 48 B（49,919 × 48 = 2,396,112）；每类 kernel 在 L2 launch 780 次（78 个 wave × 10 个
  context），L3 为 858 次（85 × 10 + 8）。
- status D2H 的 API 时间几乎全部与 GPU 工作重叠，实际拷贝约 1 ms。这与报告的结论一致：同一个 host 线程先提交一个
  wave 的全部 lane，再逐 lane 用 pageable 拷贝读取 status，每次都等该 lane 的 stream 上此前的工作完成；它不是传输开销，
  也不能与 kernel 时间重复累加。L2 的 780 次、L3 的 858 次调用都来自同一 thread id。

## 2. 静态占用上限

RTX 4090（sm_89）：128 个 SM，每 SM 最多 1,536 个线程（48 个 warp）、102,400 B shared、65,536 个寄存器、24 个
block；每个 block 另占 1,024 B 保留 shared。按 CUPTI 记录的 launch 资源，并按寄存器分配粒度、SM 子分区与 shared
分配粒度计算：

| kernel | 线程/block | 寄存器 | 动态 shared | 每 SM 上限 | 全卡常驻上限 | 实际计算线程 |
|---|---:|---:|---:|---:|---:|---|
| bidiagonal | 32 | 100 | 19,656 B（7 × 351 × 8） | 4（shared 限制；寄存器允许 16） | 512 | 每 block 只有 thread 0 |
| recover | 64 | 66 | 101,088 B（36 × 351 × 8） | 1（shared 限制，剩余约 256 B） | 128 | — |

- 每个 wave 有 640 个 component（10 个 lane × 64 个 block），bidiagonal 的 640 个 block 不能同时常驻。recover 一个
  block 几乎占满一个 SM 的 shared，这个 SM 在此期间无法驻留 bidiagonal。
- bidiagonal 的 shared 是 d、e、q 各 351 个 double 加 4 × 351 的 work。装载后其余 31 个线程返回，每 SM 至多 4 个线程
  在执行 `recorded_dbdsqr_`。单个 component 内的串行依赖和 shared 访问延迟能否被掩盖，只能由硬件计数器测量；这里
  只是审计依据的假设。

**wave 内 lane 时序（观察）**：L2 的 78 个 wave 中有 77 个、L3 的 85 个 10-lane wave 全部是同一形态：8 个 lane 的
bidiagonal 在 wave 开始后 50 ms 内开始，另 2 个 lane 在 223–347 ms（中位约 310 ms）才开始；每个这样的 wave 中，
10 个 launch 区间都有同时重叠的时刻。例外是 L2 的第一个 wave
（首次执行，873.9 ms）：1 个 lane 在 5 ms、8 个在 277 ms、1 个在 540 ms 开始；L3 的尾 wave 只有 8 个 lane，近乎同时
启动。抽样的 L2 第 10/39/60 个 wave 和 L3 第 10/43 个 wave 中，前 8 个 lane 在约 3 ms 开始、361–378 ms 结束，后 2 个在
268–336 ms 开始、620–703 ms 结束，wave 跨度 675–751 ms。各 lane 的 bidiagonal launch 时长相近（每个 wave 最长/中位数
的中位为 1.04），但在 bidiagonal 阶段平均只有 56% 的 lane 同时处于 bidiagonal。这与 512 个 block 的常驻上限相符：
8 个 lane 正好 512 个 block。数据来自 launch 级的开始/结束时刻，没有 block 级记录。

- 按同样的粒度计算，每 SM 放下 5 个 bidiagonal block，动态 shared 需不超过 19,456 B，比现在少 200 B；那时全卡可常驻
  640 个，正好一个 wave。
- 这只是假设，不能推出可节省的时间：现在早启动 lane 的 recover 与晚启动 lane 的 bidiagonal 重叠；若 640 个 block 都能
  早启动，recover（每 SM 1 个、与 bidiagonal 互斥）可能成为新的限制。同 SM 的 block 增多也可能改变每个 block 的时长。

## 3. 对下一项的意见

同意 Codex 的建议：只做 bidiagonal/DBDSQR 的依赖链与 scratch 生命周期审计，recover 作为共存资源约束一并考虑，再决定
是否做原型。审计宜给出：

1. **串行依赖与有效执行线程**：`recorded_dbdsqr_` 在 thread 0 上的依赖链结构，以及哪些步骤可以由其余线程参与。
   静态步数乘估计延迟只能形成假设，不能据此判断瓶颈。
2. **scratch 的同时活跃范围与复用条件**：bidiagonal 的 d/e/q/work，以及 recover 的 36 × 351 缓冲。shared 降到多少时
   每 SM 的上限会变化（例如 ≤19,456 B 时为 5），只作为假设记录。
3. **每个候选改动是否保持运算顺序**：逐项写明。隔离原型正是完成逐位资格的地方；未通过逐位资格的改动不得进入生产集成。

报告已写明本轮不授予 kernel 改写或新的 speedup GO，我同意。

## 4. 硬件计数器

驱动设置 `RmProfilingAdminOnly=1`，当前用户没有权限，非交互 sudo 需要密码，所以占用率与 stall 原因未测。若要补测，
需要用户决定：

- 由用户在空闲时段以 sudo 交互运行 ncu，只针对 captured 输入；
- 或设置 `NVreg_RestrictProfilingToAdminUsers=0` 并重新加载驱动或重启，这会中断 GPU0 上其他用户的任务。

另外，ncu 的 per-kernel replay 会串行化 kernel，测得的是单个 kernel 的占用率，不能当作原并发 Stage B 的性能。

## 证据

`attempt3/trace/{events.bin,names.tsv,markers.tsv,trace-summary.json}`、`attempt3/analysis.json`、`closure.json`、
`manifest.json`、`attempt3/protocol.json`、`attempt3/capture/report.json`。计算为只读一次性脚本；没有修改生产源码、
native 或默认配置，没有 commit 或 push。
