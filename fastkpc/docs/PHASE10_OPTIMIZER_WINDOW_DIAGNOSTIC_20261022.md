# Exact-primary optimizer：固定窗口并发诊断

日期：2026-10-22。执行 [L2 runtime 后续评审](PHASE10_L2_PREFILL_RUNTIME_REVIEW_20261022.md)
限定的 16/32/64 worker 对照；只增加外部诊断工具，生产源码、native 和默认配置不变。

## 结论

**诊断完成；降低 worker 数为 STOP_FOR_CAPTURED_WINDOWS。** 四个固定窗口的三次主计时
中位数均以 64 worker 最快，16/32 均未显示收益。全部捕获和重放逐位门通过。

跨窗口滚动尚未获得 runtime 集成 GO，但有了更具体的下一项实验依据：L5 示例窗口
最后四分之一时间几乎只剩一个 optimizer kernel，其完整 grid 仅有四个 block，后续
窗口还没有提交。可以在同一 native 上比较“两个相邻窗口顺序运行”与“128 个相同 setup
由 64 worker 动态领取”的有界隔离重放。该实验尚未执行；不能把它的潜力写成方法级收益。

## 输入、执行与身份

生产 native SHA-256：

```text
8e457e811eaa01ae85326b8ff3cc0bcfa56f5b907ad6840a302a8205370d3874
```

在运行前固定选取 canonical 351×48、numCol=35、S2 路线的 L3 W31/W32、L5 W7/W8。
四个窗口各有 64 个 setup，共 256 个 setup、2,501 个 target state；L3 的 q=28，L5
的 q=46。捕获的是实际 prepared geometry、Y、target key/order、初始 log-SP、完整
optimizer control 与生产输出，不从只包含已消费 CI 的子集重新拼组。

外部 interposer 调用原生产实现，再读取诊断 shadow。捕获运行截至 L5 的 239,574 个
CI、pMax、任务/删除字段、图、计数与 RNG 均与归档 S2 前缀逐位一致；native 资源计数
闭合，CPU numerical authority / unknown fallback 为零。捕获有额外 D2H 和文件输出，
其耗时不能当作完整方法性能，亦不声称捕获过程没有大块 D2H。

重放直接调用原 `multi_penalty_gcv_capacity_optimize_multi`，只改变 worker 参数。
每个窗口一次创建 64 个 prepared，重复使用、每轮释放 residual lease；所有请求的
cohort、Y、几何与控制位不变。设置 GPU0、CUDA connections=32、BLAS/OMP/MKL 线程为1；
GPU1 无计算任务，CPU affinity 固定为同一组20个逻辑 CPU。

每窗口先做一次 64-worker 预热，主计时顺序为：

```text
64 → 32 → 16 → 32 → 16 → 64 → 16 → 64 → 32
```

每档三次。随后另做带 host/event 观测的 64/32/16 对照；另进程采集 L3 W31、L5 W7 的
Nsight 轨迹。观测与 profiler 耗时不混入主计时。主边界包含原 optimizer 调度、上传、
GPU 求解、selected fitted/residual 和正常 metadata 返回；诊断 shadow、比较与序列化
都在边界外。prepared 创建/释放单独报告，没有实现新的池化或异步分配。

## 正确性

所有 52 次完整窗口调用（主计时、预热、event 观测）均与捕获生产输出逐位一致。
额外的8次完整窗口 profiler 进程调用（含预热）也一致。比较包含：

- selected/initial log-SP、RSS、EDF、score、condition、gradient/Hessian、系数等全部导出的浮点状态；
- device shadow 的 coefficients、fitted、residual；
- effective rank、aggregate rank、solver info、route、optimizer status；
- iteration、objective/score call、step-halving、Newton/boundary 等轨迹计数。

每次还要求 GPU exact-primary、stable-SVD、CPU authority/fallback 为零，并确认输出槽
已释放。L3/L5 各取前两个真实 setup 的 CUDA memcheck：0 errors、0 bytes leaked；
这是缩小的资源检查，不是64-setup性能测试或完整方法资格。观察事件均显式销毁。

## 固定窗口性能

单位秒，三次未加 event 观测的窗口 wall-time 中位数。

| 窗口 | targets | 64 worker | 32 worker | 16 worker |
|---|---:|---:|---:|---:|
| L3 W31 | 903 | **0.709** | 1.346 | 1.617 |
| L3 W32 | 754 | **1.254** | 1.386 | 2.672 |
| L5 W7 | 337 | **6.301** | 10.284 | 18.763 |
| L5 W8 | 507 | **5.242** | 17.297 | 22.162 |

重复波动仍明显。例如 L5 W8 的 32-worker 范围为 7.460–19.783 秒，64-worker 为
5.087–6.121 秒。原始三次值、范围和中位数都保留在 `analysis.json`；不报告置信区间，
不把这些窗口的比例外推到完整 optimizer 或 outer wall-time。

64 个 prepared 的一次创建为 0.108–0.112 秒、释放为 0.018–0.027 秒。这是当前
隔离环境中的成本；运行中穿插创建/释放对其他 stream 的影响尚未测量。进程内
`cudaMemGetInfo` 在释放 prepared 后没有回到初始值，因此不据此声称所有 CUDA context
占用归零；memcheck 无泄漏，所有进程结束后 GPU0 回到原有空闲占用。

## Host 与 device 归因

以下只来自一次各档 Nsight 观测，不能替代主计时中位数。

| 64-worker 观测项 | L3 W31 | L5 W7 |
|---|---:|---:|
| 完整窗口 | 0.553 s | 7.106 s |
| 无任何 kernel 执行的时间 | 35.372 ms | 34.167 ms |
| 存在 kernel 的时间比例 | 93.60% | 99.52% |
| 最后四分之一无 kernel 时间 | 3.687 ms | 2.382 ms |
| 最后四分之一平均同时执行的 optimizer kernel | 2.143 | 0.999 |
| 最后四分之一 active kernel 完整 grid 合计的时间均值 | 35.639 blocks | 3.995 blocks |

“存在 kernel”不是 SM 利用率。完整 grid 的和也只是在途 blocks 的上界，不能当作实际
resident blocks；kernel 内部分 targets 可能已经完成。当前 GPU 有128个 SM，L5 尾部
的一个四-block kernel 明确显示少量长任务限制窗口结束，不能称为整卡已饱和。

64-worker 观测中，两窗口最多同时有32个 optimizer kernel；这不单独证明硬件队列
映射或队头阻塞因果。L5 的 optimizer enqueue-to-start 中位数约146 ms、最大约2.870 s，
确有 device 排队，但不能把这些相互重叠的等待相加作为节省量。

Host 等待主要记录在 pageable 结果 D2H 的 `cudaMemcpyAsync` 内。L5 W7 的64个 D2H
API 累计 host 时间为186.319秒，而窗口只有7.106秒，真正的 device copy 累计约0.161 ms；
64次显式 stream synchronize 累计约0.727 ms。该现象说明只统计显式 synchronize 会
漏计等待，也说明这里不能把 host API 时间误当传输成本或可消除时间。

轨迹图：[optimizer timeline](../artifacts/full_cuda_ci/optimizer_windows_20261022/optimizer-timeline.png)。
该图显示原 setup 顺序、host launch、enqueue-to-start delay 和实际 optimizer kernel 区间。

## 下一步边界

保留当前64-worker设置。下一项仅建议有界双窗口隔离对照：原来的两个64-setup调用，
与原调度器接收128个相同setup、最多64个worker的队列比较。两档使用同样驻留的prepared，
保留原target cohort与输出次序，重新核对逐位状态、重复耗时和显存增量；先回答去除
窗口屏障本身是否有收益。即使通过，也不能跳过后续真实生命周期、登记所有权和完整
runtime资格。

本轮没有执行跨窗口滚动、L2 Stage A 并发、Stage B overlap、pinned I/O重开或生产参数
修改。没有新完整 outer、真实数据或PDAG资格；541.1665秒仍是此前S2 development结果。
Phase10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false。

工具：[optimizer_windows_20261022](../tools/research/optimizer_windows_20261022/README.md)。
证据：`fastkpc/artifacts/full_cuda_ci/optimizer_windows_20261022/` 的 capture/replay回执、
逐setup事件、Nsight rep/sqlite、memcheck、`analysis.json`、`closure.json` 和 `manifest.json`。

manifest SHA-256：

```text
b846bcb3b99ed33f1445651ad0398206ac99e24b6998a74a7889a731dfc1212e
```
