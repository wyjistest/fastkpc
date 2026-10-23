# Exact-primary optimizer：相邻双窗口隔离对照

日期：2026-10-22 开始，2026-10-23 收口。执行
[窗口复核](PHASE10_OPTIMIZER_WINDOW_DIAGNOSTIC_REVIEW_20261022.md)之后限定的 A/B/C
实验，沿用[固定窗口捕获](PHASE10_OPTIMIZER_WINDOW_DIAGNOSTIC_20261022.md)及其生产 native。

## 结论

**当前调度器直接接收 128 个请求：STOP_FOR_CAPTURED_PAIRS。** 两组真实相邻窗口中，
64-worker 合并与 32-worker 合并在六轮主计时里全部慢于顺序双窗口；所有数值、状态、
target 身份与输出次序逐位一致。可以产生真实跨窗口 device overlap，但没有产生完整
调用的收益。

保留现有 64-setup / 64-worker 窗口形状。完整 runtime 滚动集成仍为 NOT_YET_GO；不据
这次实验重开排序、固定 stream 池、pinned I/O、L2 Stage A 或 component overlap。
这些不同机制并未在本轮测试，也不能被本轮一并判定有效或无效。

生产源码、native、默认配置均不变。没有新完整方法耗时；`541.1665s` 仍是此前 S2 的
development 结果。Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false。

## 固定输入与公平边界

生产 native SHA-256：

```text
8e457e811eaa01ae85326b8ff3cc0bcfa56f5b907ad6840a302a8205370d3874
```

原捕获 manifest SHA-256：

```text
b846bcb3b99ed33f1445651ad0398206ac99e24b6998a74a7889a731dfc1212e
```

| 输入 | setup 数 | target 数 | q |
| --- | ---: | ---: | ---: |
| L3 W31 + W32 | 128 | 1,657 | 28 |
| L5 W7 + W8 | 128 | 844 | 46 |

使用实际捕获的 prepared geometry、Y、初始 SP、完整 optimizer control、原 target
cohort 和顺序；原 fixture、逐位比较器和 observer 直接复用且核对 SHA。所有计算仍调用
原 `multi_penalty_gcv_capacity_optimize_multi`，不替换数值实现。

| 档位 | 调用形状 | worker |
| --- | --- | ---: |
| A | 64 请求返回后，再调用后续 64 请求 | 每次 64 |
| B | 原序 128 请求由现有 atomic queue 动态领取 | 64 |
| C | 原序 128 请求由现有 atomic queue 动态领取 | 32 |

三档均在预热前创建同样的 128 个 prepared 并常驻复用。A 第一窗口的 residual token
保留到第二窗口返回，三档统一在完整计时结束后检查并释放输出。这样避免输入驻留量、
输出生命周期和中途诊断下载混入调度对照。这里没有模拟 runtime 的 setup 创建/淘汰、
coefficient arena 登记或 component consumer 生命周期。

A/B/C 各预热一次，然后按 `ABC / BCA / CAB / CBA / BAC / ACB` 交错执行，六次/档，
各档在每个执行位置出现两次。请求构造、诊断 shadow D2H、比较和文件输出均在计时外。
计时包含原 native 上传、optimizer、selected fitted/residual、正常 metadata D2H、
调度及返回包装。event 观测和 Nsight 分开执行，不混入主计时。

仅 GPU0 执行，启动各进程前两卡 compute-app 查询均为空。主计时期间 GPU1 的全部采样
均为0% utilization、425 MiB桌面占用。独立 L5 profile 有一个1%采样（210 MHz），
profile期间显存最高427 MiB；原始记录保留，不把profiler耗时当性能结果。CPU affinity
固定 0–19，CUDA connections=32，BLAS/OMP/MKL 线程数为1。32 worker 表示最多32个
并发 host 调用，不代表独占32个硬件队列；128个 prepared 仍各自持有 stream。

## 主计时结果

单位秒，六次未加 event 观测的完整双窗口调用。

| 输入 | A 中位数（范围） | B 中位数（范围） | C 中位数（范围） |
| --- | ---: | ---: | ---: |
| L3 W31 + W32 | **1.837402**（1.794–1.995） | 2.671594（2.157–5.622） | 4.537548（4.105–5.191） |
| L5 W7 + W8 | **12.409458**（11.464–12.687） | 24.072498（20.548–29.837） | 35.733501（25.603–40.152） |

B 相对 A 分别增加45.40%、93.99%；C 分别增加146.95%、187.95%。每个输入的 B、C
均为0/6轮胜过同轮 A，且两档各自最快的一次也慢于 A 最慢的一次。波动明显，但没有
通过挑选最快值或单次 profile 改写结论。原始值、范围、中位数与逐轮差值保存在 `analysis.json`。

这些是两组 captured pairs 的结果，不外推为整层或完整方法的加速/回退比例。

## 数值与执行身份

主进程两组各24次完整调用（预热、主计时、event观测），profile进程各6次（含预热），
共60次完整128-setup重放、75,030个 target 输出实例全部通过逐位比较。覆盖256个
不同 setup、2,501个不同 captured target state，验证：

- selected/initial log-SP、RSS、EDF、score、condition、gradient/Hessian 等全部导出浮点状态；
- coefficients、fitted、residual 的全部位；
- route/rank/info/status、收敛及迭代、objective/score、step-halving、boundary 等轨迹字段；
- 原 setup 输出下标和逐 target key/order，GPU exact-primary / stable-SVD authority；
- CPU objective/optimizer/solve、fallback、CUDA error、stability replay、terminal confirmation 为零。

每次均释放全部 residual lease，销毁诊断事件。当前实验没有重新执行完整 CI、pMax、
graph、PDAG 或真实数据资格；其输入 authority 来自已经归档并校验的 capture。

## 独立 Nsight 归因

下表每档只有一次观测，只解释执行形状，不能替代上述六次主计时。
overlap 是第一窗口与第二窗口 optimizer kernel 区间并集的交集长度，不是可节省时间。

| 输入/档位 | 完整调用 s | 跨窗口 optimizer overlap s | optimizer 峰值并发 | H2D 在 API 返回后等待 >5 ms |
| --- | ---: | ---: | ---: | ---: |
| L3 A | 1.916 | 0 | 32 | 30 |
| L3 B | 1.965 | 0.807 | 21 | 0 |
| L3 C | 3.020 | 0.014 | 16 | 0 |
| L5 A | 12.001 | 0 | 32 | 64 |
| L5 B | 19.776 | 4.323 | 32 | 35 |
| L5 C | 20.460 | 0.758 | 32 | 0 |

B 的两个输入均有31个第二窗口 optimizer kernel 在第一窗口最后一个 optimizer
kernel 结束前启动；C 均只有5个。现有调度器确实跨窗口领取了工作，但提前领取不等于
充分补位，更不等于净收益。kernel 存在比例、并发数量或完整 grid 总和都不是实测 SM occupancy。

两组 profile 的同 setup optimizer kernel 时长在 B/C 中并未明显变长：相对 A 的中位
比例，L3 为0.9966/0.9948，L5 为0.9954/0.9919。因此不能把当前回退简单解释为
每个 optimizer kernel 都因竞争而翻倍；更明显的是任务链启动/完成与补位的时间分布变化。

结果返回路径出现了重要等待：

| 输入 | 本 setup 最后 kernel 结束 → 结果 D2H 开始，中位数 ms：A / B / C |
| --- | ---: |
| L3 | 357.702 / 465.378 / 501.571 |
| L5 | 3671.873 / 7768.906 / 4193.393 |

L5 每档128个结果 D2H 的实际 device copy 累计仅约0.375–0.386 ms，copy结束到 API
返回的中位数约27–32 µs；长等待在 copy真正开始之前。worker 需要等原 batch 返回，
这些完成路径等待会影响后续请求领取。L5 B 的 D2H API 累计 host 时间约1243.516秒，
但完整调用只有19.776秒，不能把重叠的线程等待相加作为收益池。

这给出了比“32个硬件队列”更具体的观测边界，但尚未证明内部硬件队列映射、驱动依赖
或 pageable transfer机制的唯一因果。L3 B 和两组 C 消除了 >5 ms H2D 等待，仍没有
完整调用收益，也直接说明该单一指标不能作为推进门。六个profile计时范围均无
`cudaMalloc*` / `cudaFree*`，没有中途 prepared 创建/释放混入这次回退。

逐 setup host、H2D、launch、optimizer、最后kernel、D2H与返回时间在 `*-timeline.csv`；
图见 [双窗口时间线](../artifacts/full_cuda_ci/optimizer_double_windows_20261022/double-window-timeline.png)。
灰线包含 launch 返回后的同 stream 前置工作，不能统称硬件排队时间。

## 容量与适用边界

L3/L5 的128个 prepared 创建分别约201.094/207.760 ms，释放约52.261/39.725 ms，
都在 GPU optimizer 不运行时测量。第二窗口的增量空闲显存下降分别为1230/1038 MiB。
主计时进程的采样峰值分别为4169/3511 MiB，属于无完整 component cache/arena 的
隔离进程，不构成完整 S2 runtime 的容量资格。

释放 prepared 后仍有 CUDA context/library 保留内存，不能据 `cudaMemGetInfo` 声称
进程内所有 device bytes 归零。L3/L5 各取每窗口前两个 setup，共四个 setup 的 A/B/C
CUDA memcheck 均为0 errors、0 bytes leaked；这是缩小的资源检查，不是完整128-setup
memcheck。所有实验结束后 compute-app 查询为空，GPU0/GPU1恢复50/425 MiB原桌面占用。
资源检查结果由同目录 `closure.json` 绑定。

## 路线决策与复现

```text
bounded A/B/C diagnostic                       COMPLETE
direct 128-request concatenation              STOP_FOR_CAPTURED_PAIRS
32-worker combined configuration              STOP_FOR_CAPTURED_PAIRS
full runtime rolling integration               NOT_YET_GO
production source / native / defaults          unchanged
```

若将来重开，需要一个针对已观测完成路径等待的新机制及独立诊断，不能仅把128请求改名
为 rolling、把32 worker称为独占队列，或套用两窗口总 kernel duration 的理想列表调度
公式预测整层收益。本轮不进入 runtime 集成，也没有执行 D 时长排序。

工具：[optimizer_double_windows_20261022](../tools/research/optimizer_double_windows_20261022/README.md)。
证据：`fastkpc/artifacts/full_cuda_ci/optimizer_double_windows_20261022/`，包括同 native
build/input SHA、各次原始计时、逐setup数据、GPU telemetry、Nsight rep/sqlite、资源检查、
`analysis.json`、`closure.json` 与 `manifest.json`。它们是 source-closure development
diagnostic，不是 clean-producer 性能中位数或 promotion evidence。

完成 audit：`PASS_DIAGNOSTIC`。最终 manifest SHA-256：

```text
969e9cd611e1b80a3b8807490bcff3d82e6a7b28ceaea11f5a8298e307b4fc3c
```
