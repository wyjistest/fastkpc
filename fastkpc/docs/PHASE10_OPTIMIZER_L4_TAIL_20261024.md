# L4 原窗口尾部归因与有界补位机会

历史证据说明：下文保留本次归因时的结论与身份。后续隔离原型、cache/arena 共存及
默认关闭 runtime 已另行通过资格，见 [L4 runtime 报告](PHASE10_OPTIMIZER_L4_SLOT_RUNTIME_20261024.md)；
新的同 native 完整方法 OFF/ON 中位数为 463.288/407.531 秒。

日期：2026-10-24。承接 [L4 原窗口 32-worker runtime](PHASE10_OPTIMIZER_L4_RUNTIME_20261024.md)。
本轮仅运行外部诊断，生产源码、native、默认配置和原完整方法结果不变。

结论：**`GO_L4_BOUNDED_REFILL_EXPERIMENT_ONLY`**。两份完整时间线都显示窗口尾部
存在显著机会；保持观测到的逐 setup 服务时间不变时，32-slot 原顺序连续补位的模型
分别减少 **52.578 / 53.008 秒**。这是进入隔离原型的筛选结果，尚未测得实际 refill
或完整方法加速。L4 runtime 集成仍为 `NOT_YET_GO`，既有全层常驻容量 STOP 保持。

## 输入、生产身份与预声明门

生产源码来自 runtime commit `33e2d3e`，当前仓库 HEAD 为 `9bacecf`。全部 98 个
生产输入与 [source attestation](PHASE10_SOURCE_ATTESTATION_20261024.json) 及 Git blob
一致，使用既有、已资格化的 native：

```text
2c8dd8af4ea021d8a784588717476f091a0132b9415067e4a3e115ca65bcd0ee
```

没有重新编译生产 CUDA；仅编译链接实际 native 的外部 NVTX/audit companion。
原始 capture manifest 为 `d5c77e9c…d386ceb`，全部 34 个 fixture 的 SHA 逐个核对。
每次保留 34 个原窗口（33×64+42）、2,154 setups、17,767 targets；形状为
`n=351, q=37, penalty_count=4`，覆盖全部 14 种实际 target cohort 大小，最大为18。
没有排序、拆分、补齐或只选部分样本。

执行前冻结 [README 协议](../tools/research/optimizer_l4_tail_20261024/README.md)
及 `protocol.json`：两个独立 GPU1 进程，每个完整预热一次，再完整 profile 一次。
固定 CPU affinity 0–19、CUDA connections=32、status-wait=1 和单线程 BLAS/OMP/MKL。
每个窗口只创建自己的 prepared；以32 workers调用生产 optimizer，完整校验后释放
leases 和对象。两卡进程采样未观察到外部计算任务，但这不是新 isolated canonical
性能 campaign；时钟并未锁频，且 shadow 比较会打断窗口之间的连续执行。

主模型使用完整 host setup 范围，包含已观测上传、等待和 metadata 返回时间。
预声明要求两次模型对当前分段 native 时间的相对误差均不超过5%，模型屏障机会
均至少达到实测 native 的10%，且全部数值、覆盖和资源门通过，才支持下一独立实验。
较小机会为 STOP，模型不成立或两次结论冲突为 CONDITIONAL。门没有因结果修改。

## 实测窗口形状

以下来自 NVTX 范围和 CUDA API correlation；每份 trace 的12,924个 kernels及
4,308次 native copies 均归属到唯一 setup，没有遗漏或重复归因。

| 项目（秒） | Trace 1 | Trace 2 |
| --- | ---: | ---: |
| 未 profile 的完整预热 native 调用总和 | 87.189 | 86.088 |
| profile 中34个 native 窗口总和 | **84.554** | **84.925** |
| native 内没有任何 kernel 的时间 | 0.612 | 0.655 |
| 每窗口最后一个 setup 开始 device 工作至最后 kernel 结束，合计 | 63.586 | 63.765 |
| 上述尾部中仅1–8个 kernel同时执行的时间 | **53.090** | **53.353** |
| prepared 创建 host 时间 | 3.466 | 3.426 |
| prepared 销毁 host 时间 | 0.840 | 0.859 |
| shadow 数值校验（计时外） | 0.356 | 0.355 |

native 范围端点与 C++ wall clock 的完整累计差各小于1毫秒。建销、packaging、shadow
及 lease release 单独报告，未混成 optimizer device time。预热与 profile 时间不是
平衡性能对照，也不据此声称 profiler 加速了执行。

约63.6秒尾部内仍在做有效计算，53秒低并发也不是 GPU 空闲或可直接删除的时间。
实测空洞不足1秒，说明仅减少空洞不是主要机会。值得测试的是：当前窗口长任务
收尾时，能否开始后续窗口的独立 setup，且不会被资源竞争吞掉收益。kernel 个数
不是实际 resident blocks、SM occupancy 或 FP64 利用率。

![两次实测原窗口尾部](../artifacts/full_cuda_ci/optimizer_l4_tail_20261024/window-tails.png)

## Host 与传输归因

每份 trace 的 native H2D 都是2,154次、49,889,736 bytes，device copy总计约
**7.13 ms**；native D2H是2,154次、68,367,416 bytes，约 **7.7 ms**，不含计时外
shadow。创建 prepared 时另外上传 geometry 共587,033,928 bytes，device copy约
**71.7 ms**。这些短 copy 时间不代表对应 API 的 host 边界也一样短。

`cudaEventSynchronize` 的 worker 时间累计分别为846.648/844.461秒，多个 workers
同时等待，其时间并集为83.947/84.276秒；不能把累计等待相加当作节省量。
kernel结束到 host返回的中位间隔约0.411/0.409毫秒。

另有一项必须保留的观测：setup NVTX 进入到首个**被 trace 记录的** runtime API，
worker 时间累计140.026/141.222秒，并集约8.976/9.007秒；中位数仅26.7/35.8微秒，
P95约249/250毫秒。它包含未被记录的 API、库工作或宿主调度的可能性，不能直接
归因于某个锁或 `cudaSetDevice`。这些间隔已经保留在主 host-span 模型中，其并集
还与其他 setup 的 GPU 工作重叠，不能另加为收益。详见 `submission-gaps.json`。

## 模型筛选结果与限制

模型按原 setup 顺序、最多32个槽位领取工作；对照保留34个窗口屏障，候选只在
全层结束时等待。每个 setup 的服务时间固定为该 trace 的实测完整 host 范围。

| 主 host-span 模型（秒） | Trace 1 | Trace 2 |
| --- | ---: | ---: |
| 当前窗口模型 | 84.134 | 84.474 |
| 当前窗口实测 | 84.554 | 84.925 |
| 拟合误差 | **0.497%** | **0.531%** |
| 原顺序32-slot连续补位模型 | **31.557** | **31.466** |
| 模型中的窗口屏障差值 | **52.578** | **53.008** |
| 差值 / 实测 native | 62.18% | 62.42% |

用第一份 trace 的服务时间解释第二份窗口总量，误差0.931%；反向为0.095%。
这支持模型对当前**有屏障**形状的解释，仍不是对不存在的 refill 实现的验证。
仅使用首末 kernel span 的辅助模型，拟合误差4.72%/4.95%，连续补位为27.244/
27.177秒；它未完整保留 host 成本，不用它宣称更大的生产收益。

单个 setup 的服务时间并非稳定常数：第二次 / 第一次 host span 的中位比约1.000，
P95为1.316，329个超过1.10、320个低于0.90；device span 的P95为1.117，144个超过
1.10。持续补位可能让更多 blocks 竞争 FP64、寄存器和内存，不能假设其服务时间仍
与当前尾部稀疏执行时相同。

模型差值除以2,154个 setups，约为24.4–24.6 ms/setup。这只是“新增成本全部串行
进入关键路径时”的算术预算，不是实测槽位开销。若假设每个 setup 新增5 ms串行
成本，累计为10.77秒，模型仍有机会；但新 geometry upload、输出 detach、event/
lease和资源竞争均尚未实测。因此本轮不提供保守净收益保证或新的 outer时间预测。

## 数值、资源与容量边界

两个进程各包含完整预热/profile，共四次全层重放，比较 **71,068个 target 输出
实例**（唯一 targets仍为17,767）。selected/initial SP、完整 optimizer状态和轨迹、
route/rank/info、coefficients/fitted/residual全部逐位一致，GPU exact-primary及
CPU authority/fallback门通过。没有重新运行240,498个CI、pMax、最终PDAG或真实数据。

受审计 native调用内 allocation/free、stream/event和BLAS handle建销为零；每个
窗口及最终 application allocations/bytes/streams/events/handles与输出leases关闭。
application buffer峰值600,340,480 bytes，创建后最小可用显存21,960,130,560 bytes，
两次采样的GPU1整程峰值均3,174 MiB。没有 production component cache/arena共存，
这不是新槽位池的容量资格。四项手算模型/区间并集回归通过；本轮未新跑memcheck。

## 下一项的明确范围

允许进入 **L4 32-slot有界补位隔离原型**，从原始2–18-target cohorts出发，不继续
尝试2,154个完整prepared同时常驻。不能直接打开L5的capacity-eight路径。

原型必须测量并比较：当前原窗口32-worker完整生命周期，对比同顺序、同cohorts的
有界连续补位完整生命周期；把geometry重绑定、上传、最终系数detach、lease释放及
pool建销全部纳入。输出必须保存后才允许slot复用，并验证stale token、失败清理与
完整数值身份。先验证全部14种capacity，再做重复全层对照；只有实际净收益成立，
才进一步做生产cache/arena共存、至少2 GiB余量及完整canonical资格。

现有完整方法开发中位数保持 **461.663秒**。生产默认、Phase 10 ACTIVE、candidate
NOT_FROZEN、holdout SEALED、recommended=false均不改变。本次诊断未commit/push。

## 证据

工具：`fastkpc/tools/research/optimizer_l4_tail_20261024/`。`build.py`冻结执行工具、
生产源码及capture身份；`run.py`保存两次进程、GPU采样和Nsight输出；`analyze.py`
保存完整setup/window/API表及模型；`submission_gaps.py`单独分析未记录区间；
`plot.py`导出实测图，`archive.py`绑定全部产物。

artifact：`fastkpc/artifacts/full_cuda_ci/optimizer_l4_tail_20261024/`。
`protocol.json`、`analysis.json`、`closure.json`、两份原始trace和全部校验日志均保留。

```text
manifest SHA256
6c365f8d7a3be41dd6b8fc7a8549f822459ef906c63ae6c319ef1357f3622b79
```
