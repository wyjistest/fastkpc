# L5 整层 optimizer 补位诊断

日期：2026-10-23。承接 [status-wait 复核及勘误](PHASE10_OPTIMIZER_STATUS_WAIT_REVIEW_20261023.md)。
本项只做真实整层捕获重放，不实现 prepared 槽位池或 runtime rolling。

结论：`GO_L5_RESIDENT_DIAGNOSTIC`。同 native、等待开启，32-worker 连续补位的
六轮 native-call 中位数为 `31.787s`，原 64/64 常驻窗口为 `106.154s`，
完整输出逐位一致，并由独立 trace 确认跨原窗口的实际 kernel 重叠。
这支持继续做有界槽位池隔离原型；完整 runtime 补位仍为 `NOT_YET_GO`。

## 输入、身份与范围

使用已资格化 status-wait 的 source-closure native：

```text
ba8f806bfc1ebbf8ec90b28dcb095a57d163c986836ac33ddaf8fb0a3ed9298f
```

所有配置均开启 prepared-owned status wait；GPU1、CPU affinity 0–19、
`CUDA_DEVICE_MAX_CONNECTIONS=32`，BLAS/OMP/MKL 单线程。GPU0 的外部任务被允许，
两卡 telemetry 和 compute PID 均保存，因此性能属于共享主机上的 development 证据。

外部 capture companion 在生产 S2 canonical 351×48 前缀中保存 L5 的全部输入和输出。
15 个原窗口为 `14×64 + 59` 个 setups，共 955 setups、6,816 targets；形状为
`n=351, q=46, penalty_count=5`，各 setup 保留实际 target capacity、geometry、
Y、initial SP、optimizer control、target identity/order 及完整结果。
native 和生产源码不因本项修改；capture 的 shadow D2H 只用于诊断，CPU 不参与数值生产。

捕获前缀通过：239,574 CI/pMax 位值、task/deletion、sepsets、adjacency、
计数、RNG、原窗口身份、CPU authority/fallback 为零和资源归零均与已资格化 S2 一致。
这不构成新的完整 canonical 性能或最终 PDAG campaign。

## 计数勘误与实际容量

`residual_batches` 计数后续最多 16-target 的 materialization cohort，不是 optimizer
setup 数。四份原窗口 receipt 的真实账本为：

| 层 | optimizer setups | 原窗口 | residual batches |
|---|---:|---:|---:|
| L3 | 4,065 | 64 | 5,573 |
| L4 | 2,154 | 34 | 2,156 |
| L5 | 955 | 15 | 955 |
| L6 | 245 | 4 | 245 |
| L7 | 44 | 1 | 44 |
| L8 | 9 | 1 | 9 |

容量检查实际创建全部 955 个 prepared，并在每次创建后要求仍有至少 2 GiB 可用。
结果 PASS：application-visible buffer 共 `6,002,917,360 bytes`，955 次分配、
955 streams、1,910 events、955 cuBLAS handles。CUDA 可用显存从
`24,426,381,312` 降到 `8,938,913,792 bytes`；完整增量约 14.42 GiB，
不能用约 5.59 GiB 的 buffer 账本替代实际容量成本。
创建/释放分别为 `1.520410/0.316364s`，均在没有 optimizer 执行时测量。
六轮主 campaign 每次创建后的最小可用显存为 `7,733,051,392 bytes`，仍高于 2 GiB 门。
该 campaign 的 GPU1 telemetry 采样峰值为 `16,742 MiB`；采样峰值和单次容量检查
的 free-memory 差值是不同口径，不能相互替代。
释放后 application-visible 资源全部归零，CUDA context/library 仍保留部分显存，
不能声称进程所有 device bytes 都归零。此重放没有 11 GiB component cache 或生产 arena，
不构成 runtime 容量资格。

实际 target capacity 分布：2/3/4/8 targets 分别有 65/42/56/792 个 setups，
单个 prepared 的 application-visible buffer 为 `2,031,520–7,000,312 bytes`。

## 三档对照与计时边界

| 档 | 调度 | prepared 生命周期 |
|---|---|---|
| P / A-prod | 原 64/64 窗口 | 每窗口创建、完成、释放 |
| A / A-resident | 原 64/64 窗口 | 全层常驻 |
| R / R-resident | 32 worker，原顺序连续领取 | 同一轮与 A 共用全层常驻对象 |

每轮 A/R 使用同一批 955 个对象；P 执行时没有额外 resident pool。所有数值运算、
target cohort、optimizer control 和结果索引保持原样；R 将并发 worker 从 64 改为 32，
并允许跨原窗口连续领取。这测量的是两项组合的收益，不分解各自的独立贡献。
本次没有按时长排序，也没有修改 worker 领取实现。

A/R 在全部 native 调用完成后才下载完整 shadow，作为主要调度对照。现有公开 consumer
view 不暴露 fitted 指针，P 为验证完整 fitted 位值，必须在窗口之间下载 shadow 后
再释放 prepared。P 同时报告 raw total、diagnostic time 和扣除诊断的分段工作时间；
这些中间停顿可能影响调度和温度，所以 P 只作为生命周期参照，不能当作无诊断开销的
端到端性能基线。所有配置的调用、packaging、建销、token release 和校验时间单列。

每档先预热一次，然后执行六轮固定顺序：
`PAR / RAP / ARP / PRA / PAR / RAP`。A/R 先后顺序各三轮，P 首尾各三轮。
每次均对 captured 完整 optimizer 状态、coefficients/fitted/residual、route/rank/info、
target identity/order 和迭代轨迹逐位比较。optimizer 调用期间不允许出现可拦截到的
CUDA 分配/释放、stream/event 创建/销毁或 cuBLAS handle 创建/销毁。

预先声明的诊断门：所有硬门及容量通过，R 的 native-call 中位数 `<=0.90× A`，
且计入 resident 建销后的 R 时间仍小于 P 的分段工作参照，才建议继续评估有界槽位池。
这不是 runtime integration GO。单独 Nsight profile 用真实 kernel interval 判断跨原窗口
重叠，不把 profile 时间混入主计时。

## 验证与结果

首尾窗口各两个真实 setups 的三档 memcheck 已通过：共 30 个 target 输出实例逐位一致，
CUDA 0 errors、0 bytes leaked。它是缩小范围的 sanitizer 检查；全层容量、输出和
生命周期由普通重放另外验证。

主计时全部完成：含预热共 21 次整层重放，143,136 个 target 输出实例逐位一致；
每轮输出 leases、application-visible allocations/streams/events/cuBLAS handles 关闭，
optimizer 调用期间的资源创建/销毁计数为零。所有 955 个 setups、6,816 个 targets
均参与每次重放，没有因慢样本或容量问题删减输入。

六次正式计时（单位秒；profile 不在内）：

| 档 | native-call 中位数 | native-call 范围 | 分段工作中位数 | 计入一次完整建销 |
|---|---:|---:|---:|---:|
| P / A-prod | 106.124 | 105.261–111.142 | 107.993 | 107.993 |
| A / A-resident | 106.154 | 101.942–110.304 | 106.161 | 107.953 |
| R / R-resident | 31.787 | 31.631–31.822 | 31.791 | 33.580 |

R/A 的 native-call 中位数比为 `0.299438`，约 `3.3396×` 加速、下降 `70.06%`；
六轮 R 均胜同轮 A。A/R 每轮共用的 pool 创建/释放中位数为 `1.476592/0.313930s`，
最后一列对每档都计入一次完整的实测建销成本，作为保守的单次使用估计。
P 的创建/释放中位数为 `1.487826/0.356162s`，完整 shadow 校验中位数 `0.115264s`
另列；P 的窗口间诊断停顿限制仍然适用。

### 独立 GPU trace

A/R 使用单独进程和 profile-only companion，在采集前创建全部 955 个 prepared，
每档预热一次、采集一次，采集结束后校验完整输出并释放资源。它们不替换主计时 binary，
也不计入上表中位数。通过 setup NVTX 范围及 CUDA API correlation 关联所有 kernel：

| 项目 | A / A-resident | R / R-resident |
|---|---:|---:|
| 覆盖 setups | 955 | 955 |
| 已关联 kernel 次数 | 5,730 | 5,730 |
| 未归因 kernel 次数 | 0 | 0 |
| profile 范围时间 s | 100.699 | 31.590 |
| 跨原窗口 kernel 重叠时间 s | 0 | 27.042 |
| 同时活跃的原窗口峰值 | 1 | 6 |
| 前窗口未全完成即启动的后续窗口 | 0 / 14 | 14 / 14 |

重叠时间为“至少两个不同原窗口各有 kernel 正在执行”的时间并集，
不累加 kernel 对或 setup 对，不能把 `27.042s` 当成可直接兑现的节省。
两档同时执行的 kernel 峰值均为 32；这不证明 stream 到硬件队列的映射，
也不把 worker 数变化和消除窗口屏障各自的作用拆开。

最初合并 P/A/R 的 Nsight 采集在 P 之后以 139 退出，kernel log 将 fault 定位在
`libcupti.so.12.5`；确切触发原因未证实。原始 trace、日志、process receipt 和
`PROFILE_V1_FAILURE.md` 全部保留，并标为 `NOT_COMPLETED_NOT_PASS`。
后续分开的 A/R 采集均完成、输出逐位一致且资源关闭，不把失败尝试计入通过次数。

有效完整运行合计 25 次（主 campaign 21 次、独立 A/R profile 各 2 次），
加上缩小范围 memcheck，累计比较 `170,430` 个 target 输出实例，全部逐位一致。
所有已完成 campaign 的输出 leases 和 application-visible 资源关闭，
native optimizer 调用期间受审计的资源创建/销毁计数为零。

### 决策和下一步边界

预声明的容量、正确性、资源及性能门均通过，六轮 R 都胜同轮 A；
计入一次完整常驻建销的 `33.580s` 也小于 P 的 `107.993s` 分段参照。
因此本轮结论为 `GO_L5_RESIDENT_DIAGNOSTIC`，允许继续评估有界 prepared 槽位池。

下一项应只在隔离原型中验证槽位重绑定、原 target cohort/顺序、完整输出位值、
结果保存与 token/event 生命周期，以及补位期间无资源建销；再测几何上传与输出转移的
净成本和显存峰值。本轮没有实现该池，未证明它能保持全层常驻对照的收益。
未来 runtime 集成仍需同时容纳 component cache/arena、验证其他层及真实输入，
并通过完整 canonical 的数值和 outer wall-time 门。不能把 L5 的 `3.3396×`
称为完整方法加速，也不能由此更新约 541 秒的历史 S2 完整方法结果。

证据是 GPU1、共享主机、未提交 source-closure 的 captured-layer development 结果，
不是 isolated、clean-producer、promotion 或最终 PDAG 证据。

## 复现

工具：[optimizer_layer_refill_20261023](../tools/research/optimizer_layer_refill_20261023/README.md)。
artifact：`fastkpc/artifacts/full_cuda_ci/optimizer_layer_refill_20261023/`。

机器归档为 `closure.json` 和 `manifest.json`，绑定全部 fixture、primary/profile companion、
native/source SHA、原始计时、两卡 telemetry、有效及失败 trace 和上游 evidence 索引。

manifest SHA-256：

```text
dfc6c04ba077235466ba052513efc346b5b6c2a1ca2bfaf81439c0b94050ea68
```

Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false；
runtime rolling NOT_YET_GO。
