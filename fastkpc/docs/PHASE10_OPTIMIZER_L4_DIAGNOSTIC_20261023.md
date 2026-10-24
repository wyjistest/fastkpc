# L4 captured-layer 调度与容量诊断

日期：2026-10-23。承接 [L5 opt-in runtime 资格](PHASE10_OPTIMIZER_SLOT_RUNTIME_20261023.md)。
本轮只诊断 L4，生产源码、native、默认配置及 L5 实现不变。
预声明协议和复现入口见
[optimizer_l4_diagnostic_20261023](../tools/research/optimizer_l4_diagnostic_20261023/README.md)。

最终结果：整层 prepared 常驻为 `STOP_FULL_LAYER_CAPACITY`；原窗口内 worker 数调整
为 `GO_L4_WINDOW_WORKER_DIAGNOSTIC`。64/32 workers 的六次 native-call 中位数为
**92.044950 / 80.695489 秒**，下降 **12.33%**，六组配对全部改善。
这是下一项独立 runtime 资格的依据，当前没有集成 L4 调度改动。

## 捕获身份

使用上一项已通过完整构建和六次 canonical 资格的原始 native：

```text
b0d3523285bc8098286d00350c5d4204edec31aab13f6336090fab2d203cdf58
```

external capture/replay companion 直接调用该生产实现，未复制数值 kernel。
98 文件 source closure 与 L5 runtime manifest 保持一致。GPU1、CPU affinity 0–19、
CUDA connections=32、status-wait=1、BLAS/OMP/MKL 各一线程；GPU0 的外部训练保留。
证据属于 shared-host、uncommitted source-closure development，不能称为 isolated、
clean producer 或 promotion。

canonical S2 前缀通过 L4 的 234,152 个 CI/pMax 位值、task/deletion、skeleton、
sepsets、n.edgetests、RNG 和窗口身份与 L5 runtime ON1 的对应前缀完全一致。
CPU 数值 authority/fallback 为零，暴露的资源计数归零。额外 shadow D2H 只用于捕获
和离线比对；捕获总时间不作为完整方法性能结果，也没有新增最终 PDAG 资格。

全部输入为 2,154 setups、17,767 targets、34 个原窗口（33×64+42），
`n=351, q=37, penalty_count=4`，penalty root rank 总和均为 32。
每个 setup 的 geometry、Y、initial SP、control、target 顺序、完整 optimizer 状态与
coefficients/fitted/residual 位值均已保存。

| 原始 target capacity | setup 数 |
| --- | ---: |
| 2 | 64 |
| 3 | 112 |
| 4 | 170 |
| 5 | 137 |
| 6 | 129 |
| 7 | 210 |
| 8 | 326 |
| 10 | 2 |
| 11 | 999 |
| 12 | 1 |
| 14 | 1 |
| 16 | 1 |
| 17 | 1 |
| 18 | 1 |

容量超过 8 的 1,006 个 setups 覆盖 11,086 targets；不能套用 L5 的 capacity-eight
槽位，也不能拆成不同 target cohorts 后继续声称原数值合同已资格化。

## 整层常驻容量 STOP

先逐个创建/销毁全部 2,154 个真实 prepared，得到 application-visible buffers 总和
15,297,769,040 bytes（14.247 GiB），最大单 cohort 15,176,176 bytes。
随后按原顺序同时保留 prepared；创建第 1,298 个时可用显存降至
2,142,044,160 bytes，低于预声明的 2,147,483,648-byte 余量门，立即停止继续创建。
该次没有运行 optimizer，也没有 CUDA OOM。

| 实测项 | 值 |
| --- | ---: |
| 全层所需 setups | 2,154 |
| 触门时已创建 | 1,298 |
| 常驻前 free bytes | 24,424,284,160 |
| 触门时 free bytes | 2,142,044,160 |
| 已创建前缀的物理显存增量 | 20.752 GiB |
| 已创建前缀的 application bytes | 9,655,282,944 |
| streams / events / BLAS handles | 1,298 / 2,596 / 1,298 |
| 创建 / 释放 | 1.999663 / 0.445646 秒 |

这是不含生产 component cache/arena 的独立进程；完整 prepared 成本不能用 buffer
账本代替。释放后 application allocations/bytes、streams、events、BLAS handles
全部归零；driver/library 保留部分内存，因此不声称整个进程 device bytes 归零。

本轮关闭全层 resident A/R 实验，没有把前 1,298 个 setups 当成全层性能样本。
按事先声明的分支，后续只比较全部 34 个原窗口中的 A=64 workers、W=32 workers。
每个窗口的一组 prepared 被 A/W 共用，完成两档 shadow 比较和 lease release 后释放，
再进入下一个原窗口。这一分支只改变 worker 数，不产生跨原窗口补位资格。

## 重放门和计时口径

A 的实际 worker 数为 `min(64, setups)`，因此最后一个原窗口为 42 workers；
W 的全部完整窗口均为 32 workers。两档各预热一次，随后六组按 A/W、W/A 交替。
相同窗口的两档共用原始 prepared，完整比较后释放输出 lease；窗口之间销毁对象。
native API call 总和、packaging、完整 shadow 比较、token release 以及建销分别记录。
每档生命周期估计都计入一次完整的共享建销成本。

窗口间 shadow 比较会打断连续执行，因此 native-call 总和是 captured diagnostic，
不是未经插桩的 outer wall time。该分支既未去除窗口屏障，也未采集 GPU overlap，
不能把它作为整层 refill 的实测结果。

预声明性能门为 W/A 六次中位数比 `<=0.90` 且六组均改善，才支持下一项独立资格；
比率落在 `[0.90,0.97)` 或配对不稳定为 CONDITIONAL，`>=0.97` 则 STOP 当前策略。
任一数值或资源硬门失败立即停止，不能用性能抵消。

缩小范围 memcheck 已覆盖全部 14 种 target capacity 及首尾样本：15 setups、
136 targets，A/W 两次共比较 272 个 target 输出实例，全部逐位一致。
optimizer state、selected-SP、coefficients/fitted/residual、rank/status、轨迹和
target identity/order 均纳入比较；CUDA memcheck 为 0 errors、0 bytes leaked。
受审计的 optimizer 区间内资源建销为零，输出 leases 和 application-visible 资源关闭。
该 sanitizer 结果不替代全层 ordinary replay，也不声称对完整 canonical 做过 sanitizer。

## 六组正式结果

| 组 | 顺序 | A：原窗口 64 workers（秒） | W：原窗口 32 workers（秒） | 下降 |
| --- | --- | ---: | ---: | ---: |
| 1 | A/W | 92.313378 | 80.959929 | 12.299% |
| 2 | W/A | 93.323565 | 80.398927 | 13.849% |
| 3 | A/W | 91.457737 | 80.875274 | 11.571% |
| 4 | W/A | 96.020865 | 80.333885 | 16.337% |
| 5 | A/W | 89.020633 | 80.870667 | 9.155% |
| 6 | W/A | 91.776521 | 80.520311 | 12.265% |
| 中位数 | — | **92.044950** | **80.695489** | **12.330%** |

W/A 比率 `0.876696546` 通过 `<=0.90` 门，六组均改善；中位数差为 11.349460 秒。
预热的 91.982857/80.868689 秒未进入正式中位数。
全部 14 次整层重放（两档预热加十二次正式运行）各保留 2,154 setups、17,767 targets、
216,194 次 optimizer 迭代和原 target identity/order；完整输出全部逐位一致。
全层共比较 248,738 个 target 输出实例，加上 memcheck 为 **249,010** 个；
这些是重复比较实例，不是同样数量的唯一 targets。

| 其他计时（六次中位数，秒） | A | W |
| --- | ---: | ---: |
| input packaging / token release 纳入后的分段工作 | 92.058665 | 80.708700 |
| 每档计入完整 prepared 建立/销毁后的分段工作 | 96.231291 | 84.884677 |
| 完整 shadow 比较（在工作计时外） | 0.274722 | 0.275083 |

每组 A/W 共用的 prepared build/release 中位数为 3.347939/0.826551 秒；两档都收取
一次该成本，不把对象常驻造成的生命周期差异算成 worker 数收益。各项中位数独立计算，
不能要求它们相加严格等于另一个总量的中位数。

全部 optimizer 调用的受审计资源建销为零；每个窗口和最终进程边界的输出 leases、
application allocations/bytes、streams、events、BLAS handles 都关闭。
正式 campaign 的 application buffer 峰值为 600,340,480 bytes，prepared 创建后最小
free memory 为 21,989,490,688 bytes。该 campaign 不含生产 cache/arena。

| 运行 | GPU1 整程采样显存峰值（MiB） |
| --- | ---: |
| canonical L4 捕获 | 21,624 |
| 独立容量检查 | 20,430 |
| 缩小范围 memcheck | 2,202 |
| 14 次整层重放 campaign | 3,146 |

采样间隔约 200 ms。容量创建/释放很快，telemetry 峰值可能漏过瞬时最高占用；
判断 2 GiB 门采用每个 prepared 创建后的 `cudaMemGetInfo`，不能用采样值推翻
已实测的容量 STOP。application bytes、创建增量、整程采样峰值分别保留。

## 决策与后续

本轮只支持继续资格验证 **L4 原窗口内 32-worker 的 opt-in runtime**：原 34 个窗口、
全部 target cohorts、prepared 生命周期及其他层调度继续保持；完整 canonical
数值/顺序/资源和 outer wall-time 必须重新验证，才能给 runtime 性能结论。

11.349 秒差值相当于当前 472.080 秒完整方法基线的约 2.4%，只说明潜在收益量级，
不是已测得的完整方法改善。本轮不更新该 472.080 秒基线，也不声称已证明 GPU kernel
overlap 或消除了相同长度的纯 device time。

全层 resident A/R 没有进入计时；其容量 STOP 不能外推成“所有有界补位都无效”。
若以后另开 L4 槽位原型，必须覆盖容量 2–18 的真实 cohorts、与生产 cache/arena 共存、
完整输出和 lease/event 生命周期；L5 的 capacity-eight 结果不能作为它的资格。
当前优先级是原窗口 worker 数这一窄改动，其他 L4 架构探索仍未获 runtime GO。

artifact：`fastkpc/artifacts/full_cuda_ci/optimizer_l4_diagnostic_20261023/`。
`closure.json` / `manifest.json` 绑定原始 fixtures、capture receipt、两档逐窗口/整层时间、
容量和 memcheck、两卡 telemetry、进程、工具、生产 source/native 以及参考 receipts。
最终 manifest SHA-256：

```text
d5c77e9c8bcd89c5ee5ac3fcd93fc8421a60231000bd355273b90e631d386ceb
```

GPU1 测试进程已退出，GPU0 外部训练 PID 845631 保留。

生产源码/native、L5 资格及默认配置未改，本轮未 commit/push。
Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false。
