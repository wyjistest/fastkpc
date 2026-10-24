# L5 opt-in 槽位池 runtime 集成资格

日期：2026-10-23。承接 [32 槽隔离原型](PHASE10_OPTIMIZER_SLOT_POOL_20261023.md)。
最终状态：`GO_L5_OPT_IN_RUNTIME_DEVELOPMENT`。六次完整 canonical skeleton 硬门
全部通过，OFF/ON 三次中位数为 **550.152 / 472.080 秒**，总耗时下降 **14.19%**。
仅资格化默认关闭的 canonical S2 L5 runtime 路径。

本轮只把 L5 原型接入 canonical 351×48、default-Inf、dcc.gamma S2 路径。
开关 `FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS=1` 默认关闭，需要 exact-primary reuse、
L2/structured component prefill 和 status-wait=1。q=46、五 penalties，实际 target
cohort 保持 2/3/4/8；不匹配时 fail closed，不回退 CPU 或静默改变 cohort。

R 主线程先取得原顺序的全部 L5 setup、复制有所有权的 host geometry packet，并为
现有 coefficient arena 预留整层系数空间。32 个固定容量槽位各有 stream、handle 和
输出 lease；worker 只调用现有 exact-primary 数值实现。保存系数时只串行化 arena
的 metadata/D2D 入队，每个槽位在自己的 blocking event 完成后才释放旧 lease。
packet 保留至所有 workers 完成或失败排空后。上传错误会终止并销毁整个池，不复用
部分更新的槽位。

优化完成后仍按原 setup/target 顺序发布 target state。Stage A 从同一 coefficient
arena gather 系数、重绑定池中一个槽位、运行现有 finalizer，然后按原顺序保存 residual。
Stage B 开始前整个槽位池已排空并销毁。其他层不使用该池。L5 诊断记录一次真实整层
optimizer boundary，不伪装成原来的 15 个窗口。

本轮数值、计数、资源、显存和性能门在 GPU 运行前写入
[执行协议](../tools/research/optimizer_slot_runtime_20261023/README.md)。
完整 A/B 使用同一 source-closure native，OFF/ON/ON/OFF/OFF/ON 六次 fresh-process
运行，分别取三次中位数；ON 总耗时需不超过 OFF 的 97%，三组相邻配对均改善，且
全部硬门通过，才给出 development 性能 GO。

GPU1 执行，GPU0 外部训练保留；证据为 shared-host development。
不升级为 clean producer、候选冻结或 promotion。Phase 10 ACTIVE、candidate
NOT_FROZEN、holdout SEALED、recommended=false；默认开关保持关闭。

工具目录：`fastkpc/tools/research/optimizer_slot_runtime_20261023/`。
证据目录：`fastkpc/artifacts/full_cuda_ci/optimizer_slot_runtime_20261023/`。

## 已完成的前置资格

98 个输入、43 个对象从空构建目录完整重编译通过。native SHA-256：

```text
b0d3523285bc8098286d00350c5d4204edec31aab13f6336090fab2d203cdf58
```

五个容量桶的 74 个 GPU 函数指令均与此前已资格化实现一致。
普通与 memcheck 两次检查各比较 68 个 optimizer/Stage A target 输出实例，均逐位一致；
每次 24 项负向/旧 token 检查通过，取消清理和 application-visible 资源关闭通过。
受审计区间内资源建销为零。缩小范围 CUDA memcheck 为零错误、零泄漏；没有把
缩小范围 sanitizer 描述成全 canonical sanitizer。

测试程序最初未设置 Rlapack 的动态库搜索路径，因此在进入 GPU 工作前启动失败；
修正 runner 环境后通过。该日志保留在 `loader-attempt/`，不计入通过运行；生产
source/native 没有因此改变。

## 完整 canonical 对照

同一 native、98 文件源码闭包，按预先声明的 OFF/ON/ON/OFF/OFF/ON 顺序执行。
每次 fresh R process，重置 dataset-specific result/target caches；CPU affinity 0–19，
BLAS/OMP/MKL 各一线程，CUDA connections=32，11 GiB retained component budget、
residual bypass=1、eig lanes=8、status-wait=1。只切换 L5 slots。
完整 skeleton 调用作为 outer elapsed；artifact 持久化及离线 authority 比较在计时外。

| 执行顺序 | 模式 | outer elapsed（秒） | L5 optimizer host boundary（秒） | 硬门 |
| --- | --- | ---: | ---: | --- |
| 1 | OFF1 | 554.703 | 107.075796 | PASS |
| 2 | ON1 | 472.080 | 31.827456 | PASS |
| 3 | ON2 | 474.505 | 31.800844 | PASS |
| 4 | OFF2 | 550.152 | 109.892481 | PASS |
| 5 | OFF3 | 548.008 | 104.332973 | PASS |
| 6 | ON3 | 470.261 | 31.814829 | PASS |

ON/OFF outer median 比率为 `0.8580901278`，通过预先声明的 `<=0.97` 门；
三组相邻配对均改善，分别下降 14.895%、13.750%、14.187%。中位数节省
78.072 秒，加速 1.1654×。

| 阶段（三次中位数，秒） | OFF | ON |
| --- | ---: | ---: |
| 完整 outer | 550.152 | 472.080 |
| 全部 selected-SP optimizer | 345.312 | 267.473 |
| L5 optimizer host boundary | 107.075796 | 31.814829 |
| Stage A | 16.568508 | 16.573718 |
| Stage B | 144.642183 | 144.789824 |
| native setup | 23.811181 | 23.696339 |
| prepared build | 14.205273 | 12.730752 |

全部 optimizer 中位数减少 77.839 秒，与 outer 节省接近；其中 L5 boundary 减少
75.261 秒。Stage A/B 基本持平。这里测量的是 host API 边界，不是纯 device time；
prepared build 等计数与父阶段存在包含关系，各行和独立中位数不能直接相加。
收益来自 64→32 workers、整层持续取任务、槽位复用的组合，不能全部归因于减少分配。

## 数值、顺序和资源硬门

六次各完成 240,498 个 CI，CI/pMax 相对已资格化 S2 及本轮匹配基线均逐位一致。
最大 CPU authority CI/pMax 误差均为 `3.05425129631942e-12`，满足 `1e-10`。
`[0.05,0.15]` 内 1,337 个 CI 同样逐位不变，最大 authority 误差相同。
task identity/order、deletion、skeleton adjacency、sepsets、n.edgetests、RNG endpoint、
physical cohort sequence 和 Gram resolution sequence 哈希全部保持一致。

exact-primary target 账本、零 replay/terminal confirmation、stable-SVD route、
component route/status 检查全部通过。生产 CPU 数值 authority、fallback、residual/
component 大 D2H 均为零；CPU authority 仅在 GPU 结果产生后供离线比较。
所有暴露的 application resource counters、coefficient arena 和槽位生命周期关闭。
本轮没有新增最终 PDAG 或三套大型真实数据资格，不能用 skeleton 门替代它们。

每次 ON 覆盖一个 L5、955 setups、6,816 targets；32 槽创建/销毁均为 32，
optimizer binds、Stage A binds、detached 均为 955，peak workers=32。
准备耗时 222.662–225.438 ms，销毁 10.841–10.929 ms；worker bind/detach 的
累计时间是并发 worker 的求和，不能作为额外串行阶段加入 wall time。

三个 ON 的实际生产 cache/arena 共存容量结果相同：

```text
free_before_create       2,913,796,096 bytes
free_after_create        2,372,730,880 bytes（约 2.21 GiB，满足 >=2 GiB）
physical creation delta    541,065,216 bytes（516 MiB）
```

六次整程 GPU1 采样峰值均为 **22,366 MiB**，采样间隔约 200 ms；这是采样峰值，
不是瞬时最大值证明。516 MiB 只是创建增量，不能当作整次占用。
容量余量有限，本轮不资格化其他 workload、其他层或多个池共存。

## 证据身份与后续边界

仓库 HEAD 为 `d982dcf6ee441353cd14d7c7f6b46e116f2e6499`；本轮 runtime 仍包含
未提交改动，真正生产输入由 98 文件 source SHA map 与上述 native SHA 绑定。
GPU1 UUID 为 `GPU-9c6883e3-fe8f-1a96-ebac-c8496069cba2`；六次均未出现其他
GPU1 compute process。GPU0 的外部训练 PID 845631 保留，因此这是 **shared-host、
uncommitted source-closure development evidence**，不是隔离性能或 clean producer 中位数。

`closure.json` 保存六次结果、分层中位数、显存采样峰值、进程环境和最终门判定；
`manifest.json` 绑定源码、native/build、完整 receipts、工具、authority、历史参考及日志。
`qualified-run.R` 保留实际运行脚本；最后工具清理仅移除 EOF 的一空行，归档逐字节
验证除此之外无差异。原始 receipts 的 runner SHA 保留并绑定该副本，未改写运行证据。
最终 manifest SHA-256：

```text
77e07ad20ac59a1dd03cb2af485bfbb0edb2d4ebb4055c6dce9f119557925173
```

保留首次 loader 失败日志，不计入 PASS。默认开关、Phase 10、candidate、holdout、
recommended 状态均维持前述边界。本轮未 commit/push。

ON 的剩余 optimizer API boundary 中位数为 L3 49.777s、L4 90.315s、L6 43.694s、
L7 10.711s、L8 16.626s。后续最值得单独诊断的是 L4 captured-layer 调度和容量；
当前 8-target 槽位合同不能直接推广。其他层的通用 rolling runtime 继续 NOT_YET_GO。
