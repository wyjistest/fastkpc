# Exact speedup 执行记录（2026-10-20）

按 [提速研究文档](PHASE10_EXACT_SPEEDUP_FINDINGS.md) §8 执行。该文件记录
实际实验，研究文档中的机制假设不能代替这里的生产测量。

本轮基于 `main@6a6a4b3` / runtime `dcb1749`。全部候选数值计算仍在 GPU；
已有 CPU oracle 和分量 shadows 只用于离线资格比较。没有 CPU fallback。
Phase 10 仍为 ACTIVE，candidate 未冻结，holdout 未打开，recommended=false。

本轮完整 canonical 对照得到明确收益的是 **`CUDA_DEVICE_MAX_CONNECTIONS=32`**：
1159.691 → 880.659 s，下降 24.061%，240,498 个 CI 与默认 connections
逐位一致。pinned I/O 没有额外完整方法收益，实验已移出生产源码；eig 扩 lane
和当前 structured eig 实现均不进入生产。所有性能结论仍是 development evidence。

## 1. Pinned optimizer I/O

实验开关：`FASTKPC_MULTI_PENALTY_PINNED_IO=1`。每个 prepared handle 持有
Y 和 optimizer state 的 pinned host buffer，复用期间地址稳定，异常销毁前
等待本 stream。GPU kernels、运算次序和既有正常同步点均不变。关闭开关
使用原 pageable 路径。

全部容量（base、small64、192、384、559）已编译；完整 A/B 使用相同 binary：
`c4f599ef5267c9ff1c7a5a81d7e550330b730d84679a94328bf690c0bb51ffc0`。
它是未提交源码闭包构建，不能标记成 clean producer commit evidence。

先行 subset22 / conditioning depth ≤3 使用 base+small64 构建
`2ff60ac0c919554066c7da3a87b286eb9b49a61604153ec41985f9656417af0f`：

| 指标 | pageable | pinned |
|---|---:|---:|
| outer elapsed | 62.109 s | 61.845 s |
| multi-penalty optimizer | 10.88854 s | 10.92824 s |
| all optimizer | 13.60971 s | 13.62824 s |
| dCov host boundary | 37.52467 s | 37.29362 s |

task/order/deletion、所有 CI p、pMax、adjacency、sepsets、n.edgetests、RNG
逐位一致；GPU authority 和资源门通过。持久 handle / residual token 原回归通过。
这组单次子集结果没有可归因的提速。

retained 配置的 pinned nsys 剖析确认：

- optimizer memcpy 内累计 mutex wait 由旧剖析的 247.87 s 下降至约 0.02 s；
  memcpy API 累计时间由 258.52 s 下降至约 0.09 s。
- optimizer kernel 并发仍为 4.55（原剖析 4.58），窗口约 10.75 s
  （原 10.82 s），launched-block concurrency 上界 34.3（原 34.6）。
- 等待主要移到既有 `cudaStreamSynchronize`，不能把累计 host wait 当作收益池。
- pinned H2D 设备排队延迟中位数 42.317 ms，p90 287.204 ms，最大
  1293.418 ms；尚不能把原因唯一归给 kernel 算力或主机线程。
- 全进程 `cudaMallocHost` / `cudaFreeHost` 均为 7,415 次，所有返回值成功。

设备排队诊断后，增加了低成本的 connections 配置对照。全容量 binary 下：

| subset22 配置 | outer | multi-penalty optimizer |
|---|---:|---:|
| 32 connections，pageable | 57.700 s | 6.432629 s |
| 32 connections，pinned | 57.318 s | 6.104471 s |

两档均与原子集逐位一致。主要收益来自 connections 配置，不能归给 pinned；
pinned 的子集增量较小。这里的 connections 指每个进程的
`CUDA_DEVICE_MAX_CONNECTIONS=32`，没有新增 GPU 或更改计算任务语义。

完整 canonical 顺序验证已完成：默认 connections/pageable →
32 connections/pageable → 32 connections/pinned。三档使用同一全容量
binary；后两档分别与前一档进行逐位比较，并各自直接检查 oracle。

第一档有效 receipt：`canonical-default.rds`，outer **1159.691 s**，
all optimizer 615.780 s（multi-penalty 591.598 s），dCov 437.197 s，
exact residual 34.360 s，native setup 22.585 s。240,498 CI 通过；CI/pMax
最大误差均为 3.0547508966805026e-12；authority、RNG、资源关闭门全部通过。

第二档 `canonical-queues32.rds`：outer **880.659 s**（下降 24.061%），
all optimizer 335.361 s（multi-penalty 311.157 s），dCov 437.978 s，
exact residual 34.431 s，native setup 22.592 s。outer 节省 279.032 s，
optimizer 节省 280.418 s；其余主阶段基本不变。所有 CI p、pMax、task/order/
deletion、adjacency、sepsets、n.edgetests、RNG 与第一档逐位一致。

第三档 `canonical-pinned-queues32.rds`：outer **884.672 s**，没有优于第二档。
三档完整结果如下；每档一次 fresh-process、default-Inf canonical，使用同一
native、同一 GPU、相同 retained cache/bypass/8 eig lanes，另一张 GPU 无重任务。

| 配置 | outer | all optimizer | multi-penalty optimizer | dCov | exact residual |
|---|---:|---:|---:|---:|---:|
| 默认 connections，pageable | 1159.691 | 615.780 | 591.598 | 437.197 | 34.360 |
| 32 connections，pageable | **880.659** | **335.361** | **311.157** | 437.978 | 34.431 |
| 32 connections，pinned | 884.672 | 338.329 | 313.876 | 438.023 | 34.431 |

单位均为秒。32 connections 的 outer 节省 279.032 s，optimizer 节省
280.418 s；pinned 相对 pageable 多用 4.013 s，单次差异不足以断言稳定回退，
但没有证据支持保留生产开关。全部 CI/pMax、task/order/deletion、adjacency、
sepsets、n.edgetests、RNG 在三档间逐位一致。

三档还通过 `audit_saved_canonical.R` 对原正式审计逻辑的离线复核，包含
exact-primary、系数复用、component-cache accounting、semantic sepset、
authority 与资源关闭。最大 CI/pMax oracle 误差保持 3.0547508966805026e-12。
这证明 connections 配置有完整方法级收益；不是 pinned 缓冲的收益，
仍是每档一次的 source-closure development evidence。

第一轮 canonical 计算后的 R 审计脚本曾因运行期间修改文件导致读取错误，
没有保存 receipt，已排除出性能证据。修复后冻结运行脚本，并在审计之前
先保存 `.pending` 原始 receipt；离线审计函数已通过子集正例和误用子集作
canonical 的拒绝测试。新的 campaign 不修改运行中的脚本或 native。
未更改数值 runtime 的默认值。pinned 改动保存为研究目录中的
`pinned_optimizer_io_experiment.patch`，可在本轮 HEAD 上重新应用和构建；
生产 `mgcv_multi_penalty_gcv.cu` 已恢复到 HEAD。实验 runner 在检测不到
pinned native 时拒绝执行 pinned=1，避免把无效开关误记为对照。

## 2. Eig 的大批次 lane 限制

对现有 benchmark 增加仅诊断用的 NVTX 与 profiler capture range。
分配、warmup 和最后的大矩阵 D2H 都不在 steady-state capture 范围。
单 host thread 和 per-lane host threads 都测量；所有 eig 输出逐位一致。

48 components、3 次 profiled trial 的证据：

| 指标 | 8 lanes | 16 lanes |
|---|---:|---:|
| 三次 eig.lanes range 总长 | 415.295 ms | 415.489 ms |
| range 内有任意 kernel 执行 | 99.88% | 99.87% |
| kernel 平均并发 | 3.354 | 3.368 |
| sytrd4_gpu 时间并集 | 349.396 ms | 352.854 ms |
| sytrd4_gpu 同类并发 | 1.043 | 1.040 |
| steady-state syevd 内 malloc/free | 0 | 0 |

该 batch 已有足够宽度，`sytrd4_gpu` 覆盖约 84% 的 range，增加 lanes
没有缩短它。改用多个 host threads 也没有降低 makespan。kernel 并发与
grid 数不等于 SM utilization；本证据没有声称获得了硬件 occupancy 百分比。

带 cuSOLVER/cuBLAS 范围的 subset22 production 剖析进一步确认：9,041 次
syevd（含初始化）内部的 malloc/free 均为 0。全进程 21,913 次同步 malloc
中，10,215 次在 `cublasCreate_v2` 内；25,327 次同步 free 中，3,405 次在
该创建调用内。其余发生于 syevd 之外，不能作为“syevd 内分配串行”的证据。
未取得的原生调用栈不作猜测归因：本机 perf_event_paranoid=4，Nsight
CPU backtrace 采样不可用。

非 profiled 的 9/16/48 batch、各 7 次重复结果如下（device median）：

| components | 8 lanes | 扩展 lanes | 扩展时间 |
|---|---:|---:|---:|
| 9 | 30.151 ms | 9 | 29.959 ms |
| 16 | 48.261 ms | 16 | 48.465 ms |
| 48 | 139.041 ms | 16 | 139.431 ms |

扩 lane 没有实质收益；这些 microbenchmark 不替代 canonical outer wall-time。

## 3. Structured eig GPU 隔离原型

实现 GPU 排序/合并 ties、Brownian-bridge 闭式双对角、LAPACK DBDSQR
右旋转记录、极点偏移坐标 secular roots/deflation、旋转逆序重放与 O 映射。
没有生成完整 P 或 n×n 距离矩阵。反复访问的 QR 状态和向量恢复 tile 使用
shared memory，旋转日志有显式容量限制。

实现复用仓库 vendored LAPACK 的数值迭代；generator 只替换 VT 的四种
更新操作并记录其变换。二值输入显式处理 ±lambda 的 tie，k=1 选择正根
（对应 ascending eigenvalue 中较高的 index）。

16 类 stress、n=351/1000、k=35、每组 120 个分量对：

| 资格项 | 结果 |
|---|---:|
| n=351 最大 abs Δp | 3.9524e-14 |
| n=1000 最大 abs Δp | 3.2696e-13 |
| statistic / mean / variance 合同容差 | PASS |
| alpha=0.1 flips | 0 |
| 重复输出逐位确定 | PASS |
| 常数 / 非有限 / rotation overflow fail closed | PASS |
| 等计数二值输入 k=1 tie-break | PASS |

这些 p 值来自 GPU 分量的显式 diagnostic shadow 与离线 dense oracle 比较；
不是完整 GPU method 或 production cache 的资格证据。

同输入的 7 次重复结果：

| components | retained dense eig-only | structured component | structured / dense |
|---|---:|---:|---:|
| 9 | 30.151 ms | 277.794 ms | 9.21× |
| 16 | 48.261 ms | 277.810 ms | 5.76× |
| 48 | 139.041 ms | 285.461 ms | 2.05× |

48 个分量中约 247.15 ms
花在串行 bidiagonal QR/log 中。将其小数组放进 shared memory 后仍明显
慢于现有路径，因此该原型不进入生产集成；没有运行全量 canonical/三真实
数据 method gate，也没有赋予生产 cache solver identity。

表中 dense 只计 eig，structured 计完整 component device pipeline；比值不能
解释成同边界的完整方法降速。该诊断没有提供足够的生产提速依据，因此停止
当前实现的集成，而不把它外推成所有结构化算法均不可行。`cuobjdump` 也确认
QR kernel 为 100 registers、stack/local memory 为 0，没有 local spill 可供
直接消除。

`compute-sanitizer --tool memcheck --leak-check full` 对 n=351 的全部 16 类
分量、重复执行、二值 tie 和 fail-closed 路径检查为 0 errors、0 bytes leaked。
sanitizer 的插桩时间不纳入性能表；该检查也不代替完整 method 资格。

## 4. R2 / R3 决策

完整 A/B 已否定“消除 pageable 锁等待就会获得同规模 outer 收益”的推断。
提高 connections 已在不改写调度器的条件下消除约 280 s optimizer boundary。
后续评估必须以这条更快的配置为比较起点，不能沿用默认 connections 的收益池。

恢复原 native `fe8286f7…` 后，32 connections/pageable 的 subset22 nsys
复核显示 optimizer kernel 时间并集为 7.24 s，平均 kernel 并发 6.72；
marker 窗口为 7.44 s，launched-block concurrency 上界 50.3。相比原默认
connections 的约 10.82 s / 4.58 / 34.6，设备排队确有改善。仍存在的
183.89 s 累计 memcpy mutex wait 是多线程累计值，不能再当作 wall-time
机会；本剖析没有测得 SM occupancy，也没有证明剩余差距全可由常驻内核消除。

- **R2 常驻 optimizer：生产重构 NOT YET GO。** 仅有 block 独立性分析不足以
  证明完整 optimizer 的吞吐。重开需先用 captured complete-optimizer window
  比较 32 connections，覆盖容量 bucket、workspace/lifetime 和长尾，并通过
  selected-SP、route/status、coefficients 逐位门。
- **R3 lane-per-target：生产重构 NOT YET GO。** 已有 QR/Q 微基准未覆盖
  stable-SVD、Newton、line search 与目标间分支发散，不能换算成 canonical
  optimizer 收益。先有完整 evaluator 的 GPU 原型与逐位对照，才考虑集成。

880.659 s 中 dCov 为 437.978 s，已大于 all optimizer 的 335.361 s。
即使把 optimizer 全部消除，其余部分仍约 545.298 s；目前没有 180/120 s
门的可信实现路径。本轮没有重写 optimizer、降低容差或替换生产 eig solver。

## 复现工具与证据边界

研究目录：`fastkpc/tools/research/exact_speedup_20261020/`。

- `run_pinned_optimizer_ab.R`：fresh R process、同 native A/B、oracle 与逐位门。
- `analyze_eig_lanes.py`：NVTX 精确调用范围归因，不把启发式窗口当 solver 时间。
- `generate_recorded_dbdsqr.py` / `structured_eig_gpu.cu`：独立 GPU 原型。
- `verify_structured_eig_gpu.py`：分量 oracle、确定性、tie 与 fail-closed 检查。
- `benchmark_structured_eig.py`：同输入的隔离 steady-state 对照。

完整 campaign 已归档至：
`fastkpc/artifacts/full_cuda_ci/exact_speedup_20261020/`。
`manifest.json` SHA-256：
`8968e7247795e348bd94b7bf011b4aee80817e5da7d3720316cfb9584ea205aa`。
该索引包含压缩后的 receipts、原始 receipt SHA、native、source closure、
逐项硬门和 nsys/分量实验证据。归档时仍是 pinned 实验源码，准确绑定本轮
`c4f599ef…`；不能把 880.659 s 重新标成 restored native 的完整 canonical 运行。

恢复原 native 后另有 subset22 receipt：**57.554 s**，与实验 binary 的
32-connections/pageable 子集在全部 p/pMax、task/deletion、图、sepset、
计数和 RNG 上逐位一致，authority 和 resource 门通过。该结果只用于恢复
校验，不外推为恢复后 native 的全量性能证据。

随后从恢复后的源码重新编译 base/small64/192/384/559 全部五个 optimizer
对象并链接，native SHA 精确恢复为：
`fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50`。
`fastkpc/src` / `fastkpc/R` 与 HEAD 无差异，没有遗留实验对象等待增量链接。
恢复后的 profile、子集 receipt、显式 IEEE 位检查（含有符号零/NaN payload）、
sanitizer、冻结 runner 和重建日志由独立的 `restoration/manifest.json` 索引；
SHA-256 为
`5c0a0af8294462d940c8c046c934f849c3ca2711689c07c5375ecc3c0b64e3ce`。
所有 R/Python/shell 工具语法检查与 `git diff --check` 通过。

便捷开发入口：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  sh fastkpc/tools/with_phase10_exact_development_env.sh \
  Rscript fastkpc/tools/run_phase10_pvalue_contract_audit.R NEW_OUTPUT.rds
```

入口在创建 CUDA context 前设置 connections=32 及 retained 配置，只影响
子进程，保留调用者显式环境覆盖，不改变全局机器或库默认值。11 GiB cache
按本机 RTX 4090 资格记录设置，不构成其他显存容量的资格证明。

所有当前时间均是单次 source-closure development evidence，不是 clean
producer 重复中位数或 promotion；三套大型真实数据未按 connections=32 重跑。
冻结 candidate 前仍须 clean producer 重复 canonical 及真实数据重新资格。
