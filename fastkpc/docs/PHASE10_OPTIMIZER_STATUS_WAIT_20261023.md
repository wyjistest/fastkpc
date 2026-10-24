# Optimizer status D2H 前等待：原窗口资格验证

日期：2026-10-23。承接 [D2H 复核](PHASE10_OPTIMIZER_DOUBLE_WINDOW_DIAGNOSTIC_REVIEW_20261023.md)。

## 范围与预先声明的门

本项只资格验证 prepared-owned blocking event。保留原 64-request/64-worker
窗口、target cohort、数值 kernel、selected-SP/fixed-SP 语义和下游 producer event。
不包含跨窗口补位、按时长排序或整层 prepared 常驻，也不改变生产默认配置。

开关为 `FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H`：未设置、空串或 `0`
关闭，`1` 开启，其他值报错。在 prepared 创建时读取并固定；开启时在该对象的设备上
创建独立的 `cudaEventDisableTiming | cudaEventBlockingSync` event，随对象释放。
worker 内不创建 event，不使用 TLS。optimizer 原 residual completion event 之后、
首次 pageable status D2H 之前记录并等待新 event。原 D2H、末尾同步、trace 下载和
fixed evaluator 的完成路径保留。

预先声明的接受条件是所有硬门通过，且开启后的两次 outer 中位数不高于关闭后的
`1.03×`。这是当前原窗口的有限 development 无实质回退检查，不是统计置信区间、
clean-producer 中位数或 promotion。无论结果如何，开关默认保持关闭。

## 统计修正

原 D2H 占用等待模型补上本 API 的进入时间：

```text
max(0, own_kernel_end - max(previous_D2H_return, own_API_enter))
```

首个 D2H 单独报告，不进入“after first”合计。原 trace 和实测 wall time 未变。
patched 六档修正后的推断占用等待均为零，旧公式误计的 patched L5 C
`6.330606s` 已移除；这仍是 trace 模型，不是驱动内部锁测量。
旧 manifest、脚本及输出保留在 review artifact 的 `attribution-v1/`。
修正后 review manifest SHA-256：

```text
e1a70a0b5940c12d7463ed365fa62b4d2782815663be5a7fc24df6b291d4608e
```

## 资格验证设计

从空目录中的 production-only source snapshot 全量构建，包含全部五个容量桶。
`qualify.cpp` 直接调用实际 native，使用仅存在于测试 executable 的 CUDA API
拦截验证 event flags、设备/stream 归属、创建/复用/销毁和 record/wait/allocation
失败。生产源码不含故障注入或测试计数器。

真实 q=28/46 捕获验证完整 optimizer 状态、coefficients/fitted/residual 的逐位一致；
q=73/82/193/385 的非退化合成问题覆盖其余容量桶的宿主控制路径。这些合成问题
不代表大型真实输入或容量极限。另检查原 producer event、fixed evaluator、trace、
已租用 output slot、配置不可变性、失败后恢复及多 worker 取消。CUDA memcheck
检查真实 q=28/46 完整路径、五桶的 prepared 分配/失败/销毁及多 worker 取消。
首次扩大到全部大型数值核的 memcheck 在 q=193 中主动终止，记录保留于
`validation-v1`，不能当作完成的 sanitizer PASS。五桶的完整数值与 event 门仍全部
运行；所有 GPU kernel 指令与归档版本一致。最终局部检查使用空闲 GPU1，允许 GPU0
另有训练任务，不作为性能证据。

完整 canonical 为 351×48、default Inf、S2 配置，按 off/on/on/off 执行；每档两次，
同一 native、fresh R/cache、CPU 0–19、
`CUDA_DEVICE_MAX_CONNECTIONS=32`、BLAS/OMP/MKL 单线程，保留 11 GiB cache、
all-hit bypass 和八条 eig lanes。持久化及离线 authority 比较不计入 outer elapsed。

最初设计要求两卡无其他计算任务。GPU0 被其他用户的训练占用后，用户明确同意使用
GPU1；因此本次四次对照固定在 GPU1，要求该卡没有其他计算进程，并记录两卡所有
compute PID。GPU0 的外部训练被允许但不能视为隔离环境：本次时间只构成共享主机上的
development 对照，可能受 CPU、PCIe 和调度竞争影响。`1.03×` 门保持不变，其结论
仅适用于本次环境。先行 `semantic-off/on` 两次语义运行另行保留，不计入四次性能中位数。

每次要求 240,498 CI 和 pMax 相对当前 S2 逐位一致，全部值对归档 mgcv authority
误差不超过 `1e-10`；task/deletion/graph/sepset/count/RNG、optimizer 迭代计数、
窗口 cohort、physical group 和 Gram 顺序相同，GPU-only authority、cache 元数据及
资源归零门通过。第一档 off 还与历史 S2 native 的 receipt 比较。

## 结果

全量构建通过：97 个源码/构建输入、43 个目标文件；native SHA-256：

```text
ba8f806bfc1ebbf8ec90b28dcb095a57d163c986836ac33ddaf8fb0a3ed9298f
```

五个容量对象的 74 个 GPU 函数 SASS 与归档 S2 一致，只归一化匿名编译单元的 hash
标识，保留函数名与所有指令。局部逐位输出、trace、fixed evaluator、event 生命周期、
故障恢复和多 worker 取消检查通过。

局部检查 `checks.json` 为 PASS；上述范围的 memcheck 为 0 errors、0 bytes leaked，
等待归因四项回归及 build-lock 检查通过。先行 GPU1 `semantic-off/on` 完整 canonical
均通过全部硬门：各 240,498 CI，CI/pMax 逐位变化数为零，最大 authority 误差均为
`3.05425129631942e-12`。在 authority p-value 为 `[0.05,0.15]` 的 1,337 个 CI 中，
同样没有任何相对基线的位变化。

四次 GPU1 开发对照完成，按实际运行顺序列出（单位秒）：

| 运行 | 等待开关 | outer elapsed | optimizer host |
|---|---|---:|---:|
| off1 | 关闭 | 545.707 | 342.323953 |
| on1 | 开启 | 543.962 | 338.923532 |
| on2 | 开启 | 550.379 | 346.225670 |
| off2 | 关闭 | 541.632 | 337.362202 |
| 关闭中位数 | | 543.6695 | 339.843078 |
| 开启中位数 | | 547.1705 | 342.574601 |

开启/关闭 outer 中位数比为 `1.006439574`，即开启增加 `3.5010s`（`0.644%`）；
optimizer host 中位数增加约 `2.732s`。通过预先声明的 `<=1.03×` 无实质回退门，
但没有证明原 64/64 窗口获得加速，也不能从两次重复得出统计上的等效结论。

四次完整 canonical 均通过全部硬门，连同先行语义对照共六次完整通过：每次
240,498 个 CI，CI/pMax 相对 S2 的位变化数均为零，最大 authority 误差均为
`3.05425129631942e-12`。task/deletion/adjacency/sepsets/count/RNG、optimizer
迭代计数、窗口 cohort、physical group/Gram 顺序、GPU-only authority、cache
元数据及资源归零门全部通过。四次 GPU1 采样显存峰值均为 `22,366 MiB`。

四轮均未发现 GPU1 上的其他计算进程；GPU0 的外部训练进程持续存在，采样利用率
峰值依次为 `32%/32%/71%/32%`。因此这些时间只能作为本次共享主机上的
source-closure development 对照，不能与历史 GPU0 的 `541.1665s` 中位数直接
拼接成新的加速结论。先行语义运行不纳入上述性能中位数。

最终 machine closure：`PASS_SHARED_HOST_ORIGINAL_WINDOW_OPT_IN`。
保留默认关闭的 prepared-owned status wait；生产数值 kernel、64/64 调度和
fixed evaluator 不变。本项完成了等待机制的局部及完整 canonical 资格验证；
后续跨窗口补位仍需独立的整层捕获重放、容量/生命周期和 outer wall-time 证据，
runtime rolling 继续为 `NOT_YET_GO`，本次没有实现或资格化它。

## 复现与证据

工具：[optimizer_status_wait_20261023](../tools/research/optimizer_status_wait_20261023/README.md)。
artifact：`fastkpc/artifacts/full_cuda_ci/optimizer_status_wait_20261023/`。
构建工具保存完整源码 SHA 映射、43 个目标文件和 native SHA；campaign 保存完整
receipts、门、分阶段计时和 GPU telemetry，最终 manifest 绑定这些文件。

本次运行对应 `d982dcf` 之上的未提交源码闭包，不是 clean producer commit。
最终 manifest SHA-256：

```text
6449669f37760976a55da27269eed32db593e6fa4dc00ed6f8cdb09b42dc6e10
```

Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false。
本项不声称新的真实数据或最终 PDAG 资格。
