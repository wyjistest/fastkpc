# Phase 10 开发交接：构建一致性与 structured 行和容量优化

日期：2026-10-30。本次将已验收的构建修复、容量优化、工具与报告收口到 Git。
当前安装 native 为 `b2b79fc6…e135b12b`，结论为 **GO_CAPACITY / speed NOT_PROVEN**。
容量版本的 canonical 隔离开发中位数为 **283.328 秒**；Phase 10 **ACTIVE**、
candidate **NOT_FROZEN**、holdout **SEALED**、recommended **false** 保持不变。

## 提交与运行身份

| 提交 | 范围 |
| --- | --- |
| `89a8099341d3c7a7b39fed2e2d73bfe5986ebb61` | method-batch 全量/增量编译参数统一、两处 R 测试清理修复、10 个构建资格工具文件 |
| `ae3687f06c94fca2ec0828cfcc3f9986170dbbad` | structured 专用行和 kernel、删除矩阵分配、15 个原资格文件及 2 个新 Git 核对工具文件 |

[Git source attestation](PHASE10_SOURCE_ATTESTATION_20261030.json) 的状态为
`PASS_COMMITTED_SOURCE_CLOSURE`。修复后父版本的全部 **107 个生产输入**与构建提交
逐项一致，ON 的全部 **107 个生产输入**与 runtime 提交逐项一致。**25 个原资格
文件**保持已封存字节，两处 R 测试修复与实际回归使用的文件一致。两个新 Git 核对
工具另行记录，不冒充运行前已有的工具。

原隔离运行 HEAD 为 `e15be9dda379687a4a57d74291cf8a36ca6c35da`，当时生产改动未提交。
本次证明新 Git blobs 包含相同的已测源码闭包；没有重编 native 或重跑 GPU campaign，
也不把原 receipts 重标为从新提交 clean checkout 运行。源码入库不升级性能证据等级。
历史报告中的“未提交”“未安装”和“待执行”描述各自当时状态；当前仓库身份以本交接为准。

## 构建与二进制

[构建资格](PHASE10_BUILD_CONSISTENCY_20261029.md) 统一的是
`full_cuda_ci_method_batch.cu` 在 full/fixed-SP incremental 两个入口的 `--fmad=false`。
其他 TU 的 FMA 策略不变。真实全量和增量构建最终 native SHA 一致：

```text
修复后 OFF: 639d598fe23db8cc08571f6e3f7d1e153a6c273d17a13ac8ecd806302179bab8
当前 ON:    b2b79fc6219841a7b9de193e312de27bbdb8bca77a9d3a9d4a626aa1e135b12b
历史版本:   a1e16a9c0c8089ea90491426c56475394449efb03317be4a87f1925c7b73bd34
```

原始 NVCC 对象的局部临时文件名差异与诊断归一化副本分别保留；最终 native 比较
没有归一化。ON 仅重编 `full_cuda_ci_vertical.o`，其余 **43 个** full-build 父对象
不变。本次重新核对安装的 native、**44 个链接对象**及两个 generated includes，
均与验收记录一致；安装目录里额外的诊断对象不计入链接清单。

构建修复已通过三条受影响方法 `dcc.perm/hsic.gamma/hsic.perm` 的跨 binary 回归、
permutation RNG、四项已有回归，以及完整 canonical 的数值资格。行和改动保持
`fabs`、线程内累加、归约树及 totals 顺序；dense consumer/fallback 使用原独立矩阵。
尾批、lane 复用、640 component payload、混合 fallback、失败零发布和 sanitizer
资格见[原型记录](PHASE10_STRUCTURED_ROWS_20261029.md)。

## 已接受的容量与完整耗时

[原协议隔离验收](PHASE10_STRUCTURED_ROWS_ISOLATED_20261030.md) 保留
OFF1/ON1/ON2/OFF2/OFF3/ON3 顺序、fresh R/cache、物理 GPU1、CPU affinity 0–19、
数学库单线程及既有 postback/work-pair/lanes/slots；两卡均无外部 compute。

| 配对 | OFF 秒 | ON 秒 |
| --- | ---: | ---: |
| 1 | 283.985 | 283.328 |
| 2 | 285.091 | 290.024 |
| 3 | 283.307 | 283.214 |
| 中位数 | **283.985** | **283.328** |

中位比值 **0.997686498**，三个配对均 ≤1.05，通过原中位 ≤1.02 的不回退门。
观察下降 **0.657 秒 / 0.23135%**、2/3 配对获益，未通过速度门（中位 ≤0.98 且
三组全胜），因此保持 **speed NOT_PROVEN**。历史 `c469ac5/e15be9d` 的 290.491 秒
继续绑定 `a1e16a9c…`，不与本轮 283.328 秒相减来宣称矩阵删除的加速。

每轮 **240,498 个 CI/pMax 位模式**、task/order/deletion/skeleton/sepsets/counts/RNG
全部不变；最大 authority 误差 **3.05425129631942e-12**。CPU numerical authority、
fallback、生产 large D2H 为零，最终 allocations/bytes/streams/events 全部归零。

各层 requested structured pool 节省 **630,789,120 B = 601.5673828125 MiB**；
solver workspace 不变。三个配对的 L3/L4/L6 原始最低空闲检查点均改善 **620 MiB**；
L6 高于 2 GiB 门槛的余量由 **79.125 → 699.125 MiB**。分配请求账本与实际空闲
检查点分别报告；它们不证明连续峰值上界或更大输入资格。释放空间保留为余量。

## 证据核对与复现边界

本次提交前后核对最终 manifest 的 **962 项**：107 个生产输入、18 个原工具、
275 个行和证据文件、562 个父版本文件。另完整核对共享主机 manifest 的 **419 项**。
这些是两个 manifest 的条目数，包含重复文件，不能相加为独立文件数。

共享主机历史 manifest 的旧 `fastkpc/build/fastkpc_cuda.so` 依赖，仅映射到已封存的
`build_consistency_20261029/parent/fastkpc_cuda.so`，仍要求原 `a1e16a9c…` SHA；
其余项按原路径和 SHA 核对。旧 receipt、manifest、pending 协议和原始失败记录
均未改写。[共享主机报告](PHASE10_STRUCTURED_ROWS_SHARED_HOST_20261029.md)及其
[修订复核](PHASE10_STRUCTURED_ROWS_SHARED_HOST_REVIEW_20261029.md)继续保留独立证据等级。

```text
isolated closure:  cdc26f74cab09449dcfd47ef2d3f6b710246dd4682dc93122fca5855081bfaa3
isolated manifest: 196a0a84da7535a65927dbc4d8a49db5b192950fc7372848a5e970d1e4a8c87c
shared manifest:   ad13ec1733094e815d7b92c2121f367a4d762367b4198cf1827c29c6efeacd9d
```

[只读 Git 核对工具](../tools/research/structured_rows_git_20261030/README.md)可复算上述
身份对应，输出与入库 attestation 一致；它不编译或运行 GPU。入库语法检查通过
**17 个 Python AST、6 个 R parse、1 个 shell syntax**；14 份 Markdown 的 92 个本地
链接全部可解析。whitespace 检查唯一例外是
已封存 `structured_rows_20261029/lifecycle.cu` 的末尾空行：保留原字节，对该文件
仅关闭 `blank-at-eof` 检查，其余 whitespace 检查照常执行。

数据集、native、captures、原始 receipts 与 profiler traces 保留在本地，没有入库。
fresh clone 可取得源码、工具和身份索引，但不能仅凭 clone 完整复核 GPU 成绩；
研究工具要求其说明中已认证的本地输入，并对缺失输入失败，不把 SKIP 当 PASS。

## 后续范围

本次仓库收口不启动新优化实验。下一项性能工作应复用已有 Stage B/work-pair 归因，
先确认当前基线仍缺少哪项证据，再决定窄诊断或原型。q55 继续后排；recover tile16/8、
L8 STOP、默认开关、数值合同及所有 promotion 状态不变。冻结候选前仍需冻结 producer
上的 clean-checkout 重复性能与完整语义资格，180/120 秒门仍未通过。
