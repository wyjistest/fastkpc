# Phase 10 开发交接：optimizer status wait、L5 slots 与 L4 worker cap

最新仓库收口见 [2026-10-28 开发交接](PHASE10_DEVELOPMENT_HANDOFF_20261028.md)：
runtime `1d8a11d`、工具 `2920ab8`，accepted opt-in development median 为 311.440 秒。
本页及原 attestation 保留 10-24 的历史身份；最新结果仍为 source-closure development evidence。

后续进展：L4 有界槽位的 runtime 与工具现已分别提交为 `4a1ff6b` / `1c27d6f`。
[后续 runtime 报告](PHASE10_OPTIMIZER_L4_SLOT_RUNTIME_20261024.md)记录隔离 OFF/ON
中位数 **463.288/407.531 秒**及完整硬门；[新的 Git attestation](PHASE10_L4_SLOT_SOURCE_ATTESTATION_20261024.json)
绑定 99 个源码输入与 44 项工具记录。下文及原 attestation 保留先前交接的历史身份，
其 461.663 秒不是新 campaign 的 OFF 对照。新增选项仍默认关闭，Phase 10 状态不变。

日期：2026-10-24。本次收口覆盖已完成的 D2H 等待归因、status-wait 资格、
L5 全层补位与有界槽位实验、L5 opt-in runtime，以及 L4 容量诊断和原窗口 runtime。

## 当前结果与适用范围

最新 canonical 351×48、default-Inf、S2 配置中，L4 worker cap 的同 native
OFF/ON 三次中位数为 **475.439 / 461.663 秒**，下降 **2.90%**，三组配对均改善。
两组都启用已资格化的 L5 slots。L4 optimizer host 中位数为
**90.617233 / 80.233098 秒**，下降 **11.46%**；通过预声明的 outer `<=0.99`
及 L4 `<=0.95` 比率门。

此前独立 L5 campaign 的 OFF/ON 中位数为 **550.152 / 472.080 秒**，下降
**14.19%**。这是 worker 数、全层补位和槽位复用的组合收益，不能称为仅消除了分配
开销。不同 campaign 的历史时间不能代替本轮同 native OFF 对照。

最新六次运行每次全部 240,498 CI/pMax 位值、task/order/deletion/skeleton/sepsets/
n.edgetests/RNG、原 cohort 和 Gram 顺序一致；对 mgcv authority 的最大 CI/pMax
误差为 `3.05425129631942e-12`。CPU 数值 authority、fallback、large D2H、cache
route/status mismatch 为零，application resources 关闭。每次 prepared build/release
为 7,691/7,691，采样峰值为 22,366 MiB，L5 pool 创建后余量约 2.21 GiB。

这些结果仅为 **GPU1 共享主机、提交前 source-closure development evidence**。
GPU0 外部训练继续运行；本次未补做隔离 clean-checkout 性能、真实数据或最终 PDAG
资格。Phase 10 **ACTIVE**、candidate **NOT_FROZEN**、holdout **SEALED**、
recommended **false**；180/120 秒门仍未通过。

## 提交与源码身份

| 提交 | 内容 |
| --- | --- |
| `33e2d3e941eb25529b350474de713f7fbf741cc9` | 默认关闭的 status-wait、L5 slot runtime、L4 worker cap 和配置检查 |
| `27c34ea59a1debb4e4a0d00360af940ff55ed365` | 八组诊断/资格工具、复现说明及 D2H 归因回归测试 |

[源码 attestation](PHASE10_SOURCE_ATTESTATION_20261024.json) 将最终 L4 资格的全部
98 个生产源码/构建输入逐个绑定到 runtime commit 的 Git blob，并核对八份 manifest
中的 74 项工具记录与 tools commit 一致。它保留完整 source SHA-256 清单及原始
manifest/closure/build 索引。本地 native 与最终 L4 资格一致：

```text
2c8dd8af4ea021d8a784588717476f091a0132b9415067e4a3e115ca65bcd0ee
```

最终 L4 native 来自审计过的增量构建：父 L5 native 为 `b0d35232…cdf58`，其
98-input/43-object 全量构建有独立回执；本轮仅 one-call host 源码变化，41 个保留
对象完全一致，重编 structured eig 的规范化 SASS 不变，生成的 DBDSQR 内容一致。
这不等于本轮新执行了一次全量 build。

原始回执的 HEAD 仍为 `d982dcf6ee441353cd14d7c7f6b46e116f2e6499`，各历史运行
保留其自身 native/source 身份。此次 Git attestation 证明最终提交源码与最终资格源码
一致，不将旧运行改称从新 commit 的 clean checkout 运行。历史报告中的“未提交”
仍描述当时状态，当前仓库身份以本交接为准。

## 后续开发入口

| 主题 | 结论与边界 | 复现入口 |
| --- | --- | --- |
| D2H 归因 | [复核](PHASE10_OPTIMIZER_DOUBLE_WINDOW_DIAGNOSTIC_REVIEW_20261023.md)：排除 API 进入前时间；推断等待不等于驱动锁测量 | [工具](../tools/research/optimizer_d2h_review_20261023/README.md) |
| status-wait | [资格](PHASE10_OPTIMIZER_STATUS_WAIT_20261023.md)：543.6695/547.1705 秒，通过 3% 无实质回退门，未证明独立加速 | [工具](../tools/research/optimizer_status_wait_20261023/README.md) |
| L5 全层补位 | [诊断](PHASE10_OPTIMIZER_LAYER_REFILL_20261023.md)：支持有界槽位实验；常驻诊断不能代替 runtime 性能 | [工具](../tools/research/optimizer_layer_refill_20261023/README.md) |
| L5 有界槽位 | [原型](PHASE10_OPTIMIZER_SLOT_POOL_20261023.md)：32 个 capacity-eight slots，原 cohort 保留 | [工具](../tools/research/optimizer_slot_pool_20261023/README.md) |
| L5 runtime | [资格](PHASE10_OPTIMIZER_SLOT_RUNTIME_20261023.md)：`GO_L5_OPT_IN_RUNTIME_DEVELOPMENT`，限 canonical S2 L5 | [工具](../tools/research/optimizer_slot_runtime_20261023/README.md) |
| L4 容量/原窗口 | [诊断](PHASE10_OPTIMIZER_L4_DIAGNOSTIC_20261023.md)：全层常驻 `STOP_FULL_LAYER_CAPACITY`，原窗口 32-worker 可进入资格 | [工具](../tools/research/optimizer_l4_diagnostic_20261023/README.md) |
| L4 runtime | [资格](PHASE10_OPTIMIZER_L4_RUNTIME_20261024.md)：`GO_L4_ORIGINAL_WINDOW_WORKERS_DEVELOPMENT`，两档均启用 L5 slots | [工具](../tools/research/optimizer_l4_runtime_20261023/README.md) |

先读最终 runtime 结果，再读较早的 proposal/review。历史 L5 `runtime rolling
NOT_YET_GO` 已被限定范围的 L5 runtime 资格推进；其他层和通用 rolling scheduler
仍未资格化。L4 全层常驻 STOP 保持，不应由原窗口 worker-cap 收益推导出 L4 slots
或跨窗口调度 GO，也不能把 host calls 峰值称为实测 GPU overlap。

三个新增选项默认关闭：

```text
FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H
FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS
FASTKPC_DCC_GAMMA_L4_OPTIMIZER_WORKERS32
```

复现须遵循对应 README 的完整 S2、exact-primary coefficient reuse、retained cache/
bypass/eig lanes、status-wait、CPU/thread 和容量约束。L4 保留 34 个原窗口、2,154
setups 和 17,767 targets；L5 保留 955 setups、6,816 targets 及原 2/3/4/8 cohorts。
本次没有扩大这些开关的支持范围或改变默认配置。

## 提交前检查和证据存放

本次仓库收口核对了 98/98 committed source hashes、74/74 committed tool-map
entries 以及本地 native SHA。四项 D2H 归因测试、42 项 Python AST、七项 R parse
和两项 shell syntax 检查通过；没有为提交操作重新构建 CUDA 或运行 GPU campaign。
runtime 的既有数值、配置、资源和限定范围 sanitizer 证据见各资格文档。

receipt 绑定的研究工具保留原字节；D2H build helper 的一处历史行尾空白也保留，
避免改变原 manifest 身份。新增交接文档与路线图变更通过 whitespace 检查。

datasets、native binaries、capture fixtures、Nsight traces 和大型 receipts 继续在本地
artifact 存储，未纳入 Git。相关脚本需要 README 指定的归档输入及父构建，不能将
源码归档等同于无需外部 fixtures 的完整可重现环境。后续候选冻结仍需冻结 producer
上的 clean-checkout 重复性能及对应完整语义资格。
