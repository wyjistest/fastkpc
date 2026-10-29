# Phase 10 开发交接：properties cache、L3/L6 slots 与 Stage B work pair

后续交接：见 [2026-10-29 vector-postback handoff](PHASE10_DEVELOPMENT_HANDOFF_20261029.md)。
当前 runtime/工具为 `c469ac5`，accepted opt-in development median 为 290.491 秒。
本页及原 attestation 保留 10-28 的历史身份；证据仍为 source-closure development。

日期：2026-10-28。本次将已验收的 runtime、验证工具与复核文档收口到 Git。
当前 accepted opt-in development median 为 **311.440 秒**，对应
**GO_DEFAULT_OFF_WORK_PAIR_RUNTIME**。新增选项保持默认关闭；Phase 10 **ACTIVE**、
candidate **NOT_FROZEN**、holdout **SEALED**、recommended **false**、L8 optimizer **STOP**。

## 提交身份与证据边界

| 提交 | 内容 |
| --- | --- |
| `1d8a11d75d5f4beb2b0b70fdddf13bf12adb25aa` | 默认关闭的 device-properties cache、L3/L6 有界槽位、fixed-VT work pair 与入口计数 |
| `2920ab8182c9f0ad3edf1c574db4e64a438dfd39` | 150 个研究工具/说明文件与两个 L3/L6 配置回归测试 |

[Git 源码 attestation](PHASE10_SOURCE_ATTESTATION_20261028.json) 核对了全部 **104 个生产输入**
与 runtime commit 的 Git blob，以及全部 **152 个新归档验证文件**与 tools commit。
最新 runtime manifest 的 **341 项**与 deep manifest 的 **168 项**重新计算哈希后均通过；
其中 runtime 的 12 项工具映射也与 tools commit 一致。

当前 native SHA-256 保持为：

```text
12acc3cd01cce2dcac5f2bfb5ec02f27056e92154cbab2534c7f03bdc22caa7d
```

本次没有重新编译或运行 GPU campaign。最后一次构建是基于已验证父对象的增量 source closure，
仅重编 structured eig 与 one-call 两个对象并 relink；其他对象逐个保持父构建身份。
原始性能运行的 HEAD 仍为 `d27e59b5d7baaabd60b28a419fa0d0bd52c76b53`。
Git attestation 证明提交源码与已测源码相同，不将原 receipts 改称新 commit 上的
clean-checkout 运行，也不升级为 promotion evidence。

各历史报告/复核中的“未提交”“未 push”描述当时状态，当前 Git 身份以本交接为准。
[前次 L4 attestation](PHASE10_L4_SLOT_SOURCE_ATTESTATION_20261024.json) 与原始失败、
中断、共享主机及隔离续跑证据全部保留。历史工具版本仍按各自封存快照核对；不宣称
当前工具与每一份历史工具映射都相同。

## 性能账本

以下是四个独立 campaign 的同 native 对照，不把不同 campaign 的中位数差相加：

| 独立改动 | OFF 中位数秒 | ON 中位数秒 | 本 campaign 下降 |
| --- | ---: | ---: | ---: |
| device-properties cache | 408.882 | 392.416 | 4.03% |
| L6 bounded slots | 395.741 | 368.851 | 6.79% |
| L3 bounded slots | 370.553 | 329.078 | 11.19% |
| Stage B fixed-VT work pair | 328.464 | **311.440** | **5.18%** |

最后一项的正式完整方法收益为 **17.024 秒**；Stage B 中位数
**144.350986 → 127.801689 秒**，其 **16.549297 秒**差值单列为阶段归因。
最慢 ON 的 318.820 秒仍低于最快 OFF 的 327.667 秒，三个预声明性能门均通过。
L7/L8 的波动位于 optimizer host 计时边界，与旧诊断相符；本轮未重新采集 trace
确诊机制，也不扣除这些波动或重开 L8。

六次最新运行各覆盖 **240,498 CI**：CI/pMax 位模式、task/order/deletion、skeleton、
sepsets、n.edgetests、RNG、原物理 cohort 与 Gram 顺序一致。对保存的 mgcv authority
最大误差为 **3.05425129631942e-12**；CPU 数值 authority/fallback、生产 residual/component
payload D2H、route/status mismatch 和 dense fallback 为零，最终 application resources 归零。
没有新增真实数据或最终 PDAG 资格。

L3 creation/growth/raw-total 容量门及 L4/L6 守卫保持不变；L6 最紧检查点有
**105.125 MiB** 的门槛余量。检查点和外部采样不是连续峰值证明。
work-pair GPU 数值证据限 canonical n=351 与已列合成案例；实际 SM occupancy 尚未测量。

## 后续开发入口

| 主题 | 当前结论与入口 |
| --- | --- |
| properties cache | [完整资格](PHASE10_DEVICE_PROPERTIES_CACHE_20261025.md)、[复核](PHASE10_DEVICE_PROPERTIES_CACHE_REVIEW_20261025.md)：成功查询按逻辑设备缓存，默认关闭；共享主机补跑不混入隔离中位数 |
| 晚层 optimizer | [实验](PHASE10_LATE_OPTIMIZER_EXPERIMENTS_20261025.md)、[复核](PHASE10_LATE_OPTIMIZER_EXPERIMENTS_REVIEW_20261026.md)：保留 L8 STOP 与 trace 因果边界 |
| L6 slots | [runtime](PHASE10_OPTIMIZER_L6_SLOT_RUNTIME_20261026.md)、[复核](PHASE10_OPTIMIZER_L6_SLOT_RUNTIME_REVIEW_20261026.md)：32 个 capacity-5 slots，原 245 setups/1,148 targets 与 Stage A 顺序 |
| L3 slots | [runtime](PHASE10_OPTIMIZER_L3_SLOT_RUNTIME_20261028.md)、[复核](PHASE10_OPTIMIZER_L3_SLOT_RUNTIME_REVIEW_20261028.md)：32 个 capacity-33 slots，原 4,065 setups/54,878 targets，限定 canonical S2 |
| L3 容量证据 | [原 STOP](PHASE10_OPTIMIZER_L3_COEXISTENCE_20261027.md)、[stack 归因](PHASE10_OPTIMIZER_L3_STACK_ATTRIBUTION_20261027.md)、[独立 v2 合同](PHASE10_OPTIMIZER_L3_COEXISTENCE_V2_20261027.md)：不作 stack 扣除或 baseline 重置 |
| Stage B 归因 | [activity 诊断](PHASE10_STAGE_B_ATTRIBUTION_20261028.md)、[work 审计](PHASE10_STAGE_B_DBDSQR_WORK_AUDIT_20261028.md)：host 等待不能另算收益池；实际 occupancy 未测 |
| fixed-VT work pair | [隔离资格](PHASE10_STAGE_B_WORK_PAIR_20261028.md)、[runtime](PHASE10_STAGE_B_WORK_PAIR_RUNTIME_20261028.md)、[最新复核](PHASE10_STAGE_B_WORK_PAIR_RUNTIME_REVIEW_20261028.md)：完整 canonical 默认关闭集成 GO |

四个新增选项均默认关闭，复现须遵循对应工具 README 的完整配置：

```text
FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE
FASTKPC_DCC_GAMMA_L6_OPTIMIZER_SLOTS
FASTKPC_DCC_GAMMA_L3_OPTIMIZER_SLOTS
FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR
```

properties cache 在首次使用时固定模式；work pair 在 plan 创建时固定模式与入口/shared 配对。
L3/L6 slots 的合法配置和容量门没有扩大。不得以默认关闭的 canonical GO 推导任意形状
默认开启、完整 Phase 10 结束或推荐路线成立；180/120 秒门仍未通过。

## 快照位置与仓库检查

本次源码快照在
`fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_runtime_20261028/runtime/build-source/`；
父源码快照在
`fastkpc/artifacts/full_cuda_ci/optimizer_l3_slot_runtime_20261028/build-source/`。
父 native 为本次 `runtime/previous-native.so.gz`。deep 阶段的六个历史工具版本保存在
本次 `runtime/deep-tool-source/`。原实施报告已同步纠正父源码快照位置。

本次提交检查通过 **70 个 Python AST、12 个 R parse、3 个 shell syntax** 检查。
一个已测生产头文件和 12 个封存工具保留末尾空行，避免改变源码哈希；其他源码/工具
whitespace 检查通过。文档改动单独检查。

数据集、native binaries、captures、profiler traces 和大型 receipts 留在本地 artifact 存储。
工具仍需要对应 README 声明的输入与父构建；源码入库不等于完整外部 fixtures 已发布。
今后冻结候选仍需冻结 producer 上的 clean-checkout 重复性能与完整语义资格。
