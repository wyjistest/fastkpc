# L3 实际替换共存：分点原始容量合同

日期：2026-10-27。独立诊断工具：
`fastkpc/tools/research/optimizer_l3_coexistence_v2_20261027/`；证据：
`fastkpc/artifacts/full_cuda_ci/optimizer_l3_coexistence_v2_20261027/attempt1/`。

结果：**`GO_L3_REPLACEMENT_COEXISTENCE_SPLIT_CONTRACT`**。新合同下，真实
32-slot 替换路径完整通过 optimizer、Stage A 和 canonical 硬门。旧
`STOP_L3_REPLACEMENT_POOL_INCREMENT` 不变；本轮是运行前另行声明的新协议，
门槛依据旧实验提出，不追认旧 receipt。生产 runtime 集成仍为 `NOT_YET_GO`，
本轮不作性能 GO。

## 合同与范围

基线位于真实 coefficient arena reserve 之后、pool 创建之前。

| 测量边界 | 原始测量硬门 |
|---|---:|
| pool 创建完成，任何 L3 launch 之前 | 相对原基线增量 ≤1.5 GiB |
| optimizer 完成，Stage A 之前 | 相对建池完成增量 ≤1.25 GiB |
| 同一个 optimizer 完成边界 | 相对原基线总增量 ≤2.75 GiB |
| 整个 L3 optimizer + Stage A | 每个 checkpoint 和采样点的 free ≥2 GiB |

Stage A 的原始总增量必须记录，但**没有 2.75 GiB 总增量上限**。W/S 最小
free 的比较仅作描述，不设 noninferiority 门，不据此宣称 S 显存需求更低。
没有 stack 模型扣减、基线重置、新 warmup 或 `cudaDeviceSetLimit`。

限定 canonical 351×48、default-Inf、S2 exact-primary；L3 n=351、q=28、
三个 penalty、total root rank=24、原 cohort 2–33 targets、32 个 capacity-33
slots、decomposition trace 关闭。超出 envelope 时诊断 slot 路径 fail closed。
采用实际 coefficient arena，未额外分配全层 GPU 输出 arena。

本轮只编译独立 host overlay：复用旧 `bridge.o/rebind.o`，生产数值入口和
native 均保持不变。链接后 14 个 GPU 函数的 instruction words 与冻结生产
参考逐一相同。15 项容量边界检查通过，包括阈值本身、一字节超限，以及
Stage A 总增量超过 2.75 GiB、但 free 恰好 2 GiB 时仍符合合同的情形。
生成器确认 L4/L6 的 1 GiB 守卫与生产实现逐字相同。

## 执行条件

两次均为 fresh R process，GPU1 独占，GPU0 外部 Python PID 1918904 在开始前
已声明允许。CPU affinity 0–19、BLAS/OMP/MKL 单线程、connections=32；
原 component cache / bypass / 8 eig lanes、L4/L5/L6 与 device-properties cache
配置不变。采集两卡进程与资源记录；这些是共享主机上的语义/容量诊断，
不是隔离性能证据。

10 ms 是采样线程请求的间隔；实际调度/query 间隔保留在 CSV 和 closure 中。
W/S 分别记录 **11,986 / 7,903** 次观测，其中周期采样 **6,347 / 2,322** 次；
最大周期采样间隔分别为 **42.379 / 13.672 ms**，全部观测点满足 free 门。
通过门只能保证观测点满足合同，不证明未采样瞬间没有更低的 free。
诊断 shadow D2H 单列计数，仅供比对，不作 CPU 数值 authority 或 fallback。

## 实际 L3 容量与共存

S 的原始基线 free 为 **5,836,177,408 B**；建池完成为 **4,617,732,096 B**；
optimizer 完成为 **3,411,869,696 B**。原始测量如下，未扣除任何首次执行成本。

| S 测量 | bytes | GiB | 门 |
|---|---:|---:|---:|
| 建池增量 | 1,218,445,312 | 1.134765625 | ≤1.5 |
| 建池完成至 optimizer 完成新增 | 1,205,862,400 | 1.123046875 | ≤1.25 |
| optimizer 完成时总增量 | 2,424,307,712 | 2.2578125 | ≤2.75 |
| Stage A 最大观测原始总增量 | 2,424,307,712 | 2.2578125 | 只记录 |
| 整个 L3 / Stage A 最小观测 free | 3,411,869,696 | 3.177551270 | ≥2 |

本轮 S 在 Stage A 的观测点没有进一步降低 free；这不排除采样间隙中的瞬时变化。
W 的 L3 最小观测 free 为 **3,273,457,664 B / 3.048645020 GiB**，仅作描述性
对照。不得由这一对数值推出 S 的生产显存需求更低或显存 noninferiority 成立。

S 的应用 pool 为 **880,205,568 B / 839.429443 MiB**。建池完成至 optimizer
完成，应用 bytes、allocations、streams、events、BLAS handle 数均不变，worker
资源建销计数为零。pool 关闭后，全部 pool 应用资源增量恢复；这不要求 CUDA
context 的首次执行保留内存立刻退回基线。

W/S 都完成全部 **4,065 setups / 54,878 optimizer targets** 的 captured 输出
逐位比较。S 检查全部 geometry/Y 与原 cohort 身份，并经真实 coefficient arena
detach/gather 完成原序 **5,573 Stage A cohorts / 54,878 targets**：coefficients、
fitted、residual、route/rank/info 与 W 全部逐位一致，W Stage A 二进制与冻结参考
SHA 相同。每个模式的诊断 shadow D2H 为 **653,267,712 B**，不计作生产 authority。

## 完整方法资格与限制

W/S 各自通过全部 **67/67** canonical 门：

- **240,498 / 240,498** CI 与全部 pMax 对冻结 reference 逐位一致；changed CI、
  changed pMax、alpha flips 均为零。对保存的 mgcv authority，最大 CI/pMax
  绝对误差仍为 **3.05425129631942e-12**，满足 `1e-10`。
- task identity、删除决策与轨迹、骨架、sepsets、n.edgetests、RNG endpoint、
  optimizer trajectories、physical cohorts 和 Gram key/order 一致。
- exact-primary、exact-fixed stable-SVD、cache metadata、各层 ledger、arena
  ownership 与 L4/L5/L6 守卫通过；CPU 数值 authority/fallback 和生产 large
  residual/component D2H 为零。诊断 shadow 是上述单列的测试读取。
- 最终应用 allocations、bytes、streams、events、BLAS handle 和 worker
  resource mutations 均为零；生产资源 snapshot 也归零。

W/S outer 观测为 **383.627 / 342.635 s**，仅是共享主机、带诊断 shadow 的各一次
运行，不是隔离重复中位数。不能把差值从 accepted median 中扣除。当前 accepted
canonical development median 继续是 **368.851 s**。

这次没有新的真实数据、final-PDAG、clean-producer 或 promotion 资格。
`GO_CAPTURED_L3_SLOTS` 与旧容量 STOP 保留；新 GO 只关闭实际替换共存这一门。
Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 不变。

下一项推荐是**默认关闭的 L3 runtime 集成与资源/数值资格**：保持本轮 envelope、
三项分点原始增量及整个 L3 的 free 守卫，保留 L4/L6 原门。通过后，另行预声明
并执行隔离的完整方法重复计时，才决定性能 GO；不再需要重复 stack 归因。

## 身份

| 项目 | SHA-256 |
|---|---|
| 生产 native | `313d9f3cc6dc8512b5624aae84cc459c6083ff04fd7883c32812fdb081510a1b` |
| 新 diagnostic overlay | `6c716f3f312843e07bb9bb866c15eefcfae04d5a73d1f3fad4522f1607063639` |
| protocol | `355b36b899e37b055e2a9c9990dfb9f2555b84b4a8b5cab0e4ab200b9ac36338` |
| build | `36dc8705def17af06e828888a594194aee5ab37b7c9650c61b4c67f4cc636593` |
| 原 Stage A reference | `dd7ac0fb31d278c763f9a7780cae0e1fbc800364fc456312d63e3a532f200a17` |
| closure | `1596aadf60da4fd9ec7421f1817303d4c5720e0c3820ecb89751691b00007dec` |
| manifest | `9a4b626a01ae155f74536edc7427da34033643d6078cf209e2dd967aee10f1cd` |

101 个生产输入与 native 在构建、每次运行前后核验。旧共存和 stack attribution
证据保留；没有生产源码/默认配置修改，没有 commit 或 push。我方 GPU 任务和
监控进程均已结束；GPU0 的外部任务未被干预。复现命令见新工具目录 README；
机器账本见 `protocol.json`、`build.json`、`{W,S}/probe.json`、`{W,S}/report.json`、
`{W,S}/memory-samples.csv`、`{W,S}/checkpoints.csv`、`{W,S}/cleanup.json`、
`linked-device-code-audit.json`、`closure.json` 和 `manifest.json`。
