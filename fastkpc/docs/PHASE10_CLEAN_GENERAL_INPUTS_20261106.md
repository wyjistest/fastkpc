# 干净构建、原版 R 参考与多输入验收（2026-11-06；11 月 12 日收口）

状态：`PASS_CLEAN_ORIGINAL_REFERENCE_QUALIFICATION`。14 次正确性运行、
42 次隔离重复计时和 11 月 12 日的原始 receipt 独立审计均已完成。
七套输入全部通过原数值、图/RNG、资源合同与预注册完整耗时不回退门；
只有 real300 达到速度门，其余六套仍为 `NOT_PROVEN`。本轮冻结已推送提交
`2be6a816d3039a5f9340ba7018166d737ab11d9a`，在独立 detached checkout
全量编译，不复用父对象，不修改生产源码或原始 `kpcalg` 数值实现。
这是逐输入开发验收，未完成 Phase 10 performance promotion。

构建记录 `build/build.json` 确认 44/44 个对象全量重编，217 份源码身份
一致、checkout 无修改，耗时 2,912.895 秒。新 native SHA-256：

```text
d8793173b00c89d971f9210ae4f7020f7bef786dab5e48ec43dfc8fc310d7350
```

该时间是编译耗时，不是算法性能。没有复用父对象或替换主工作区安装。

当前已完成的阶段：

- 七套新的原版 R 参考通过，共 7,882 次 skeleton CI、18,322 次 orientation CI。
- 五项补充 GPU 回归、其他三种方法的 28 个用例 / 792 次 CI、独立 rank-one
  夹具以及 CUDA memcheck/leak check 均通过。
- 历史 351×48 回归的 240,498 次 CI、原 GPU CI/pMax 位模式和全部 34 个门
  通过；对封存原参考最大误差 `3.05425129631942e-12`。285.395 秒为单次
  correctness run，运行时有本轮 CPU 参考任务，不是性能中位数。
- 七套输入均完成 OFF/ON：14 次完整运行共核对 15,764 次 skeleton CI、
  36,644 次 orientation CI。最大 CI 误差 `1.83623949379097e-13`，最大
  pMax 误差 `1.44911860289199e-13`；所有数值、图、定向、PDAG、RNG、
  GPU skeleton authority 与资源门通过。
- 11 月 10 日 09:51:23–15:06:38（Asia/Shanghai）完成 `timing1` 的
  42 次 fresh-process 运行，计时 campaign 共 18,914.588 秒。42 份
  process records 均通过两卡隔离检查；全部数值与行为门通过。
- 11 月 12 日的 `closure1` 独立重读 14+42 份原始 receipts，合计
  63,056 次 skeleton CI、146,576 次 orientation CI，最大误差仍为上述值；
  从原始计时重算中位数、比例、速度门并核对运行顺序。505 项证据文件
  全部通过哈希核对。此审计没有加载 native 或重新执行 GPU/CPU 数值求解。

dense500 的同一次 OFF correctness run 中，完整调用为 2,913.113 秒，
skeleton 为 13.383 秒，CPU orientation 为 2,898.216 秒。这些是同次观察，
不是重复中位数；显示该输入完整流程的成本主要位于原版 CPU 定向。
dense500 ON 的完整调用为 3,009.579 秒，其中 skeleton 10.798 秒、
CPU orientation 2,997.175 秒。GPU0 在 OFF 末段及 ON 运行时有外部用户
任务，两份 process receipt 均记录了共享主机状态。允许继续独占 GPU1 的
正确性验收，但这些单次时间不能用于计算加速或不回退结论。

补充夹具修正保留原失败目录：supplement1 缺少历史 shard；supplement2
缺少 ABI 身份核对使用的历史 manifest；supplement3 的 rank-one 夹具漏写
`route="full_cuda"`，被 authority 门拒绝。修复仅涉及工具和 artifact
路径，最终通过目录为 supplement4，未降低数值或资源门槛。

## 参考与安装身份

主工作区 `kpcalg/R/dcovgamma.R` 已按用户同意的执行路线恢复为原版，SHA-256：

```text
d0620dd9b06f006170b5e142b892c202619296797b2bf9ca842a7c2b5b3b715f
```

此前 scalar-diagonal 补丁只保留在历史 artifact 及独立 rank-one fixture 中。
既有 `numcol_20261106` receipts、closure 与 source attestation 保留原身份，
不改标为原版 R 的运行结果。原版 R 在 rank one 的错误仍是公开边界。

主工作区安装的 native 不由本轮替换，SHA-256 仍为：

```text
2727ef51899688b54da697d0008f99a4908922dbcacec1c30ed58cef09198301
```

## 冻结的验收输入与口径

下表 CI 数为每次运行的数量，误差为两档对原版 R 的最大值；七行均 PASS。

| 输入 | 完整形状 | 默认 numCol | 最大层 | skeleton CI | 定向 CI | 最大 CI 误差 |
|---|---:|---:|---:|---:|---:|---:|
| real179（真实） | 179×8 | 17 | 3 | 320 | 0 | 5.962e-14 |
| real243（真实） | 243×9 | 24 | 4 | 413 | 7 | 8.793e-14 |
| real300（真实） | 300×10 | 30 | 4 | 655 | 2 | 1.449e-13 |
| public500 | 500×50 | 50 | 3 | 2,789 | 89 | 1.836e-13 |
| dense500 | 500×8 | 50 | 6 | 3,559 | 18,000 | 7.862e-15 |
| blocks256 | 256×12 | 25 | 1 | 98 | 32 | 1.810e-14 |
| dense1025 | 1025×4 | 102 | 2 | 48 | 192 | 3.359e-45 |

保留全部输入行列；公共入口真正省略 `numCol`，使用默认 `Inf` 自然停止。
dense500 要求到达 L6；dense1025 要求实际执行 structured 容量之外的 dense
component producer。351×48 仅作为另外一项历史回归。

14 次 correctness 的实际覆盖确认：dense500 ON 的 L3/L4/L5/L6 分别创建并释放
32/32/32/27 个 slots，处理 280/280/168/54 个 targets；level prefill 产生
950 个 components。全部已创建 pool 的原始最低空闲检查点高于 2 GiB，
本组最小值为 7,243,628,544 B（约 6.75 GiB），这是检查点，不是连续峰值。
dense1025 ON 保留 rank 102，structured 关闭，dense producer 实际产生
12 个 components。blocks256 自然止于 L1，未执行深层 prefill/slots。

独立 `check_receipts.R` 从原始 receipts 重新匹配 R 参考值，核对 CI/pMax、
root identity rejection=0、reuse fallback=0、arena 分配/释放相等、
live bytes=0，以及 slots 创建/销毁和容量检查点。
本组 OFF/ON 也不是统一逐位相同：五套出现至多 `6.21724893790088e-15`
的 CI/pMax 差异，blocks256 与 dense1025 逐位相同。它们全部满足预定
`1e-10` 数值合同；不能把 canonical 的旧 GPU 位模式不变推广为所有输入逐位不变。

CPU 参考由新的原版 R `kpcalg`、`pcalg`、mgcv 与 RSpectra 运行产生，
不以 native 数值后端替代。逐 CI 与有限 pMax 的绝对误差门为 `1e-10`，
同时核对任务、删边、图、sepset union、计数、完整定向轨迹、PDAG、RNG、
GPU authority 与资源闭环。原版 pcalg 的 node-major 原始调用顺序与 native
调度不同：native 顺序使用独立 R 值进行 host-only replay 验证，不能称为
原始 pcalg 调用顺序逐项相同。

已执行的冻结计时协议：逐输入按 `A1 C1 C2 A2 A3 C3` 运行 OFF/ON。
A 为普通 exact-primary/coefficient-reuse/retained GPU，C 增加现有
prefill/structured/slots/work-pair/postback 开关。采用 GPU1、CPU 0–19、
数学库单线程；正式计时要求两张 GPU 都无外部计算任务。

主时间是完整公共 WAN-PDAG 调用，另列 skeleton 与 CPU orientation。
每次 fresh R process，只预热无关的 80×3 数据，然后清空输入缓存；artifact
读写与核对位于计时边界外。逐输入中位数比值 `C/A <= 1.05` 才是不回退；
只有 `C/A <= 0.95` 且最慢 C 快于最快 A 才登记速度收益。各阶段中位数
不能相加作为同次完整调用账本。

2026-11-06 18:27（Asia/Shanghai）的历史正确性收口预检记录 GPU0 上仍有外部
Python 进程 `159105`，两卡空闲条件不成立，未启动 timing1。GPU1 上的
本轮任务在该预检时均已结束。此阻塞随后解除，11 月 10 日按原协议完成
42 个 fresh processes；未将此前共享主机下的 correctness 时间混入计时。

首个 timing run 之前另修正了一处未执行过的预热分支：改用无关 80×3
数据的 full-CUDA WAN-PDAG 入口，避免 skeleton facade 的 legacy 默认
route，并显式核对已加载 native 路径。此前全部为 correctness phase，
没有计时结果受到该分支影响。每个已完成阶段保留原工具快照及哈希。

## 隔离重复性能结果

下表均为完整公共 WAN-PDAG 调用的三次中位数，单位秒。A、C 是两种 GPU
配置；不表示 GPU 对原版 CPU 的加速比。判定沿用预注册的 `C/A <= 1.05`
不回退门，以及 `C/A <= 0.95` 且最慢 C 快于最快 A 的速度门。

| 输入 | A OFF | C ON | 耗时变化 | 不回退 | 速度判断 |
|---|---:|---:|---:|---|---|
| real179，179×8 | 1.040 | 1.053 | +1.25% | PASS | NOT_PROVEN |
| real243，243×9 | 2.096 | 2.028 | −3.24% | PASS | NOT_PROVEN |
| real300，300×10 | 3.353 | 2.893 | −13.72% | PASS | PROVEN |
| public500，500×50 | 24.986 | 25.265 | +1.12% | PASS | NOT_PROVEN |
| dense500，500×8 | 2798.063 | 2797.599 | −0.02% | PASS | NOT_PROVEN |
| blocks256，256×12 | 1.496 | 1.493 | −0.20% | PASS | NOT_PROVEN |
| dense1025，1025×4 | 230.835 | 231.239 | +0.18% | PASS | NOT_PROVEN |

real300 的最慢 C 为 2.930 秒，仍快于最快 A 的 3.299 秒；real243 虽然
三次 C 均快于 A，但中位数降幅不足 5%，仍保持 `NOT_PROVEN`。
public500 的三次 C 均略慢于 A；1.12% 的中位数回退在预定容限内，不能
表述为提速。不同输入不合并成一个总体加速比。

阶段中位数单独列出；它们不能相加成某次完整调用的精确账本：

| 输入 | A skeleton | C skeleton | A CPU orientation | C CPU orientation |
|---|---:|---:|---:|---:|
| real179 | 1.009 | 1.019 | 0.003 | 0.003 |
| real243 | 1.726 | 1.661 | 0.343 | 0.340 |
| real300 | 3.096 | 2.631 | 0.233 | 0.233 |
| public500 | 10.593 | 11.001 | 14.360 | 14.235 |
| dense500 | 14.362 | 11.146 | 2784.555 | 2786.425 |
| blocks256 | 0.267 | 0.267 | 1.210 | 1.206 |
| dense1025 | 1.367 | 1.335 | 229.455 | 229.884 |

dense500 的 skeleton 中位数减少 22.39%，但完整调用由原版 CPU 定向主导，
完整耗时没有达到速度门；dense1025 也主要耗时于 CPU 定向。
这支持后续按实际输入分别诊断，而不是把某一组的收益推广到全部输入。

最终 receipt 审计还核对了已创建 pool 的容量检查点和释放，56 次运行的
最低 pool 空闲检查点为 7,192,903,680 B，仍高于 2 GiB 门槛。
这是原始检查点证据，不是连续显存峰值。审计工具核对的原始 R 值、
task/deletion order、sepsets、最终 PDAG、RNG、authority、stable-SVD、
cache/reuse 状态及资源闭环均通过；本轮未改变数值合同。

## 复现入口

工具与预注册协议位于
[`clean_general_inputs_20261106`](../tools/research/clean_general_inputs_20261106/README.md)。
本地证据根目录为
`fastkpc/artifacts/full_cuda_ci/clean_general_inputs_20261106/`，包含独立
`checkout/`、构建日志、冻结输入、CPU references 和后续 GPU receipts。
大型 native、原始真实输入及 receipts 仍是本地材料。

11 月 6 日已验收目录为 `references1`、`qualification1`、`supplement4`，
当时独立封存为 `correctness_closure1`。该历史 manifest 含 275 项证据文件，并验证 217 份
源码的 Git 字节身份、44 个对象、generated includes、native、原版 R
以及未替换的主安装。失败夹具记录也保留在 manifest 中，未算作通过。

```text
correctness summary SHA-256
6ecbacd1ed31a0384ec2a87bbe3124203f683b3fbb9a6c5693762f3cddca4e97

correctness manifest SHA-256
5bdbd0018da12d686e4272f7795871d952588e22dd7a6a249867d2c5e1084b66
```

该正确性阶段的身份索引见
[`PHASE10_CLEAN_GENERAL_INPUTS_ATTESTATION_20261106.json`](PHASE10_CLEAN_GENERAL_INPUTS_ATTESTATION_20261106.json)。
它保留当时的计时未运行状态和 GPU 预检，不改写历史收口。

11 月 12 日最终 `closure1` 将 `timing1` 纳入，复核全部 56 份原始 receipts，
重算逐输入性能判定，并封存 505 项文件。217 份源码与 Git、44 个对象、
两个生成文件、构建日志、native、原版 R、未替换的主安装及历史封存哈希
均再次核对。最终审计工具另存快照；旧运行工具快照仍绑定原执行字节。

```text
final summary SHA-256
fa72661739dc92d61a5cd6c6000e679ba2da65bba7ea602afd49055199aad7e7

final manifest SHA-256
89306730984f8be645e14a2bbbbdca16598c31a530c956ff30411444763245a0

raw receipt audit SHA-256
c020792233748094ca0c5b3e3ad9c7303947b7d544cc9c48a73a08a69a8ba2c4
```

当前身份及逐输入结果索引见
[`PHASE10_CLEAN_GENERAL_INPUTS_ATTESTATION_20261112.json`](PHASE10_CLEAN_GENERAL_INPUTS_ATTESTATION_20261112.json)。
旧 `PHASE10_SOURCE_ATTESTATION_20261106.json` 和历史 numCol closure 保留原状。
复核命令见工具 README；已完成的计时无需重跑才能审计。

本轮不冻结 candidate、不打开 holdout、不做正式 promotion，也不把逐输入
开发实验合并成普遍性能承诺。Phase 10 `ACTIVE`、candidate `NOT_FROZEN`、
holdout `SEALED`、recommended `false` 保持。
