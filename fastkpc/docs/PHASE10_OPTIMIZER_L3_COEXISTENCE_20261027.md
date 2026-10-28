# L3 实际 replacement coexistence：预声明增量门 STOP

日期：2026-10-27。承接 [L3 captured-cohort](PHASE10_OPTIMIZER_L3_SLOTS_20261027.md)
和[复核](PHASE10_OPTIMIZER_L3_SLOTS_REVIEW_20261027.md)。本次只做诊断，不接入生产默认配置。

正式状态：**`STOP_L3_REPLACEMENT_POOL_INCREMENT`**。32 槽模式在 optimizer 完成后的
预声明物理增量门停止，尚未进入 Stage A。这里的 STOP 针对本轮容量合同；
`GO_CAPTURED_L3_SLOTS` 仍成立，生产 runtime 仍为 `NOT_YET_GO`。
没有新的完整方法性能结果，canonical development 中位数继续为 **368.851 s**。

## 预声明合同与实际替换

完整生产源码闭包、native 和默认配置保持不变。另建可审计的诊断 host overlay：

- W 使用原来的 64 个 L3 窗口，保存真实 Stage A 参考。
- S 以 32 个 capacity-33 槽位替换 4,065 个原 prepared 请求，保持原 cohort 和
  请求身份，使用实际 coefficient arena；计划沿用原 Stage A cohort/Gram 消费顺序。
- n=351、q=28、三个 penalty、总 root rank=24、target count 2–33。
  每个 S 请求均检查原 prepared 为空，并核对全部捕获几何与 Y；没有同时保留旧窗口 handle。
- 不增加整层 GPU 输出备份区。optimizer shadow 和 Stage A shadow 仅用于诊断对照，
  不提供 CPU 数值 authority。W 诊断 D2H 为 **653,267,712 bytes**；S 完成的 optimizer
  shadow 对应 **320,487,520 bytes**，尚未执行 Stage A shadow。
- 保持真实 S2、11 GiB component cache、bypass、8 eig lanes、L4/L5/L6 slots、
  properties cache/status wait；GPU1 独占、两卡无外部 compute 任务、affinity 0–19、
  connections=32、BLAS/OMP/MKL 单线程。每次运行均保存进程与 GPU 监控。

在任何 GPU 运行之前固定：

```text
L3 池创建和 optimizer 完成时的物理增量 <= 1.5 GiB
基准 = 真实 coefficient arena 已 reserve 后、建池之前
所有记录的 L3 检查点及 10ms 样本的 free >= 2 GiB
worker 期间受审计 CUDA 资源建销 = 0
数值/身份/所有权/资源硬门不得放松
```

1.5 GiB 是本次 L3 的独立资格额度，现有 L4/L6 的 1 GiB 守卫没有改变。
物理增量按这个基准计算，包含建池之后首次执行带来的公共成本，不能事后扣除再宣布通过。
样本与检查点不保证未观测瞬间的连续峰值。

## W 完整通过，S 在容量门停止

W 完整 canonical 通过 **67/67** 硬门：240,498 个 CI 和全部 pMax 位值不变；
最大 CI/pMax authority 误差均为 **`3.05425129631942e-12`**。task、deletion、
adjacency、sepsets、计数、RNG、Gram 顺序、authority 和资源门全部通过。
W 的 `377.585s` 含诊断，只用于说明运行完成，不作为性能证据。

W 的全部 **4,065 setups / 54,878 targets** optimizer 输出与冻结 capture 逐位一致；
**5,573 个真实 Stage A cohorts / 54,878 targets** 保存为原顺序参考，同时检查了
coefficient arena gather 与原 selected coefficients 的逐位身份。
W 在 L3 期间观测到最小 free **3.048645 GiB**。

首个 S 在 `pool.after_work()` 的容量守卫退出。保留该失败后，另做一次**仅增加日志**的
S 重放：同一 overlay、同一输入、同一 32 槽形状、同一 1.5/2 GiB 门，无新增预热。
它再次命中同一守卫，记录到以下精确检查点数值：

| 时点 | free | 相对建池前的物理增量 |
|---|---:|---:|
| arena reserve 后、建池前 | 5.435364 GiB | 0 |
| 32 槽创建完成 | 4.300598 GiB | **1.134766 GiB** |
| optimizer workers 全部结束 | 3.177551 GiB | **2.257813 GiB** |

应用可见的池分配为 **880,205,568 bytes / 839.429443 MiB**。optimizer 期间：

```text
完成且逐位一致的 setups / targets       4,065 / 54,878
新增应用可见 device bytes              0
新增物理占用                          1,205,862,400 bytes = 1,150 MiB
worker 期间受审计 CUDA 资源建销          0
```

全部 floating/integer optimizer 状态、selected-SP、coefficients/fitted/residual、
route/rank/info 和轨迹通过冻结 capture 的逐位检查。新增物理占用没有出现在应用分配账本中。
命中的是 **1.5 GiB 增量门**；该检查点仍有 **3.177551 GiB** 空闲，未发生 OOM。
上述精确值属于日志重放，不冒充首个失败 S 未记录的具体数值。

异常展开后，应用可见 allocations、bytes、streams、events、BLAS handles 全部为 **0**。
S 没有发布结果，没有回退 CPU。S 的 coefficient detach 已实际执行，但保存后的 gather、
Stage A 输出和完整 CI/pMax/图资格尚未到达，**不能声称它们已通过**。

## 新发现：首次执行成本与独立原型的测量基准不同

独立 captured-cohort 容量实验先执行并验证一个 33-target 原生请求，再测量池增量，
将首次执行后保留的 **1,152 MiB** 公共 context 成本另列。真实 canonical 没有增加这次预热。

W 的检查点也提供了关键旁证。以 W 的 L3 起点为基准，用
“物理占用增长减应用可见 bytes 增长”计算，在每个原窗口 prepared 释放之后：

| W 时点 | 应用账本之外的物理增长 |
|---|---:|
| 首个窗口完成并释放 prepared | 1,161.067 MiB |
| 64 个窗口关闭检查点的范围 | 1,158.679–1,164.025 MiB |
| 最后一个窗口关闭 | 1,162.362 MiB |

这些检查点的 BLAS handle 数均回到 L3 起点的 38。原窗口路径同样保留了约 1.13 GiB 的
额外物理占用。源码中 conditioning size <=2 使用一个 smooth，L3 才进入这里的
multi-penalty 执行形状；见 `full_cuda_ci_native_setup.cpp:1194`、`:1333`。

因此，证据支持先区分**共同首次执行成本**与**槽位自身成本**。具体 allocator、
device local memory、库或 driver 的贡献尚未分离，不能把它直接归因于某一种分配机制。
也不能将 2.258 GiB 称为相对原生产路径新增的净显存成本，或据此证明 32 槽无法容纳。

本轮容量门仍然失败。不能在看见结果后把公共成本从原门槛中扣掉，也不能把 1.5 GiB
改大后复用同一 receipt 宣布 PASS。S 未到 Stage A，真实联合生命周期资格仍未完成。

## 决策与下一项

保留 captured-cohort GO；停止当前合同下的 runtime 推进。下一项仅建议对首次执行的
非应用物理占用做窄归因，并据此为**新的**实验预声明公共成本、池增量和联合余量的
口径。不要立即改变 32 槽数量、做分级容量池或隐藏一次额外预热；若将来加入预热，
其成本与显存必须计入完整生命周期与 outer。

只有新的容量合同通过实际 Stage A 共存及完整语义门，才考虑默认关闭的生产接入、
故障资格和重复 outer 计时。本轮没有生产源码/native/default 修改，没有 commit/push。
Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 保持不变。

## 身份、工具与证据

工具：`fastkpc/tools/research/optimizer_l3_coexistence_20261027/`。
证据：`fastkpc/artifacts/full_cuda_ci/optimizer_l3_coexistence_20261027/attempt2/`。
`attempt1` 是 GPU 运行之前被校验拒绝的构建生成尝试，也已保留。

生产 native：`313d9f3cc6dc8512b5624aae84cc459c6083ff04fd7883c32812fdb081510a1b`。
诊断 host overlay：`14f6a8b1d7ae152b969c3e98056bd34e93f49906fad559b7fd82011ab800eef9`。
其 rebind 适配对象的 14 个 GPU 函数完整指令字均与生产对象一致；overlay 不导出或替换
生产 small64 optimizer/finalization 数学入口。原生源码闭包与冻结 L3 工具哈希保持一致。

| 证据 | SHA-256 |
|---|---|
| protocol.json | `1f23b2828592e32b875a416500e1958f05db8b9d81a85231a1f3ba807c2d475d` |
| build.json | `0b79e0f9a861cdb661826acd8b1b9a0813b106f73eba89b04d77fcd082a73203` |
| W/stage-a.bin | `dd7ac0fb31d278c763f9a7780cae0e1fbc800364fc456312d63e3a532f200a17` |
| failure-attribution/closure.json | `969387d0fee218b46e98f10f368b26ac419c9993a756dcd3270a9759e2ba2f78` |
| closure.json | `d3cd26dd3ae8d8ea838d3b87045b5fbeeb0f2fc413679a5d5f644c0c3b5d650b` |
| manifest.json | `6baa0fde7262bd61e5341bde8efe61a8b1e0d5b3c4fc89d74a686ffa209a3da1` |

manifest 绑定 50 个 artifact、工具及早期构建尝试。失败原始日志、进程身份、
GPU 样本、W 完整 receipt、Stage A 参考、S 失败归因及生成 overlay diff 均保留。
