# Structured eig：真实残差、有界批次与缓存共驻留诊断

日期：2026-10-20。补充
[大批次合成输入测量](PHASE10_STRUCTURED_EIG_SCALING_20261020.md)。

## 决策

```text
真实残差分量级数值资格                 PASS（本次样本范围）
256/640 分量吞吐与 11 GiB 容量共驻留   GO
隔离性能剖析 / 内存检查                COMPLETE
完整方法 / 整层预计算生产集成           NOT YET GO
```

真实输入上的阶段优势仍然成立。当前证据支持下一步做按层 component
生命周期与有界预计算原型；不支持直接替换生产 eig，也不能预测 canonical
总耗时。Phase 10、candidate、recommended route 和 sealed holdout 均不改变。

## 输入与数值身份

生产 native 保持为：

```text
fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50
```

从该 native 的 22 列、level 3 子集回执选择 320 个不同 S 的真实任务：
64 个最接近 alpha=0.1 的 development pair，再从其余 S 按任务顺序等距选择
256 个。每个 cohort 两 target，共 640 个不同 `(target,S)` 分量，n=351、q=28。
这是独立两-target重放，不宣称逐字节捕获了原 one-call 的全部运行状态。

调用现有 GPU capacity dispatcher，执行 exact-primary optimizer 和 exact-fixed
stable-SVD evaluator，下载显式 diagnostic residual shadow。CPU 不产生 residual
或替代任何 GPU 计算；kpcalg 仅作离线 oracle。全部 target 的 solver info=0、
stable-SVD route成立、CPU optimizer/solve/fallback计数为0，生产资源计数闭合。

相同 residual 的 kpcalg p 与原 GPU task 最大差为 `1.107309e-13`。
两种 eig 从同一份 little-endian FP64 文件读取输入；artifact 绑定 residual、
task、数据、源码、native 与研究库的 SHA-256。

本轮主数值结果：

| 项目 | 结果 |
|---|---:|
| 真实同-S CI pairs | 320 / 320 PASS |
| 其中 p 在 [0.05, 0.15] | 25 |
| 结构化 p 对 kpcalg 最大绝对误差 | 1.0948186801584825e-13 |
| 结构化 p 对原 GPU task 最大绝对误差 | 6.467049118441537e-15 |
| statistic / mean / variance 绝对与相对门 | PASS |
| alpha=0.1 翻转 | 0 |
| 各档三次重复、共驻留前后输出字节 | exact |
| 原型旋转数最大值 | 110,964 |
| 固定旋转容量 | 170,662 |
| 容量溢出注入 | fail closed，未发布输出 |

另外每个规模抽取 2,048 个跨 cohort 的诊断配对候选，去掉自配对后，对相同
输入的 GPU dense 分量作 statistic/moment/p 比较，全部通过。它们不是新增
canonical CI 或 graph trace 行。所有 pair/gamma 比较均为离线诊断，尚未将
结构化分量接入生产 GPU pair/gamma consumer。

## 吞吐与容量

单张 RTX 4090（GPU0），connections=32，其他 GPU 无计算任务；CPU 数学库
线程数为1。每档一次预热、三次重复取中位数。共驻留档和非共驻留档交替
dense/structured顺序。结构化计时包含 H2D、四阶段 GPU pipeline、诊断 D2H、
线程启动/回收；dense 为 retained 八 lane 的 eig-only device time，排除距离
矩阵构造。两边均不包含 context 创建。

| 分量数 | 额外驻留 | 结构化 wall | dense eig-only | 比值 | 结构化吞吐 |
|---:|---:|---:|---:|---:|---:|
| 256 | 0 | 0.340460 s | 0.726338 s | 0.469 | 752/s |
| 256 | 11 GiB | 0.336436 s | 0.725519 s | 0.464 | 761/s |
| 640 | 0 | 0.719815 s | 1.809565 s | 0.398 | 889/s |
| 640 | 11 GiB | 0.690426 s | 1.789319 s | 0.386 | 927/s |

不能从小幅差异推断“占用更多显存能加速”。结果说明本次显存容量压力没有
消除原型的吞吐优势。比值也不是相同边界的完整方法 speedup。

结构化显式 workspace：256 分量为 `1,090,779,136 bytes`（1.016 GiB），
640 分量为 `2,726,947,840 bytes`（2.540 GiB）。640 + 11 GiB 时，边界采样
的设备已用显存最大值为 `15,020,261,376 bytes`（13.99 GiB，含 CUDA context
等开销）。释放后回到同一 `474,415,104 bytes` 水平；首次加载 cuSOLVER 有
约4 MiB的一次性变化。这里不是连续采样的 vendor transient allocation峰值。

11 GiB 使用 cudaMalloc 后完整 cudaMemset，并检查头尾哨兵。它仅模拟现有
component cache 的容量驻留，未运行真实 cache 的插入/读取、optimizer、
Prepared-S或其他生产上下文。因此通过的是容量共驻留门，非整机流水线容量门。

## 深层与三套真实数据 witness

复用之前 guarded-QR 诊断中已认证的 **exact-fixed** residual shadows，完全
不使用失败的 QR residual。原 archive manifest 和 unchanged native 都已核对；
这些72个cohort已通过独立 production capacity dispatcher 的 residual-bit parity。
本次不重复 selected-SP 优化，只重新资格验证结构化 eig。

| 输入 | n | pairs | 最大 p 误差对 kpcalg | 最大旋转数 |
|---|---:|---:|---:|---:|
| 20240102191119 | 242 | 10 | 2.7964e-15 | 53,426 |
| 20230331145508 | 253 | 12 | 1.6263e-19 | 58,278 |
| canonical hard/deep cases | 351 | 40 | 2.6146e-14 | 109,130 |
| 20241215091815 | 364 | 10 | 4.4756e-16 | 118,193 |

q覆盖28、37、46、55、64、73；全部 statistic/moment/p、重复确定性、零翻转门
通过。这是 witness 资格，不是三套数据的完整 skeleton/PDAG 资格。

## 独立剖析与资源检查

Nsight Systems 对 640 分量 + 11 GiB、预热后的三次重放记录：

| profiled trial | wall | 所有 kernel 时间并集 | QR 时间并集 | recover 时间并集 | 最后 kernel 后尾部 |
|---:|---:|---:|---:|---:|---:|
| 0 | 770.76 ms | 717.12 ms | 598.38 ms | 207.98 ms | 26.65 ms |
| 1 | 731.99 ms | 712.75 ms | 673.58 ms | 323.80 ms | 17.38 ms |
| 2 | 683.19 ms | 667.89 ms | 615.16 ms | 308.00 ms | 13.51 ms |

最多10条kernel并发。各阶段时间并集有重叠，不能相加；这是kernel级并发，
没有测量SM occupancy，也未证明1000/s是硬上限。profiling时间不纳入性能表。
当前设备依赖链仍由双对角QR和向量恢复主导，没有支持把主机提交当成下一
主要收益池的证据。

compute-sanitizer memcheck 覆盖四种 n 各一个真实 pair、dense/structured重复
以及 rotation overflow：`0 errors`、`0 bytes leaked`。它不等价于完整640分量
及生产流程的全部错误路径审计。

## 下一门

1. 用当前 exact producer 的逐层 target/component 身份，重放预计算、消费与
   释放生命周期，量出需要驻留的峰值及未消费项比例。现有119,817条只证明
   可以缓存110,683个已消费分量；不能容纳预计132,974个全部预计算分量。
   当前缓存没有淘汰，满后停止插入，必须明确按层回收或其他有界策略。
2. 保持原 residual 物化运算与cohort形状，先做256/640有界分量队列的子集
   原型。给新 solver 独立 cache identity，并完成真实 GPU pair/gamma、全部
   pMax、task/deletion/graph/sepset/count/RNG与资源门，再比较完整outer时间。
3. 子集明确获益后才进入canonical与三真实数据全量资格及重复性能campaign。

保持 current native与生产默认配置；不重新启用QR optimizer，不使用CPU fallback，
不据本次分量吞吐承诺180/120秒。

## 复现与证据

工具：[structured_real_20261020/README.md](../tools/research/structured_real_20261020/README.md)。
原始数据与回执目录：`fastkpc/artifacts/full_cuda_ci/structured_real_20261020/`。
`manifest.json` 索引输入、320个cohort、研究库、源码快照、数值回执、Nsight报告
和memcheck日志；`benchmark.json`、`witness-qualification.json`与
`profile-analysis.json`分别记录性能/数值、补充witness和剖析结果。

证据等级为隔离 development diagnostic；不是clean-checkout重复完整方法中位数、
candidate冻结或promotion evidence。
