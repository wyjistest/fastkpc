# Structured 行和原型：共享主机补充对照

2026-10-30 后续收口：[原协议隔离验收](PHASE10_STRUCTURED_ROWS_ISOLATED_20261030.md)
已完成，最终 `GO_CAPACITY / speed NOT_PROVEN`，已安装 `b2b79fc6…e135b12b`。
本报告继续保留共享主机这一轮的独立证据；下文未安装/待执行状态描述的是本轮结束时。
原 419 项历史依赖已核对，旧 live native 唯一映射到封存的同 SHA 副本，详见后续报告。

日期：2026-10-29。状态：`PASS_SHARED_HOST_DEVELOPMENT`，独立重算审计通过。
用户明确授权 GPU0 有外部任务时在独占 GPU1 上继续测试。本报告单独记录共享主机
开发观察，不替代[原先的隔离验收](PHASE10_STRUCTURED_ROWS_20261029.md)；本轮未安装原型。
六轮数值、容量及共享主机耗时不回退门通过；中位数下降 1.6281%，未达到速度门。

## 协议与身份

两版均使用已通过全量/增量一致性及数值资格的构建修复。OFF 与 ON 只有 structured
consumer TU 不同；其他 43 个对象完全相同。

```text
OFF: 639d598fe23db8cc08571f6e3f7d1e153a6c273d17a13ac8ecd806302179bab8
ON:  b2b79fc6219841a7b9de193e312de27bbdb8bca77a9d3a9d4a626aa1e135b12b
```

运行顺序保持 OFF1、ON1、ON2、OFF2、OFF3、ON3。每轮 fresh R/cache，固定物理 GPU1、
CPU affinity 0–19、数学库单线程，保留 postback、work-pair、lanes/slots。复用完整
canonical 351×48/default-Inf evaluator 和原硬门；仅改变证据标签，并在原计时边界
外记录 epoch。artifact 保存与对比不进入 skeleton 计时。

门槛直接绑定原 pending protocol，数值、容量与耗时阈值不变：

- 所有原 CI/pMax、任务/删除/图/sepset/计数/RNG、authority、资源门通过。
- 每层 structured pool 减少恰好 630,789,120 B，solver workspace 不变；每组
  L3/L4/L6 最低空闲检查点分别改善至少 600 MiB。
- 观察到的耗时不回退：ON/OFF 中位数比值 ≤1.02，每个配对比值 ≤1.05。
- 观察到的速度门：中位数比值 ≤0.98，三组全部更快。

这些判据的本轮结果仅适用于共享主机。即使全部通过，原隔离 acceptance 仍待执行，
不会更换 accepted median。所有完成轮次均保留，不根据外部负载或耗时挑选结果。
外部 GPU1 任务会中止本测试自己的进程组；GPU0 任务不被干预。

## 负载与计时边界

每 200 ms 采样两卡的利用率、显存、功耗、SM 时钟及温度；约每 3 秒记录 CPU ticks、
负载、内存、CPU/memory/I/O pressure，以及 GPU 进程的 CPU 时间与 affinity。
摘要限定在该轮 skeleton 的 compute window 内，原始采样全部保留。
这些离散样本不能证明不存在主机争用，也不是连续峰值证明。

六轮均记录到 GPU0 外部进程 2323839、2335739；GPU1 始终没有外部计算进程。
下表按实际执行顺序排列，CPU 为 affinity 0–19 的 busy 比例：

| 轮次 | CPU busy | GPU0 平均利用率 | GPU1 平均 SM MHz | GPU1 采样最高温度 |
| --- | ---: | ---: | ---: | ---: |
| OFF1 | 31.32% | 68.61% | 2749.73 | 48°C |
| ON1 | 31.37% | 69.63% | 2741.29 | 50°C |
| ON2 | 31.51% | 69.01% | 2741.49 | 50°C |
| OFF2 | 31.17% | 69.43% | 2739.09 | 51°C |
| OFF3 | 31.09% | 70.46% | 2739.40 | 51°C |
| ON3 | 31.19% | 69.74% | 2738.81 | 51°C |

平均负载接近不等于不存在 CPU/内存/PCIe 干扰，也不升级这组记录的证据等级。

## 结果

| 配对 | OFF 秒 | ON 秒 | ON/OFF | 变化 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 286.718 | 293.149 | 1.022430 | +2.2430% |
| 2 | 295.873 | 289.468 | 0.978352 | -2.1648% |
| 3 | 297.619 | 291.056 | 0.977948 | -2.2052% |
| 中位数 | **295.873** | **291.056** | **0.983719** | **-1.6281%** |

中位数比值 ≤1.02，三个配对均 ≤1.05，因此共享主机下的预声明不回退门 PASS。
速度门要求比值 ≤0.98 且三组全胜；实际比值为 0.983719、仅 2/3 胜，因此
`observed_speed_threshold=NOT_MET`。4.817 秒是观察差值，不宣称为已证明的加速。
没有删除或重跑任何已完成轮次；本轮结束时，原六轮隔离 campaign 尚未启动。

分阶段独立中位数：

| 阶段 | OFF 秒 | ON 秒 |
| --- | ---: | ---: |
| optimizer | 108.090227 | 103.091558 |
| Stage B | 128.427917 | 128.347316 |
| Stage A | 14.539045 | 14.564590 |
| native setup | 24.420619 | 24.492803 |
| consumer dCov | 14.353744 | 14.350164 |

Stage B 只相差约 0.081 秒；主要观察差值落在未修改的 optimizer 阶段。
这不足以把总时间差归因为删除矩阵的收益。各阶段中位数独立计算，不相加为同轮账本。

## 数值、容量与资源

六轮各覆盖 240,498 次 CI，全部 CI/pMax 位模式与固定父参考一致，所有 task/deletion/
skeleton/sepset/count/RNG、authority、route/status、cohort/缓存元数据门通过。
首次 OFF/ON 各 74 个门，四次重复运行各增加三个重复身份/位模式门，共 77 个门，
全部 PASS。最大 CI/pMax authority 误差仍为 **3.05425129631942e-12**；1,337 个
authority p ∈ [0.05,0.15] 无新增位差异。CPU 数值 authority/fallback/large D2H 为零。

三个配对的 L2–L8 structured pool 均减少 **630,789,120 B = 601.5673828125 MiB**，
solver workspace 全部不变。三个配对的原始最低空闲检查点完全相同：

| 检查点 | OFF bytes | ON bytes | 改善 MiB |
| --- | ---: | ---: | ---: |
| L3 | 3,404,857,344 | 4,054,974,464 | 620 |
| L4 | 2,662,465,536 | 3,312,582,656 | 620 |
| L6 | 2,230,452,224 | 2,880,569,344 | 620 |

L6 超出 2 GiB 门槛的余量为 **79.125 → 699.125 MiB**。请求的分配字节与实际可用
显存检查点分开记录；两者的差值不混为同一指标，也不宣称扩大了输入资格范围。
每轮 live allocations/bytes/streams/events 最终均为零。OFF 分配/释放 169/169，
ON 为 159/159；stream 18/18、event 60/60 均不变。释放的显存继续保留为余量。

## 证据

本地产物：`fastkpc/artifacts/full_cuda_ci/structured_rows_shared_host_20261029/`。
执行及审计工具见[共享主机补充说明](../tools/research/structured_rows_shared_host_20261029/README.md)。
本轮 SHA-256：

```text
protocol: 38935ddb868750ef47da4429d3a934c8127210b02dfff5f21f02b91171dca726
closure:  7e6a4814f8e8b047b04eb31050294b64023c8060a0d55fb7dc4af8d636541135
manifest: ad13ec1733094e815d7b92c2121f367a4d762367b4198cf1827c29c6efeacd9d
```

`audit.json` 为 `PASS_RECOMPUTED_SHARED_HOST`：重新核对所有 source/native/object/
runner/receipt 身份、原 pending protocol、数值硬门、三个配对及中位数、容量检查点、
compute-window 采样摘要和未安装状态。manifest 封存六轮原始记录及工具版本。
这是未提交源码闭包上的开发补充，没有 clean producer 或 promotion 主张。

原 pending protocol、closure、manifest 均保留。本轮结束时安装的 native 仍为
`a1e16a9c…7b73bd34`，accepted opt-in development median 仍为 290.491 秒。
q55 后排，recover-tile/L8 STOP、Phase 10 ACTIVE、candidate NOT_FROZEN、holdout
SEALED、recommended=false 保持原状态。无新增真实数据或 final-PDAG campaign。
六轮测试进程均已结束。当时剩余验收为两卡无外部计算时的原协议隔离对照，
已由上述 2026-10-30 后续收口完成；共享主机记录不并入其耗时统计。
