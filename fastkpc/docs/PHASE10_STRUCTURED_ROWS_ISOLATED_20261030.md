# Structured 行和容量优化：原协议隔离验收

日期：2026-10-30。最终状态：`GO_CAPACITY / speed NOT_PROVEN`。
原协议六轮隔离验收、独立重算审计、安装及封存均已完成。已安装 ON native
`b2b79fc6…e135b12b`；数值、容量、资源及预声明完整耗时不回退门全部通过。
本项接受为容量优化，未证明速度收益。

提交后身份见[2026-10-30 开发交接](PHASE10_DEVELOPMENT_HANDOFF_20261030.md)与
[Git attestation](PHASE10_SOURCE_ATTESTATION_20261030.json)：构建修复 `89a8099`、
runtime `ae3687f` 与已测源码闭包一致。原运行仍保留提交前身份，不重标为 clean checkout。

[复核报告](PHASE10_STRUCTURED_ROWS_SHARED_HOST_REVIEW_20261029.md)的四项修订已包含在
执行口径中。两卡无外部 compute 时，按原先冻结的协议执行，保留已认证源码、两个
native、原工具及所有门槛。[共享主机补充](PHASE10_STRUCTURED_ROWS_SHARED_HOST_20261029.md)
独立保存，不混入本轮。当前容量版本的隔离开发中位数为 **283.328 秒**；历史
**290.491 秒**仍绑定原 `a1e16a9c…7b73bd34`，不能把两者之差作为矩阵删除的收益。

## 身份与验收口径

OFF 为已通过全量/增量构建一致性和数值资格的修复后父版本；ON 仅重编 structured
consumer TU 删除无消费者矩阵，其他 43 个对象完全相同。

```text
OFF: 639d598fe23db8cc08571f6e3f7d1e153a6c273d17a13ac8ecd806302179bab8
ON:  b2b79fc6219841a7b9de193e312de27bbdb8bca77a9d3a9d4a626aa1e135b12b
```

顺序为 OFF1、ON1、ON2、OFF2、OFF3、ON3。每轮 fresh R/cache，canonical 351×48、
default-Inf，使用物理 GPU1、固定 CPU affinity 0–19、数学库单线程；保留 postback、
work-pair、lanes/slots。两卡不得出现外部计算进程，artifact 保存在原计时边界外。

验收仍使用原完整耗时：中位 ON/OFF ≤1.02 且三个配对均 ≤1.05。速度 GO 额外要求
中位比值 ≤0.98 且三组全胜；速度 GO 不是容量优化的安装条件。
容量门要求每层请求池字节减少恰好 630,789,120 B、solver workspace 不变，每个配对
L3/L4/L6 原始最低空闲检查点分别改善至少 600 MiB，并通过原数值及生命周期门。

新增的[收口工具](../tools/research/structured_rows_completion_20261030/README.md)
只在全部运行结束后记录每组 L7/L8、L4/L5、Stage B 及其余耗时差。该账本用于描述
差值位置，不建立因果关系，不产生扣减后的验收时间，不调整原门槛或筛选轮次。

## 完整耗时与判定

六轮均使用物理 GPU1；隔离监测中两卡没有外部 compute。全部完成轮次保留，没有
按耗时挑选、删除或补跑。下表按原配对编号排列，执行顺序仍为上述交错顺序。

| 配对 | OFF 秒 | ON 秒 | ON/OFF |
| --- | ---: | ---: | ---: |
| 1 | 283.985 | 283.328 | 0.997686498 |
| 2 | 285.091 | 290.024 | 1.017303247 |
| 3 | 283.307 | 283.214 | 0.999671734 |
| 中位数 | **283.985** | **283.328** | **0.997686498** |

中位数观察差值为 **−0.657 秒（−0.23135%）**，2/3 配对更快。中位比值 ≤1.02，
每个配对均 ≤1.05，因此 `outer_nonregression=PASS`。这些是预声明的实际不回退
容差，不是绝对零回退证明。速度门所需的 ≤0.98 与三组全胜均未达到，故
`speed=NOT_PROVEN`。不剔除第二组较慢的 ON，也不用扣除阶段波动的时间重新验收。

## 数值、行为与资源

| 检查 | 六轮结果 |
| --- | --- |
| 每轮 logical CI | 240,498 / 240,498 |
| CI / pMax 位差异 | 0 / 0 |
| task identity/order、deletion、skeleton、sepsets、counts、RNG | exact |
| 最大 authority CI / pMax 误差 | 3.05425129631942e-12 |
| authority p ∈ [0.05, 0.15] | 1,337 个，原身份及位模式不变 |
| route/status、原 cohort 与缓存元数据门 | PASS |
| 首次 OFF/ON 硬门 | 各 74 / 74 PASS |
| 其余四轮（含重复身份/位模式门） | 各 77 / 77 PASS |
| CPU numerical authority / fallback / large D2H | 0 |
| 最终 live allocations / bytes / streams / events | 0 / 0 / 0 / 0 |
| OFF allocation / free | 169 / 169 |
| ON allocation / free | 159 / 159 |
| stream create/destroy、event create/destroy | 18/18、60/60 |

先前[GPU 边界资格](PHASE10_STRUCTURED_ROWS_20261029.md)中的满批、尾批、lane 连续
复用、640 component payload、混合 dense GPU fallback、dense failure 零发布和
Compute Sanitizer 结果继续绑定原已认证源码/native。本轮复用该资格，不将 canonical
中的零 fallback 当作故障路径覆盖。数值容差、rank/route/info、target/cohort 顺序、
lanes/slots、postback 和 work-pair 配置均未改变。

## 容量与生命周期

三个配对在 L2–L8 的 requested structured pool 均为
**3,422,507,520 → 2,791,718,400 B**，恰好减少
**630,789,120 B = 601.5673828125 MiB**。solver workspace 逐项不变。
lane 跨层存活，节省量在这些层持续存在。原始最低空闲检查点三组一致：

| 检查点 | OFF bytes | ON bytes | 改善 MiB |
| --- | ---: | ---: | ---: |
| L3 | 3,404,857,344 | 4,054,974,464 | 620 |
| L4 | 2,662,465,536 | 3,312,582,656 | 620 |
| L6 | 2,230,452,224 | 2,880,569,344 | 620 |

每个配对、每个检查点均通过至少改善 600 MiB 的门。L6 高于既定 2 GiB 门槛的
余量由 **79.125 → 699.125 MiB**。分配请求账本、原始空闲检查点和外部采样分别
报告；601.567 MiB 与 620 MiB 不混成同一指标，不推断为连续峰值保证或更大输入
资格。释放的空间保留为余量，没有增加并发。

## 描述性阶段账本

各阶段独立中位数如下，不相加为某次运行的完整账本：

| 阶段 | OFF 秒 | ON 秒 |
| --- | ---: | ---: |
| optimizer | 99.583658 | 99.498159 |
| Stage B | 127.985948 | 127.759821 |
| Stage A | 14.283428 | 14.251579 |
| native setup | 22.316976 | 22.403908 |
| consumer dCov | 14.142745 | 14.147178 |

每组 ON−OFF 的差值位置为：

| 配对 | outer 秒 | L7+L8 optimizer 秒 | L4+L5 optimizer 秒 | Stage B 秒 | 余数 秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | −0.657 | −0.007227 | −0.012825 | −0.417676 | −0.219272 |
| 2 | +4.933 | +4.954363 | +0.080681 | −0.219137 | +0.117093 |
| 3 | −0.093 | −0.058179 | −0.039453 | −0.034436 | +0.039068 |

第二组的较慢时间主要落在 L7/L8 optimizer；该记录不识别波动原因。Stage B 的
三次小幅下降也不足以证明完整方法加速。余数按原 outer 差值减去表中三项阶段差值
计算，仅描述差值位置；验收始终使用未经扣减的原始 outer elapsed。

## 证据与安装边界

本地产物根目录：`fastkpc/artifacts/full_cuda_ci/structured_rows_20261029/`。

| 产物 | 状态或内容 |
| --- | --- |
| `canonical/closure.json` | 六轮原协议验收，GO_CAPACITY |
| `audit.json` | PASS_RECOMPUTED |
| `installation.json` | PASS_INSTALLED；安装 native、44 个链接对象及对应 generated includes |
| `completion_20261030/attribution.json` | PASS_DESCRIPTIVE_ACCOUNTING |
| `completion_20261030/historical-verification.json` | PASS_PRESERVED_HISTORICAL_IDENTITY；419 项 |
| `closure.json` / `manifest.json` | 最终状态、输入/工具/运行记录封存 |

原 `audit → install → verify-history → seal` 顺序完成。当前
`fastkpc/build/fastkpc_cuda.so` 与 ON SHA 一致，安装前 native 及其 44 个原对象保留
在 `build_consistency_20261029/parent/`。最终 SHA-256：

```text
closure:  cdc26f74cab09449dcfd47ef2d3f6b710246dd4682dc93122fca5855081bfaa3
manifest: 196a0a84da7535a65927dbc4d8a49db5b192950fc7372848a5e970d1e4a8c87c
```

共享主机 manifest 记录了当时安装路径的旧 native 身份。本次安装后，历史校验只为
这一项显式建立映射：

```text
原路径: fastkpc/build/fastkpc_cuda.so
归档:   fastkpc/artifacts/full_cuda_ci/build_consistency_20261029/parent/fastkpc_cuda.so
SHA:    a1e16a9c0c8089ea90491426c56475394449efb03317be4a87f1925c7b73bd34
```

419 项历史依赖全部通过，其余项仍按原路径及 SHA 核对。映射单独留证，原共享主机
manifest、receipts、pending 协议/closure/manifest 字节均不改写。运行旧共享主机
审计时不能把已合法替换的 live native 当作旧版本；历史身份核对使用上述明确映射。
`c469ac5/e15be9d` 与 290.491 秒继续作为原 `a1e16a9c…` 的历史证据，本轮匹配对照
只比较 **283.985 → 283.328 秒**。

本项是完整构建父版加单 TU 重编的重复 source-closure development evidence；运行完成时
源码尚未提交，后续 Git 绑定见上文。不是 clean committed producer、promotion、新真实数据或 final-PDAG
campaign。README 与主路线图已同步当前安装状态及容量版本的 283.328 秒中位数。
默认配置、q55 后排、recover/L8 STOP、Phase 10 ACTIVE、candidate NOT_FROZEN、
holdout SEALED 与 recommended=false 保持原状态。六轮及审计进程均已结束。
