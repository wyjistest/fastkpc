# Structured 行和原型：2026-10-29 数值与容量资格记录

2026-10-30 后续收口：[原协议六轮隔离验收](PHASE10_STRUCTURED_ROWS_ISOLATED_20261030.md)
已完成，最终 `GO_CAPACITY / speed NOT_PROVEN`，已安装 `b2b79fc6…e135b12b`。
匹配 OFF/ON 中位数为 **283.985/283.328 秒**，数值、容量、资源及不回退门通过。
下文保留 2026-10-29 安装前阶段快照，pending 结论与原产物身份不作事后改写。

## 2026-10-29 安装前快照

日期：2026-10-29。当前状态为
`NUMERICAL_AND_CAPACITY_QUALIFIED_PERFORMANCE_PENDING`。
构建入口修复已先完成，见[构建一致性资格](PHASE10_BUILD_CONSISTENCY_20261029.md)。
行和原型通过 GPU 边界测试及完整 canonical 数值/容量资格；**尚无隔离 outer
non-regression 或 speed GO，未安装原型**。GPU0 的外部推理任务仍在运行，因此
预声明的六次隔离对照尚未启动。q55 扩展、lane/slot 数、recover tile 和 L8 STOP 不变。

随后经用户授权完成[六轮共享主机补充对照](PHASE10_STRUCTURED_ROWS_SHARED_HOST_20261029.md)：
OFF/ON 中位数 **295.873/291.056 秒**，全部数值、容量及共享主机不回退门 PASS，
速度门 NOT_MET；machine audit `PASS_RECOMPUTED_SHARED_HOST`。该结果独立保存，
不替代下述原隔离协议、不安装原型、不更新 accepted median。

## 唯一 runtime 改动

在 `full_cuda_ci_vertical.cu` 新增 structured 专用的 row-only kernel，保留原
`fabs`、每线程 column 累加、共享内存归约树和 totals kernel。删除
`StructuredComponentLane` 的未消费距离矩阵分配、成员及写入。dense consumer 和
dense GPU fallback 继续使用原距离矩阵 kernel 与独立 workspace。

父 native 为完整构建及增量一致性验证后的：

```text
639d598fe23db8cc08571f6e3f7d1e153a6c273d17a13ac8ecd806302179bab8
```

原型只重编 consumer TU，保留其余 43 个认证父对象，native 为：

```text
b2b79fc6219841a7b9de193e312de27bbdb8bca77a9d3a9d4a626aa1e135b12b
```

没有改变其他 TU 的 FMA 策略、数值容差、rank/route/info 规则、cohort、target 顺序、
solver、cache identity 或输出发布边界。structured 仍使用既有 opt-in 入口。

## 数值、fallback 与生命周期

GPU source-shared 测试直接比较原距离矩阵 kernel 和生产 row-only 路径的行和及总和。
同一 lane 连续处理 **64/1/63/17/64/9/2** 个 component，改变输入偏移，比较全部
有效输出及未使用尾部的位模式。单 lane 分配账本为 279,171,840 B，其中 solver
workspace 272,701,440 B。

640 个归档 component 的三字段 payload 逐字节一致，Gram 生命周期和 pending reset
保持原行为。混合批次含 **56 个真实 small-gap dense GPU fallback / 14 个 structured
成功**；dense 分块为 48+8，fallback payload 与独立 dense GPU 路径相同，14 个成功
payload 的原 target 顺序及位模式与归档相同。强制 dense failure 时当前 wave 发布
条目为零，后续访问 fail closed。rotation overflow、abandoned submit 和最终资源
释放均通过。Compute Sanitizer：**0 errors / 0 bytes leaked**。

首次独立测试链接缺少 R 库搜索路径，在任何 GPU 测试前失败。补齐测试程序的库路径
后通过；未改生产数学或添加 CPU fallback。原失败日志与工具版本保留。

完整 `dcc.gamma` canonical（351×48、default-Inf、fresh R/cache）结果：

| 检查 | 结果 |
| --- | --- |
| logical CI | 240,498 / 240,498 |
| CI / pMax 位差异 | 0 / 0 |
| task / deletion / graph / sepsets / counts / RNG | exact |
| 最大 authority CI / pMax 误差 | 3.05425129631942e-12 |
| authority p ∈ [0.05, 0.15] | 1,337 个，无新位差异 |
| 原有完整硬门 | 74 / 74 PASS |
| CPU numerical authority / fallback / large D2H | 0 |
| 最终 live allocations / bytes / streams / events | 0 / 0 / 0 / 0 |
| tracked allocation / free 次数 | 159 / 159；父版为 169 / 169 |

数值资格独占 GPU1，同时记录 GPU0 外部任务。该次运行的 elapsed 不作为性能证据。
没有新增完整真实数据或 final-PDAG campaign。

## 实测容量

L2–L8 每层的常驻 structured pool 均由 **3,422,507,520 B** 降至
**2,791,718,400 B**，减少 **630,789,120 B = 601.5673828125 MiB**；solver
workspace 逐项不变。lane 跨层存活，因此该差额在后续层仍然存在。

| 原始检查点 | 父版最低空闲 MiB | 原型最低空闲 MiB | 增量 MiB |
| --- | ---: | ---: | ---: |
| L3 | 3247.125 | 3867.125 | 620 |
| L4 | 2539.125 | 3159.125 | 620 |
| L6 | 2127.125 | 2747.125 | 620 |

L6 原始字节为 **2,230,452,224 → 2,880,569,344 B**；超出既定 2 GiB 门槛的
余量由 **79.125 → 699.125 MiB**。请求字节账本与 `cudaMemGetInfo` 可用显存属于
不同口径，分别报告；没有把静态预测直接作为实测值。这些是既有离散检查点，不是
连续显存峰值证明或更大输入资格。原型释放的空间保留为余量，没有增加并发配置。

## 原隔离性能门（2026-10-29 尚未执行）

待两张 GPU 均无外部 compute 后，在独占 GPU1 执行 OFF1/ON1/ON2/OFF2/OFF3/ON3，
每次 fresh R/cache，固定 CPU affinity 0–19、数学库单线程，保留既有 postback、
work-pair、lanes 和 slots。两个 native 都包含构建修复，只隔离矩阵删除这一项差异。

预声明判据保持原样：ON/OFF 中位数比值 ≤1.02，所有配对比值 ≤1.05，视为满足
本实验实际不回退门；该噪声余量不等于证明绝对零回退。速度 GO 还要求中位数
比值 ≤0.98 且三组全胜。容量接受必须同时通过数值、容量、资源和完整耗时门。
原隔离协议目前 **speed NOT_TESTED / outer non-regression PENDING**，不因为容量改善而越过
耗时门，不安装原型，不把共享主机运行混入隔离对照。

下一条命令已经固定：

```sh
python3 fastkpc/tools/research/structured_rows_20261029/campaign.py
```

通过后才执行工具说明中的 `archive.py audit / install / seal`。工具和协议见
[行和原型说明](../tools/research/structured_rows_20261029/README.md)。当前仍安装
`a1e16a9c…7b73bd34`；accepted opt-in development median 仍为 **290.491 秒**。
Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 不变。

## 证据索引

本地产物：`fastkpc/artifacts/full_cuda_ci/structured_rows_20261029/`。
`candidate/build.json` 绑定 107 个生产输入、44 个对象及 native；`qualification/`
保存 unit/fallback/sanitizer，`semantic/` 保存完整 canonical、容量检查点及共享主机
标记。`qualification-build-failed/` 保留首次测试链接失败。

`pending-outer-protocol.json` 固定未执行的六轮顺序与全部门；`pending-closure.json`
的 machine audit 仅针对已完成范围，明确未安装、未验收耗时。SHA-256：

```text
pending closure:  6d00308fbd01fcae46a2b2b12328ce647027a2dc56b26e3d7d0d42e3015cd331
pending manifest: 007d78bb81ad010e0d0d43e9855ee51ee087aaeae8c3f93713100e539b7a2e60
```

manifest 同时绑定父构建、原始/归一化对象、历史失败工具副本、receipts 和当前工具。
这些是 source-closure development evidence；源码尚未提交，不重标成 clean producer。
