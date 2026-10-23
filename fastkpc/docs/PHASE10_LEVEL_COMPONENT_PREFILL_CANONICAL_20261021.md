# Canonical dense level component prefill：默认关闭资格

日期：2026-10-21。承接[22 列原型](PHASE10_LEVEL_COMPONENT_PREFILL_RESULT_20261021.md)
与[代码审阅及重复运行](PHASE10_LEVEL_COMPONENT_PREFILL_REVIEW_20261021.md)。
本轮修正空任务层与调用方设备恢复，将原型的显式形状范围扩为 `351×22` 或
`351×48`，执行新 native 的 22 列回归与 canonical `351×48/default-Inf` 对照。
数值计算仍使用 GPU dense full eig，prefill 开关默认 0。

结论：**默认关闭的 canonical dense 原型资格 PASS**。完整 off/on 数值与资源
硬门、外部 payload 全量归因均通过；单次 outer elapsed 下降 5.56%。
这关闭本项窄工作，不构成生产默认启用或 Phase 10 promotion。

## 源码修正与回归

- one-call 的 `begin/build/finish` 共用 `prefill_this_level`，零任务层统一跳过。
  test-only planner 注入空 level 3，前三级仍执行真实 GPU 工作；off/on 的低层
  输出一致，资源闭合。这个测试验证生命周期，不冒充新的真实数据资格。
- 所有 prefill cache 工作入口使用局部设备作用域，成功和异常退出均恢复调用方
  CUDA device。source-shared smoke 用 GPU0 工作、GPU1 作为调用线程设备，覆盖
  成功、异常与析构；compute-sanitizer 为 0 errors、0 bytes leaked。这不是
  双卡算法，也不是完整 R 方法的全程 sanitizer。
- 65 项 level 数组未改动：现有 `p <= 64` 入口与 `level <= p - 2` 已约束索引。
- 新 native 的 22 列 default-Inf off/on 共 24,264 个 CI、pMax、task/deletion、
  图、sepsets、计数、RNG 全部逐位一致；on 的物理分组与 Gram key/顺序一致。
  7,270 个生产分量、6,632 个消费分量账本通过。108.513 s / 92.047 s 仅作为
  本轮回归记录，构建与部分回归同时运行，不用于新的性能中位数。
- 配置拒绝、既有 legacy component-cache 回归与资源闭合均 PASS。

## 同一 native 的 canonical 单次计时

两次均为 fresh R process、fresh dataset caches，GPU0、connections=32、
11 GiB component cache、residual bypass、8 eig lanes、数学库线程 1；GPU1
未运行重任务。大型回执持久化在计时边界外，外部 payload 观察另开进程执行。

| 项目 | off | on | on − off |
|---|---:|---:|---:|
| outer elapsed | 892.468 s | 842.873 s | −49.595 s（−5.56%） |
| selected-SP optimizer host | 336.221 s | 341.538 s | +5.317 s |
| consumer dCov host | 437.078 s | 181.243 s | −255.835 s |
| 阶段 B host | 0 | 233.354 s | +233.354 s |
| consumer dCov + 阶段 B | 437.078 s | 414.597 s | −22.481 s |
| 合计 eig device（含阶段 B） | 408.410 s | 396.835 s | −11.575 s |
| multi-penalty prepared 构建 | 61.561 s | 32.434 s | −29.127 s |
| prepared 构建次数 | 32,452 | 17,188 | −15,264 |
| 阶段 A host | 0 | 0.890 s | +0.890 s |

构建次数的减少精确等于 off 的 15,264 个 exact-primary consumer reuse 批。
on 在 prefill 窗口句柄释放前完成 finalization，consumer 不再重建这些句柄。
eig 的消费阶段计时必须加回阶段 B 的 231.306 s；不能把移走的 242.881 s
消费阶段 eig 全部算作收益。表中 host/device 项有包含关系，不能逐行相加。

这是单次、同 binary 的 source-closure development 对照；不能将 5.56% 写成
重复中位数或 promotion 性能。optimizer 的单次差异不归因于 prefill。
旧 `c4f599ef…` canonical-queues32 回执只用于输出参照，不作为本轮计时基线。

## 完整数值门与差异归因

两种模式均通过 240,498 个 CI；对归档 kpcalg authority 的最大 CI/pMax 绝对
误差均为 `3.0547508966805026e-12`，低于 `1e-10`。
off 对历史 canonical 输出逐位一致。
on/off 有 906 个 CI p 值逐位不同，最大绝对差 `1.63757896132211e-15`；
pMax 有 4 个单元格逐位不同。

task/deletion identity、adjacency、sepsets、n.edgetests、RNG endpoint、物理
分组序列与 Gram key/顺序摘要全部一致。native CPU authority、fallback、
residual/component 大 D2H 为零，route/status mismatch 为零，资源计数闭合。
本轮是 skeleton 资格，没有新增最终 PDAG campaign。

外部观察器先在 level-3 子集验证 4,148 个实际消费分量的残差与四字段逐位一致，
再对完整 canonical off/on 捕获系数、fitted/residual、四字段及 pair 引用。
诊断运行增加的同步与 D2H 单独标识，不属于生产路径或性能证据。
全量归因结果如下：

- 实际消费 65,694 个唯一分量，完整 key/request/prepared/Y identity、系数、
  route/rank/info 全部一致。off 的 54 个 ≥17 宽 cohort 对应 951 个首次消费
  分量；on 的 finalization cohort 均为 1–16 宽。
- 这 951 个分量的 fitted/residual 发生低位变化；其余 **64,743 个分量的
  fitted/residual 和四字段全部逐位一致**。任何四字段变化都有实际 residual
  变化，未使用“同一 target 列号”作为放宽依据。
- 60,333 个物理 pair 的身份与顺序一致，两档物理 p 值均逐位等于映射后的
  逻辑 CI。906 个变化的 CI 全在 level 3，全部实际消费了变化的 payload；
  **未归因 CI = 0**。
- `[0.05, 0.15]` 区间只有 1 个 CI 变化，最大差 `5.8286708792820718e-16`。
  pMax 的 4 个变化单元格对应无向边 `(3,12)`、`(3,24)`，各自最大值仍由
  同一逻辑 CI（第 185,417 / 189,444 行）提供，最大绝对差分别为
  `1.3877787807814457e-16`、`1.6375789613221059e-15`。
- 两个完整诊断运行都与各自未插桩计时回执的 CI/pMax 逐位相同，全部硬门通过。

| 字段 | 逐位变化分量数 | 最大绝对差 |
|---|---:|---:|
| fitted | 951 | 5.32907051820075e-15 |
| residual | 951 | 5.32907051820075e-15 |
| vectors | 951 | 6.65861810134061e-12 |
| selected_values | 951 | 7.95807864051312e-13 |
| total | 252 | 5.82076609134674e-11 |
| self_moment | 923 | 5.96628524363041e-10 |

后四项是 component 中间量，不是 p 值；其尺度也不同，不能直接套用 p-value
的 `1e-10` 门。这里没有放松 p 值合同：全部最终 CI/pMax 对 authority 独立
验收通过；不受残差变化影响的四字段继续要求逐位一致。

## 逐层账本与容量

| level | planned / produced | resolved | unused pending | 阶段 A cohorts | 阶段 B batches | Gram groups |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 54,878 | 42,662 | 12,216 | 5,573 | 1,144 | 9,477 |
| 4 | 17,767 | 15,433 | 2,334 | 2,156 | 371 | 3,963 |
| 5 | 6,816 | 6,377 | 439 | 955 | 142 | 1,459 |
| 6 | 1,148 | 1,080 | 68 | 245 | 24 | 312 |
| 7 | 124 | 124 | 0 | 44 | 3 | 44 |
| 8 | 18 | 18 | 0 | 9 | 1 | 9 |
| 合计 | 80,751 | 65,694 | 15,057 | 8,982 | 1,685 | 15,264 |

未消费比例为 18.65%。reset 6 次，丢弃 complete 110,665、pending 15,057，
最终驻留 18；level 0–2 按需插入 44,989，满足
`44,989 + 80,751 = 110,665 + 15,057 + 18`。
consumer 的 exact-primary finalization target 数归零，全部转入阶段 A。

本层 residual 存储峰值 154,097,424 字节；累计 residual D2D 226,748,808 字节，
gather D2D 6,474,800,640 字节。gather device 合计 0.355 s，Gram/scalar store
device 合计 1.130 s；这些是已包含的分项成本。缓存仍受原 11 GiB 预算约束。

## 身份与证据边界

```text
native SHA-256
  b1e7eab4297a13b7ff29f0bf40b2fc3ce4d6950542dff4318ee3e2dc76694ee9
one-call source
  f4c3a4df2c4fa1a5bd23bcbd6a2d7f8e03f1c58c3fde21ecf8dfe7540d8dca7a
vertical CUDA source
  d7fa0d4134da3eef5d9d8a3be23c44a2ac7b6153cafea93678d7e264d5affbb6
vertical header
  d015c6f3c61e92022bad02238cc5608217333d1c9a4ca3c65d0f07f589f4d398
```

增量构建重编 one-call、CUDA vertical 与 R API；其余对象来自之前保留的 build。
证据是精确 native/source 绑定的 development run，不是 clean-checkout producer
run。旧 22 列 `1d09d177…` 的三次 off/三次 on 仍属于它自己的证据，不迁移到
新 native。工具见[canonical runner 与观察器](../tools/research/level_component_prefill_canonical_20261021/README.md)。

完整回执、分量/pair 哈希、951 个分量的原始数组、CI/pMax 归因、测试日志与
源码/native 快照归档至 `fastkpc/artifacts/full_cuda_ci/level_component_prefill_canonical_20261021/`。
机器回执为
[closure.json](../artifacts/full_cuda_ci/level_component_prefill_canonical_20261021/closure.json)，
全量归因为
[attribution.json](../artifacts/full_cuda_ci/level_component_prefill_canonical_20261021/attribution.json)，
逐文件清单为
[manifest.json](../artifacts/full_cuda_ci/level_component_prefill_canonical_20261021/manifest.json)。

```text
closure status
  PASS_CANONICAL_OPT_IN_DENSE_PROTOTYPE
manifest SHA-256
  17952f42a6a7ee2875ddcc00026b947a77969e239f260ee2fb637624fd8d756f
```

诊断耗时 915.500 s / 860.175 s 单独标识，不能替代未插桩的 892.468 s /
842.873 s 对照。GPU telemetry 从 off 计时中途开始，不能用它宣称完整 off
区间的采样峰值或连续监控。

structured eig 未集成，其他数据未经本轮资格；默认开关、Phase 10 ACTIVE、
candidate 未冻结、sealed holdout 与 recommended=false 均保持不变。
