# 实际缓存 miss 形状与 22 宽分量 payload：第 0 道门 STOP

日期：2026-10-21。执行
[level 起始分量预填充方案](PHASE10_LEVEL_COMPONENT_PREFILL_PROTOTYPE_PLAN_20261021.md)
的前置字节门。先验证 actual persistent 8-lane cache payload，只有通过才进入
22 列预填充实现和完整方法开关对照。

## 决策

```text
accepted-native 实际缓存捕获 / 完整回归               PASS
按原 cache-miss cohort 复现全部 6,632 个 payload      PASS
改为 22 宽后四字段逐位一致                            FAIL：self_moment 不一致
当前 LSCP 原型集成                                   STOP_PAYLOAD_BITS
production 默认 / canonical / structured 集成         未推进
```

这是接口前置资格的有证据 STOP。没有新增 level-prefill 生产开关，没有改生产
源码或 accepted native，也没有放松 payload 或 p-value 合同。

## 为什么前面的 p 级实验没有发现它

[单 lane 批形状实验](PHASE10_COMPONENT_BATCH_SHAPE_20261021.md)覆盖的分量宽度
最低为 2，且不使用 persistent component cache。原生产运行却可能在一次包含
多个 target 的 residual cohort 中只遇到 **一个 component miss**；已有 hit
不参与这次 eig/Gram 构建。因此必须区分：

```text
残差物化 cohort 宽度
实际待构建 component 的 miss_count
```

先做的共享源码 smoke 在 174 个真实 target 上发现：宽度 1 改变 self_moment，
其他三个字段不变。随后补做实际 accepted-native cache commit 捕获，并以真实
miss cohort 重放，避免把较宽的 residual cohort 误当成生产分量参照。

## 生产参照与身份

accepted runtime 为 `dcb1749`，native SHA：

```text
fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50
```

使用同一 351×22 输入、default-Inf，实际到 level 6；connections=32、11 GiB
cache、residual bypass、8 eig lanes，数学库线程为 1。在真实
`run_full_cuda_ci_phase35_legacy_eig_batch_from_exact_primary_reuse_residual`
中捕获 cache D2D 写入的四字段、对应 residual 字节及实际 miss_count。
每次调用仍完整执行原生产代码，所有字段在原 completion 后读取。

捕获运行与已有未插桩基线相比：

- 24,264 个 CI 的 p 位、pMax、task/deletion identity 全部相同；
- adjacency、sepsets、n.edgetests、RNG endpoint 相同；
- 原生 CPU numerical authority / fallback 为零，native 资源计数闭合。

外部探针明确执行 diagnostic residual/component D2H，原生计数不统计这些
探针下载；它们不是生产无大块 D2H 的运行。CPU 只复制、比较和归档数据，
不产生 residual、component 或 CI 数值。捕获未新增最终 PDAG campaign。

共捕获 6,632 个不同 component，对应 2,592 个实际 miss cohort，覆盖
q=28/37/46/55。原 miss 宽度的 target 计数：

| miss_count | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| targets | 1,057 | 734 | 1,323 | 1,416 | 1,020 | 690 | 322 | 32 | 18 | 20 |

## GPU 字节重放

独立 probe 直接包含生产 `full_cuda_ci_vertical.cu`，采用同一 execution
context、deterministic cuSOLVER、pedantic cuBLAS、atomics 配置和缓存布局。
构建工具从生产函数原文提取 distance → eig → selection → Gram/self_moment
→ 四字段 D2D 的语句，不重写数学算法，并记录提取段 SHA。构建不覆盖 native。

先逐个重放实际 miss cohort，要求 probe 与上述真实缓存字节一致。全部
2,592 个 cohort / 6,632 个 component 通过，才改变批形状。输入 residual 的
每个字节同时与此前真实 finalizer 捕获核对。

随后把全部 6,632 个已消费 component 跨 S 分为 22 宽：自然序、反序及两次
额外重复，最后一批为 10；不加入重复列或虚构 padding。上下文保持 8 条
persistent eig lanes、容量 22，并实际分配 11 GiB cache。

| 字段 | 不一致的唯一 target | 不一致的比较次数 | 最大绝对差 |
|---|---:|---:|---:|
| vectors | 0 | 0 | 0 |
| selected_values | 0 | 0 | 0 |
| total | 0 | 0 | 0 |
| self_moment | **1,002** | **4,008** | **2.9103830456733704e-10** |

总计 26,528 次 component-output 比较，包含四次对照，不是新增 CI 数量。
全部变化来自原 miss_count=1 的条目；1,057 个此类条目中 1,002 个改变。
原 miss_count≥2 的 5,575 个条目在本次 22 宽对照中全部逐位一致。

所有重复及反序均重现上述差异。scope 内另有 174-target smoke，覆盖
1/2/3/7/8/9/16/17/19/21/22 宽和产生的尾批，四字段门同样检出 singleton
差异。原生产全部条目的按原 miss 形状重放则始终一致。

## 含义与边界

1. 差异定位到 Gram/self_moment 阶段的批形状依赖；vectors、selected_values
   与 total 都保持原比特。没有用 profiler 推断或命名具体 cuBLAS kernel。
2. 避免新批次的 singleton 尾批不能解决问题：原生产有 1,057 个 singleton
   miss，把它们并入 22 宽仍会改变既有 self_moment。
3. 2.91e-10 是 **self_moment 差异，不是 p-value 差异**。本次没有将改变后的
   payload 接入完整 CI，不能宣称 p 超过或满足 1e-10，也不能宣称图发生漂移。
   当前阻断项是方案自己要求的四字段逐位门。
4. 不能直接把所有 Gram 改成 singleton 或统一多分量形状：两种原形状都真实
   存在。若继续，应先单独设计并资格验证如何保留原首次 miss 的 self_moment
   语义，再考虑预填充；不按此结果直接重写生产消费者。

生产集成、canonical 重跑和 structured 接入均未进入；Phase 10、candidate、
recommended route 与 sealed holdout 的状态不变。

后续单独完成了
[原 GPU pair/gamma 的 scalar-only 差分诊断](PHASE10_SINGLETON_GPU_PVALUE_REPLAY_20261021.md)：
全部 1,186 个受影响 CI 最大 |Δp|=3.0531e-16，alpha 翻转为 0。该新证据量化了
本输入上的 p 影响；本报告的四字段逐位失败与 STOP 决定保持不变。

## 资源与复现

捕获与 GPU 重放的 tracked live allocations/bytes/streams/events 均闭合。
Compute Sanitizer memcheck（含 full leak check）完成 174-target smoke：
4,176 次比较重现同一数值 STOP，内存检查为 0 errors、0 bytes leaked / 0
allocations，进程退出码为 0。内存检查通过不改变 payload 字节门的失败结论。

工具：[component_payload_20261021/README.md](../tools/research/component_payload_20261021/README.md)。
证据目录：`fastkpc/artifacts/full_cuda_ci/component_payload_20261021/`；包含真实
cache payload/residual 捕获、逐文件 SHA、回归 receipt、完整比较报告、源码
提取段与 probe/interposer binary。它依赖已归档的 frozen production-cohort
输入，不依赖 CPU 重算 residual。

归档审计为 `PASS`，数值决策仍为 `STOP_PAYLOAD_BITS`。19,896 个原始捕获
文件全部压缩归档，并逐项解压核对长度与 SHA-256。
[归档 manifest](../artifacts/full_cuda_ci/component_payload_20261021/manifest.json)
的 SHA-256：

```text
e5c7fa254094a5fcde593b5efed36f4275b0000f2e1d989214fd71a1e0932d1e
```

另一个 GPU 当时存在其他任务。所有时间仅记录为诊断耗时，不作性能比较；
证据等级是 accepted-native 捕获与隔离 GPU 字节资格，不是性能或 promotion。
