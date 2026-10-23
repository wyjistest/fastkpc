# 真实生产 cohort 捕获与固定系数 GPU 重放

日期：2026-10-21。完成 component 生命周期评审要求的窄实验：捕获真实
one-call 的原 cohort，固定同一份 exact-primary 系数，只改变 fitted/residual
物化的宽度和列顺序。补充此前
[独立 dispatcher cohort 实验](PHASE10_RESIDUAL_COHORT_IDENTITY_20261021.md)。

## 决策

```text
真实生产 cohort 捕获、原形状 GPU 重放               PASS
所有拟合宽度均可任意重组而保持逐位一致              FAIL：17 列出现真实反例
22 列输入，每个 S 按不超过 16 列分块物化            PASS（本次实际形状与目标范围）
受限 multi-penalty dense 分量对照原型               GO（仍须完整方法门）
canonical / 三真实数据 / 生产默认 / structured 集成  NOT YET GO
```

不需要根据这次结果立即重写拟合 GEMM。残差物化宽度与后续 eig 分量批宽是
两个不同参数：可以先按每个 S 的窄 cohort 生成残差，再收集多个 S 的独立
分量给 640 宽的 eig producer。这次只验证前一项的数值身份，没有实现该流水线。

## 捕获边界

accepted native 保持为：

```text
fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50
```

通过独立 `LD_PRELOAD` 诊断库调用原始函数，未修改生产源码、对象文件或 native：

1. 在真实 multi-setup optimizer 返回时，捕获即将进入 coefficient arena 的
   GPU 系数、Y、selected log-SP、target key、route/rank/info。每个 target
   只取这次生产运行的值，重放阶段不重新优化。
2. 在真实 `finalize_selected_coefficients` 中，原 cuBLAS GEMM 照常执行；捕获
   实际 X、系数、fitted，等原 completion event 后捕获实际 GPU residual。
3. 用 base target key 和 log-SP 的原始比特重建 residual-v2 key；逐列确认
   finalizer 的系数、Y 和 metadata 与对应 producer 完全一致。
4. 用独立 GPU 重放库执行相同 GEMM 与 `__dadd_rn(Y, -fitted)`。先要求每个原
   cohort 的 fitted/residual 全部逐位复现，再改变分组与列顺序。

捕获包含明确的 diagnostic D2H 和额外同步，不是生产无大块 D2H 的运行。
原生 authority counters 不统计这些外部探针拷贝；CPU 不产生系数、fitted、
residual 或 CI 数值。捕获耗时不用于性能比较。

单张 RTX 4090，n=351，connections=32，CPU 数学库线程 1，retained cache / bypass /
eig lanes=11 GiB / 1 / 8。实际捕获的 cuBLAS version=120503、math mode=0、
atomics mode=0；重放保持这些配置。另一张 GPU 没有本实验任务。

## 真实运行资格

先复用归档的 22 列、max-level-3 基线；再为同一 22 列输入建立一次未插桩的
default-Inf 基线，捕获运行分别与对应基线比较。

| 范围 | CI | producer cohorts / targets | finalizer cohorts / target 出现次数 | 实际消费唯一 target |
|---|---:|---:|---:|---:|
| max-level-3 | 22,373 | 628 / 4,662 | 1,589 / 5,425 | 4,148 |
| default-Inf，实际到 level 6 | 24,264 | 1,242 / 7,270 | 2,592 / 8,306 | 6,632 |

两次捕获均通过：所有 CI p 值与 pMax 逐位一致，task/deletion identity、
adjacency、sepsets、n.edgetests 和 RNG endpoint 一致；原生 CPU numerical
authority / fallback 为零，native resource counters 闭合。两行覆盖重叠，
不能相加作为独立 CI 数量。这里验证 skeleton，未新增最终 PDAG campaign。

default-Inf 的 7,270 个 producer target 中，638 个没有实际消费参考输出；
没有为它们虚构原 residual。实际消费的 6,632 个唯一 target 全部参与比较。

## 固定系数结果

2,592 / 2,592 原 cohort 的原形状重放逐位一致，原宽度内反转列顺序也全部
逐位一致。所有重放均复用已捕获的 GPU 系数和 Y 字节。

| q | 实际 finalizer cohorts | 实际消费唯一 target | 原最大宽度 | 本次最大测试宽度 | 不超过 16 列分块 |
|---:|---:|---:|---:|---:|---|
| 28 | 1,589 | 4,148 | 10 | 17 | PASS |
| 37 | 816 | 1,981 | 6 | 9 | PASS |
| 46 | 166 | 461 | 4 | 4 | PASS |
| 55 | 21 | 42 | 2 | 2 | PASS |

重组测试只用同一真实 producer 中的不同 target，不加重复列或虚构 padding。
测试包括自然序、反序、轮转、不同宽度窗口，以及一个完整覆盖的确定分块方案：
按 target 列号排序，均衡分为每块不超过 16 列的 cohort（17 列分为 9+8）。
该方案的每块重复两次。

default-Inf 共 75,683 次 target-output 比较，其中包含重复与重叠窗口，不能
视为独立 CI。具体反例：

```text
改变为 17 列后不一致的比较                  168
涉及唯一 target                            32
涉及 producer / 原 finalizer cohort         2 / 11
最大 fitted 与 residual 绝对差              4.440892098500626e-15
上述反例的系数 / log-SP / route / rank/info  一致
```

因此，批宽可以在系数完全不变时改变拟合与残差低位。这里证明了物化形状的
影响，没有通过 profiler 单独证明具体 cuBLAS kernel 的切换机制。

完整的“不超过 16 列”分块方案覆盖全部 6,632 个实际消费唯一 target：
两次重放共 16,612 次原 target-output 比较，fitted/residual 零比特差异。
本次需物化的已消费 S 对应 1,243 个分块；未测其集成后的调度和耗时。

## 边界与下一门

- 这是本机、本库、本输入的实际形状证据，不是“任意 q、任意 GPU 上 16 列以内
  永远逐位一致”的定理。q=37/46/55 的测试宽度分别只到 9/4/2。
- 不能把 4.44e-15 residual 差直接换算为 p 值误差。这次没有把变化后的残差
  接入整套 CI 重算，也未证明该 17 列路线满足所有 p/pMax <=1e-10。
- single-penalty 的 level 1–2 投影与系数阶段未做重组资格；后续窄原型须保留
  这些阶段的现行执行。canonical 已知存在原宽度 >=17 的组，不能直接推广
  “一律不超过 16 列”并声称仍复现 canonical 原比特。
- 允许进入的下一项是 22 列 multi-penalty 范围的 dense 对照：采用已验证分块，
  保持 canonical 逻辑重放与现有 11 GiB 预算，验证分量收集与消费接口。完整
  p/pMax、task/deletion、图、sepset、计数、RNG 与资源门仍须通过。
- 在该对照通过并测出净收益以前，不接入 structured eig 生产默认，不改 Phase
  10、candidate、recommended route 或 sealed holdout。

## 资源与证据

原 native SHA 与编译时绑定的生产源码闭包均保持一致。GPU 重放的最宽和最深
真实 source cohort 经 Compute Sanitizer memcheck：0 errors，0 bytes leaked。
这是一项 smoke 检查，不是对全部 captured case 的 sanitizer 遍历。

工具：[production_cohort_frozen_20261021/README.md](../tools/research/production_cohort_frozen_20261021/README.md)。
证据目录：`fastkpc/artifacts/full_cuda_ci/production_cohort_frozen_20261021/`。
目录保存原始捕获的压缩包与逐文件 SHA、基线和捕获 receipts、GPU 重放库、
全部比较 CSV、机器 summary、memcheck 日志和 source/native/binary manifest。
归档工具逐项核验原始压缩包内容及身份。证据等级为隔离 GPU 身份诊断，
不是性能 campaign 或 promotion evidence。

机器归档 audit=PASS；manifest SHA-256：
`b89596a42266f21759a5f3eb0082bbae1bdecfd7a06af9340e1c2866fd023c8c`。
