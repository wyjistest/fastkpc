# 残差 cohort 逐位身份实验：物化宽度决定比特类

日期：2026-10-21。回应
[分量生命周期重放报告](PHASE10_COMPONENT_LIFECYCLE_REPLAY_20261021.md)
评审确定的前置实验："使用同一份 exact-primary 系数，只改变残差物化的 cohort"。

## 决策

```text
系数与 selected log-SP 比特是否随 cohort 变化        否：1,140 个 target 全部只有一种摘要
fitted / residual 比特是否随 cohort 变化             是，且只取决于 cohort 宽度类：T ≤ 16 一类，T ≥ 17 另一类
是否取决于列顺序或列位置                              否：同宽度不同顺序的 80/80 列逐位相同
差异量级                                             fitted 与 residual 最大 |Δ| = 6.217e-15
与归档生产 dispatcher 输出的锚定                      24/24 个 pair cohort 的系数、残差、log-SP 逐位相同
覆盖范围                                             multi-penalty 路线（level ≥ 3，q = 28–73）；single-penalty（level 1–2）未覆盖
整层残差前移能否与现行路线逐位一致                   只有当每个 target 的物化宽度类与现行 cohort 相同时才行；
                                                     现行 cohort 由动态调度决定，一般不可静态复现
生产集成                                             NOT YET GO
```

这是在 accepted native 上的隔离 GPU 诊断，不是数值资格、性能证据或 promotion
证据。Phase 10、candidate、recommended route 和 sealed holdout 均不改变。

## 方法

只使用 accepted native `fe8286f7…` 与其公开入口
`C_full_cuda_ci_multi_penalty_capacity_qualify`。该入口对一个 cohort 依次执行
exact-primary optimizer（每 target 一个 block）、cohort 整体的
`cublasDgemm(n, target_count, q)` 拟合与逐元素残差核（与生产
`finalize_selected_coefficients` 的调用形状相同，`mgcv_multi_penalty_gcv.cu`），
以及 exact-fixed stable-SVD 评估，并返回全部主机 shadow。没有修改任何生产
源码、目标文件或库。

实验设计满足评审要求的边界：

- **同一份 X 比特**：每个 case 的 prepared-S 几何由同一 `full_cuda_ci_native_setup_native`
  + `fastkpc_full_cuda_phase6_prepare` 生成；22 列 case 的几何 SHA-256 与真实
  重放归档中的 `geometry_sha256` 相同。
- **同一份系数比特**：每个 cohort 调用都重新运行 optimizer，但实验直接验证
  系数比特与 cohort 无关（下文），因此跨 cohort 比较的 fitted/residual 差异
  只能来自物化形状。
- **只改变 cohort**：同一组 target 在 15 种形状下物化：归档 pair、pair 重复、
  pair 交换、pair 优先序的前缀宽度 3/5/7/10/16/24/32、两个位置不同的 5 宽
  窗口、重排的 3 宽前缀，以及全部非 S 列的自然序、反序、随机序（19 或
  40–44 个 target）。
- **真实 S 与 q 覆盖**：24 个 case 取自真实重放归档的 level-3 pair（22 列
  数据，q = 28），16 个 case 取自 canonical 32-connection 回执的 level 4–8
  S（48 列数据，q = 37, 46, 55, 64, 73）。
- **门**：每次调用要求 solver info 全 0、stable-SVD 路线、`cuda_exact_primary_target_count`
  等于宽度、CPU/fallback 计数为 0；全程前后资源快照相同；native SHA 不变。

每列记录 optimizer 系数、fitted、residual、exact-fixed 系数/fitted/residual、
selected/fixed log-SP 的 SHA-256，以及相对 pair cohort 的最大绝对差。共 40 个
case、592 次 cohort 调用（另有 28 次宽度阈值调用）。

## 结果

| level | q | cases | targets | 系数多摘要 | log-SP 多摘要 | fitted 多摘要 | residual 多摘要 | 最大 \|Δ\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 28 | 24 | 456 | 0 | 0 | 384 | 384 | 5.33e-15 |
| 4 | 37 | 6 | 264 | 0 | 0 | 96 | 96 | 4.44e-15 |
| 5 | 46 | 4 | 172 | 0 | 0 | 64 | 64 | 6.22e-15 |
| 6 | 55 | 3 | 126 | 0 | 0 | 48 | 48 | 5.77e-15 |
| 7 | 64 | 2 | 82 | 0 | 0 | 32 | 32 | 6.22e-15 |
| 8 | 73 | 1 | 40 | 0 | 0 | 16 | 16 | 5.33e-15 |

每个 case 恰有 16 个 target 出现两种 fitted 摘要：正是同时出现在窄 cohort
（前缀 ≤ 16）与宽 cohort 中的 16 个 target；只出现在宽 cohort 中的 target
全部只有一种摘要。按形状与宽度对 pair cohort 的 fitted 摘要不一致率：

| 形状 | 列数 | 不一致 |
|---|---:|---:|
| pair / pair 重复 / pair 交换 | 80 / 80 / 80 | 0 / 0 / 0 |
| prefix 3 / 5 / 7 / 10 / 16，重排 prefix 3，5 宽窗口 ×2 | 各 80 | 全部 0 |
| prefix 24 / 32 | 26 / 26 | 26 / 26 |
| 全部列：自然序 / 反序 / 随机序 | 80 / 80 / 80 | 80 / 80 / 80 |

按宽度：2、3、5、7、10、16 全部一致；19、24、32、42、43、44 全部不一致。
同宽度内不同顺序逐位相同（prefix 3 对重排 prefix 3、pair 对交换 pair、全列
自然序对反序和随机序，均 80/80），同宽度下每个 target 只有一种摘要；宽
cohort 之间（19 至 44）彼此相同。因此只有两个比特类，由宽度决定。

宽度阈值（6 个 case，q = 28/37/46/55，宽度 16–20，比较前两个 target）：

| 宽度 | 等于窄类（pair） | 等于宽类（全列） |
|---:|---|---|
| 16 | 6/6 | 0/6 |
| 17 | 0/6 | 6/6 |
| 18 | 0/6 | 6/6 |
| 19 | 0/6 | 6/6 |
| 20 | 0/3 | 3/3 |

阈值在 17：`cublasDgemm` 在 N ≥ 17 列时切换了 kernel（规约顺序）。差异量级
最大 6.2e-15，residual 与 fitted 的差完全相同（残差核是逐元素相减，不引入
新差异）。exact-fixed 路线的 fitted/residual 呈现相同的宽度类行为。

附带观察：同一 cohort 内 optimizer 路线残差与 exact-fixed 路线残差有
5,144 / 6,716 列逐位相同（76.6%）。生产路线依赖 refresh 后的"bitwise-qualified"
系数 arena 处理这一差异，不在本实验范围内。

调用耗时中位数 0.09 s，最大 4.6 s（q = 64–73、44 target）。

## 现行生产 cohort 的宽度

残差 cohort 宽度取自回执 wave trace 中有 miss 的组的 `component_request_count`：

| 数据 / level | 有 miss 的组 | 宽度 p50 / p90 / p99 / max | 宽度 ≥ 17 的组 |
|---|---:|---|---:|
| canonical level 2（single-penalty 路线） | 9,668 | 8 / 18 / 29 / 46 | 1,123 |
| canonical level 3 | 9,477 | 4 / 12 / 16 / 21 | 54 |
| canonical level 4 | 3,963 | 3 / 9 / 11 / 11 | 0 |
| canonical level 5–8 | 1,824 | ≤ 8 | 0 |
| 22 列子集 level 2 | 876 | max 20 | 37 |
| 22 列子集 level 3 | 1,589 | max 10 | 0 |

## 对设计的含义

1. **整层残差前移与逐位一致的关系已经明确。** 残差比特只取决于物化宽度类。
   把整层物化切成 ≤ 16 宽的 cohort，可以在 level ≥ 4 完全复现现行比特，在
   level 3 复现除 54 个宽组之外的全部比特；但现行 cohort 的宽度由动态
   frontier 调度决定，没有静态切分能复现每个 target 的宽度类。所以对 level 3
   （以及未测的 level 2），"与现行路线逐位一致"不是一个一般可达的验收门。
2. **两条路线供决策**：
   - (a) 先让生产 finalization 与 cohort 形状无关：用确定性的逐列核替换三处
     `cublasDgemm` 拟合（exact-primary 系数 finalization、optimizer batch、
     exact-fixed 评估），使比特不再依赖 T。这会在 ~1e-14 量级改变现行比特，
     需要对 canonical 与三组真实数据做一次对 kpcalg 权威的完整重新资格
     （p ≤ 1e-10、graph/sepset/n.edgetests 精确、资源闭合）。之后 dense 对照
     原型可以用逐位一致作为验收门，后续所有调度改动也都能逐位门控。
   - (b) 保持 cuBLAS，原型按数值合同验收：p 对权威 ≤ 1e-10、graph/sepset/
     n.edgetests 精确、残差漂移记录在 ≤ 1e-14 量级；不要求逐位一致。
   (a) 更干净但改动生产；(b) 不改生产但放弃逐位门。两者都需要评审决定。
3. **level 2 必须单独测**。single-penalty 路线的投影 GEMM（`X^T Y`、
   transform × RHS）同样以 `target_count` 为维度，而且影响的是系数而不仅是
   fitted；现行 level 2 有 1,123 个 ≥ 17 宽的组。应通过
   `C_full_cuda_ci_single_penalty_gcv_multi_cuda` 做同样的 cohort 实验。
4. 若选 (a)，逐列核必须同时用于 optimizer 路线与 exact-fixed 路线，否则
   refresh/arena 的 bitwise-qualified 逻辑失去意义；其时间开销可忽略
   （n × q × T 规模）。

## 复现与证据

工具与说明：[residual_cohort_identity_20261021/README.md](../tools/research/residual_cohort_identity_20261021/README.md)。
原始数据：`fastkpc/artifacts/full_cuda_ci/residual_cohort_identity_20261021/`
（`cases/NNN.rds`、`summary.csv/rds`、`analysis.rds`、`width_threshold.csv/rds`、`run.log`）。

```text
native                                  fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50
run.R                                   477f0a1968136b1977aaacf320fd5ffec43a2eeb5121fb020bf34a18efaef25d
analyze.R                               e3caeb8ce5594e87c5638baface2814430226f738e385263b8026c2bb7888c3d
threshold.R                             3a042474bad638effdb8169b96576e932ede8dc841c97ed6fdba66b4220c7913
summary.csv                             6db71d029331d3f97ff29ff20aa635d8f96f22f24ba0d9d7183f1efdb2940051
summary.rds                             c2f89315052606ef41ba28c7717dcc85cbaed04af11102921e060f99c4090ce8
width_threshold.csv                     0aa7ebbc002f956a065d0652b564f051e05fb7ac35dd4c7fd1faee6c803114fd
fitted_mismatch_by_width.csv            59c690aa9529772173a55b0b8962be5939b6bcfc783aaaed6f98893574b38a0d
```

环境：`CUDA_VISIBLE_DEVICES=0`、`CUDA_DEVICE_MAX_CONNECTIONS=32`、数学库线程 1、
`FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY=1`（与真实重放捕获相同）。GPU1 无本实验
负载。证据等级为隔离 GPU 诊断。
