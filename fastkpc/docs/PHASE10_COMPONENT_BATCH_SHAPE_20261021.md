# legacy eig 批形状对 p 值比特的影响：第 0 道门（p 级）

日期：2026-10-21。回应
[level 分量预填充提案](PHASE10_LEVEL_COMPONENT_PREFILL_PROTOTYPE_PLAN_20261021.md)
评审要求的第 0 道门："先验证完整分量字节"。生产入口不下载分量 payload，本文
先做 p 级的必要条件检验：同一残差 token 上，分量批宽与 pair 批数是否改变任一
pair 的 p / statistic / mean / variance 比特。

## 决策

```text
同一有向 pair 在不同批形状下的 (p, statistic, mean, variance) 比特   全部相同：2,088 / 2,088 个有向 pair 只有一种摘要
覆盖的分量批宽 / pair 批数 / 顺序                                    2, 4, 8, 16, 17, 19 / 1, 2, 4, 8, 10, 171 / 自然序与反序
同 token 重复请求                                                    逐位相同
方向互换 (x,y) 与 (y,x)                                              p 的低位不同（23 个无向 pair），statistic/mean/variance 相同；
                                                                     方向由任务固定，不属于批形状问题
payload 字节级比较                                                   未做；需要独立比较四字段，不能用 p/pMax 一致替代
结论                                                                 有限形状的 p 级检查 PASS；后续实际 cache payload 门发现 singleton 反例并 STOP
```

证据等级为 accepted native 上的隔离 GPU 诊断，不是数值资格或性能证据。

## 方法

accepted native `fe8286f7…`，只用公开入口，不改生产代码。每个 case 取真实
重放归档的一个 level-3 S（22 列数据，q = 28），19 个非 S 列全部作为 target：

1. 用 capacity qualification 入口得到每个 target 的 selected log-SP（系数与
   log-SP 比特已证明与 cohort 无关）。
2. 用 persistent exact-fixed evaluator 对全部 19 个 target 建立一个残差 token，
   所有请求共用同一 token，残差比特固定（shadow SHA-256 记录在 summary 中）。
3. 对同一 token 发 29 个 legacy eig 批请求
   （`C_full_cuda_ci_phase35_legacy_eig_batch_from_optimizer_residual`）：
   9 个单 pair；4 个 4 宽 2 pair；2 个 8 宽 4 pair；1 个 16 宽 8 pair；
   1 个 19 宽 9 pair；1 个 19 宽 171 pair；同一 171 pair 反序；1 个 17 宽
   10 pair 窗口；9 个单 pair 重复。每个请求要求 status 全 OK、
   `component_build_count` 等于引用 target 数、residual/component D2H 为 0。
4. 每个 pair 行记录 (p, statistic, mean, variance) 比特的 SHA-256。

12 个 case，348 个请求，4,848 个 pair 行，2,088 个有向 pair。其中真正跨
不同分量批宽及 pair 批数比较的是 108 个有向 pair；有单 pair 参照的比较为
924 行，其余主要覆盖相同全批请求的顺序变化。分量构建的
Gram 矩阵来自 `cublasDgemmStridedBatched(batchCount = miss_count)`，pair 阶段
来自 `cublasDgemmBatched(batchCount = pair_count)`（`full_cuda_ci_vertical.cu`），
两者的批数都在本实验中变化。

## 结果

| 指标 | 值 |
|---|---:|
| 有向 pair 数 | 2,088 |
| 有向 pair 出现多种摘要 | 0 |
| 跨不同分量批宽的有向 pair | 108 |
| 对单 pair 参照不一致的行 | 0 / 924 |
| 无向 pair 出现多种摘要 | 23（全部为方向互换） |
| 分量批宽 | 2, 4, 8, 16, 17, 19 |
| pair 批数 | 1, 2, 4, 8, 10, 171 |

方向互换的例子（case 1，pair 1/22）：两个方向的 statistic、mean、variance
打印到 17 位有效数字完全相同，p 为 0.037796772312177998 与
0.037796772312177797，相差约 2e-16。现行路线中 x→y 与 y→x 是不同任务，各自
方向固定；LSCP 不改变任务方向，因此这不影响原型的逐位门。但它说明 pair
核在方向上不是逐位对称的，任何把两个方向合并计算的设计都必须单独资格验证。

## 对原型的含义

- 在上述 108 个跨形状有向 pair 上，分量批宽从 2 到 19、pair 批数从 1 到
  171 未改变 p 比特。该公开入口不传 component cache，实际每请求创建临时
  单 lane 上下文；persistent 8 lanes、22 宽和 canonical 的 48 宽均未测。
- 这是必要条件：payload 的 vectors、selected_values、total、self_moment 若有
  比特差异而不影响任何 p，本实验看不到。字节级确认必须直接比较四字段；
  同一新 binary 开关关闭/打开的 p/pMax 一致是另一道端到端门，不能替代它。
- 后续[实际 cache payload 字节门](PHASE10_COMPONENT_PAYLOAD_GATE_20261021.md)
  已发现本实验未覆盖的 singleton-miss 反例：1,002 个真实条目在 22 宽重组后
  self_moment 改变。当前 LSCP 因该门 STOP；本报告保留为有限形状的 p 级证据。

## 复现与证据

工具：[component_batch_shape_20261021/README.md](../tools/research/component_batch_shape_20261021/README.md)。

```text
native          fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50
run.R           b6ef5c4fffcd344023c354c2e0ade3802122c545aaa5efe8ecdb564b1c10a014
summary.csv     5081c038945deb5c1c3fd0567a2f5e364b7e5d72d8f5dfdf8d6af89a37d841ec
pair_rows.csv   b7d51a26bd4dde5590cd4b683e3220f6aa774c603186b2f68c23d464f5983e21
results.rds     204e85cfbe76a4970479e3bc7999bdfa39ffa5988dab37aa23be9f9c0d4bf907
```

环境：`CUDA_VISIBLE_DEVICES=0`、`CUDA_DEVICE_MAX_CONNECTIONS=32`、数学库线程 1、
`FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY=1`；前后 native 资源计数相同。
