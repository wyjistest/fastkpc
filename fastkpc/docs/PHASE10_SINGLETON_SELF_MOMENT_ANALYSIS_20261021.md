# singleton-miss self_moment 的量级、p 影响估计与可选路径

日期：2026-10-21。回应
[分量 payload 第 0 道门 STOP](PHASE10_COMPONENT_PAYLOAD_GATE_20261021.md)
提出的问题："重新推进前，需要先解决原 singleton-miss 的数值语义如何保留"。
本文保留最初的离线量化，并按评审修正执行建议。随后完成了
[原 GPU pair/gamma 差分重放](PHASE10_SINGLETON_GPU_PVALUE_REPLAY_20261021.md)：
全部 1,186 个受影响 CI 的最大 |Δp| 为 3.0531e-16，alpha 翻转为 0；
原 payload 的 STOP 决定仍保持，生产源码未改。

## 结论

```text
差异来源                      只在 self_moment；原 miss_count = 1 的条目（单 lane、batchCount = 1 的 Gram GEMM）
                              与多分量批不同类；vectors / selected_values / total 不变
差异量级                      绝对最大 2.91e-10；原 self_moment 取值 1.68e4–1.61e5（1,002 个改变条目），
                              保守相对界 ≤ 1.73e-14；补齐候选后逐条实测最大 3.47e-15、27 ULP
                              20 个旧 witness 的逐条相对差最大 2.56e-15（1–14 ULP）
先前对 p 的影响估计           variance ∝ sm_left × sm_right，相对扰动 ≤ 3.5e-14；
                              在 320 个真实 pair 上，相对 variance 扰动 1e-13 时 max |Δp| = 9.1e-15，
                              1e-12 时 9.0e-14；此 CPU 敏感度只用于量级分析
后续原 GPU 实测               1,186 个受影响 CI，最大 |Δp| = 3.0531e-16，翻转为 0
为什么 p 级实验没发现          p 级实验的最小分量批宽是 2，没有 batchCount = 1 的情形
"保留 singleton 语义"是否可静态实现   否：某个分量在生产里是否为 singleton miss 取决于动态调度历史
推荐                          GPU 差分诊断已完成；先评审保留原 miss 形状或新 Gram 的独立设计
                              不自动采用验收例外，不解除原 payload STOP
```

## 量化

Codex 的 payload 门报告给出改变条目数与最大绝对差；本文从其归档的原始
捕获（`native/raw.tar.gz`，逐文件 SHA 已在 manifest 中）读取 1,002 个改变
条目的原 self_moment，并与 300 个未改变条目对比：

| 项目 | 改变条目（1,002） | 未改变样本（300） |
|---|---:|---:|
| self_moment 最小 / 中位 / 最大 | 16,836 / 41,698 / 161,198 | 13,487 / 37,702 / 109,945 |
| 捕获元数据 | residual_targets = 2–6，component_misses = 1，active_lanes = 1 | — |

改变条目按 residual_targets=2/3/4/5/6 分别为 876/70/50/4/2；共同特征是
component_misses=1，不能把 residual cohort 宽度与 miss_count 混同。

最大绝对差 2.91e-10 对最小改变条目的相对差为 1.73e-14，对中位数为 6.98e-15；
报告的 20 个 witness 逐条相对差最大 2.56e-15，范围为 1–14 ULP。这种低位
差异与 cuBLAS 在 batchCount = 1 与 ≥ 2 时选择不同 kernel 的解释相容（未用
profiler 证实具体 kernel）。

pair 核中 self_moment 只进入 variance：
`variance = factor × sm_left × sm_right / n²`，mean 来自 totals，statistic
来自 vectors 与 selected_values（三者不变）。因此 Δp 只通过 variance 的
相对扰动 r ≤ 2 × 1.73e-14 产生。用 CPU gamma 近似（对 320 个真实 pair，
与原 GPU p 的最大差 1.1e-13，足以估计量级）：

| variance 相对扰动 r | max \|Δp\|（320 对，双向） | max \|Δp\| / p |
|---:|---:|---:|
| 1e-15 | 4.4e-16 | 1.5e-13 |
| 1e-14 | 1.0e-15 | 1.1e-12 |
| 1e-13 | 9.1e-15 | 1.1e-11 |
| 1e-12 | 9.0e-14 | 1.1e-10 |
| 1e-10 | 9.0e-12 | 1.1e-08 |

alpha 附近 0.02 以内的 12 对在 r = 1e-12 时 max |Δp| = 9.0e-14。按实测相对
差的保守界 ≤ 3.5e-14，最初估计 |Δp| ≲ 1e-14。该 CPU 表不能替代 GPU
差分资格；原 GPU 的后续实测见上述报告。

## "保留 singleton 语义"的可实现性

生产里一个分量是否以 singleton 形状构建，取决于它首次被请求时所在 frontier
组的其余分量是否已在缓存中，即取决于动态调度历史。整层预填充没有"首次
miss"的概念，单凭预填充静态分块不能预判这一类别。可以另行研究：提前
生产 eig 的 vectors/selected_values/total，把 Gram/self_moment 延后到消费，
用独立首次访问账本恢复原 miss 分组。其实现、成本与逐位身份仍需验证；
当前证据不支持把"保留原 singleton 语义"宣布为不可达。

这同时说明：现行路线的 self_moment 比特本身依赖调度历史。任何改变物理
分组的改动（包括此前讨论的 round-robin 消费者、wave 合并等）都会遇到同一
风险。因此物理调度不同不能直接推断数值逐位相同；也不能反向断言物理
调度不同必然改变结果。需分别资格验证 eig、Gram 与 pair 的数值身份。

## 可选路径

1. **原 GPU 差分重放（已完成）**：旧报告没有保留全部候选 scalar，所以先
   重放并保存 22 宽下的完整 self_moment' 比特；捕获全部受影响的原消费批次，
   包含 all-hit。原 payload 必须逐位复现 pair-cross、compact 与 canonical p。
   然后只替换 scalar，用原 pair 核的运算顺序与 GPU gamma 重放；不采用 CPU
   gamma 或 variance 比例缩放。1,186 个受影响 CI 最大 |Δp|=3.0531e-16。
2. **可归因例外门（不改生产）**：LSCP 子集验收改为"p/pMax 逐位一致，例外
   必须逐条归因到 singleton 类 self_moment 变化，且 |Δp| ≤ 1e-13、无
   alpha 翻转"，canonical 仍按 1e-10 合同独立验收。优点是零生产改动；缺点
   是逐位门有例外，且每次调度改动都要重新归因。
3. **生产侧批宽无关 Gram（改生产，一次性重新资格）**：把 Gram/self_moment
   改为逐分量的确定性计算（例如逐分量 `cublasDgemm` 或自写归约核），使 payload
   与批形状和调度历史无关，这是需要实测的目标。改动可能影响多分量批条目的
   self_moment 低位（本子集原多 miss 条目占约 84.1%），需要对 canonical 与三组真实数据按合同
   完整重新资格，并绑定新的 producer/cache 数值身份。只有完整硬门通过，
   才能建立新 baseline；其后的生产者或调度器改动仍须逐项验证。
4. **停止 LSCP**：保留现行路线，structured eig 走别的接法。

第 1 步已完成。剩余选项还包括上文的"eig 预生产、按原 miss 形状延后 Gram"
设计。GPU 差分很小支持评审这些选项，不自动采用第 2 条，也不免除第 3 条
对原 producer、kpcalg 与完整方法的重新资格。本轮只执行诊断。

## 证据

- payload 门报告与归档：`fastkpc/artifacts/full_cuda_ci/component_payload_20261021/`
  （manifest SHA-256 `e5c7fa25…`），本文读取其 `native-full/report.json` 与
  `native/raw.tar.gz` 中改变条目的原 payload。
- p 敏感度计算输入：`fastkpc/artifacts/full_cuda_ci/structured_real_20261020/pairs.csv`
  的 320 个真实 pair（statistic、mean、variance）。
- 上述 CPU 敏感度表仍属于离线分析；后续 GPU 回执与完整比特见
  `fastkpc/artifacts/full_cuda_ci/singleton_pvalue_20261021/` 和 GPU 重放报告。
