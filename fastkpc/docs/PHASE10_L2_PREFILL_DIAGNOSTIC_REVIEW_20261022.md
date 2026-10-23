# L2 prefill 隔离诊断：复核与 runtime 集成要点

日期：2026-10-22。复核 [L2 隔离诊断](PHASE10_L2_PREFILL_DIAGNOSTIC_20261022.md)
（native `89a2fab0…` 未变）。本文不改代码、不改状态；下文的节省量是情景换算，
不是承诺，须由完整运行确认。

> **执行回填（2026-10-22）：** 上述默认关闭的 L2 opt-in 已完成
> [完整 runtime 资格](PHASE10_L2_PREFILL_RUNTIME_20261022.md)。同 binary S3/S2/S2/S3
> 为 687.491/547.191/535.142/694.347 秒，两次中位数 690.919 → 541.1665 秒，
> 减少 21.6744%。四轮全量数值、图、顺序、RNG 和资源门通过，各模式内逐位重复；
> L2 consumer 22,720 批全部旁路，Stage A/B 中位数 15.099/54.566 秒。
> 此处是 canonical opt-in development GO；默认、candidate、holdout、Phase 10 不变。
> 下文保留实施前的情景换算，不应再作为当前实测性能引用。

> **评审修订（同日）：** 采纳 Codex 的三处更正。
>
> 1. 初版写“L2 没有 all-hit 批”，这是错的。当时把 all-hit residual 时间为零误读成
>    没有 all-hit 批；实际上 13,052 批已经走旁路。
> 2. 初版称冻结重放“可以代表完整运行的输出”，说法过强，现已改为“支持推进集成”。
> 3. consumer 侧全命中断言原先遗漏，现已补上。
>
> 情景换算改用 structured canonical 回执中逐层 trace 的实测值，替换初版按分量数
> 分摊的估计；481 s 门的口径也已更正。

## 结论

```text
诊断结论            成立；closure、timing.csv、manifest 与文档一致，生产 native 未变
完整方法情景         L2 约 −130 s（逐层 trace 实测值换算；阶段 A 只计隔离循环）
                    structured 两次中位数 691 s → 约 560–570 s；需要完整运行确认
同意的下一步        默认关闭的 L2 runtime opt-in 完整资格；首版沿用 ≤16 分组与串行阶段 A
计时对照            同一 binary 上 S3 → S2 → S2 → S3（structured L≥3 / L≥2）交错
```

## 复核要点

- **数字**：Stage B（dense 144.34 / structured 54.46 s 中位数）、阶段 A 14.79 s、
  structured 吞吐约 917/s、峰值 12,393 / 15,701 MiB 均与 `timing.csv`、`closure.json`
  一致；manifest SHA `1af2e8c1…`。
- **残差归因**：L2 只有 q = 30 一种形状。原宽度 ≥ 17 的 cohort 发生低位变化，其余
  逐位相同，与 multi-penalty 在 `cublasDgemm(n, target_count, q)` 上的宽度分类现象一致。
  L2 宽 cohort 比例高：消费分量中 18,539 / 42,685（43%）发生变化，但 p 最大只变
  1.05e-14（structured），按合同验收没有问题。
- **冻结重放的作用**：L2 零决策翻转，说明后续各层的任务集预计保持不变，这支持
  推进集成。但它不能代表已验证的完整 runtime 输出。残差位模式取决于数据、S、
  target、selected-SP 位模式与物化分组；保持物理分组也不足以验证新增的缓存与
  生命周期实现。完整 240,498 个 CI、pMax、trace、RNG 与资源门须重新通过。

## L2 consumer 现状（structured canonical，structured1 的逐层 trace）

| 批类别 | 批数 | residual host |
|---|---:|---:|
| all-hit（已走旁路） | 13,052 | 0 |
| mixed | 8,454 | 29.879 s |
| all-miss | 1,214 | 4.475 s |
| 合计 | 22,720 | 34.353 s |

- L2 consumer 的 dCov host 为 171.061 s，其中 eig 158.980 s，component build（距离、
  eig、选择、Gram）167.978 s。structured2 与此几乎相同（158.949 / 34.348 s）。
- L1 的 residual 只有 0.065 s，所以 single-penalty residual 34.44 s 几乎全在 L2，
  初版的 32.5 s 分摊估计偏低。
- 集成后的预期：22,720 批全部命中，其中 9,668 批（8,454 mixed + 1,214 all-miss）
  新增走旁路；consumer residual 求解归零；阶段 A 求解 49,919 个 target。

## 完整方法情景

单位为秒。

| 项 | 现状 | L2 prefill 后 |
|---|---:|---:|
| L2 consumer dCov host | 171.1 | 约 5 |
| L2 consumer residual host | 34.4 | 0 |
| 阶段 A（隔离循环，含句柄创建与释放） | — | 约 14.8 |
| 阶段 B（structured，49,919 个分量） | — | 约 54.5 |
| 净变化 | | 约 −131 |

- “约 5 s”按全命中 consumer 组的实测成本换算：L2 现有 all-hit 组每组 0.128 ms，
  L3–L5 带延后 Gram 的组每组 0.22 ms；22,720 组 × 0.22 ms ≈ 5 s。
- 阶段 A 的 14.8 s 来自隔离循环，不含 runtime 的初始化与 49,919 个条目的缓存登记。
  L2 其余约 52 s（optimizer 7.3 s、consumer 主机开销等）假定不变。
- 因此约 −130 s、整体约 560–570 s 只是情景，须由完整运行确认。

## runtime 集成要点

1. **阶段 A 的位置与句柄**：single-penalty prefill 窗口只把 selected SP 写入
   target state cache，不持有 fixed-SP prepared 句柄（`complete_single_penalty_prefill_window`）。
   因此阶段 A 需要自建句柄。应直接使用窗口已持有的 `work->contexts`，不要再经
   `setup_cache.get` 取 context，否则 LRU 状态和 frontier 偏好会变，物理分组摘要
   将不一致。selected log-SP 直接沿用原位模式，不重新计算。
2. **入口与 fail-closed**：三处都要扩到 level 2：
   - `begin_level` 目前拒绝 `level < 3`（vertical.cu:1896）；
   - one-call 的 `prefill_this_level` 写死 `level >= 3`；
   - consumer 的全命中断言（`full_cuda_ci_one_call.cpp:6170`）只覆盖 `level >= 3`。
     L2 开启后同样必须断言全命中，避免预填遗漏时静默回到 consumer 求解。

   L2 的 reset 会丢弃 L0–L1 的 2,304 个条目；L2 计划生产 49,919 个，低于容量 119,817。
3. **账本**：验证 L2 的 mixed / all-miss residual 时间归零、22,720 批全部 all-hit、
   阶段 A 求解数等于 49,919。
4. **rotation 余量**：L2 最大 147,167，容量 170,662，余量 13.8%，比深层的 22% 小。
   溢出时会回退到 dense，结果仍正确；回执已记录逐层最大值，继续监测即可。
5. **显存**：pool 改为在 L2 创建，但整轮峰值出现在运行末尾（缓存累积），预计
   仍约 21,991 MiB；L2 的条目在预分配的 11 GiB cache 内。须看实测。
6. **实施顺序**：首版沿用已验证的 ≤16 分组与串行阶段 A，先完成完整资格与计时；
   跨 stream 流水作为下一项独立实验，便于分清收益来源。
7. **计时对照**：同一 binary 上按 S3 → S2 → S2 → S3 交错，阶段 A、阶段 B 与 L2 elapsed
   分开报告。optimizer 的同模式波动可达 ±8 s，不能据此作判断。

## 之后

若 L2 实测约 −130 s，canonical 约 560 s，其中 optimizer 约 333 s（约 60%），
structured eig 约 144 s。剩下的杠杆是阶段 B 与 optimizer 重叠（上界约 84 s，有 FP64
争用风险，需先测），然后是 optimizer 本身。

关于门槛：481 s 只是用历史 CPU 基准（601 s）换算的 0.8×，正式门仍是同一轮正确
CPU baseline 的 0.8×。120 s 门在这条线上达不到。
