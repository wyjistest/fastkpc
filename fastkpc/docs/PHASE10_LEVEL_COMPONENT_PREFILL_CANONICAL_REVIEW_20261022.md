# Canonical dense prefill：复核、成本归因与 structured 阶段 B 下一步

日期：2026-10-22。复核 [canonical 资格](PHASE10_LEVEL_COMPONENT_PREFILL_CANONICAL_20261021.md)
（native `b1e7eab4…`）。本文不改代码、不改状态，也没有新的 GPU 计时；
数字全部取自 Codex 归档的回执。下文的节省量是按吞吐换算的上界，不是预测。

后续窄资格已完成，见 [structured device 适配器结果](PHASE10_STRUCTURED_DEVICE_ADAPTER_20261022.md)：
1,735 个分量、1,294 个原 GPU consumer 配对、批形/并发逐位门和 GPU dense
fallback/resource 门通过。完整方法仍未集成，下一门为 opt-in 22 列资格；
下文 L2 residual 的可消除成本及约 495 秒组合场景尚未证实，不进入正式预算。

## 结论

```text
资格结论             成立；源码/native 绑定、两处修正、账本恒等式与计时表复核无误
单对收益构成         L3–L7 合计 −57.7 s（其中 optimizer +0.7 s）；L2 +3.5 s、L8 +4.6 s 为单次波动
prepared 句柄重建    −29.1 s（15,264 次 × 1.91 ms），比此前 −23 s 的估计大
on 模式成本图        optimizer 341.5 s | 阶段 B 233.4 s | consumer dCov 181.2 s（含 L0–2 eig 165.5 s）| 其余约 86 s
建议的下一步         structured 阶段 B（level ≥3，默认关闭，dense 阶段 B 保留为对照）
之后的两条杠杆       阶段 B 与 optimizer 窗口重叠；level 2 single-penalty prefill
120 s                eig 全部归零仍约 446 s；optimizer 是剩余主项
```

## 复核

- **绑定**：工作树三份源码的 SHA 与 `closure.json` 一致；
  `fastkpc/build/fastkpc_cuda.so` 为 `b1e7eab4…`；manifest SHA `17952f42…` 与文档一致。
- **修正 diff**（相对 22 列原型 `1d09d177…` 的源码快照）只有三处：
  1. one-call 用 `prefill_this_level`（`level >= 3 && task_count > 0`）统一控制
     begin/build/finish；
  2. `ScopedPrefillDevice` 覆盖 `begin_level`、`store_residual`、`build_level`、
     `resolve_gram`、`finish_gram_timing`、`finish_level`，析构时恢复调用方设备且不抛异常；
  3. 形状门放宽为 351×22 或 351×48。

  其余代码逐字不变，与回执一致。
- **推广限制**（当前不触发，不算缺陷）：`begin_level` 在 `planned > capacity_entries_`
  时 fail closed；canonical 最大单层 54,878，低于 119,817。推广到更大输入之前，
  需要分块生产或部分预填充策略，不能只靠报错。
- **账本**：`44,989 + 80,751 = 110,665 + 15,057 + 18 = 125,740`；
  `110,665 = 44,989 + 65,694 − 18`（level 8 的 18 个条目在结束时仍驻留）。
- **计时口径**：`timing.csv` 的 consumer dCov 取 `cuda_dcov_host_ms`（vertical 内
  `dcov_start` 边界）；prepared 构建计时位于 one-call 的句柄创建处
  （`full_cuda_ci_one_call.cpp:2870–2892`），两者不重叠，差值可以分别归因。
  optimizer、native setup、prepared 与 residual solve 之间有包含关系，整张表不能求和。

## 单对收益的逐层构成

单位为秒；optimizer 是 `prefill_batches` 的逐层 `optimizer_host_ms` 合计。

| level | off | on | on − off | on optimizer | on 阶段 B |
|---|---:|---:|---:|---:|---:|
| 0–1 | 7.607 | 7.641 | +0.034 | 0.241 | — |
| 2 | 245.959 | 249.433 | +3.475 | 7.168 | — |
| 3 | 250.613 | 223.577 | −27.036 | 48.186 | 158.565 |
| 4 | 170.966 | 153.151 | −17.815 | 91.876 | 51.350 |
| 5 | 138.156 | 130.045 | −8.111 | 105.247 | 19.705 |
| 6 | 48.361 | 44.616 | −3.745 | 39.780 | 3.317 |
| 7 | 12.280 | 11.324 | −0.957 | 10.604 | 0.362 |
| 8 | 16.760 | 21.329 | +4.569 | 21.164 | 0.055 |

- 预填充只作用于 level ≥ 3。L3–L7 合计 −57.66 s，其中 optimizer 合计 +0.72 s。
- L2 不受开关影响，+3.5 s 是运行间波动。L8 的 +4.6 s 等于 multi-penalty 单个
  setup 最大 host 时间的增量（16.529 → 21.164 s）；两次 optimizer 的迭代总数
  （890,171）与最大迭代数（187）相同。两者都属于单次运行波动，与 Codex
  “optimizer 单次差异不归因于 prefill”的判断一致。
- dense 形状在 canonical 上的结构性收益可能接近 −58 s，而不是 −49.6 s；
  这需要重复运行才能确认，本文不作此主张。

## on 模式的成本图

| 项 | 秒 | 占 842.873 s |
|---|---:|---:|
| selected-SP optimizer host | 341.5 | 40.5% |
| 阶段 B host（其中 dense eig 231.3 s） | 233.4 | 27.7% |
| consumer dCov host | 181.2 | 21.5% |
| 其中 level 0–2 按需 eig | 165.5 | 19.6% |
| 其余（setup、residual solve、调度等，存在包含关系） | 约 86 | 约 10% |

eig device 合计 396.8 s，约占 47%。

## 三条杠杆（吞吐换算上界）

### 1. structured 阶段 B（level ≥ 3）

- dense 阶段 B 对 80,751 个分量用了 231.3 s eig，逐层吞吐 346–349 分量/s。
- structured 在真实残差上的 640 分量吞吐为 889–927/s
  （[real replay](PHASE10_STRUCTURED_EIG_REAL_REPLAY_20261020.md)，n = 351，
  含 11 GiB 共驻留）。按 900/s 换算为 89.7 s，约省 140 s。
- 其中 15,057 个未消费分量：dense 花费 43.1 s，structured 约 16.7 s。

### 2. 阶段 B 与 optimizer 窗口重叠

- 当前阶段 B 在全部 prefill 窗口结束后串行执行（`build_level`，
  `full_cuda_ci_vertical.cu:1854`）。阶段 A 已按窗口执行，所以阶段 B 可以在每个
  窗口的阶段 A 之后，在自己的 stream 上提交，frontier 开始前统一同步一次。
- 理想重叠上界为 Σ_L min(阶段 B_L, optimizer_L)：dense 123.0 s，structured 76.9 s。
  structured 的上界受 L3 的 optimizer 只有 48.2 s 限制。
- dense 能否与 optimizer 共处尚不清楚：`sytrd4_gpu` 的同类并发只有 1.04
  （[execution 文档 §2](PHASE10_EXACT_SPEEDUP_EXECUTION_20261020.md)），可能占满设备。
  structured 的 kernel 是 32 线程块、延迟受限，更适合与同样延迟受限的 optimizer
  共处。实际收益取决于 optimizer 被拖慢多少，只能实测。
- 比特：dense 三字段与批组成无关，已在 canonical 上成立（64,743 个未受残差
  影响的分量四字段逐位一致）。structured 原型每个分量独占一个 block 和一份
  workspace，`tile` 不依赖批大小，按构造应当批无关，但仍须作为 gate 0 实测。

### 3. level 2 single-penalty prefill

- L2 planned 49,919，consumed 42,685（未消费 14.5%）。按需 eig 约 157–159 s，
  single-penalty residual solve 34.4 s（L1–L2 合计）。
- structured 预填充按 900/s 约 55 s；consumer eig 与 residual solve 移除后，
  粗略量级约 −130 s。这需要新的 single-penalty 阶段 A（其残差 cohort 语义尚无
  实验），容量 49,919 低于 119,817。

三条都达到上界时约为 842.9 − 140 − 77 − 130 ≈ 495 s，其中 optimizer 占 341.5 s。

## 建议：第一步做 structured 阶段 B

范围：level ≥ 3，新开关默认关闭；dense 阶段 B 原样保留为对照；不改调度，
不改延后 Gram。

设计要点：

1. **输出合同与 dense 相同**：`selected_values` 按 |λ| 降序，并列时取较大的升序下标
   （`select_center_legacy_eigencomponent_kernel`，`full_cuda_ci_vertical.cu:828`）；
   向量为 centered vectors；`total` 为 Σ D_ij。self_moment 继续由 consumer 按原
   miss 分组延后 Gram 计算。
2. **独立 structured 上下文**：原型每个上下文最多 64 个分量，每个分量 workspace
   约 4.26 MB（640 分量为 2.54 GiB）。本次 campaign 观察到 GPU0 显存峰值
   18,681 MiB（卡容量 24 GB）。阶段 B 运行时 prefill 句柄已释放，但 setup cache
   仍驻留，峰值需要实测。
3. **逐分量 GPU dense 回退**：info ≠ 0、rotation 容量溢出、k 边界上 |λ_k| 与
   |λ_{k+1}| 的相对间隙低于阈值时，该分量改走 dense；在条目元数据与回执中记录
   solver identity 和回退计数。不允许 CPU 参与。
4. **cache identity**：条目带 solver 标识。按层 reset 已保证不会跨层混用，但
   key 或元数据仍需显式区分。
5. **门**：
   - gate 0：structured 三字段对批宽、批内位置与顺序逐位不变（为第二步重叠做准备）。
   - 22 列子集与 canonical：全部 CI/pMax 对 kpcalg authority ≤ 1e-10；
     task/deletion、图、sepsets、n.edgetests、RNG 精确；物理分组与 Gram key/顺序
     摘要与 dense-on 相同；开关关闭时对已接受回执逐位一致；memcheck 与资源闭合。
   - 与 dense-on 的 p 差异逐分量归因（预期在 1e-14 量级，以实测为准）。
   - 计时：同一 binary 上的 off / dense-on / structured-on。
   - 三套真实数据（n = 242/253/364）需要放开 351×{22,48} 形状门，另列一步。

第二步（重叠）和第三步（level 2）在第一步通过之后再决定。

## 关于 120 s

即使 eig device 全部归零，on 模式仍约 446 s；三条杠杆都达到上界也约 495 s。
剩余主项是 optimizer 341.5 s（multi-penalty 316.9 s，890,171 次迭代）。

- 120 s 门在这条线上达不到。
- ≤ 0.8× CPU（481 s）门只有在三条杠杆都接近上界、optimizer 也另有进展时，
  才可能进入范围。

最近一次 optimizer 占用剖析（launched-block 并发上界约 34）是在默认 8 connections
下做的，改成 32 connections 之后没有重测。若要继续追 120 s，应先在 32 connections
下重测 optimizer 的 SM 占用，再判断保持比特的执行层改造是否值得做。

## 工作区状态

生产源码改动（vertical.{cu,hpp}、one_call.cpp、lane benchmark）与 `goal-5.6.md`
的修改仍未提交。`fastkpc/build/fastkpc_cuda.so` 是 `b1e7eab4…`，不是 accepted
`fe8286f7…`（副本在 `fastkpc/artifacts/full_cuda_ci/exact_speedup_20261020/baseline-fe8286f7.so`）。
是否先把默认关闭的原型、文档与研究工具提交为检查点，由用户决定。
