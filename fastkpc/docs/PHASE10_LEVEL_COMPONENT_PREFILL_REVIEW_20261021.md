# 22 列 level component prefill 原型：代码审阅、计时归因与重复运行

日期：2026-10-21。审阅
[默认关闭原型的资格结果](PHASE10_LEVEL_COMPONENT_PREFILL_RESULT_20261021.md)，
对象是未提交的 `full_cuda_ci_vertical.{cu,hpp}` 与 `full_cuda_ci_one_call.cpp`
改动和 prototype native `1d09d177…`。本文不改代码、不改状态；重复运行只读
使用已有 runner。

后续执行已完成，见[canonical 资格与归因](PHASE10_LEVEL_COMPONENT_PREFILL_CANONICAL_20261021.md)：
新 native `b1e7eab4…` 修正零任务层与设备恢复，22 列回归及 canonical 全量门
PASS，单次 canonical 892.468 s → 842.873 s。本文六次重复仍只属于旧
`1d09d177…`，不迁移为新 native 的重复证据；下文 canonical 估计是当时的历史预估。

## 结论

```text
代码审阅                         未发现会产生错误结果的缺陷；关闭路径是逐语句等价的重构
增量构建 ABI 风险                无：包含已改头文件的编译单元只有 3 个，全部重编
潜在问题（推广前须处理）          2 项：零任务层与设备恢复；level 数组已有入口边界保护，见下
单对计时的结构性归因              component 净 −9.8 s，consumer prepared 句柄重建消除 −4.0 s，
                                optimizer −3.1 s（重复运行证实为噪声）
重复运行（3 对交错，同一 binary）  off 中位 108.013 s → on 中位 94.214 s，−13.8 s（−12.8%）；
                                3×3 个交叉组合差全部为正（10.7–17.4 s）；optimizer 中位无差异；
                                扣除 optimizer 后稳定 −14.0 s；六次全部门 PASS
canonical 量级（本审阅时的估计）   dense 形状约 −30 s 量级；后续实测与证据边界见上方链接
```

## 代码审阅

审阅范围：`git diff` 中 vertical 357 行、header 31 行、one-call 198 行。

已确认正确的部分：

- **关闭路径等价**：consumer 原内联的距离、eig、选择语句移入
  `submit_legacy_three_fields`，Gram/归约移入 `submit_legacy_gram`；参数、
  handle、stream、stride、启动配置与原文逐项相同。Codex 的 off 回执对
  accepted subset 回执逐位一致，与此相符。
- **pending 与 miss 分离**：`eig miss_count` 只计真正的 miss，继续控制
  residual bypass、eig 与 workspace；`pending_gram_keys` 独立收集，按 consumer
  的 component 迭代顺序排列，与 baseline 中 `miss_keys` 的构造顺序相同。
  level ≥ 3 出现任何真正 miss 时在两处 fail closed（批构建与 `execute_group`）。
- **流序依赖**：`resolve_gram` 的 gather、Gram、归约与 self_moment 提交全部在
  persistent stream 上排队于 pair metadata 与 pair 核之前；pending 槽位写入
  NaN 哨兵，若被误读，pair 核会产生 `kDcovStatusNonfinite`，`require_ok` 抛错。
- **主机缓冲生命周期**：`build_level` 每批末尾 `cudaEventSynchronize` 后才释放
  `targets` / `solver_info` 主机向量；`store_residual` 每个 cohort 同步后才复用
  finalizer 槽位与 arena 系数 staging。
- **账本**：原 `entries == insert_count` 与 arena gather 等式按分项改写，
  `insert − prefill_produced`、`insert − discarded_complete − discarded_pending`、
  `reuse_hit + finalization_targets` 三处改写与实现一致；阶段 A 时间从
  optimizer host 边界中扣除，避免重复计时。
- **增量构建**：`full_cuda_ci_vertical.hpp` 只被 `full_cuda_ci_vertical.cu`、
  `full_cuda_ci_one_call.cpp`、`r_api_cuda.cpp` 直接包含，没有其他头文件转引；
  `FASTKPC_INCREMENTAL_EIG_ONLY=1` 正好重编这三个单元，结构体布局变化不会
  造成跨单元不一致。

推广到 canonical 或其他输入之前应处理的潜在问题：

1. **零任务层**：`begin_level` 在 `planned == 0` 时抛错。按 pcalg-stable 语义，
   上一层只要有一条存活边的快照邻居数大于 level 就会进入下一层；若该层删边
   降低了度数，下一层可能零任务。现有 one-call 对零任务层是允许的（跳过即可）。
   108 份归档回执中没有出现零任务层，22 列与 canonical 也没有，所以当前不触发；
   但推广后会表现为假性失败。建议 `planned == 0` 时跳过本层预填充。
2. **按 level 下标的定长数组（复核后不列为缺陷）**：`std::array<…, 65>`
   以 level 为下标（分组摘要、Gram 摘要、阶段 A 时间）。现有 one-call 入口已
   限制 `p <= 64`，结合 `level <= p - 2`，当前可达 level 最大为 62，已有完整
   边界保护。本轮不需要扩容；将来若扩大入口支持范围，才须一并审视数组边界。
3. **设备选择未恢复**：`begin_level` 调用 `cudaSetDevice` 后不恢复调用方设备，
   与本文件其余函数的恢复约定不一致。单 GPU 下无害；若将来启用多 GPU 目标需要修正。

另外两点属于性能而非正确性：阶段 A 每个 cohort 一次主机同步（子集 1,244 次、
合计 97.6 ms），阶段 B 每批一次主机同步（子集 332 次）；canonical 规模下
预计在秒级以内，届时以实测为准。

## 单对计时归因（Codex 归档的 off/on 回执）

| 项目 | off | on | 差 |
|---|---:|---:|---:|
| outer elapsed | 110.949 s | 93.461 s | −17.488 s |
| consumer dcov host（含 consumer eig） | 50.125 s | 18.364 s | −31.761 s |
| 阶段 B host（含提前的 eig） | — | 21.872 s | +21.872 s |
| 阶段 A host | — | 0.098 s | +0.098 s |
| multi-penalty prepared 句柄构建 | 7.182 s（4,732 次） | 3.190 s（2,140 次） | −3.992 s |
| selected-SP optimizer host | 45.998 s | 42.904 s | −3.094 s |
| 其余 | — | — | 约 −0.6 s |

逐层看，差值全部来自 level 3–5（−8.12 / −4.70 / −4.51 s），level 0–2 与 6
变化在 0.14 s 以内。

- **component 净 −9.8 s**：consumer eig 46.169 → 16.643 s（level 0–2 仍按需），
  阶段 B eig 21.533 s，合计 eig 净 −7.99 s；其余约 1.8 s 来自 consumer 批中
  finalize 与主机开销的减少。eig 的净收益来自批宽：level ≥ 3 的 consumer 批
  吞吐 225 分量/s，22 宽阶段 B 为 338 分量/s。
- **prepared 句柄重建 −4.0 s，结构性**：消失的构建次数 4,732 − 2,140 = 2,592，
  恰好等于 off 运行的 exact-primary 复用批数。off 运行中 consumer 每个复用批
  都要重建一次 multi-penalty 句柄做残差 finalization；on 运行中阶段 A 复用
  prefill 窗口已持有的句柄，consumer 全部走 all-hit 旁路。原结果文档没有单列
  这一项。
- **optimizer −3.1 s，性质未定**：并发 setup 主机时间合计下降 2%，最大单 setup
  时间不变；没有已知机制让 prefill 加速 optimizer，需要重复运行区分噪声与效应。

## 重复运行

同一 prototype native `1d09d177…`、同一 runner（`run.R`，SHA `c53f6930…`，与
Codex 归档逐字节相同）、同一开发环境（GPU0、connections 32、11 GiB cache、
bypass、8 lane、数学库线程 1）。顺序按 ABBA-BA 交错以抵消漂移：
off1、on1、on2、off2、off3、on3；on1/on2 以 off1 为分组摘要参照，on3 以 off3
为参照，off 运行以冻结 default-Inf 基线回执为参照。运行期间 GPU0 无其他
任务，GPU1 空闲；主机上另有其他用户的 CPU 进程，两种模式承受相同干扰。

| 运行 | 模式 | 门 | elapsed | optimizer host | consumer dcov host | 阶段 B host | 合计 eig | prepared 构建 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| off1 | off | 全部 PASS | 106.661 s | 42.136 s | 49.672 s | — | 45.751 s | 7.172 s |
| on1 | on | 全部 PASS | 92.370 s | 42.059 s | 18.094 s | 21.877 s | 37.927 s | 3.192 s |
| on2 | on | 全部 PASS | 95.968 s | 45.680 s | 18.142 s | 21.885 s | 37.981 s | 3.192 s |
| off2 | off | 全部 PASS | 109.786 s | 45.498 s | 49.539 s | — | 45.626 s | 7.101 s |
| off3 | off | 全部 PASS | 108.013 s | 43.763 s | 49.550 s | — | 45.635 s | 7.118 s |
| on3 | on | 全部 PASS | 94.214 s | 43.939 s | 18.146 s | 21.879 s | 37.978 s | 3.182 s |

| 汇总 | off 中位 | on 中位 | 差 |
|---|---:|---:|---:|
| outer elapsed | 108.013 s | 94.214 s | −13.799 s（−12.78%） |
| optimizer host | 43.763 s | 43.939 s | +0.18 s |
| elapsed − optimizer | 64.288 s | 50.288 s | −14.000 s |
| 合计 eig device | 45.635 s | 37.978 s | −7.66 s |
| prepared 句柄构建 | 7.118 s | 3.192 s | −3.93 s |

- 3 次 off 与 3 次 on 的 3×3 个交叉组合差全部为正，范围 10.69–17.42 s。
  这些组合重复使用同一批六次运行，不能当成 9 次独立重复或 9 对实验。
- optimizer 在两种模式下都在 42.1–45.7 s 之间波动，中位数没有系统差异。
  单对中 −3.1 s 的 optimizer 差值是运行间噪声，不是 prefill 的效应。
- 扣除 optimizer 之后的下降 14.0 s 在各次之间很稳定，与结构性归因相符：
  component 净 −9.5 s（consumer dcov host −31.4 s，阶段 B +21.9 s），prepared
  句柄重建 −3.9 s，其余约 −0.5 s。
- 六次运行的 24,264 个 CI、pMax、task/deletion、图、sepset、计数、RNG 门全部
  通过；on 运行的物理分组与 Gram 顺序摘要等于各自 off 参照，也等于 Codex
  归档的 final-on。运行前后 native SHA 不变。

结论：在这一 22 列输入、这一开发配置上，dense 形状的稳定收益约 14 s
（约 13%），比单对的 15.76% 小，主体是结构性的。这仍是开发证据，不是
clean-checkout 重复中位数或 promotion 证据。

## canonical 量级估计（未实测）

以下只是按已有回执的线性外推，不是预测：

- **prepared 句柄重建**：canonical 回执中 multi-penalty 句柄构建 32,452 次、
  49.7 s，exact-primary 复用批 15,264 个。若"每个复用批省一次重建"在 canonical
  上同样成立，按子集实测的约 1.5 ms/次，约 −23 s。
- **eig 批宽**：canonical level ≥ 3 的 consumer 批已较满（270 分量/s，65,694
  个分量 243.7 s）；48 宽阶段 B 按 338–355 分量/s、含 18.65% 浪费的 80,751
  个分量约 227–239 s，净收益约 5–17 s。
- **合计**：dense 形状在 canonical 上约 −30 s 量级（880 s 的 3–4%）。真正的
  杠杆是把阶段 B 换成 structured producer：按真实残差 640 宽吞吐约 900 分量/s，
  80,751 个分量约 90 s，对 243.7 s 的 consumer eig 约 −150 s。这仍需要
  structured 的完整资格与独立 cache identity。

## 建议

1. 推广前修正零任务层与设备恢复；level 下标数组已有入口保护，无需改动。
   修正后的源码/native 需单独记录并重跑 22 列回归，旧 native 的证据保留为历史。
2. canonical dense 资格：放开 `p == 22` 限制、上下文 48 宽；按 1e-10 合同验收，
   并要求对关闭运行逐位不同的 p 都归因到 54 个 ≥ 17 宽 cohort 中首次消费的
   完整残差/component identity，不能仅按 target 列号判定例外。
   同时实测 prepared 句柄重建的消除量，验证上面 −23 s 的估计。
3. structured 替换：阶段 B 已与 consumer 解耦，可以单独建 640 宽 structured
   上下文；延后 Gram 保持不变，singleton 语义继续保留。

## 复现与证据

重复运行证据：`fastkpc/artifacts/full_cuda_ci/level_component_prefill_repeat_20261021/`
（六次运行的 report.json、gzip 回执、运行日志，交错驱动 `drive.sh`、汇总脚本
`analyze.R`、`repeat-summary.csv`、native SHA 前后记录，以及逐文件 SHA 的
manifest）。

```text
prototype native   1d09d177b35a538474364faf11bfe2e5383bec56c83d562cfde0c24c1bbff1cf
runner run.R       c53f69306d4a9b3b…（与 Codex 归档相同）
manifest.json      1eae3b0fa8dfb68b3e006cc95e9ca0005bf4cfa18a53f81e08a9600306b871fd
```

复现（仓库根目录，GPU 空闲）：先 `mkdir -p /tmp/claude-lscp-repeat` 并把归档的
`drive.sh`、`analyze.R` 放入其中，再 `sh drive.sh`（按 off1、on1、on2、off2、off3、on3
调用 `with_phase10_exact_development_env.sh` 与 `level_component_prefill_20261021/run.R`），
然后 `Rscript analyze.R`。证据等级为同一 binary 的交错开发重复运行。
