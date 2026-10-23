# 22 列 level component prefill：默认关闭原型资格

日期：2026-10-21。承接[路径 (i) 设计](PHASE10_LEVEL_COMPONENT_PREFILL_PROTOTYPE_PLAN_20261021.md)
与[gather→Gram 第 0 步](PHASE10_GATHER_GRAM_STEP0_20261021.md)。

本文保留 `1d09d177…` 的 22 列历史结果。后续生命周期修正与新 native 的
canonical 351×48 资格已完成，见[canonical 执行结果](PHASE10_LEVEL_COMPONENT_PREFILL_CANONICAL_20261021.md)；
其证据身份与本文分开记录，生产默认开关仍关闭。

本轮输入是 `cancer_RD-causalDiscoveryInput.rds` 前 22 列，351×22、default-Inf、
`dcc.gamma`。实现了窗口内残差物化、整层三字段
预填充与首次消费时的原形状 Gram。开关默认关闭，显式拒绝其他输入形状。
canonical 351×48、其他数据、structured eig、生产默认启用均未进入本轮。

```text
22 列默认关闭 dense 原型实现与资格    PASS
完整四字段 / CI / pMax 逐位门        PASS
canonical 与生产默认启用             NOT YET GO
```

## 实现与边界

- 阶段 A 位于 `complete_multi_penalty_prefill_window` 内部：arena 系数存储及原
  residual 租约释放之后、prepared handles 释放之前。复用窗口句柄，cohort 均衡
  切到 ≤16；D2D detach 完成后才复用 residual 槽位与 coefficient staging。
- 层开始先检查缓存容量并等待上一层最后读者，再 reset；阶段 B 只运行原距离、
  deterministic eig、选择与 total 语句，写入三字段。原按需路径与新路径共享
  同一数值实现，self_moment 显式 pending，槽位写入 NaN 哨兵。
- 真正 eig miss 与 pending Gram 分开。pending 独立检查/取得 workspace，在
  原有持久 stream 上完成 D2D gather、原 Gram/reduction 与 scalar store，随后
  pair 继续在同一 stream 消费。元数据 complete 表示同 stream 已排队，不是
  跨 stream 无依赖读取的许可。没有增加每个 pending 消费的显式中间 host wait。
- level≥3 意外 miss 立即 fail closed；不实现混合 MISS/PENDING 回退，也没有 CPU
  numerical fallback。阶段 B 的 compact eig status 校验失败时同样不发布结果。
- 每层物理分组摘要包含 conditioning set、有序 selected/executed tasks；Gram
  摘要包含 level、完整 request identity、计数和有序 component keys。两者都
  在同一新 binary 的 off/on 运行之间比较。
- stage A、stage B、gather、Gram、三字段/scalar 写入及 reset 丢弃分别记账。
  旧的 consumer counters 保持其消费阶段含义；总 eig 工作必须另加阶段 B。

## 最终 native 与证据类别

```text
accepted reference native
  fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50
prototype native
  1d09d177b35a538474364faf11bfe2e5383bec56c83d562cfde0c24c1bbff1cf
```

重编译全部包含变更 vertical header 的 translation units：one-call、CUDA
vertical 与 R API，其余对象来自 accepted build。源码、native 与回执精确绑定，
证据类别是 **source-closure development**，不是 clean-checkout 重复中位数或
promotion evidence。之前 `e03613bc…` 的开发检查没有混作最终 SHA 的回执。

最终 native 的 off/on default-Inf 两次运行均已完成，24,264 个 CI 与 pMax 全部
逐位相同；off 另对 accepted subset receipt 逐位相同。task/deletion、adjacency、
sepsets、n.edgetests、RNG endpoint、物理分组及 Gram key/顺序摘要全部通过。
native CPU authority、fallback、residual/component 大 D2H 为零，资源闭合。
本轮不新增 CPU oracle 或最终 PDAG campaign。

## 完整账本

| level | planned / produced | resolved | pending at end | 阶段 A cohorts | 阶段 B batches | 原形状 Gram batches |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 4,662 | 4,148 | 514 | 630 | 212 | 1,589 |
| 4 | 2,093 | 1,981 | 112 | 444 | 96 | 816 |
| 5 | 473 | 461 | 12 | 149 | 22 | 166 |
| 6 | 42 | 42 | 0 | 21 | 2 | 21 |
| 合计 | 7,270 | 6,632 | 638 | 1,244 | 332 | 2,592 |

消费阶段 exact-primary finalization targets 为 0；物化全部移入阶段 A。
layer reset 4 次，累计丢弃 complete 11,173、pending 638，最终驻留 42。
level 0–2 按需插入 4,583；`4,583 + 7,270 = 11,173 + 638 + 42`。
route/status mismatch 均为 0。

## 单次成本对照

| 配置 | outer elapsed |
|---|---:|
| 同 binary，off | 110.949 s |
| 同 binary，on | 93.461 s |

观测下降 17.488 s（15.76%）。这只是一对完整 development 运行；selected-SP
optimizer host 时间同时从 45.998 s 变为 42.904 s，这部分单次差异不能归功于
component prefill，也不能据此宣布稳定 15.76% 或预测 canonical 收益。

阶段 B host 合计约 21.872 s，其中 eig device 约 21.533 s。消费阶段 eig
从 46.169 s 变为 16.643 s，但必须把提前执行的 eig 加回：新总 eig device 约
38.176 s，观测下降约 7.993 s。不能将消费阶段移走的约 29.526 s 全部记成收益。

阶段 A host 合计约 97.6 ms，gather device 约 36.5 ms，Gram/scalar store device
约 176.5 ms。新增 gather 为 653,649,920 字节；层 residual 存储峰值为
13,090,896 字节。缓存仍使用原 11 GiB 预算，332 个 prefill 批的额外工作包含
638 个最终未消费分量。以上时间有嵌套，不能直接相加推导 outer elapsed。

## 验证与归档

最终 native 的独立四字段观察运行已完成：实际 pair 消费的 6,632 个分量四字段
全部逐位等于 accepted 捕获。`archive.R` 另对这 6,632 份 reference 文件逐个
核对 accepted raw manifest 的 SHA，全部通过。外部观察器有显式诊断 D2H，
该运行不用于性能比较；数值结果始终由原 GPU consumer 产生。

最终源码的 source-shared sanitizer 冒烟覆盖 singleton/双分量 gather、重复解析、
未消费 pending reset、零消费层、容量不足、非法 rank/重复 key 与异常资源清理：
0 errors、0 bytes leaked。它不是完整 R 方法的全程 sanitizer 资格。
新配置拒绝/资源闭合测试与既有 legacy component-cache 回归测试均 PASS。

归档校验为 PASS，最终 native/source/回执身份一致。完整结果见
[closure.json](../artifacts/full_cuda_ci/level_component_prefill_20261021/closure.json) 与
[manifest.json](../artifacts/full_cuda_ci/level_component_prefill_20261021/manifest.json)。

```text
manifest SHA-256
  bb26901e47935df1b86cc510fd564ddd9986aa026d266c79d805867a129bedb0
```

工具：[level_component_prefill_20261021](../tools/research/level_component_prefill_20261021/README.md)。
证据目录：`fastkpc/artifacts/full_cuda_ci/level_component_prefill_20261021/`。

Phase 10 保持 ACTIVE；candidate 未冻结，sealed holdout 未打开，recommended=false。
历史 eager 四字段的 STOP_PAYLOAD_BITS 保留。当前结论只用于默认关闭的 22 列
dense 原型，后续扩展仍需各自的完整资格。
