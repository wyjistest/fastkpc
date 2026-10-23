# L2 structured component prefill：完整 runtime 资格

日期：2026-10-22。前置证据为 [L2 隔离诊断](PHASE10_L2_PREFILL_DIAGNOSTIC_20261022.md)
及其 [runtime 集成评审](PHASE10_L2_PREFILL_DIAGNOSTIC_REVIEW_20261022.md)。

## 当前状态

**PASS_CANONICAL_OPT_IN。** 同 binary 完整 S3 → S2 → S2 → S3 全部通过。
两次中位数从 **690.919 秒降至 541.1665 秒**，减少 **149.7525 秒（21.6744%）**。
各模式内两次 CI/pMax 均逐位一致，S2 满足全部 authority 1e-10 合同和资源门。
源码构建、配置拒绝、retained cache 回归、生命周期、CUDA memcheck 和小型 runtime
检查全部通过。该结论只覆盖当前 canonical opt-in；默认配置、Phase 10、candidate、
holdout 和 recommended route 均未改变。

## 实现边界

- 新开关 `FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL=1` 默认关闭；只接受
  `dcc.gamma / n=351 / p=48 / numCol=35`，并要求既有 structured/level prefill、
  fresh caches、exact-primary reuse、component cache、bypass、八 eig lanes 和原 scheduler。
- S3 为原 structured L>=3；S2 在完全相同的执行栈上增加 L2。
- 阶段 A 直接使用 single-penalty optimizer 窗口已有的 context，不额外经过 setup LRU。
  输入 Y 在 optimizer 输入移动前保留；selected log-SP 沿用 optimizer 原始位模式。
- 每个 S 创建一次 exact-fixed prepared handle，将 target 平衡分为宽度不超过 16 的
  cohort，按顺序执行原 stable-SVD evaluator。residual 在 workspace 复用前完成 D2D
  detach，结束后释放 handle。没有新增跨 stream 流水或改变数值内核。
- 阶段 B 使用既有 structured pool。consumer 保持原物理分组和 Gram key/order，
  对启用预填的层强制检查全命中、完整 residual bypass、consumer residual/eig 时间为零。
- 新增逐层 consumer 账本和 L2 input-copy / prepared-create / evaluate / release 计时。
  完整阶段 A 另计 D2D detach、缓存登记和控制成本，避免只报 evaluator 的局部时间。

## 构建和接入检查

最终 native SHA-256：

```text
8e457e811eaa01ae85326b8ff3cc0bcfa56f5b907ad6840a302a8205370d3874
```

先从空目录构建全部 43 个对象，耗时 2470.969 秒。首个 runtime smoke 捕获了一处
新接入的计数错误：旧 coefficient-arena 断言将全部 prefill finalization 都算作
coefficient gather，而 L2 使用独立 exact-fixed evaluator。该版本以 fail-closed
终止；错误 native `c051bd48…f6c1d4`、源码和日志保存在 `failed-ledger-revision/`。

修正增加独立 `l2_component_residual_targets`，coefficient-arena 继续严格核对深层
gather。只重编对应 C++ translation unit 并链接，其他 42 个对象逐一验证哈希不变。
本轮最终 binary 的全部对象均来自本轮从空目录建立的 source closure，没有借用历史
campaign 的对象。构建回执记录此次修正；这仍是未提交的 development evidence。

最终版本重新通过：

- L2/structured/level 非法配置和 resource closure；retained component cache 回归。
- 640 个真实组件的逐位 payload；56/70 mixed GPU dense fallback、48-capacity chunking、
  不完整 producer、重复 resolve、pending reset、失败不发布、rotation overflow、abandoned submit。
- L2 只允许 structured 入口，L1 和 dense L2 拒绝。
- CUDA memcheck：0 errors、0 leaked bytes。
- 真实 runtime 小型对照：保留 canonical 前八列，其余 40 列独立行置换；维度仍为 351×48，
  最大层数为 2。S3/S2 均完成 2,622 CI，任务/图/sepsets/计数/RNG/物理分组/Gram 顺序一致，
  最大相互 p 差为 `8.3544282603043e-15`，时间为 7.555/5.547 秒。
  L2 生产 477 个组件，220 个 consumer 批次、780 个 target 全部旁路，consumer residual/eig 为零。
  此小型用例仅是 GPU 对照接入检查，不是 CPU authority 或完整性能证据。

## 完整 campaign 口径

同一 native、source closure、GPU 0、CUDA connections=32、BLAS/OMP 线程数=1，
11 GiB component cache、bypass=1、eig lanes=8。每轮 fresh R process 和 fresh semantic caches；
开始前两张 GPU 均无 compute jobs，GPU 1 保持空闲，每 200 ms 采样显存/利用率等。

顺序为 `s3a → s2a → s2b → s3b`。第一个 S3 要求与历史 S3 的全部 CI/pMax 逐位一致；
S2 要求独立归档的 kpcalg authority 全部 240,498 个 CI 和所有 pMax 均在 1e-10 内。
只有 L2 允许 p 位模式变化，每个配置的重复运行须逐位一致。另检查完整任务/删除轨迹、
skeleton、sepsets、n.edgetests、RNG、物理分组、Gram key/order 和全部资源/authority 门。

计时边界为完整 skeleton 调用；序列化、压缩和离线 authority 比较不进入该边界。
报告分别列出完整 outer、L2 elapsed、阶段 A、阶段 B、begin/reset、finish 和 consumer 成本。
prepared-build 计时与阶段 A 部分重叠，不能再次加总。

## 完整正确性和账本

两次 S3 的所有 CI/pMax 与历史 S3 逐位一致。两次 S2 均完整通过 240,498 个 CI，
最大 CI/pMax authority 误差为 `3.05425129631942e-12`。相对同 binary S3，
125,078 个 CI 和 732 个 pMax 元素发生低位变化，最大为 `1.0547118733939e-14`；
全部 CI 变化都在 L2，其他层逐位相同。L2 的 125,293 个结果与先前 isolated structured
诊断逐位相同。没有 decision flip，任务/删除轨迹、skeleton、sepsets、计数、RNG、
physical-group digest、Gram key/order digest 全部一致。S2 两次 CI/pMax 逐位一致；
四轮 optimizer iteration sum/max 均为 890,171/187，exact-primary targets 均为 80,751，
stability replay 和 terminal confirmation 均为零。

authority p 位于 0.05–0.15 的 1,337 个 CI 全部通过，其中 L2 为 527 个，最大误差同为
`3.05425129631942e-12`。该合同是连续 p 值误差不超过 1e-10，不能把 S2 描述为相对 S3
全局逐位不变。CPU numerical authority、unknown/CPU fallback、residual/component 大块
D2H、cache route/status mismatch 和退出时 CUDA 资源均为零。

| L2 项目 | S3 | S2 |
|---|---:|---:|
| consumer batches | 22,720 | 22,720 |
| all-hit batches | 13,052 | 22,720 |
| mixed / all-miss | 8,454 / 1,214 | 0 / 0 |
| bypassed target occurrences | 60,066 | 147,176 |
| consumer residual target occurrences | 87,110 | 0 |
| Stage A prepared handles | 0 | 1,126 |
| Stage A residual cohorts / targets | 0 / 0 | 3,356 / 49,919 |
| produced / consumed unique components | — | 49,919 / 42,685 |
| pending unused components | — | 7,234 |

S2 L2 的 49,919 个 structured components 全部接受，dense fallback 为零，78 waves，
peak inflight=640。最大 rotations 为 147,167，容量 170,662，保留约 13.8% 余量。
全层共 produced/resolved/pending 为 130,670/108,379/22,291。

## 完整性能

单位为秒；同一台机器、同一 GPU、同一 binary，按下表顺序执行。

| 顺序 | 模式 | outer elapsed | L2 elapsed | optimizer host |
|---|---|---:|---:|---:|
| 1 | S3a | 687.491 | 248.941 | 330.477 |
| 2 | S2a | 547.191 | 93.943 | 344.737 |
| 3 | S2b | 535.142 | 93.803 | 333.504 |
| 4 | S3b | 694.347 | 248.633 | 337.113 |
| 两次中位数 | S3 | **690.919** | **248.787** | 333.795 |
| 两次中位数 | S2 | **541.1665** | **93.873** | 339.120 |

S2 相对 S3 的完整 outer 加速为约 **1.277×**。最慢 S2 仍比最快 S3 少 140.300 秒，
明显大于各模式内部的 12.049/6.856 秒跨度；这是两次 development 对照，不是置信区间
或正式 promotion 中位数。两个完整 L2 elapsed 只相差 0.140 秒。

| L2 分项（两次中位数） | S3 | S2 |
|---|---:|---:|
| 完整 Stage A | 0 | 15.099 |
| 其中：保留 Y | 0 | 0.027 |
| 其中：prepared 创建 | 0 | 2.230 |
| 其中：exact-fixed evaluator | 0 | 12.297 |
| 其中：prepared 释放 | 0 | 0.395 |
| Stage B | 0 | 54.566 |
| begin/reset | 0 | 0.000404 |
| finish | 0 | 0.001554 |
| consumer residual host | 34.180 | 0 |
| consumer dCov host | 170.309 | 3.791 |
| consumer eig device（包含于上述 dCov） | 158.304 | 0 |

完整 Stage A 还有约 0.150 秒的 D2D detach、缓存登记和控制成本，不能只取 evaluator
计时。以上明确列出的执行边界净减少约 131.031 秒；L2 实际整层减少 154.914 秒，剩余
约 23.883 秒来自这些边界之外的工作变化。全方法 prepared-build 由 32.393 降至
13.931 秒，减少 18.462 秒，支持 consumer handle 生命周期减少的解释；该计数同时
覆盖 optimizer 和 fixed 两侧，不能将全部 23.883 秒都归给它，也不能再重复加到收益中。

整体 outer 节省 149.753 秒，与 L2 整层节省 154.914 秒相符；optimizer host 中位数反而
增加 5.325 秒。此次收益因此不能归因于 optimizer 变快。S2 当前主要累计 host boundaries
为 optimizer 339.120 秒、全层 Stage B 144.048 秒；后续并发机会仍需独立诊断，本轮未实施。

## 完整运行的容量

四轮 200 ms 采样的 GPU 0 峰值均为 **21,991 MiB**，没有增加完整运行采样峰值。
GPU 1 维持 425 MiB 桌面占用，采样利用率峰值为 1%/0%/0%/0%；本轮没有在另一张卡
安排计算。所有 one-call 退出资源计数归零，memcheck 无错误或泄漏。
采样峰值不是瞬时 allocation 上界；当前容量结论仅针对本次 canonical 配置。

## 证据和复现

工具：[l2_prefill_runtime_20261022](../tools/research/l2_prefill_runtime_20261022/README.md)。
回执目录：`fastkpc/artifacts/full_cuda_ci/l2_prefill_runtime_20261022/`。
`build.json` 绑定全部生产源码和 43 个对象，`checks.json` 保存接入检查；
`closure.json`、`timing.csv` 和 `manifest.json` 索引四轮完整回执、telemetry、工具和 native。
manifest SHA-256：

```text
f2a6e5b674dbbf87deda8acf2c359bb2f274b6e4238fb00e6cf6fd0317e29ec2
```

本轮不包含新真实数据或最终 PDAG 资格，也不构成 clean producer 重复中位数或 promotion。
Phase 10 继续 ACTIVE；candidate NOT_FROZEN；holdout SEALED；recommended=false。
