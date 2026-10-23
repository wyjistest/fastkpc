# Phase 10 开发交接：prefill runtime 与 optimizer 窗口诊断

日期：2026-10-23。本次提交边界是已完成的 L2/structured/level prefill 资格和相邻
optimizer 双窗口诊断；同时进行的 optimizer D2H 新实验及其新评审留待独立收口。

## 当前状态

`dcc.gamma` canonical 351×48、default-Inf 的 S2 opt-in 两次 outer 中位数为
**541.1665 秒**，同 binary S3 为 **690.919 秒**。这是提交前源码上的 development
对照，尚未完成 clean checkout 性能重跑、新真实输入和最终 PDAG 资格。

全部 240,498 CI 和 pMax 满足 authority `1e-10` 合同，S2 最大误差为
`3.05425129631942e-12`。S2 相对 S3 的低位差异最大为 `1.0547118733939e-14`，
只在 L2；每个配置内重复运行逐位一致。不能将 S2 宣称为相对 S3 全局逐位相同。

三个 prefill 开关默认均关闭。Phase 10 **ACTIVE**、candidate **NOT_FROZEN**、
holdout **SEALED**、recommended **false**；180/120 秒性能门仍未通过。

## 提交与源码身份

| 提交 | 内容 |
|---|---|
| `bfc5b1016b2e227cd0ca913f348b986c2940ae2a` | level/L2 prefill、structured GPU adapter、独立构建与配置测试 |
| `57ed1e5933076a6c49245db2580e887a1bec85fa` | 已完成的组件实验、capture/replay/authority/资源工具和 development 环境 wrapper |
| `51cf1059ccfe7e238c4fdba033578e76df5fc58b` | optimizer 单窗口/双窗口诊断、逐位验证和 STOP 结果 |

[源码 attestation](PHASE10_SOURCE_ATTESTATION_20261023.json) 保留完整 97 文件 SHA-256
清单，逐个用 `git show bfc5b10:<path>` 验证与 L2 runtime closure 相同，包含生产
build script 和 DBDSQR generator。当前本地 native 也与该资格运行一致：

```text
8e457e811eaa01ae85326b8ff3cc0bcfa56f5b907ad6840a302a8205370d3874
```

attestation 同时索引 L2、structured canonical 和 optimizer 诊断的原始 manifest/closure
哈希。这证明提交源码与已运行源码相同，不把旧回执改写成从该 commit 的 clean checkout
运行，也不承诺不同 toolchain 的重建 binary SHA 相同。旧报告中的“未提交”描述保留其
运行时事实；当前 Git 身份以本交接和 attestation 为准。

## 从哪里继续

| 主题 | 最终证据与范围 | 工具入口 |
|---|---|---|
| dense level prefill | [canonical 资格](PHASE10_LEVEL_COMPONENT_PREFILL_CANONICAL_20261021.md) | [复现说明](../tools/research/level_component_prefill_canonical_20261021/README.md) |
| structured L≥3 prefill | [canonical 资格](PHASE10_STRUCTURED_PREFILL_CANONICAL_20261022.md) | [复现说明](../tools/research/structured_prefill_canonical_20261022/README.md) |
| structured L≥2 prefill | [L2 runtime 资格](PHASE10_L2_PREFILL_RUNTIME_20261022.md)，当前 canonical opt-in GO | [复现说明](../tools/research/l2_prefill_runtime_20261022/README.md) |
| optimizer worker 数 | [四个实际窗口](PHASE10_OPTIMIZER_WINDOW_DIAGNOSTIC_20261022.md)，降低 worker 数 STOP | [复现说明](../tools/research/optimizer_windows_20261022/README.md) |
| 相邻双窗口 | [A/B/C 结果](PHASE10_OPTIMIZER_DOUBLE_WINDOW_DIAGNOSTIC_20261023.md)，直接拼接/32-worker 为 STOP_FOR_CAPTURED_PAIRS | [复现说明](../tools/research/optimizer_double_windows_20261022/README.md) |

后续优先读最终结果，再读同主题的 proposal/review；早期建议不能覆盖后续实测 STOP。
双窗口在当前生产 native 上有真实 device overlap，但 B/C 在两组输入的六轮中全部慢于 A。
诊断没有实现生产 rolling scheduler、sortD、新 persistent-stream pool 或新的 pinned-I/O
机制，runtime rolling 仍是 NOT_YET_GO。针对已观测等待提出新机制时，必须另做隔离
因果验证和完整资格，不能把累计 host waits 当成可兑现收益。

生产构建使用 `sh fastkpc/tools/build_cuda_native.sh`；recorded DBDSQR 从 vendored
LAPACK 生成，structured translation unit 单独采用 `--fmad=false`。不依赖 research
目录的历史实验实现。开发环境 wrapper 只设置 retained cache/bypass/eig lanes 等缺省
环境值，尊重调用者已设值；它不会自动打开 prefill，也不会改变库默认配置。

原始 datasets、native、capture fixtures、Nsight traces 和大型 receipts 保留在本地
artifact 存储，未纳入本次 Git 提交。部分历史重放工具需要这些已归档输入和 authority；
clone 仓库即可检查源码，但不能据此声称无需 fixtures 就能重现全部 campaign。

## 本次提交前检查

生产 closure 的 97/97 文件已同时核对工作区和 runtime commit，native SHA 一致。
Python AST、R parse-only、shell syntax 和 CUDA build-lock contract 检查通过。
本次提交操作没有重跑 CUDA build 或 GPU campaign，沿用源码身份完全一致的既有
完整资格；正在进行的独立 GPU 实验没有被中断。

历史研究工具按原字节归档以保留 receipt 中的哈希：两个 helper 的已有空白及实验 patch
的 context 空白未改写；其余新增内容通过 whitespace 检查。后续冻结 candidate 仍需
在冻结 producer 的 clean checkout 上运行正式重复性能及相应完整语义门。
