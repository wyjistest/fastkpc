# CUDA 构建入口一致性资格

日期：2026-10-29。起点为已推送的 `c469ac5` / `e15be9d`。本项结论为
`PASS_BUILD_FIX_PARENT`；这是完整源码闭包构建及数值回归，不是 clean committed
producer 的性能 campaign，不修改 Phase 10、candidate、holdout 或推荐状态。

## 改动与构建证据

`build_cuda_native.sh` 为 `full_cuda_ci_method_batch.cu` 集中定义
`METHOD_BATCH_NUMERICAL_FLAGS="--fmad=false"`，全量与
`FASTKPC_INCREMENTAL_FIXED_SP_ONLY` 两个入口共用。其余 CUDA TU 的 FMA 策略保持原样。
根 README 撤销历史 v1 correctness 表述，两层 README 指向现有交接并区分 PASS/SKIP。

在独立 production-only 源码快照中从空 build 目录构建全部 44 个链接对象，然后在
相同目录实际运行 fixed-SP 增量入口。没有用静态命令检查代替真实构建。
两个入口的 method-batch 编译命令一致，最终 native 均为：

```text
639d598fe23db8cc08571f6e3f7d1e153a6c273d17a13ac8ecd806302179bab8
```

44 个对象去除调试/局部符号后的诊断副本全部逐字节一致。7 个原始 CUDA `.o` SHA
不同；NVCC 的局部 STT_FILE 包含带进程号的 `tmpxft_...cudafe1.cpp` 文件名。
原始对象和 SHA 均保留，诊断副本使用 `objcopy --strip-debug --discard-all`，
不参与链接或安装。最终 native 使用实际构建产物直接比较 SHA，没有归一化。

初版构建比较器额外要求原始 `.o` 全部相同，因上述信息差异给出失败；原始 driver
及该结果保留。对象信息的区分规则在增量构建前写入
`object-identity-policy.json`，独立 `object-payload-comparison.json` 的三个门
（命令、全部对象内容、最终 native）均 PASS。没有放宽数值或性能门。

原 accepted postback native
`a1e16a9c0c8089ea90491426c56475394449efb03317be4a87f1925c7b73bd34`
及其 44 个链接对象、107 个源码输入已在改动前封存。历史 290.491 秒继续绑定该
native；全量重编的结果不冒充历史 receipt 的 producer。

## 数值与行为资格

受影响 TU 实际服务 `dcc.perm`、`hsic.gamma`、`hsic.perm`。原 native、全量 native、
增量 native 各跑五个 fixture × 三方法，共各 **2,055 次 CI**。fixture 包括非线性
输入的 conditioning ceiling 0/1/2、实际执行到 level 3 的 dense 输入，以及真实
351×6 子集。全量和增量均与原 native 的 p/pMax 位模式、task/deletion/graph/
sepsets/counts、route/status、permutation RNG 完全相同，authority 与资源门通过。
这是三方法跨 binary 回归，不宣称新增完整三方法 CPU-authority campaign。

四项现有回归通过：layered identity、permutation overlap、failure order、deferred
SVD。首次测试包装暴露 overlap/deferred 测试的 `Sys.unsetenv` 多参数清理错误；
两处改为字符向量，包装改为一次执行整个 expression vector，避免提前执行清理。
原失败日志、原工具及测试字节保留。修复后重跑四项回归，未修改数值判据。

全量 native 再跑完整 `dcc.gamma` canonical 351×48、default-Inf、fresh R/cache：

| 检查 | 结果 |
| --- | --- |
| logical CI | 240,498 / 240,498 |
| CI / pMax 位差异 | 0 / 0 |
| 最大 authority CI / pMax 误差 | 3.05425129631942e-12 |
| 保留的完整硬门 | 74 / 74 PASS |
| task / deletion / graph / sepsets / counts / RNG | exact |
| CPU numerical authority / fallback / large D2H | 0 |
| 最终 live allocations / bytes / streams / events | 0 / 0 / 0 / 0 |
| L6 最低空闲检查点 | 2,230,452,224 B |

构建期间 GPU0 出现外部推理任务，首次增量检查因此未启动，该协议保留为
`qualification-not-started-gpu0-busy`。后续增量和 canonical 仅在独占 GPU1 上做
数值资格，监测并记录 GPU0 外部进程。canonical 记录的 287.939 秒明确不是性能
证据、不是 accepted median。后续 OFF/ON 性能实验仍要求两卡无外部 compute。

## 本地证据与复现

产物位于 `fastkpc/artifacts/full_cuda_ci/build_consistency_20261029/`：

- `parent/identity.json`：原源码/native/44 对象。
- `full/build.json`、`incremental/build.json`：真实命令、对象及 native SHA。
- `object-payload-comparison.json`：对象内容和最终 native 一致性。
- 各 native 的 `qualification/`、`full/regressions2/`：三方法与异常/RNG 回归。
- `canonical/qualification/`：完整 canonical receipt、74 个门和资源记录。
- `closure.json`：`PASS_BUILD_FIX_PARENT`，SHA-256
  `4358903b70b35a529716b746a77cd755e598d5a2454084e081578b85e4634953`。

执行工具及本地依赖范围见
[build consistency 工具说明](../tools/research/build_consistency_20261029/README.md)。
该 native 是后续 structured 行和原型的固定父基线；矩阵删除的容量与性能单独判定。

提交后对应关系见[2026-10-30 开发交接](PHASE10_DEVELOPMENT_HANDOFF_20261030.md)：
构建修复提交 `89a8099` 的全部生产输入与本次修复后父源码闭包一致。
