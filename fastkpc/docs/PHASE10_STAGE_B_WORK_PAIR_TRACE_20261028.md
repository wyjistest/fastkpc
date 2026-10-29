# 当前 work-pair OFF/ON：Stage B 时序诊断

日期：2026-10-28。状态：**AUTHENTICATED_CURRENT_WORK_PAIR_ACTIVITY_COMPLETE**。
在当前 native 上完成各一次隔离 OFF/ON activity replay，全部输出身份、模式、资源和 trace 门通过。
结果支持下一项单独的 captured recover 分块原型；**本轮没有实施或资格化该原型**。
accepted opt-in development median 仍为 **311.440 秒**，默认关闭，L8 STOP、Phase 10 ACTIVE、
candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。

## 身份、执行范围与硬门

- 生产源码对应 runtime `1d8a11d`，当前 HEAD `c38b04b`；104 个生产输入及两份 generated include
  全部保持封存哈希。native 为
  `12acc3cd01cce2dcac5f2bfb5ec02f27056e92154cbab2534c7f03bdc22caa7d`。
- 只编译三个 host 侧 observer/replay/trace 程序，复用原 observer、输入循环和 CUPTI recorder；
  没有编译 CUDA 数值 kernel、修改生产源码或替换 native。
- 复用 `stage_b_attribution_20261028/attempt3/capture/` 的 49,919/54,878 个 L2/L3 component，
  residual、原 target/cohort 顺序、key、route/rank/info 和原 native payload hashes 均按旧 manifest 核验。
- OFF 然后 ON，各 fresh process，GPU1 独占、另一 GPU 无计算任务，CPU affinity 0–19，单 math thread，
  CUDA connections=32，11 GiB cache、8 dense lanes、10 structured contexts、width=64。
  L2/L3 原 78/86 waves 保留。协议和工具身份在 GPU 工作前写入。
- 每档 **104,797 个三字段 payload hashes 全部一致**，合计 **209,594 次比较**；零 dense fallback。
  每档选定模式计数为 **10 created / 10 destroyed / 1,638 submits / 104,797 targets**，另一模式全零。
  tracked device allocations/bytes、streams、events 关闭，显式 replay buffer/stream/event 均释放成功。
- 每档 **1,000,301 条 CUPTI activity**，零丢失、零无效时间戳；每档 **13,104 个 kernel/API correlation**
  全部匹配。两次进程 exit 0，无外部 GPU 任务，所有本任务 GPU 进程已结束。

observer 在每个 Stage B 边界外做显式 payload D2H 与哈希比较；它不承担 CPU 数值计算。
本轮没有重新执行完整 skeleton 的 240,498-CI 或最终 PDAG 资格。optimizer workspaces 与中间 CI
consumers 不在 replay 中，因此也不构成完整 runtime 容量资格。

## 时间线结果

以下使用 CUPTI begin/end 标记的阶段边界。每档只有一次带 trace 运行，属于诊断计时；不纳入正式性能中位数。

| 指标，秒 | L2 OFF | L2 ON | L3 OFF | L3 ON |
| --- | ---: | ---: | ---: | ---: |
| 完整 Stage B 边界 | 55.402762 | 48.447199 | 60.970840 | 53.419457 |
| bidiagonal 活动并集 | 51.254497 | 36.457131 | 56.403094 | 40.102347 |
| recover 活动并集 | 24.001905 | 14.179868 | 26.345826 | 15.703445 |
| bidiagonal/recover 活动交集 | 20.736713 | 2.740062 | 22.846883 | 3.032341 |
| 仅记录到 recover 的 device 活动区间 | 2.607355 | 8.769724 | 2.869082 | 9.748983 |
| 各 wave 最后一个 bidiagonal 结束后的 recover 活动并集之和 | 3.237663 | 11.412601 | 3.498943 | 12.671104 |
| 未记录到 device 活动的区间 | 0.384713 | 0.240783 | 0.502888 | 0.300198 |

kernel family 并集之间会重叠，不能相加；所有并集均从原活动时间戳计算。
另用独立的端点计数扫描复算 device/bidiagonal/recover 并集、recover 独占记录区间和两类活动交集，
四个 level/mode 的 **20 项比较全部通过**，容差为 `1e-8` 秒。

L2/L3 合计：

| 指标，秒 | OFF | ON |
| --- | ---: | ---: |
| Stage B 边界 | 116.373601 | 101.866657 |
| bidiagonal 活动并集 | 107.657591 | 76.559478 |
| recover 活动并集 | 50.347732 | 29.883314 |
| bidiagonal/recover 活动交集 | 43.583596 | 5.772404 |
| 仅记录到 recover 的 device 活动区间 | 5.476437 | **18.518707** |
| 最后一个 bidiagonal 结束后的 recover 活动 | 6.736606 | **24.083704** |

ON 的 recover 独占记录区间约占阶段边界 **18.18%**，post-bidiagonal recover 活动约占 **23.64%**。
两者有重叠，**不能相加，也不能直接当作可兑现收益**。后者还可能与 roots、status、cache stores 等活动重叠。
ON 的完整 post-bidiagonal wave 尾部为 24.804864 秒，其中约 24.083704 秒记录到 recover 活动；
这只是时间线分解，不是关键路径可删除量。

## lane 启动与 recover 前序间隔

普通 wave 定义为去掉首尾 wave，L2 为 76 个，L3 为 84 个。early start 定义为相对于本 wave
第一个 bidiagonal 开始，在 50 ms 内开始执行的 bidiagonal kernel 数。

| 指标 | L2 OFF | L2 ON | L3 OFF | L3 ON |
| --- | ---: | ---: | ---: | ---: |
| 每个普通 wave 的 early starts | 8，76/76 | 10，76/76 | 8，84/84 | 10，84/84 |
| bidiagonal 开始时间跨度中位数，ms | 324.157 | **0.490** | 324.455 | **0.493** |
| 普通 wave 边界中位数，ms | 702.363 | 616.631 | 710.545 | 622.767 |
| roots 结束到同 stream recover 开始的间隔中位数，ms | 83.744 | 22.577 | 81.989 | 23.534 |
| recover kernel 时长中位数，ms | 50.240 | 38.728 | 50.243 | 38.956 |

每档全部 1,638 个 recover 的 launch API 都在其同 stream roots 结束前返回。
所列前序间隔因而不是 roots 结束后主线程才发出 recover 的延迟；它仍不能单凭 activity trace
确诊具体硬件调度机制。

recover 的单次 kernel 时长与前序间隔实际上都下降了；变大的是它在 bidiagonal 之后暴露出来的活动。
不能将本结果写成“recover 变慢了”。同样，bidiagonal kernel 时长中位数由约 364–366 ms 增至
447–449 ms，同时其全阶段活动并集缩短；单个 kernel 时长不能替代并发阶段边界。

10 个 kernel 几乎同时开始只说明 kernel 时间区间形状改变，不证明全部 640 个 block 同时驻留，
也不证明实际 SM occupancy 达到了 occupancy API 上界。

## 资源约束与下一项建议

trace 确认 bidiagonal 的 REG/shared 从 **100 / 19,656 B** 变为 **82 / 14,040 B**，LOCAL=0。
recover 保持 **64 threads、REG=66、shared=101,088 B、LOCAL=0**。
本机每 SM shared 为 102,400 B，每 block 另有 1,024 B 保留开销；当前 recover 的资源需求
不允许它与一个 bidiagonal block 同驻该 SM。这是静态限制，尚未测量实际 block 驻留或 warp stalls。

本轮结果足以支持单独评估 **captured recover tile 原型**：

1. 以当前 work-pair ON、tile=36 为基线，只改变每个 recover block 处理的独立列数，先比较较小的16/8列。
   每列内部的 rotation 逆序、算术、归一化、累加、输出槽与 status 合同必须保持。
2. 先通过全部 captured payload/intermediate、错误与边界资格，再测包含 bidiagonal、roots、recover、status、
   joins 与 stores 的完整 L2/L3 Stage B。较小 tile 会增加 block 数、改变日志读取的缓存/合并行为及并发竞争；
   不能仅以 shared 下降或 recover 单 kernel 更快宣布收益。
3. 独立原型重复显示完整阶段净收益后，才考虑 runtime opt-in 与完整 outer 验收。

本轮没有启动该原型。bidiagonal 活动并集仍约占 ON 阶段边界 **75%**；recover 的18.5/24.1秒数字
不进入预算，不构成180/120秒门的可达性承诺。L7/L8 与 optimizer 均未在本轮重测或重开。

## 证据与复现

- 工具：[README](../tools/research/stage_b_work_pair_trace_20261028/README.md)。共六个冻结工具文件，
  四个 Python 文件通过语法检查；区间裁剪、并集与交集的基本不变量检查通过。
- 证据：`fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_trace_20261028/attempt1/`。
  `build.json`、`protocol.json`、两档 `*-process.json`、`mode.json`、payload hashes、
  `events.bin`/`names.tsv`/`markers.tsv`、`campaign.json`、`analysis.json`、`verification.json`。
- closure SHA-256：`250a17aec79e82bb9c533d675ca49a587a237fe873c9be19941693ea72f6e54c`。
- manifest SHA-256：`02b7e318b13f6a7b868e8cca67f9956279f0fe0b5f196e5b051db20829f9d9d8`。
- manifest 包含43个产物、104个生产输入、6个工具、15个依赖；旧输入和历史证据均未改写。

本次未 commit 或 push；生产 native、默认配置、正式性能与 promotion 状态保持不变。
