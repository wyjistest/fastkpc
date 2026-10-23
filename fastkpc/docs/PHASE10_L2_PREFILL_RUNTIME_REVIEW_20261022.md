# L2 prefill runtime：复核与下一项诊断

日期：2026-10-22。复核 [L2 runtime 完整资格](PHASE10_L2_PREFILL_RUNTIME_20261022.md)
（native `8e457e81…`）。本文不改代码、不改状态。

> **执行结果：** [固定窗口诊断已完成](PHASE10_OPTIMIZER_WINDOW_DIAGNOSTIC_20261022.md)。
> L3/L5各两个真实窗口的16/32/64-worker交错重复全部逐位通过，四窗口中位数均以64最快，
> 降低worker数为STOP。Nsight确认L5示例尾部几乎只剩一个四-block optimizer kernel；
> 下一项可做有界双窗口隔离重放，完整滚动runtime仍NOT YET GO。以下保留诊断前的判断。

> **评审修订（同日）：** 采纳 Codex 的五处收紧。
>
> 1. 收益不再写成确定分解：明确边界解释 131.031 s，另有 23.883 s 未独立归因。
> 2. 205–212 s 改称理想化耗时下界，补上逐层下界公式，并注明 71% 是 host 调用槽位
>    利用率，不是 GPU 利用率；80–90 s 不是已确认的机会。
> 3. 逐位不变改为验收要求，不再当作实现事实。
> 4. 补写资源生命周期约束。
> 5. L2 阶段 A 另作独立实验。
>
> 结论由“滚动调度 GO”收窄为“诊断 GO”，下一步采用 Codex 提出的窄实验。

## 结论

```text
资格结论          成立；源码 diff、计时分解与回执一致；S2 = 541.1665 s 的 development 结果成立
收益口径          L2 整层 −154.914 s：明确边界解释 −131.031 s（与修订后情景一致），
                  另有 23.883 s 未独立归因；全方法 prepared 构建减少 18.462 s，
                  支持“句柄生命周期改善”的解释，但不能与前者直接相加
现状（S2）         optimizer 约 339 s（63%）；阶段 B 全层 144 s（27%）；L2 阶段 A 15 s；其余约 43 s
窗口屏障          存在（Codex 复算 319.726 / 308.660 s 与 70.9% / 71.1% 可复现）
决定              诊断 GO；滚动 runtime 集成尚未 GO；仍先于阶段 B 重叠
```

## 复核

- **源码**：相对上一份完整快照，vertical 只把 `begin_level` 放宽到 level 2（要求
  structured 已启用）；one-call 改动如下：
  - 新增 `prefill_at_level()`，统一控制 begin/build/finish、consumer 全命中断言，以及
    新增的逐批硬检查（命中数 = target 数、miss = 0、旁路 = target 数、consumer
    residual 与 eig 时间都为 0）；
  - L2 阶段 A 直接使用窗口持有的 `request.context` 与在 optimizer 输入移动前保留的 Y，
    log-SP 取自 optimizer 原始位模式，按 ≤16 宽 cohort 串行求解；异常时释放 residual
    与句柄；
  - coefficient-arena 断言扣除 L2 的 target 数，深层 gather 仍严格核对（smoke 抓到的
    计数错误已修正）。

  未见错误。
- **计时**：L2 整层 248.787 → 93.873 s（−154.914 s）。明确边界的净变化为
  (34.180 + 170.309) − (15.099 + 54.566 + 3.791 + 0.002) = −131.031 s，其余 23.883 s
  在这些边界之外。prepared 构建是全方法累计计时，覆盖 optimizer 与 fixed 两侧，
  且与阶段 A 部分重叠；它减少 18.462 s 支持句柄生命周期改善的解释，但不能当作
  23.883 s 的确定组成部分。

## 窗口屏障：已知与未知

已知：

- `prefill_multi_penalty_level_target_states` 把一层的 S 按容量（≤64）切成窗口。
  生产配置下 `setup_optimizer_pipeline` 关闭，窗口逐个同步执行；即使开启，也只有
  一个窗口在优化。
- `multi_penalty_gcv_cuda_optimize_multi` 在窗口内用原子计数分派 setup
  （`mgcv_multi_penalty_gcv.cu:6269`），各 setup 在各自的 prepared 句柄与 stream 上
  执行。窗口恰好 64 个 setup、64 个 worker 时，窗口时长由最慢的 setup 决定。

用 S2 回执估算 host 调用槽位利用率（slot-time = 各窗口 wall × min(setup 数, 64)）：

| 运行 | multi-penalty wall | slot-time | setup host 时间合计 | 槽位利用率 |
|---|---:|---:|---:|---:|
| s2a | 319.726 s | 19,091 s | 13,542 s | 70.9% |
| s2b | 308.660 s | 18,494 s | 13,148 s | 71.1% |

需要注意：

- **71% 是 host 调用槽位利用率，不是 GPU 利用率**。不能据此认定 GPU 有约 29% 的空闲。
- **耗时下界只是理想化模型**。最低也要按层计算：

  T_ideal ≥ Σ_L max( Σ_i t_Li / K_L , max_i t_Li )

  其中 t_Li 为第 L 层第 i 个 setup 的耗时，K_L 为该层并发数。这仍假定任务耗时固定；
  实际 setup host 时间含 GPU 排队、同步与资源竞争，改变调度后也会随之变化。
  初版用 Σ t / 64 得到的 205–212 s 忽略了层间屏障与单 setup 下限，只能作为更宽松的
  理想下界，不是可达值。L7（1 个窗口、44 个 setup）和 L8（1 个窗口、9 个 setup，
  17–19 s 主要受最慢 setup 限制）没有跨窗口补任务的机会。
- **80–90 s 不是已确认的机会**。逐层的 Σ t_Li 与 max t_Li 需要诊断数据才能算出。
- **64 个 stream 共享 32 个硬件队列造成队头阻塞**，仍是待验证的假设。

## 若做跨窗口滚动：约束

- **逐位不变是验收要求，不是实现事实**。独立的 prepared 句柄与 stream 提供了良好
  条件，但仍需保持 target cohort、selected-SP、输出身份与消费顺序，并处理共享状态。
  若全部 CI/pMax 与已合格的 S2 逐位一致，且任务、轨迹、资源门通过，才可以继承
  已有的 authority 误差结论，无需重跑 CPU authority。
- **句柄生命周期**：
  - prepared 创建用 `cudaMalloc` 分配 arena，结束时有 stream synchronize；
  - 释放走传统 `cudaFree`，可能引入 host 阻塞和设备同步。

  运行中补充或释放句柄可能限制后续提交与跨 stream 重叠，抵消填补空槽的收益。
  可选做法：可复用的句柄池（稳态下不分配、不释放），或 stream-ordered
  分配（`cudaMallocAsync` / `cudaFreeAsync`）。两者都要测量容量与成本。
- **共享登记**：coefficient arena 使用共享 staging；`store_residual`
  （`full_cuda_ci_vertical.cu:1936`）修改共享容器、按当前 key 数计算 arena 偏移，
  并复用同一个 completion event，不能被多个 worker 并发调用。需要明确串行登记、
  独占缓冲或其他所有权设计。
- **L2 阶段 A**（single-penalty，15 s）与本实验无关，另作独立实验，不计为附带收益。

## 下一步：窄诊断（采用 Codex 的方案）

1. 沿用现有调度器（窗口内已有原子计数的动态分派），不重写框架。
2. 选取 L3 与 L5 的真实连续窗口，固定同一批 64 个 setup、输入与 target cohort，
   比较 16、32、64 个 worker。
3. 记录每个 setup 的 host 与 device 时间、窗口尾部、句柄准备与释放成本，以及逐位结果。
4. 要回答两件事：减少并发是否反而提高吞吐；窗口尾部是否对应真实的 GPU 空洞。

证据明确之后，再进入有界的跨窗口滚动原型。阶段 B 与 optimizer 重叠仍排在其后。
