# Optimizer status-wait 资格验证：复核

日期：2026-10-23。复核 [原窗口 status-wait 资格报告](PHASE10_OPTIMIZER_STATUS_WAIT_20261023.md)
及 Codex 对 [D2H 复核](PHASE10_OPTIMIZER_DOUBLE_WINDOW_DIAGNOSTIC_REVIEW_20261023.md) 的修订。
复核、计数勘误及整层诊断回填，不改生产数值代码或 promotion 状态。Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、
recommended=false 不变。

## 结论

```text
实现          正确。开关在 prepared 创建时解析并固定；独立的 blocking-sync event 在
              cudaSetDevice(device_id) 之后随 prepared 创建、随对象销毁；关闭时 event 为空，
              析构不会调用 cudaEventDestroy(nullptr)，因此不会留下可能被后续
              cudaGetLastError 读到的残留错误；等待位于 residual completion event 之后、
              首次 pageable 状态下载之前，之后的 trace 下载不再等待 kernel
范围          合理。fixed evaluator 也由生产 one-call 直接调用，但不在本次
              optimize_multi 多 worker 执行链中，本轮不扩大修改
结果          原 64/64 窗口对照通过预声明回退门，但不足以证明等待没有影响；
              +3.50 s（0.64%）小于同一开关两次运行之间的差距（off 541.6–545.7 s，
              on 544.0–550.4 s）。同意“通过无实质回退门、未证明加速”
统计修正      接受。独立重算修正后的推断占用等待，与 Codex 的数值逐项一致
措辞修订      接受。硬件队列部分是本机探针在所测形状下的观察，不是 CUDA 承诺；
              32 仅在明确输入、等待策略和容量条件下接受性能资格
本轮执行      L5 整层常驻诊断完成，955 setups / 6,816 targets 完整覆盖；
              A/R 六轮中位数 106.154/31.787 s，逐位及资源门通过
下一步        有界 prepared 槽位池隔离原型可继续评估，runtime rolling 仍 NOT_YET_GO
```

## 1. 代码复核

对照 `git diff fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu`（+34/−5）：

- **开关**：`optimizer_status_wait_enabled()` 在 `MultiPenaltyGcvCudaPrepared` 构造时调用。
  未设置、空串或 `0` 关闭，`1` 开启，其他值抛出，prepared 创建失败。每个对象创建后
  不再改变。
- **event 生命周期**：`Event(flags, enabled)` 的默认参数保持 `completion_event` 原有的
  `cudaEventDisableTiming`。成员声明顺序为 stream → cublas → completion_event →
  status_wait_event，与初始化列表一致。创建路径在 `make_shared` 之前已执行
  `cudaSetDevice(device_id)`；析构函数体先切换设备，成员随后销毁。关闭时 `event_`
  为空，析构跳过 `cudaEventDestroy`。
- **等待位置**：在 `completion_event` 记录之后，于同一 stream 记录 status_wait_event
  并 `cudaEventSynchronize`，然后执行原 pageable 状态下载、可选 trace 下载及末尾的
  stream 同步。等待覆盖了同 stream 上的全部数值工作，所以之后的下载不会在 kernel
  运行期间占着 D2H。
- **与队列探针的关系**：探针表明，blocking-sync event record 若紧跟长 kernel，会堵住
  共享队列。这里它排在 residual kernel 之后，而此前的 gather kernel 已经在等待 optimizer
  kernel。原窗口实测未见超过预定门的回退，但这不能证明新 event 对共享队列绝无影响。
  在途 ≤32 时探针未观察到所测队头阻塞；不是无共享的 CUDA 保证，未来 arena/Stage A
  并发时还须统计这些组件的 stream。

## 2. 结果解读

- **完成路径延迟**：修改后的 trace 中，最后 kernel 结束到 D2H 开始约 0.28–0.34 ms，
  包含唤醒、host 调度及后续 API 开销，不能全部归为纯唤醒成本，也不能作为每窗口
  回退的严格上界。双窗口对照中 patched A 与 control A 的差未超过观测到的运行波动。
- **尚不能可靠区分开关效应与运行波动**：四次运行中，L6 optimizer 时间为 41.5–49.5 s，L8 为
  16.6/21.9 s 两种取值（off2 为 21.90 s、on2 为 16.63 s）。逐层中位数见下表；
  通过 3% 门不等于证明开关没有影响；共享主机的四次运行不足以确立因果归因。

| 层 | optimizer host 中位数 s（四次） | 范围 s |
|---|---:|---:|
| L2 | 6.906 | 6.80–6.95 |
| L3 | 49.302 | 48.73–51.73 |
| L4 | 87.424 | 87.05–87.79 |
| L5 | 104.148 | 101.37–105.44 |
| L6 | 45.535 | 41.51–49.47 |
| L7 | 10.595 | 9.43–10.89 |
| L8 | 19.252 | 16.63–21.90 |

## 3. 统计修正的独立核对

用修正后的定义 `max(0, own_kernel_end − max(previous_D2H_return, own_API_enter))`（不含
首个 D2H），从同一批 Nsight 导出独立重算：

| | L5 A | L5 B | L5 C | L3 A | L3 B | L3 C |
|---|---:|---:|---:|---:|---:|---:|
| 生产 | 11.344 | 19.164 | 19.853 | 1.734 | 1.846 | 2.536 |
| patched | 0 | 0 | 0 | 0 | 0 | 0 |

与修订后的复核表一致。原公式在 patched trace 中把“尚未进入 API 的执行空档”算成占用；
复核正文当时未引用该值，但分析输出中有，移入 `attribution-v1/` 是对的。

## 4. 执行前判断：整层补位需要回答的问题

- **可作用范围**：L3–L6 的 optimizer 时间合计约 286 s（上表中位数）。L7 只有一个窗口，
  L8 只有少数长 setup，二者合计约 30 s，补位基本无法缩短。
- **已知的只是两组窗口**：同一机制下，捕获的 L5、L3 相邻窗口分别缩短 46%、30%。
  这不能直接乘到整层：各窗口的长尾分布不同，窗口之间还有 prepared 构建及 finalization。
  prepared 构建全程约 14 s；Stage A 全程约 16 s 中，15.24–15.39 s 属于 L2，
  L3–L8 合计仅 0.78–1.21 s，不能把全程数当作深层补位预算。约 286 s 是研究范围，
  不是可兑现收益池。
- **生命周期是主要约束**：
  - 四份 receipt 的 `prefill_batches` 一致：L3/L4/L5/L6/L7/L8 的 optimizer setup 数为
    4,065/2,154/955/245/44/9，窗口数为 64/34/15/4/1/1。`residual_batches` 是后续
    最多 16-target cohort 的计数，L3/L4 分别为 5,573/2,156，不能作为 setup 数。
    原文把 L3 改成 88 个窗口的“勘误”撤回；原 64 窗口正确，两种理想时间外推均非实证。
  - 双窗口增量显存给出每 prepared 约 16 MiB（L5）至 19 MiB（L3）的样本估算，不能
    替代全层真实容量核算。L3 的约 100 GiB 估计撤回，常驻仍明显超出单卡容量；L5
    约 15 GiB 仅为待验证估计。先按每个 setup 的 n/q/penalty/target capacity 核算并
    实测全层分配峰值，保留安全余量；不带 11 GiB component cache 的重放不是部署容量资格。
  - 可部署的补位需要每层一组可复用的 prepared 槽位（stream、cuBLAS handle、event、
    按该层形状确定容量的 device buffer），每个 setup 只上传几何与 Y。这样在其他 worker
    的 kernel 运行期间，就不会出现 `cudaMalloc`/`cudaFree`，也不会出现可能含隐式分配
    或释放的 `cublasCreate`/`cublasDestroy`。
  - 补位期间按 setup 完成的工作（arena 登记、stage A、residual token 释放）同样不能在
    本 stream 完成前发起 pageable D2H，也不能调用隐式同步的 API，否则会重现同类串行。
- **建议顺序**：
  1. 先只捕获整层 L5：955 setups、6,816 targets、15 个原窗口。相同 native、等待开启、
     相同输入及 target cohort，比较 P（A-prod，原 64/64，逐窗口创建/释放）、
     A（A-resident，原 64/64，全层常驻）和 R（R-resident，32 worker 连续领取，
     与 A 相同的全层常驻对象）。A→R 才是调度对照，P→A 单独解释生命周期成本。
     交错重复、完整输出逐位门、建销/调用/校验时间及资源峰值分别记录。
  2. 只有常驻诊断显示明确收益且容量通过，才评估有界槽位池；本轮不实现池或 runtime 补位。
  3. 槽位池另行通过后再做 runtime 集成与完整 canonical。按时长排序仍放在最后。

## 5. 整层 L5 执行结果

[完整结果与复现索引](PHASE10_OPTIMIZER_LAYER_REFILL_20261023.md) 已归档。
三档均使用同一 status-wait native，等待开启。六轮 P/A/R 的 native-call 中位数
为 `106.124/106.154/31.787s`，R 胜全部同轮 A，计入常驻建销后为 `33.580s`。
A/R 同池对照同时改变 64→32 workers 和窗口屏障，不能拆成各自的因果收益。
P 含窗口间诊断停顿，仅为生命周期参照。

全层实际显存增量约 14.42 GiB，application-visible buffers 为 5.59 GiB；
无 11 GiB component cache 的容量通过不等于 runtime 能容纳该形状。
独立 A/R trace 各关联全部 955 setups、5,730 kernels，跨原窗口 kernel 重叠时间
为 `0/27.042s`。25 次有效整层运行及缩小范围 memcheck 累计 170,430 个
target 输出实例逐位一致，受审计资源和输出 leases 关闭；最初失败的 CUPTI
采集另行保留、不计为 PASS。

结论为 `GO_L5_RESIDENT_DIAGNOSTIC`，下一步仅评估有界槽位池，
不能外推其他层或完整方法性能。证据仍是 GPU1 共享主机 development；
生产默认、runtime rolling NOT_YET_GO、Phase 10 和 promotion 状态不变。
