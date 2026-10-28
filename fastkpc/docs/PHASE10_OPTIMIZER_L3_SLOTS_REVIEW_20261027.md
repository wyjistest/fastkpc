# L3 原窗口／32 槽 captured-cohort 对照：复核

日期：2026-10-27。复核 [L3 槽位 captured-cohort 对照](PHASE10_OPTIMIZER_L3_SLOTS_20261027.md)
（工具 `fastkpc/tools/research/optimizer_l3_slots_20261027/`，native `313d9f3c…510a1b`）。只读复核，
没有新的 GPU 运行。`GO_CAPTURED_L3_SLOTS` 成立；canonical development 中位数仍为 368.851 s；
L3 runtime 仍为 `NOT_YET_GO`；Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false
均不变。

## 结论

```text
完整性    报告列出的 6 个 SHA-256 吻合；manifest 的 152 个文件、15 个工具按当前文件重算全部一致；
          L6 build 绑定的 101 个生产源文件与 native 均未改变。7 个进程都 exit 0、isolated，
          connections=32、属性缓存开启；计时期间 SM 时钟 2760–2775 MHz
数字      六组配对、两档中位数（54.910185/14.373670 s）、比值 0.261767、工作边界中位数、逐请求
          service 比（p50 0.461207、最大 1.064703）和 935,404 次输出比较全部复现
设计      W 调用生产 multi-setup scheduler，窗口 63×64 + 33、并发 64/33、retain_residuals=true，与生产
          L3 一致；W 的输出保存与 S 的 detach 是同样的三次 D2D 加 event 同步；S 与生产同一组 nvcc 参数，
          14 个 device 函数指令一致。未发现缺陷
trace     独立重算的 W/S 统计与报告一致。S 的 30 次晚启动全部在 0.111 s，与冷 context 下 slot_backend
          kernel 的首次加载同时，稳态为零；S 的持久 kernel device 时间合计比 W 低 4.85%，32 槽
          持续负载下没有出现服务时间膨胀
生产含义  40.537 s 是两种生命周期之差，W 比生产 L3 optimizer host 慢 1.5–1.9 s。按 optimizer 边界
          预期约 35.4–35.9 s；若 runtime 不再逐请求创建 prepared，层边界可能再多一部分。都要
          runtime 实测，不能加到完整方法上
容量      生产 L4/L6 守卫要求增量 ≤1 GiB（full_cuda_ci_one_call.cpp:5155）。按原型的物理增量扣除
          诊断输出区估算，L3 池本身约 1.13–1.14 GiB；照搬这一门会 fail closed。但 W 单窗口的物理
          占用（约 1.52 GiB）比 S 池大，替换后 L3 峰值预计下降。共存探针必须先声明 L3 的增量门
建议      按 Codex 计划先做生产共存探针，另需：预先声明 L3 增量门（或缩小池）；runtime 门以 L3
          optimizer 边界加重复 outer 判定；L3 envelope 外 fail closed
```

## 1. 完整性与身份

- `capture-closure.json`、`prototype-build.json`、`protocol.json`、`trace-analysis.json`、`closure.json`、
  `manifest.json` 的 SHA-256 与报告一致。manifest 的 152 个文件、15 个工具按当前文件重算，没有不一致。
- `common.py` 在每个阶段前后核对 L6 isolated completion 的 `build.json`：101 个生产源文件与 native
  `313d9f3c…` 当前哈希不变。
- capture、checks、memcheck、capacity、campaign、trace-W、trace-S 七个进程都 exit 0、`isolated=true`，
  没有外部 GPU 进程；CUDA_DEVICE_MAX_CONNECTIONS=32，属性缓存开启。campaign 期间活动采样的 SM 时钟为
  2760–2775 MHz。
- 协议（`protocol.json`）在 checks 阶段写入，后续各阶段逐字核对；README 的门槛（≤0.90、六组全胜）在任何
  计时之前已固定。

## 2. 数字复现

由 `campaign-runs.csv` 重算：W 生命周期 53.068722/55.244590/53.383332/55.095436/55.144127/54.724933 s，
S 为 14.324073/14.391417/14.384440/14.341890/14.428826/14.362900 s；中位数 54.910185/14.373670 s，
比值 0.261767，六组全胜。工作边界中位数 51.695134/14.290550 s；W 创建/销毁中位数 1.116464/2.044934 s。

由 `campaign-setups.csv` 重算逐请求六次中位 service 的 S/W 比：p50 0.461207、p95 0.922836（报告
0.922864，分位插值不同）、最大 1.064703；4,065 个请求中 48 个 S 更慢。输出比较：六个阶段 summary 合计
935,272 次，加上 4 个进程的原生预热（各 33 个 target），共 935,404 次，与报告一致。

## 3. 设计复核

- **W 是生产路径**：生产 L3 窗口调用 `optimize_multi_penalty_prefill_window` →
  `multi_penalty_gcv_capacity_optimize_multi`（`full_cuda_ci_one_call.cpp:5321`）。canonical 配置启用
  coefficient arena，因此 `retain_residuals=true`、`force_final_selected_evaluation=true`
  （`:5590`、`:5832`），与 W 的调用一致。我的 cache-ON trace 中，生产 L3 为 63 个 64-setup 窗口加 1 个
  33-setup 窗口，请求并发与 worker 数都是 64/33，与 W 相同。
- **输出保存对称**：W 通过符号替换在生产 scheduler 内对每个 setup 调用 `preserve_native_output`，S 调用
  `detach`。两者都在该 setup 自己的 stream 上做三次 D2D（coefficients/fitted/residual），再 record 并同步
  一个 blocking event。`visits==1` 的检查保证替换对每个 setup 都生效。
- **S 槽位**：`bind` 拒绝 lease 未释放、形状不符和 poisoned 的槽位，重传 10 个几何数组并递增 generation；
  `execute` 调用由同一源码编译进原型 DSO 的 small64 batch 函数；`detach` 完成后才释放 lease；失败路径 poison 槽位并
  drain。原始请求顺序由原子计数器保证，cohort 不变。
- **编译**：原型与生产 `build_cuda_native.sh` 使用同一组 nvcc 参数（`-O3 -arch=sm_89 --fmad=false`），
  另加 hidden visibility 以隐藏 host 符号；`device-code.json` 显示 14 个 device 函数指令一致。

## 4. trace 复现

我用原始 `kernels.bin`/`runtime.bin` 独立重算（只取 `optimize_multi_penalty_targets`）：

| 观测 | W | S |
|---|---:|---:|
| 持久 kernel 跨度 | 53.854 s | 16.049 s |
| device 时间 p50 / p95 / 合计 | 81.548 / 289.730 ms / 455.882 s | 70.725 / 287.932 ms / 433.776 s |
| launch→start p95 / 合计 | 288.600 ms / 285.243 s | 3.087 ms / 10.727 s |
| 晚启动（>50 ms） | 1,682 | 30 |
| 其中恰在另一 stream 同通道 kernel 结束时 | 801 | 0 |
| 其中只与其他通道结束对齐 | 807 | 30 |
| 同通道另一 stream 持久 kernel 仍在运行时启动 | 835 | 0 |

与报告一致；最后一行报告为 836，差 1 次，属边界取等的差别。补充三点：

- **S 的 30 次晚启动全部发生在 0.111 s**，与 slot_backend 一个 kernel 的首次加载（0.109 s 启动，等待
  103 ms）同时。这是冷 context 的启动事件，稳态中 S 没有晚启动。我此前在 L3 窗口模式中看到的“机制未明
  的跨通道等待”，在 W 冷 trace 中仍有 807 次，在 S 中只剩这一次启动事件；它也主要属于窗口模式，机制
  仍未查明。
- **首次加载**：S 冷 trace 中，除 GEMM N=3/N=2（等待 509/519 ms，分别在 3.68 s、7.90 s）外，还有 N=4/6/7
  （264–322 ms）和两个 slot_backend kernel（103/131 ms）。N=3/N=2 首次加载时没有持久 kernel 晚启动，
  只推迟了各自那一次 launch。计时 campaign 先做了整层预热，不含这些成本。我此前的 first-use 分析中，
  生产 L3 首次使用的是 N=2/3；runtime 的两种模式都会包含这项成本。
- **服务时间**：S 的持久 kernel device 时间合计比 W 低 4.85%，32 槽持续负载下没有膨胀。S 工作边界
  14.29 s 也略低于我按 trace 做的请求顺序补位模型（14.82–14.85 s）。这次复核和上一轮都把服务时间列为
  首要风险，它在这次隔离原型中没有出现；生产共存时的显存与流量不同，仍需在 runtime 中确认。

## 5. 对生产的含义

40.537 s 是 W 与 S 两种完整生命周期之差，W 不是生产 L3 optimizer 边界的精确替身：

| 边界 | 生产参照 | W | S |
|---|---:|---:|---:|
| optimizer | 49.786 / 50.225 s（L6 campaign ON/OFF 中位） | 工作 51.695 s | 工作 14.291 s |
| 窗口 batch wall | 53.338 s（单份 cache-ON trace 合计） | 生命周期 54.910 s | 生命周期 14.374 s |

- W 工作边界比生产 L3 optimizer host 高 1.47–1.91 s（约 3–4%），W 生命周期比 trace 的 batch wall 高
  1.57 s。W 的逐 setup 保存合计 80–99 s（约 20–24 ms/setup），S 的相同操作合计 0.37–0.44 s（约
  0.1 ms/setup）；拷贝量相同，差别在等待。这部分对 W 墙钟的贡献没有单独测量。
- 按 optimizer 边界，生产 49.79–50.22 s 对 S 14.29–14.37 s，预期收益约 35.4–35.9 s。
- 生产 L3 窗口 batch wall 比 optimizer 多 3.61 s（逐窗口 prepared 准备与释放等），W 的创建加销毁为
  3.16 s。若 runtime 像 L4/L6 一样以槽位重绑定代替逐请求 prepared，层边界可能再少一部分。L6 的层收益
  只比 optimizer 收益多 0.10–0.14 s，但 L6 只有 245 个 setup，L3 有 4,065 个。

这些都是预期。完整方法的收益只能由 runtime 隔离 campaign 测得，不能加到 368.851 s 上。

## 6. 容量：生产增量门

| 项目 | S 池 | W 最大窗口 |
|---|---:|---:|
| 应用可见字节（不含诊断输出区） | 839.4 MiB（32 槽） | 936.7 MiB（64 个 prepared） |
| 物理增量扣除诊断输出区（305.6 MiB） | 1,158–1,168 MiB（计时/独立容量进程） | 1,558 MiB（计时） |
| 物理减应用可见，按对象平均 | 约 10.3 MiB/槽 | 约 9.7 MiB/prepared |

- 生产 L4/L6 槽位池的守卫在每个检查点要求空闲 ≥2 GiB、最大池增量 ≤1 GiB（`full_cuda_ci_one_call.cpp:5144–5157`，
  只对 `Policy::level == 4 || 6` 生效）。按上表估算，L3 池本身约 1.13–1.14 GiB；若 L3 照搬这一门，
  会 fail closed。
- 但这一门量的是相对建池前的增量。生产 L3 目前每个窗口都持有自己的 prepared，最大窗口的物理占用约
  1.52 GiB，比 S 池大；用槽位替换后，L3 时刻的峰值预计下降约 0.4 GiB。L3 时刻的采样峰值本来就比 L6 低约
  1.1 GiB，绝对余量不是首要约束。
- 每个对象约 10 MiB 的非应用物理占用，W 与 S 一致，原因没有分项（库或 driver 的按对象分配）；我没有证据
  确定它来自哪里。
- 共存探针需要先声明 L3 的门：沿用 ≤1 GiB 就要缩小池，例如按容量分级的槽位（89% 的 setup 不超过 18 个
  target，只有 10 个 setup 有 33 个）或减少槽位数；这会改变设计，需要重测时间与输出。也可以按替换后的
  实际余量另行资格化一个 L3 上限，并说明它与 L4/L6 门的关系。

## 7. 对共存探针与 runtime 的建议

1. 共存探针按 Codex 的计划进行：明确原窗口 prepared 何时被替换，核验 selected coefficient detach、原
   Stage A cohort/Gram 顺序、输出 ownership 与释放时点。另需预先声明第 6 节的 L3 增量门。
2. runtime 门以 L3 optimizer 边界（预期约 35.4–35.9 s）加重复 outer 收益判定，L0–L2、L4–L8 和层外分开
   列账；首次加载成本在两种模式中都保留，不另加预热。
3. L3 runtime 应像 L4/L6 一样 fail closed：n=351、q=28、三个 penalty、root rank 24、cohort 2–33 之外的
   请求不进入槽位路径。
4. runtime 运行一次 ktrace，用 `optimizer_slot_line_review_20261025/same_channel.py` 的通道共用统计确认
   生产 L3 槽位模式下没有同通道共用；这是观察，不是保证。

## 证据

`fastkpc/artifacts/full_cuda_ci/optimizer_l3_slots_20261027/`：`campaign-runs.csv`、`campaign-setups.csv`、
`campaign-metadata.csv`、六个阶段的 summary/process/context、`device-code.json`、`prototype-build.json`、
`trace-W/`、`trace-S/`。生产参照：`optimizer_l6_slot_runtime_20261026_isolated_completion/closure.json`
的 L3 optimizer 中位数；`optimizer_slot_line_review_20261025/ktrace-run-cache-on/profile-result.json`
的 L3 窗口。代码：`replay.cpp`、`slot_backend.cu`、`native_observe.cpp`、`fastkpc/src/full_cuda_ci_one_call.cpp`
（`:5144–5157`、`:5321`、`:5590`、`:5832`）。计算为只读一次性脚本；没有修改生产源码、native 或默认配置，
没有 commit 或 push。
