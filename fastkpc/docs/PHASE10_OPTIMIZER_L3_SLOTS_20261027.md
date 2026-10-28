# L3 原窗口／32 槽 captured-cohort 对照

日期：2026-10-27。承接 [L6 最终复核](PHASE10_OPTIMIZER_L6_SLOT_RUNTIME_REVIEW_20261026.md)
限定的下一项工作。结论：`GO_CAPTURED_L3_SLOTS`。

六组交错配对的完整生命周期中位数为 **54.910185 / 14.373670 s**，S/W 为
**0.261766919**，下降 **73.82%**，加速 **3.820×**，六组全部获胜。所有输出逐位一致，
资源与容量门通过。这是隔离的 L3 捕获实验，不能把约 40.537 秒直接从完整方法耗时中扣除。
当前 canonical development 中位数仍为 **368.851 s**；L3 runtime 集成仍为 `NOT_YET_GO`。

## 范围、身份与预声明门

生产源码、native 和默认配置没有修改。沿用当前 source-closure native：

```text
313d9f3cc6dc8512b5624aae84cc459c6083ff04fd7883c32812fdb081510a1b
```

从该 native 捕获全部 64 个原始 L3 窗口（63×64 + 33），共 **4,065 setups / 54,878 targets**，
n=351、q=28、三个 penalties、总 root rank=24。实际 cohort 大小覆盖 **2–33 的全部 32 种**，
optimizer iterations 总和为 **563,638**。捕获的完整 canonical 运行通过 **64/64** 硬门：
240,498 个 CI 与 pMax 位值不变，最大 authority 误差 `3.05425129631942e-12`；图、任务、
删除、sepset、计数、RNG、authority 和资源门通过。该运行含诊断 shadow/文件写入，
其 `375.358s` 不作为性能证据。

- W：直接调用生产 multi-setup scheduler，保持原 64/…/33 worker 窗口及实际 target capacity。
- S：32 个固定 capacity-33 prepared 槽位，以原请求顺序领取完整 cohort，重绑定全部几何和输入。
- 两档都将 coefficients/fitted/residual D2D 保存到相同大小的整层设备区，保存与等待计入工作。
  诊断 D2H 和事后逐位比较排除在计时之外；不执行 CPU 数值求解或 fallback。
- 生命周期包括打包、创建、上传、计算、D2D 保存、释放和销毁；fixture 读取不计入。
  W 的外部输出 observer 成本计入 W。预热各一次，再执行六组 `WS / SW / WS / SW / WS / SW`。
- GO 门预先固定为 S/W 生命周期中位比 ≤0.90、六组全胜、全部数值与资源门通过。
- 容量相对已验证的 compute-warm 原生 context 测量。槽位池加完整诊断输出区的检查点增量
  ≤2 GiB、剩余 ≥2 GiB。此独立实验额度显式涵盖 33-target 槽位和整层输出区，
  不是生产容量额度，也没有代替真实 component cache/arena 共存门。

独立 DSO include 当前 small64 数值实现，隐藏生产 host entrypoint，未替换 native。
所有 **14 个 GPU 函数的完整指令字**与当前生产对象一致，仅规范化匿名 translation-unit 符号。
GPU1、CPU affinity 0–19、connections=32、status wait 和 properties cache 开启，BLAS/OMP/MKL
单线程。全部进程运行期间两张卡均未出现外部 compute 任务。200ms 活动采样时钟为 2760–2775 MHz；
采样不用于排除短暂时钟变化。

## 数值、故障与资源结果

33 个缩小范围 setup 覆盖全部 target 数量和最后一个请求。原窗口、单槽重绑定和双槽并发
均通过；10 项负向检查覆盖 busy/invalid bind、不变的 identity/generation、旧 token 拒绝、
旧 token 析构不解除新 lease、部分上传/保存失败后的 poisoned slot，以及 worker 取消和 drain。

缩小范围 compute-sanitizer：**0 errors / 0 bytes leaked**。这不等于全层 sanitizer。
普通全层容量、两档预热、12 次正式计时和两份独立 trace 都比较所有 floating/integer 状态、
selected-SP、route/rank/info、coefficients/fitted/residual 和 optimizer 轨迹。合计
**935,404 个 target 输出比较实例**全部通过，包含各进程的原生 warmup 与缩小范围检查，
不包含最初捕获本身，也不计入故障注入的未完成输出。

每个完整阶段的 application-visible allocations/bytes/streams/events/cuBLAS handles 和 leases
均关闭，worker 活动期间受审计资源建销为零。库/driver 保留的公共 context 成本单独报告，
不计入 application-visible 零资源声明。

## 无 trace 的配对计时

单位秒；下表是包含所有生命周期成本、排除事后验算的边界。

| 配对 | 执行顺序 | W 原窗口 | S 32 槽 | 节省 |
|---|---|---:|---:|---:|
| 1 | W/S | 53.068722 | 14.324073 | 38.744649 |
| 2 | S/W | 55.244590 | 14.391417 | 40.853173 |
| 3 | W/S | 53.383332 | 14.384440 | 38.998892 |
| 4 | S/W | 55.095436 | 14.341890 | 40.753546 |
| 5 | W/S | 55.144127 | 14.428826 | 40.715301 |
| 6 | S/W | 54.724933 | 14.362900 | 40.362033 |
| 中位数 | | **54.910185** | **14.373670** | 两个中位数之差 **40.536515** |

工作边界中位数为 **51.695134 / 14.290550 s**。W 创建/销毁中位数为 **1.116464 / 2.044934 s**；
S 为 **0.006598 / 0.034337 s**，另计打包 **0.020555 s**。各列独立取中位数，不能机械相加
替代每次完整生命周期再取中位数的结果。

逐请求六次 host service 中位数的 S/W 比值分布：p50 **0.461207**、p95 **0.922864**、
最大 **1.064703**。它包括 host API 与 device 排队，不是单个 kernel 的独立 contention 比。
全部请求、槽位 assignment 和 generation 保存在 `campaign-setups.csv`，按 target count
的分组结果保存在 `closure.json`。

## 容量

| 项目 | 结果 |
|---|---:|
| 公共原生首次执行后保留的 context 增量 | 1,152 MiB |
| S application-visible 总字节 | 1,200,693,088 |
| 其中 32 槽 buffer | 880,205,568 bytes（839.429 MiB） |
| 其中完整输出 arena | 320,487,520 bytes（305.641 MiB） |
| S streams / events / cuBLAS handles | 32 / 96 / 32 |
| 独立容量进程的物理检查点增量 | **1,474 MiB** |
| 独立容量进程最小检查点 free | **20.178 GiB** |
| 正式计时 S / W 物理检查点增量 | 1,464 / 1,864 MiB |
| 独立容量进程 GPU1 整卡采样峰值 | 3,455 MiB |
| 配对 campaign GPU1 整卡采样峰值（含 W） | 3,861 MiB |

这些口径分别是应用分配、相对 warm context 的检查点增量、以及整卡采样值，不能混用。
独立运行没有同时驻留生产 11 GiB component cache 和实际下游 arena；联合容量尚未资格化。

## 独立 trace：等待减少，首次使用成本仍存在

另起 W、S 两个新进程采集 CUPTI，不纳入上述性能中位数。这两档显式跳过公共原生 warmup，
使首次使用可见；其冷 context 不等同于完整方法已经执行 L0–L2 后的 context。
每档均包含 4,065 个持久 optimizer kernel、54,878 个 target blocks，CUPTI 丢失记录为零，
输出逐位一致、资源关闭。

| 观测 | W | S |
|---|---:|---:|
| 持久 kernel 首次开始至最后结束跨度 | 53.854s | 16.049s |
| kernel device duration p50 / p95 | 81.548 / 289.730ms | 70.725 / 287.932ms |
| launch-to-start p95 | 288.600ms | 3.087ms |
| launch-to-start >50ms 的持久 kernel | 1,682 | 30 |
| 上述晚启动在另一 stream 同通道 kernel 结束 ±2ms 内 | 801 | 0 |
| 上述晚启动只与其他通道 kernel 结束对齐 | 807 | 30 |
| 同通道其他 stream 持久 kernel 尚在运行时启动 | 836 | 0 |

持久 kernel 的 device duration 跨请求累计从 **455.882s** 到 **433.776s**，下降约 **4.85%**；
launch-to-start 累计从 **285.243s** 到 **10.727s**。这些都是并发相加量，不能相减后当作
wall-time 收益。结合实测跨度，证据支持主要机会来自调度、排队和窗口等待的减少。
该 A/B 同时改变了窗口补位、prepared 重用和并发数，不能从两档单独分解每项贡献，
也没有证明驱动内部的 FIFO false dependency 机制。

S 的 32 个 stream 在整个实验中各使用 1–3 个观测通道，不能声称永久一一映射。
S 仍观察到 GEMM N=3 / N=2 的首次 launch-to-start 约 **509 / 519ms**，并有其他 stream
在当时运行工作的最后结束点附近恢复。这只是时间观测，且不包含在上表“持久 optimizer
kernel 晚启动”计数里；不能宣称槽位消除了全部等待。当前不增加 preload，也不重开 L8。

## 决策与下一项

数值、完整生命周期、容量和资源门全部通过，`GO_CAPTURED_L3_SLOTS` 成立。
下一项只建议 **实际生产 cache/arena 的 L3 replacement coexistence 探针**，明确原窗口
prepared 在何时被替换，避免同时保留两套 handle 造成错误容量结论；同时核验 selected
coefficient detach、原 Stage A cohort/Gram 顺序、输出 ownership 与释放时点。

共存资格通过后，才考虑 default-off L3 runtime 接入与完整 canonical 硬门、重复 outer
wall-time。当前未进行 runtime 集成，没有新的 canonical 或真实数据性能结果。
L7/L8 继续后置；Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false。
本次工具与文档尚未提交；生产源码、native、旧证据及用户复核文档未修改。

## 复现与证据

工具：`fastkpc/tools/research/optimizer_l3_slots_20261027/`。
artifact：`fastkpc/artifacts/full_cuda_ci/optimizer_l3_slots_20261027/`。
仅允许在同一 source/native epoch、空闲 GPU1 和新 artifact 目录执行，既有运行拒绝覆盖：

```bash
L3_ART=fastkpc/artifacts/full_cuda_ci/optimizer_l3_slots_20261027_repro
L3_TOOL=fastkpc/tools/research/optimizer_l3_slots_20261027
python3 "$L3_TOOL/capture.py" "$L3_ART" build
python3 "$L3_TOOL/capture.py" "$L3_ART" run
python3 "$L3_TOOL/capture.py" "$L3_ART" verify
python3 "$L3_TOOL/build.py" "$L3_ART"
for mode in checks memcheck capacity campaign trace-W trace-S; do
  python3 "$L3_TOOL/run.py" "$L3_ART" "$mode" || break
done
python3 "$L3_TOOL/analyze_trace.py" "$L3_ART"
python3 "$L3_TOOL/archive.py" "$L3_ART"
```

构建与执行脚本绑定输入、工具和 native；manifest 的 **152 个文件、15 个工具及 build
绑定的 101 个生产输入**已另行重新计算并全部吻合。原型 DSO：
`87aaba2db8e8e0b6be4a5a7fe405405c606ae5fbb2ce2a5fb91dee2edacf032c`。

| 证据 | SHA-256 |
|---|---|
| capture-closure.json | `ecfa7c0ef306722e04b5a27fa46c1637b3451a90bef8bda091f702b2d0dde315` |
| prototype-build.json | `28d261792621d123925113361ed1a936b9b43a374f4e6f0dd38c3869b8d955d8` |
| protocol.json | `a146b47e9bb1fada93e0110aae66f18bc22573b9eb448630e68ff2b9e3a7f305` |
| trace-analysis.json | `441f70df7657ad9aa37b52fd04463b7471c6a18cb08100cdd406668de1bbbbb6` |
| closure.json | `f9b5c7e6dfba3b7fd2e6439436db7d06406df8c2079424deb84630cdf3a419fb` |
| manifest.json | `018c2f6a2900a057d24867e7075830f0b76958b07b4a508317dc0c5bedd1cabb` |
