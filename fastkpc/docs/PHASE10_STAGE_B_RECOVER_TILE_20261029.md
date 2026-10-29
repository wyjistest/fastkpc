# Stage B recover tile：隔离原型结束，性能 STOP

日期：2026-10-29。结论：**tile16、tile8 均为 `STOP_NO_STAGE_GAIN`**。
在当前 work-pair ON、64 threads/block、canonical L2/L3 captured workload 下，两种分块均通过 GPU
逐位资格，但完整 Stage B 重复计时分别回退 **6.8273% / 2.9305%**。不进入 runtime 集成。

accepted opt-in development median 仍为 **311.440 秒**；默认关闭、L8 optimizer STOP、Phase 10 ACTIVE、
candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。本轮没有新的完整方法性能结果。

## 范围与身份

本轮执行[时序复核](PHASE10_STAGE_B_WORK_PAIR_TRACE_REVIEW_20261028.md)建议的窄原型：

- 生产源码对应 runtime `1d8a11d`，HEAD `c38b04b`；104 个生产输入及两份 generated DBDSQR includes 未变。
  native SHA-256：`12acc3cd01cce2dcac5f2bfb5ec02f27056e92154cbab2534c7f03bdc22caa7d`。
- 复用封存的 49,919 / 54,878 个 L2/L3 components，原 residual、key、route/rank/info、target/cohort 顺序、
  78/86 waves、10 contexts 和 width64 全部保留。`n=351`，`numCol=35`，包含边界用的第36列。
- 独立库从当前生产源码机械生成；计时代码只增加 tile 上限及 host 身份记录，数值 kernel 源码不变。
  三档固定 **64 threads/block**，不混入32线程 launch、更多 lane、cache 或 scheduler 改动。
- 三档采用相同编译参数 `-O3 -arch=sm_89 --fmad=false`；六个 device 函数的指令字跨三档全部一致。
  此检查是三份原型库之间的比较，不冒称与整个生产 native 的对象代码完全一致。
- GPU1 独占，GPU0 无计算任务，CPU affinity 0–19、单 math thread、CUDA connections=32、11 GiB cache、
  八条 dense eig lanes。18 个执行进程全部正常退出，隔离监控未发现外部 GPU 计算任务。

计时继续调用生产 cache/consumer/wave 实现。先完成一次当前 native 对照，再执行九次交错计时，最后各采集一次
独立 activity trace。13 次完整回放都验证原封存的三字段 payload hashes，合计 **1,362,361 次比较**。
每次均为零 dense fallback、10 created/destroyed plans、1,638 submissions、104,797 targets，且只执行 work-pair ON。
计时库另记录 tile、64线程及实际申请的 shared bytes，防止测错路径。

observer 的显式 payload D2H 与哈希均在 Stage B 边界外；CPU 不提供数值 authority。此 replay 不包含 optimizer
workspace 或中间 CI consumers，不构成完整 runtime 容量、240,498-CI、pMax、图、PDAG 或 RNG 的新资格。

## GPU 正确性与工具修正

tile36 分别与 tile16、tile8 比较，均通过：

- 每档 **104,797** 个 captured components，合计 **209,594** 个比较。
- 每档 **10,515,312,078** 条 rotation records，逐字段、逐顺序相同；两档合计 **21,030,624,156** 条。
- d/e/q、rotation count/overflow/info、分支覆盖、a/b、scale/live metadata、candidates、selected values、vectors、
  means 和 boundary status 逐位一致。DBDSQR shared canary 及每个诊断 allocation 的256字节前后 guard 均通过。
- 每档 **842** 个合成 target 比较；新增101种 full-pipeline 配置覆盖 distinct/rank 小形状、重复值、signed zero、
  极端有限尺度、target 尾批、column tile 边界及完整 pipeline rotation overflow。
- 每档另注入一次 recover 非有限值故障，均返回 `info=50`、`StructuredInfo`，拒绝输出。错误 payload 可能部分写入，
  不将其当作有效数值结果比较或发布。该注入代码不存在于计时库。
- 原有四种 rotation-capacity 边界、非收敛注入及八种 prepare/precheck 输入保留；完整合成分支覆盖为2047。
- 两档缩小版 CUDA memcheck 均为 **0 errors / 0 leaked bytes**。每档包含139个合成 target、16个 captured components
  及上述故障拒绝检查；完整分支和全量捕获覆盖来自后续 qualification，不归功于缩小版 memcheck。
- 所有成功运行的 tracked allocations/bytes/streams/events 闭环，显式 replay buffer/stream/event 释放成功。

成功证据为 `attempt3`。前两次失败记录完整保留，未计入 PASS 或性能统计：

1. `attempt1` 在新增 fresh-workspace 合成用例出现前序状态差异；继承的 raw-pointer lane vector 在异常退出时还导致
   sanitizer 报告240个未释放 allocations。随后仅为诊断夹具增加 RAII 清理。
2. `attempt2` 的 sanitizer 已为零错误、零泄漏，但 fresh `k=1` 用例仍比较失败；此时两档实际 tile 都是2。
   检查发现 allocation 清零/guard 初始化使用默认 stream，而计算使用 nonblocking stream，中间缺少显式完成依赖。
3. `attempt3` 在诊断 Lane 构造结束时同步默认 stream；缩小版及两档完整资格全部通过。修正只在诊断夹具，
   未改变生产源码、计时 kernel 或计时边界。前两次均未开始正式计时。

`prior-attempts.json` 绑定这两个失败目录的全部126个文件，并验证各自九个工具的原始 source snapshots 和 build outputs。
失败记录与后续成功证据分开保留，不将其改写为成功运行。

## 预注册的完整阶段性能判定

A=tile36，B=tile16，C=tile8；固定顺序 **A1 B1 C1 / B2 C2 A2 / C3 A3 B3**，各 fresh process。
每档三次均无 CUPTI、sanitizer、diagnostic export。对每个候选独立要求：两层中位数都下降、两层中位数之和至少下降5%，
且每次候选总时间都低于每次 A 总时间。小幅或噪声内正收益为 CONDITIONAL，无净收益为 STOP。

| 运行 | tile | L2 秒 | L3 秒 | L2+L3 秒 |
| --- | ---: | ---: | ---: | ---: |
| A1 | 36 | 48.476934 | 53.255972 | 101.732907 |
| B1 | 16 | 51.756387 | 56.838061 | 108.594448 |
| C1 | 8 | 49.846328 | 54.818439 | 104.664768 |
| B2 | 16 | 51.814603 | 56.870777 | 108.685380 |
| C2 | 8 | 49.893390 | 54.869712 | 104.763102 |
| A2 | 36 | 48.441844 | 53.215277 | 101.657121 |
| C3 | 8 | 49.830449 | 54.796183 | 104.626632 |
| A3 | 36 | 48.469582 | 53.162855 | 101.632437 |
| B3 | 16 | 51.709861 | 56.896539 | 108.606401 |

| 各层中位数 | tile36 | tile16 | tile8 |
| --- | ---: | ---: | ---: |
| L2 秒 | 48.469582 | 51.756387 | 49.846328 |
| L3 秒 | 53.215277 | 56.870777 | 54.818439 |
| 两层中位数之和 | **101.684859** | **108.627164** | **104.664768** |
| 相对对照耗时增加 | — | **6.8273%** | **2.9305%** |
| 判定 | 对照 | **STOP** | **STOP** |

上述合计严格是预注册的“各层中位数之和”；三次总和的中位数分别为101.657121、108.606401、104.664768秒。
两个统计量不混用。两个候选均每次都慢于最慢的 A，回退方向完全分离。
当前 native 对照为48.475031 / 53.162087秒，全部原 payload hashes 通过，不纳入三档中位数。
九次计时采样的 GPU1 总显存峰值均为15,969 MiB；这是离散采样的 replay 使用量，不是完整方法容量资格。

## 独立 trace：recover 尾部增加

每档一份 trace，各 **1,000,301** 条 activity，零丢失、零无效时间戳；合计 **39,312** 个 kernel/API correlation
全部匹配。三档正常 waves 均保持10个 bidiagonal kernels 在首个开始后50ms内启动，不能据此推断全部 block 驻留。
合计4,914个 recover launch API 均在各自 roots 结束前返回。

| L2/L3 合计，秒 | tile36 | tile16 | tile8 |
| --- | ---: | ---: | ---: |
| CUPTI Stage B 边界 | 101.958784 | 109.000041 | 105.086747 |
| bidiagonal 活动并集 | 76.584415 | 76.465083 | 77.958616 |
| recover 活动并集 | 29.883891 | 36.969280 | 32.999291 |
| bidiagonal / recover 交集 | 5.711446 | 5.635449 | 7.147751 |
| 仅 recover 活跃区间 | 18.507937 | 28.182657 | 21.529809 |
| 每 wave 最后一个 bidiagonal 结束后的 recover | 24.145139 | 31.308573 | 25.825785 |
| 完整 post-bidiagonal wave 尾部 | 24.872300 | 32.038724 | 26.626356 |
| 未记录到 device 活动 | 0.545840 | 0.547359 | 0.592978 |

| 中位数，L2 / L3，ms | tile36 | tile16 | tile8 |
| --- | ---: | ---: | ---: |
| roots 结束到 recover 开始 | 22.105 / 23.554 | 26.665 / 27.498 | 23.105 / 23.265 |
| recover kernel 时长 | 39.043 / 39.112 | 48.933 / 49.279 | 47.083 / 47.835 |
| bidiagonal kernel 时长 | 449.653 / 450.246 | 449.130 / 450.701 | 454.540 / 456.309 |

tile16 没有取得预想的提前开始或更多活动重叠；recover 时长和尾部增加，bidiagonal 并集基本不变。
tile8 的两类活动交集有所增加，但 recover 时长、尾部及 bidiagonal 并集都增加，最终完整阶段仍更慢。
因此“shared 降低、允许同驻”在这两档实验中没有兑现成阶段收益。

recover 三档均为66 registers、LOCAL=0；动态 shared 分别为101,088 / 44,928 / 22,464 B，CUDA occupancy API
上界为1 / 2 / 4 blocks/SM。这些是资源上界；未采集实际驻留、warp stalls 或内存流量计数，不能把回退归因到某个
未经测量的缓存、带宽或硬件调度机制。所有区间存在重叠，不相加，不当作可删除的关键路径时间。

独立 endpoint-count sweep 与区间合并算法的 **30 项比较全部通过**，容差 `1e-8` 秒。
trace 只用于解释；正式 STOP 来自前面的九次无插桩完整阶段计时。

## 收口与后续

停止本轮固定64线程的 tile16/8 路线，生产继续保留当前 tile 策略。tile12、32线程 launch 或其他 recover 算法没有在本轮
测量，不将它们伪称为实测 STOP，也不自动扩大本轮实验。
下一项建议回到 **L4/L5 selected-SP optimizer 本体归因**；已有 slots/scheduler、L8 STOP 和数值合同继续保持。

- 工具：[README](../tools/research/stage_b_recover_tile_20261029/README.md)，九个冻结工具。
- 成功证据：`fastkpc/artifacts/full_cuda_ci/stage_b_recover_tile_20261029/attempt3/`。
- 关键对象：`build.json`、`protocol.json`、两档 qualification/memcheck、`timing.json`、三份原始 trace、
  `analysis.json`、`binary-audit.json`、`prior-attempts.json`、`campaign.json`、`closure.json`、`manifest.json`。
- closure SHA-256：`0c7712117daded76b44ac6f2ad6c8cd60d8bc9b3bc76bf2d31ebd20e3cadbbda`。
- manifest SHA-256：`4c1c7ea5a3c79e96c343445a4cd155527a67693c0a6a452700839cd23112dcfe`。
- manifest 包含240个产物、104个生产输入、九个工具、28个依赖，合计381项；历史失败另由绑定的索引保留。

所有本任务 GPU 进程已结束。生产源码/native 未变，本次未 commit 或 push。
