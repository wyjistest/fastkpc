# Optimizer 槽位线（L5/L4）复核与 407 秒预算

日期：2026-10-25（同日按 Codex 核对意见修订，见“修订记录”）。复核 2026-10-23 至
10-24 已提交的槽位线：
[L5 槽位原型](PHASE10_OPTIMIZER_SLOT_POOL_20261023.md)、
[L5 runtime](PHASE10_OPTIMIZER_SLOT_RUNTIME_20261023.md)、
[L4 容量诊断](PHASE10_OPTIMIZER_L4_DIAGNOSTIC_20261023.md)、
[L4 workers32 runtime](PHASE10_OPTIMIZER_L4_RUNTIME_20261024.md)、
[L4 尾部归因](PHASE10_OPTIMIZER_L4_TAIL_20261024.md)、
[L4 槽位原型](PHASE10_OPTIMIZER_L4_SLOTS_20261024.md)、
[L4 槽位 runtime](PHASE10_OPTIMIZER_L4_SLOT_RUNTIME_20261024.md)，
以及 runtime 提交 `33e2d3e`、`4a1ff6b`。生产源码与 native 只读；新增的是一次 GPU1
生产 canonical 全程 trace 和两个独立 CUDA 探针，均为诊断，不是性能证据。正式开发
基线仍为 **407.531 s**。Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、
recommended=false 均不变。

## 修订记录（Codex 核对后）

Codex 核对首版后提出四处修正，并指出 tracing 开销和 L8 预加载效果的说法过强。
我逐项对照原始记录复核，全部接受，正文已改：

1. **补位模型的顺序**：首版 `predict.py` 按各 setup 首个 kernel 的实际开始时间排序，
   而这个顺序已受线程调度和 channel 等待影响。现改为真实请求顺序（第 3 节）。
   L3 由 14.877 s 变为 14.852 s，对顺序不敏感：随机顺序为 14.8–15.5 s。
   L6 由 17.284 s 变为 **18.336 s**，对顺序敏感：随机顺序 p5/p50/p95 为 14.6/17.9/21.2 s。
   另外，用同一次运行的服务时间和间隔复现 L4/L5，不能独立证明 L3/L6 在持续补位
   负载下仍保持窗口模式的 busy 时间；这一假设现已明确写出。
2. **L8 API 归因**：首版只按线程 ID 在整个进程内过滤；线程 ID 会被复用，因此混入了
   其他层的记录（如 `first_api_s = -379.791`），该部分作废。现改为按 correlation ID
   定位每个 L8 optimizer 的启动（第 6 节）。kernel 层面的阻塞结论和探针不受影响。
3. **“GPU 全空闲 51.8 s”**：tracer 只采集 kernel 和 runtime API，没有 memcpy/memset
   activity，只能称为“未记录到 kernel 执行的间隙”。按层拆分是从首个 kernel 起累加
   各层耗时得到的，没有实际阶段时间戳，已删除。该数字不能当作可消除的预算。
4. **时间口径**：cudaGetDeviceProperties 的累计 20.973 s、调用区间并集 10.712 s
   （Codex 重算，我复现一致）和主线程 7.940 s 不能互换，也都不等于可节省量；
   L3 的部分与将来的 L3 槽位重叠，不能重复计入。表 2 的 optimizer 193.22 s 是逐层
   中位数之和；完整 optimizer host boundary 的中位数是 203.983 s。两者现分别标注。
5. **其他**：仅凭一次 traced 运行（406.981 s）落在历史 ON 范围内，不足以证明 tracing
   开销可忽略，相关说法已删除。“预加载后生产 L8 稳定在约 14.9 s”只是待验证的假设。
   首版第 10 节“约 340 秒”的合计撤回。
6. **（第二次修订）** 第 4 节不再把 span 与 busy 之差全部记为队头阻塞：其中还包含
   提交延迟、kernel 间隙等其他等待。L7 的波动来源改为“可能”。依据见设备属性缓存复核
   的修订记录。

首版输出保留为 `analysis-v1-start-order.json`、`predict-v1-start-order.json`，其 SHA
即首版所列值；修订版 SHA 见文末。

## 结论

```text
代码复核    33e2d3e/4a1ff6b 的槽位路径未发现缺陷（第 1 节）
复现        三组 campaign 中位数由 report.json 重算一致：550.152→472.080、
            475.439→461.663、463.288→407.531 s；18 次运行硬门全真，CI 变化 0
407 s 去向  逐层中位数之和：optimizer 193.2 s（L3 49.7、L6 46.5、L5 31.9、L4 27.3、
            L8 19.6、L7 10.8、L2 7.3），Stage B 144.5 s（L3 60.3、L2 54.6、L4 20.1、
            L5 7.8）。完整 optimizer host boundary 中位数另为 203.983 s，口径不同
全程 trace  当前生产配置一次 canonical：GPU1 独占，406.981 s，与 Codex ON1 逐位一致。
            tracing 开销未单独测定
模型        按真实请求顺序做 32 槽 list scheduling；用同一次运行的服务时间与间隔复现
            L4/L5 槽位实测（误差 ≤0.5%）。L3/L6 的预测假设窗口模式 busy 在持续补位下不变
下一批机会  (1) L8：24 次历史运行都等于 14.907 s 最长 setup 加 1–2 个完整 setup，
                残差 ≤0.057 s，平均多 3.93 s。原因是 CUDA lazy loading：函数首次启动
                会同步上下文；探针中事先 cudaFuncGetAttributes 可消除。生产效果待 A/B
            (2) cudaGetDeviceProperties 每个 prepared/调用各一次：6,777 次，累计 20.973 s，
                并集 10.712 s，主线程 7.940 s；可节省量待 A/B，L3 部分与 L3 槽位重叠
            (3) L6 槽位：模型 45.8 → 18.3 s（下限 12.0 s），对顺序敏感
            (4) L3 槽位：模型 49.6 → 14.9 s，对顺序不敏感；但在途约 437 个 block
                （3.4/SM，L4/L5 不超过 2/SM），单 block 是否变慢须实测
Stage B     最大的剩余桶。主 kernel bidiagonal 每个 component 只用一个 warp，
            shared memory 19.7 KB/block，按此每 SM 最多驻留 4 个 → 全卡 512 个，
            与 10-22 测得的“约 512–640 在途即饱和”一致；先用 Nsight Compute 定位
工具        nsys 2024.2.3 自带 CUPTI 在完整 canonical 上崩溃（内核日志：libcupti.so.12.5
            空指针写），并非 fastkpc 缺陷；改用最小 CUPTI 注入 tracer
勘误        我 10-23 复核写 optimizer kernel“无 shared/local memory”不对：它有 7,120 B/线程
            的 stack frame（local memory）。首次启动使上下文 local-memory 预留从 108 MiB
            升到 1,444 MiB，与 Codex L4 归因中来源未定的 1,152 MiB 同量级，
            很可能是同一项（未逐字节核对）
```

## 1. 代码复核

逐项核对 `full_cuda_ci_one_call.cpp`（`OptimizerSlotPool`、`prefill_optimizer_slot_target_states`）、
`mgcv_multi_penalty_gcv_small64.cu`（rebind/drain）及 `mgcv_multi_penalty_gcv.cu`：

| 关注点 | 结论 |
|---|---|
| rebind 是否完整 | create 只上传 10 个几何数组，并写入 n/q/penalty/capacity/total_root_rank/initial_log_sp；rebind 重传同样 10 个数组，并刷新 total_root_rank、initial_log_sp、generation。n/q/penalty/capacity 由形状检查固定；arena 按该层最大 roots 分配 |
| lease 与过期令牌 | residual 带 generation；release 和析构只在 generation 相同时清 lease；rebind 递增 generation，旧令牌不能释放新 lease |
| detach 顺序 | `enqueue_store` 让 arena stream 先等 producer completion event，再做 D2D；worker 在自己的 blocking event 完成后才 release；复用前输出已复制 |
| Stage A 重绑 slot0 | `store_residual` 在返回前 `cudaEventSynchronize`，release 后才进入下一个 cohort 的 rebind |
| 线程安全 | geometry 打包（Rcpp）在 worker 启动前于主线程完成；worker 只碰 C++ 状态；`setups[i]`、`bind_ms[i]` 按下标独占写 |
| worker 内分配 | `reserve_slot_layer` 预先保留整层系数；worker 期间 `ensure_coefficient_storage` 不增长 |
| 失败路径 | 任一 worker 异常即 cancel；排空 arena 与本槽 stream 后才释放 pending；pool 析构 drain 后 reset |
| 与窗口路径一致 | 两条路径都设 `force_final_selected_evaluation=true`（窗口路径经 `retain_residuals`） |

未发现阻塞问题。观察项：Stage A 每次 bind 都调用一次 `cudaMemGetInfo`（L4 共 2,154 次），
Stage A bind 总计约 0.12 s，可以接受。

## 2. 当前 407 秒的分层构成

L4 槽位 campaign 三次 ON 的逐层中位数（秒；other = 层总耗时减去前四列）：

| 层 | 层总计 | optimizer | Stage A | Stage B | consumer | other |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.52 | 0.24 | – | – | 6.77 | 0.51 |
| 2 | 94.29 | 7.31 | 15.32 | 54.56 | 3.81 | 13.29 |
| 3 | 127.96 | 49.67 | 0.80 | 60.33 | 2.20 | 14.96 |
| 4 | 53.43 | 27.25 | 0.27 | 20.09 | 0.86 | 4.97 |
| 5 | 43.05 | 31.89 | 0.13 | 7.77 | 0.33 | 2.94 |
| 6 | 49.27 | 46.48 | 0.03 | 1.22 | 0.07 | 1.48 |
| 7 | 11.45 | 10.81 | – | 0.29 | 0.01 | 0.33 |
| 8 | 19.96 | 19.58 | – | 0.27 | – | 0.11 |
| 逐层中位数之和 | 407.13 | 193.22 | 16.56 | 144.51 | 14.24 | |

逐层中位数之和不等于整体中位数。receipt 的完整 optimizer host boundary（`optimizer_sec`）
中位数为 203.983 s，并包含窗口准备等，与上表 optimizer 列口径不同。

在全部 24 次历史运行中，L6（39.9–49.5 s）、L7（9.4–14.3 s）、L8（16.5–22.2 s）的波动
与开关无关；L3 则稳定在 48.6–51.9 s。

## 3. 全程 trace：方法

`run_profile.R` 用当前生产配置（S2、status-wait、L5 槽位、L4 workers32、L4 槽位，
native `037b92f1…`）跑一次 canonical；GPU1 上没有其他进程，GPU0 有他人进程。
nsys 2024.2.3 第一次尝试在 L3 第 1 窗口创建 prepared 时崩溃：R 打印的是 native 调用栈，
内核日志则是 `R[…]: segfault at 4 … in libcupti.so.12.5`，崩溃线程不是主线程。
证据保存在 `nsys-crash/`；`/var/crash` 里还有一份 10-23 的 nsys-launcher 崩溃记录。

因此改用 `ktrace.cpp`：它通过 `CUDA_INJECTION64_PATH` 注入，只启用 concurrent-kernel
与 runtime-API activity，按二进制写出，包含 stream、HW channel、local memory 字段及发起
线程。没有 memcpy/memset activity。该次运行 406.981 s，落在历史 ON 的 400.9–408.8 s 内；
tracing 开销未单独测定，丢失记录 0。与 Codex ON1 receipt 比较，p 值、pMax、邻接矩阵
和 n.edgetests 全部逐位相同。共 569,524 个 kernel，4,111,581 条 API 记录。

**setup 与请求顺序**：optimizer kernel 由 worker 线程发起，Stage A/B 由主线程发起。
窗口模式下每个请求都有自己新建的 prepared 对象，主线程按请求（metadata）顺序创建
它们的 stream。窗口是该层 `std::map` 顺序的连续分块；worker 在起跑屏障后用原子计数
取请求，线程顺序不代表请求顺序。CUPTI 的 stream ID 在创建时分配：72/72 个窗口的
stream ID 都是连续区间且逐窗口递增，`predict.py` 中有断言。因此层内原始顺序为：
窗口按序，窗口内 stream ID 升序。槽位模式（L4/L5）只有一个原序队列，setup 的开始
顺序就是取号顺序；拆分 setup 的依据是同一槽 stream 上首个 kernel 名再次出现。

## 4. 模型校验与窗口模式的等待

**同一次运行内的校验**（槽位层 span 与 busy 一致，span/busy 为 1.015）：

| 层 | 实测 s | 模型：span + 0.67 ms 槽间隔 | 模型：busy×1.015 + 间隔 |
|---:|---:|---:|---:|
| L4 | 27.195 | 27.189 | 27.132 |
| L5 | 31.621 | 31.611 | 31.789 |

槽上相邻两个 setup 之间的 host 间隔（bind、status 下载、detach、release）中位数为
0.666/0.678 ms，p90 小于 1 ms。服务时间和间隔都取自同一次运行的槽位层，因此这只说明
模型结构能复现槽位模式，不是对 L3/L6 的独立验证。

**窗口模式（64 个 stream 共享 32 个 channel）**：每个 setup 首末 kernel 的 span
远大于其 busy。L3 的 Σspan 为 878 s，而 Σbusy 只有 458 s；L6 分别为 622 s 和 361 s。
其中有队头阻塞的直接证据：L3 有 260 个 setup 晚于窗口开始 100 ms 以上才启动，
最晚 1.33 s；有两个窗口的第 64 个 setup 要等其余 63 个都结束后才开始。窗口实测与
“各窗口内最长 busy”之和相比，L3 多出 10.1 s（49.59 对 39.47），L6 多出 9.2 s
（45.80 对 36.60）。这部分差值除队头阻塞外，还包含提交延迟、kernel 间隙等其他等待，
trace 不能完全分开（见[设备属性缓存复核](PHASE10_DEVICE_PROPERTIES_CACHE_REVIEW_20261025.md)）。
无论来源如何，窗口层都应用 busy 建模，不用 span。

## 5. 各层预测（模型，不是实测）

服务时间取 busy×1.015 + 0.67 ms，按真实请求顺序做 32 槽 list scheduling。前提是：
窗口模式测得的 busy 在持续补位负载下不变。这一点未经验证。

| 层 | 实测 s | 原窗口 + 32 workers | 32 槽，请求顺序 | 随机顺序 p5–p95 | 下限 max(Σ/32, 最长) | 最长者先发 |
|---:|---:|---:|---:|---:|---:|---:|
| L3 | 49.59 | 43.14 | **14.85** | 14.82–15.46 | 14.61 | 14.64 |
| L6 | 45.80 | 39.12 | **18.34** | 14.61–21.17 | 12.00 | 12.00 |
| L7 | 9.38 | 9.52 | 9.52 | 9.52–10.94 | 9.52 | 9.52 |
| L8 | 16.61 | 15.14 | 15.14 | 15.14 | 15.14 | 15.14 |

- **L6**：245 个 setup，每个只有 2/3/5 个 target（容量 5）。32 槽在途约 150 个 block，
  低于 L4/L5，但仍是外推。最长 setup busy 11.8 s，是第 167/245 个请求。结果对顺序
  敏感：随机顺序从 14.6 s 到 21.2 s；要达到 12.0 s 需事先知道时长，而 L5 已表明
  target 数不能预测时长。small64 backend，q=55，六个 penalty。原型应按原序报告。
- **L3**：4,065 个 setup，target 数 2–33（均值 13.5，18 个 target 的有 997 个）。
  32 槽补位相当于平均约 437 个 optimizer block 在途。kernel 255 寄存器、64 线程/block，
  每 SM 最多驻留 4 个，全卡 512 个；按 block-seconds 估算，驻留上限对应的下限是 12.7 s。
  L4/L5 槽位约为 2 block/SM，没有观察到变慢；L3 约 3.4/SM，FP64 和 local memory
  共享是否拖慢单个 block 没有数据。须先测量补位负载下的逐 setup 服务时间。
- **L7/L8**：每层一个窗口，时间就是最长 setup 的延迟（L7 一个 3-target setup 9.4 s；
  L8 一个 2-target setup 14.9 s）。补位无效，只有提高单 target 的 kernel 速度才有用。
  L7 的波动可能来自 44 个 stream 共享 32 个 channel 的队头阻塞，未单独验证。

## 6. L8：首次启动 kernel 时的上下文同步

按 correlation ID 对齐的 L8 optimizer 启动（`predict.json` 的 `l8_optimizer_launches`）：
第一个启动 API 耗时 16.576 ms，kernel 在 +0.000 s 开始，运行 1.686 s；其余 8 个在
+0.2–0.5 ms 发起，API 约 0.006–0.017 ms，却都在 +1.699 s 才开始。9 个 setup 分别在
9 个不同的 channel 上。first-use 扫描（`first_use.py`，同样按 correlation ID）在整次运行
中只找到这一处实质性阻塞：第一个 optimizer 运行期间，base80
`form_multi_penalty_residuals_kernel` 首次启动，此后所有启动都被挡住，直到正在运行的
kernel 结束，计 1.6855 s。base80 的函数在 L8 才首次使用（q=73 超出 small64）。
L3 第 1 窗口约 88 ms 的阻塞是同一机制：small64 `gather_multi_penalty_coefficients_kernel`
首次启动（API 0.695 ms）时，两个首批 optimizer 已发起但尚未开始；它直到二者都结束
才在 +86.6 ms 开始。扫描要求首次启动时已有 kernel 在运行，所以漏掉了这一例，
是逐项检查时发现的。

**历史验证**：用本次 trace 的 L8 busy（14.907、5.268、4.630、2.158、1.930、1.749、
1.715、1.699、1.615 s），把 24 次历史运行的 L8 拟合为 14.907 + 0.03 + 至多两个
完整 setup：

```text
16.6 s 类（12 次）≈ 14.94 + 1.6–1.9      19.58 ≈ 14.94 + 4.630
18.70 ≈ 14.94 + 2.158 + 1.615            20.2 s 类（2 次）≈ 14.94 + 5.268
21.2–22.2 s 类（8 次）≈ 14.94 + 5.268/4.630 + 1.6–1.9
全部残差 ≤ 0.057 s；平均比 14.94 s 多 3.93 s，中位 L8 17.79 s
```

L8 的双峰由此得到解释：哪一个或两个 setup 被串行排到最长 setup 之前，取决于哪个
worker 先触发首次加载。这是对历史时间的拟合，不是对修复后时间的测量。

**探针**（`lmem_growth_probe.cu`，GPU1，见 `lmem_growth_probe*.out`）：先让 kernel 0
运行 400 ms，再在其他 stream 上发起后续 kernel：

| 情形 | LAZY（默认） | EAGER |
|---|---|---|
| 1：后续为同一已加载函数 | 并发 | 并发 |
| 2/3：先启动一个新函数（1 KB stack 或 0 stack） | 全部等到 +400 ms | 并发 |
| 4：同 3，但无 stack 增长 | 全部等到 +400 ms | 并发 |
| 5：同 3，事先对新函数调用 `cudaFuncGetAttributes` | **并发** | – |
| 0：后续 kernel 需要更大 stack（4K→6K） | 等待 | **仍然等待** |

结论有两条。(a) lazy loading 下，函数首次启动会同步上下文，这是 L8 和 L3 第 1 窗口
阻塞的原因；事先 `cudaFuncGetAttributes` 可以消除（情形 5）。(b) 另一机制是：需要更大
per-thread local memory 的启动要等正在运行的 kernel 排空。`cudaDeviceSetLimit(cudaLimitStackSize, 8192)`
只会提前分配内存（启动期间显存下降由 962 MiB 变为 2 MiB），并不消除这种等待。
本次生产运行中没有观察到 (b) 的实质代价：L8 的 7,120→7,136 B 增长发生时没有长 kernel
在运行。

修复方向：在并发阶段之前，只对实际路径需要的 kernel 预加载（例如对相关函数调用
`cudaFuncGetAttributes`）。函数不变，数值预期不变，但生产 L8 能否稳定在约 14.9 s 须做
A/B：预加载成本计入 outer elapsed，并复查显存余量。不建议全局
`CUDA_MODULE_LOADING=EAGER`：它会把 cuBLAS/cuSOLVER 等全部模块提前装入，启动时间和
显存未经评估，可能影响 L4/L5 槽位已资格化的 2 GiB 余量门。

## 7. cudaGetDeviceProperties 与无 kernel 间隙

`cudaGetDeviceProperties` 共 6,777 次，中位 1.22 ms/次。各口径：累计 20.973 s；调用区间
并集 10.712 s；主线程 7.940 s（5,603 次）。三者都不是可节省量，须 A/B 测定。
层的归属（近似）：L2 共 2,252 次，主线程与 18 个 worker 各 1,126 次；并发调用相互争用，
p90 21 ms，最长 43 ms。L3 为主线程 4,065 次，每个 prepared 一次，计 5.24 s，落在
窗口准备阶段；这一部分与将来的 L3 槽位重叠。L6 为 245 次，0.33 s。
调用点：`mgcv_multi_penalty_gcv.cu:5648`（prepared 创建，仅用于填 `info.gpu_name`）、
`mgcv_fixed_sp_runtime.cu:2961`、`mgcv_single_penalty_gcv.cu:1964` 等七处。

全程未记录到 kernel 执行的间隙共 51.8 s，其中 >10 ms 的 26.6 s，>100 ms 的 18.0 s。
copy engine 未被 trace，这些间隙不能算作 GPU 空闲或可消除时间。另外，`cudaMalloc`/`cudaFree`
分别有 107,325/114,153 次（累计 5.6/7.0 s，主线程 1.3/2.8 s），主要来自 L2/L3 的对象建销。

## 8. Stage B 的 kernel 形状

L2 Stage B（取自 nsys 崩溃前已覆盖的部分 trace）：

| kernel | block | 寄存器 | shared memory | 每 launch 中位 | 占 Stage B kernel 时间 |
|---|---:|---:|---:|---:|---:|
| bidiagonal | 32 | 100 | 19,656 B | 365 ms | 84% |
| recover | 64 | 66 | 101,088 B | 51 ms | 12% |
| roots | 128 | 58 | 0 | 14 ms | 3% |

sm_89 每 SM 最多 100 KB shared memory，另需为每个 block 保留 1 KB。按此 bidiagonal
每 SM 最多驻留 4 个 block，即每个 SM 子分区只有一个 warp，无法隐藏延迟；全卡最多
512 个 component 并行。recover 每 SM 只能驻留 1 个。10-22 测得“在途约 512–640 个时
吞吐饱和”，与这一驻留上限吻合，说明饱和点可能来自 occupancy，而不是硬件吞吐。
这是推断，须用 Nsight Compute 核对 achieved occupancy 与 stall 原因。
Stage B 合计 144.5 s，是剩余最大的一项；若能减少 shared memory 而不改变运算顺序，
则可能在保持逐位一致的前提下提高驻留。

## 9. 投机性 prefill 浪费（数量级）

prefill 会为本层全部计划 task 产出 component，但边删除后有些 task 不再执行：
L2 有 7,234/49,919（14.5%）、L3 有 12,216/54,878（22.3%）、L4 有 2,334/17,767（13.1%）
从未被消费。按比例折算，Stage B 浪费约 24.5 s（L3 13.4、L2 7.9、L4 2.6）；若连
optimizer 和 Stage A 一起线性折算，上限约 47.5 s。要减少这部分须改变 cohort 组成，
会改变位值，需要重新做资格，不属于逐位不变的改动。

## 10. 执行次序

同意 Codex 的次序：

| 顺序 | 工作 | 判断 |
|---:|---|---|
| 1 | 按设备缓存属性，保留诊断身份与错误处理 | 窄原型 GO |
| 2 | 对实际 L8 路径做定向预加载 A/B | 实验 GO，集成待验证 |
| 3 | 按请求顺序映射，做 L6 captured-cohort 槽位原型 | GO，随后做容量与完整 runtime |
| 4 | L3 槽位 | 后置，先测持续负载下的服务时间 |
| 5 | Stage B occupancy/stall 归因 | 诊断 GO，暂不改 kernel |

对第 1 项的复核要点：

- 覆盖全部七个调用点。缓存按设备分开、线程安全，且只缓存成功的查询。
- 失败时的报错文本和类型保持不变。
- `gpu_name`、`gpu_uuid`、compute capability、`sm_count` 等诊断值与现有 receipt 逐字段相同。
- 计量包括调用次数（应降到每设备每进程一次）、主线程 API 时间和 outer elapsed A/B，
  并要求 CI/pMax 逐位一致。

第 2 项建议单独 A/B，以便分别归因。不给出合计预测。

## 工具与证据

工具：`fastkpc/tools/research/optimizer_slot_line_review_20261025/`
（`run_profile.R`、`trace.sh`、`ktrace.cpp`、`analyze.py`、`predict.py`、`first_use.py`、
`lmem_growth_probe.cu`、`profile.sh`）。证据：`fastkpc/artifacts/full_cuda_ci/optimizer_slot_line_review_20261025/`
（`ktrace-run/` 原始记录与 `analysis.json`/`predict.json`/`first-use.json`，`nsys-crash/`，
探针输出，GPU 遥测）。修订版输出 SHA-256：

```text
analysis.json        2d6101a9576664d81c0942a498a8b80050a0fff2ecd279b74d296822751f5596
predict.json         cc9fe5b8ba8b953c6935b43dc0967e3aa0cff353c9bc06ea40bf8737121acbdb
first-use.json       7673f5705011a3df2dd16076d6db7e2320b418d51d869c958ede33083899f7ef
profile-result.json  1bd7a3c0fc11d66ccb22f98c592cd8973c1b95aee0492efb13db97477ffee777
ktrace.so            fae0e66400d6a64487924000d7d29fd72ca0818ad3b32457a5382e3dd3db2adb
```

首版输出（顺序为开始时间）：`analysis-v1-start-order.json` `f09045d7…5bc020`、
`predict-v1-start-order.json` `84ad04d1…0266ac1`。

本次没有修改生产源码、native 或默认配置，没有 commit 或 push。
