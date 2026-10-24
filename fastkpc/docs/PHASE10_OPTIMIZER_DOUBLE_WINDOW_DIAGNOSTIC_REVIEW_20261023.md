# Optimizer 双窗口诊断：复核与完成路径等待的来源

日期：2026-10-23。复核 [相邻双窗口隔离对照](PHASE10_OPTIMIZER_DOUBLE_WINDOW_DIAGNOSTIC_20261023.md)。
本复核实验当时未改变生产源码、生产 native 或默认配置；research native 只在 `/tmp` 构建，
用于因果验证。Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 不变。

## 结论

```text
Codex 的测量          复现：由原始 runs.csv 重算的中位数一致，逐位门全部通过；
                      同日 control（与生产等价的 native）再次得到 B、C 0/6 胜 A
完成路径等待的来源     结果状态下载（device → pageable host_states，
                      mgcv_multi_penalty_gcv.cu:5868）在进程内串行执行，且发起它的线程
                      在本 setup 的 kernel 仍在运行时就已进入该调用并一直占着：
                      六个生产 profile 中，D2H API 都在本 setup 最后 kernel 结束前进入（128/128），
                      122–128/128 个 D2H 的开始 = max(上一个 D2H 的 API 返回, 本 setup 最后 kernel 结束)
                      + ≤~100 µs，D2H 区间两两不重叠（0/127）
驱动层复现             与项目无关的独立程序：生产写法下，50 ms kernel 的线程要等到 1000.5 ms 才返回
                      （被 1 s kernel 线程占着的 D2H 挡住）；先等待再拷贝、或 pinned 目标，为 70 ms
硬件队列               本机探针在 ≤32 条在途 stream 时未观察到所测队头阻塞；
                      C 的 0 个延迟 H2D 与之相符，D2H 串行是已验证的主要阻塞机制。
                      32 是待资格验证的性能配置，不是 CUDA 队列映射或正确性保证
定向修改的验证         只加“先在 blocking-sync event 上等待本 setup 完成，再做 pageable D2H”
                      （device SASS 不变）：C 在两组输入上都 6/6 胜同轮 A；
                      control A → patched C：L5 12.368 → 6.705 s，L3 1.863 → 1.305 s
                      （patched 同轮 C/A 中位 0.551、0.685）；
                      共 135,054 个 target 输出实例逐位一致
路线含义               Codex 的 STOP 对生产 native 成立；其重开条件（针对已观测完成路径等待的
                      新机制与独立诊断）已满足。建议先把“等待后再下载”作为单独的生产候选资格化，
                      再做在途 ≤32 的跨窗口补位原型；不据两组窗口外推整层或方法级收益
```

## 1. 复现 Codex 的测量

由 `campaign-*-runs.csv` 重算，与原文一致（单位秒，6 轮中位数与范围）：

| 输入 | A | B | C |
|---|---:|---:|---:|
| L3 W31+W32 | 1.837（1.794–1.995） | 2.672（2.157–5.622） | 4.538（4.105–5.191） |
| L5 W7+W8 | 12.409（11.464–12.687） | 24.072（20.548–29.837） | 35.734（25.603–40.152） |

`profile_summary.py` 由 Nsight 导出重算的 wall、跨窗口 overlap、峰值并发、延迟 H2D 数与
“最后 kernel → D2H 开始”中位数，也与原文的两张 profile 表逐项一致。

## 2. 生产 trace：D2H 串行，且持有者等待自身 kernel

逐个 D2H 检查（`d2h_lock_chain.py`、`d2h_wait_cause.py`、`d2h_api_after_kernel.py`）：

| profile | wall s | D2H API 在本 setup kernel 结束前进入 | 开始 = max(上一 D2H 返回, 本 API 进入, 本 kernel 结束) + ≤100 µs | D2H 区间重叠 | 推断占用等待 s（不含首个 D2H） | 已完成 kernel、排在后面等待的 worker·秒 |
|---|---:|---:|---:|---:|---:|---:|
| L5 A | 12.001 | 128/128 | 122/128 | 0/127 | 11.344 | 424.6 |
| L5 B | 19.776 | 128/128 | 128/128 | 0/127 | 19.164 | 1116.3 |
| L5 C | 20.460 | 128/128 | 127/128 | 0/127 | 19.853 | 542.7 |
| L3 A | 1.916 | 128/128 | 124/128 | 0/127 | 1.734 | 54.8 |
| L3 B | 1.965 | 128/128 | 125/128 | 0/127 | 1.846 | 92.6 |
| L3 C | 3.020 | 128/128 | 123/128 | 0/127 | 2.536 | 70.7 |

2026-10-23 统计修正：占用窗口起点补上本次 API 的进入时间，即
`max(0, own_kernel_end - max(previous_D2H_return, own_API_enter))`。
首个 D2H 没有同范围内的前驱，单独报告，未混入上表合计。旧公式会把尚未进入 API
的 GPU 执行空档算成占用，例如 patched L5 C 的 6.330606 秒；修正后 patched 六档
全部为零。原始 trace、计时和逐位结果没有变化，旧 manifest、工具和分析输出保留在
`attribution-v1/`，新 manifest 绑定修正后的分析。这仍是 trace 模型，不能测出驱动内部锁。

- **规则**：第 k 个 D2H 只有在第 k−1 个 D2H 的 API 返回之后才开始；若此时本 setup 的
  kernel 还没结束，就等到 kernel 结束。六个 profile 的中位偏差为 19–27 µs。这正是“一次只允许一个
  pageable D2H，且持有者从进入调用起一直占到自己的拷贝完成”的可观测特征。
  驱动内部用的是锁还是 staging buffer，本文不做断言。
- **占用时间**：L5 B/C 中，有 setup 在等待自身 kernel 的同时占着 D2H 的时间合计
  19.2/19.9 s，几乎覆盖整个调用（19.8/20.5 s）。其间 kernel 已完成的 worker 无法返回，
  也就不能领取下一个 setup。B/C 失去补位收益，原因在此。
- **原文的线索**：原文“长等待在 copy真正开始之前”“worker 需要等原 batch 返回”与此一致。
  这里不是队头阻塞：L5 A 被延迟的 117 个 D2H 中，只有 1 个紧接另一 stream 的 kernel
  结束而开始。B/C 中这类时间上的重合较多（L5 为 22/37 个），但不改变上述规则：这些
  D2H 同样在上一个 D2H 返回后 ≤100 µs 内开始。与之重合的多是新链的首个短 kernel
  （`target_squared_norm_kernel`），其中约一半由刚从上一个 D2H 返回的 worker 提交
  （L5 B 10/24、C 22/41，`coincidence_source.py`）。
- **A 为什么几乎不受影响**：A 有窗口屏障，窗口内最后一个返回本来就要等最长的
  kernel；串行只是把其余 setup 的返回推迟到那时。最长 kernel 结束后，排队的 D2H 不再
  等待任何 kernel，每个窗口末尾最多多出约 63×30 µs（约 2 ms）的排空。
  同日 control A 与修改后 A 相差在噪声内（见第 5 节）。
- **上一份复核的遗留问题**：32 worker 主计时方差大（L5 W8 7.46–19.78 s）、原因未明，
  现在可以解释为 D2H 获得顺序的随机性。修改后 C 的 6 轮范围为 6.665–6.763 s。

pageable H2D（Y 上传）未观察到同类长阻塞：这些生产 profile 中 H2D API 最长约 0.08 ms。

## 3. 驱动层独立复现

`pageable_d2h_convoy.cu`：线程 0 运行 1 s kernel，并最先进入 D2H；其余线程晚 20 ms
启动 50 ms kernel。每线程独占 non-blocking stream，拷贝 6 KiB。RTX 4090、driver
555.42.06、CUDA 12.5、`CUDA_DEVICE_MAX_CONNECTIONS=32`，各 3 次重复：

| 写法 | 8 线程：短线程返回 中位/最大 ms | 32 线程：短线程返回 中位/最大 ms |
|---|---:|---:|
| P：launch → pageable D2H → stream sync（生产写法） | 1000.5–1000.6 / ≤1000.9 | 1001.0–1001.3 / ≤1001.8 |
| S：launch → stream sync → pageable D2H | 70.2 / 70.2 | 70.4 / 94–122 |
| E：launch → blocking-sync event 等待 → pageable D2H | 70.6 / ≤70.8 | 71.1–71.6 / ≤72.5 |
| H：launch → pinned D2H → stream sync | 70.2 / 70.2 | 70.2 / 98–120 |

生产写法下，短 kernel 线程全部被长 kernel 线程持有的 D2H 挡住。S、E、H 都消除了这种
串行；但在 32 线程、20 个 CPU 的条件下，自旋等待（S、H）出现 94–122 ms 的长尾，
blocking-sync event（E）没有。生产实现应采用 blocking 等待，而不是更多的自旋线程。

## 4. 硬件队列：何时共享、什么会堵住

`queue_mapping_probe.cu`、`queue_policy_probe.cu`、`queue_refill_probe.cu`（同上环境）：

- **什么会堵住共享队列**：一个长 kernel 本身不会。只有排在它之后、必须等待它的命令
  才会：同 stream 的后续 kernel 与 blocking-sync event record 会堵住同队列的其他 stream；
  普通 event record、pinned D2H、host 侧 stream sync 不会。生产链中，optimizer kernel
  之后的 gather kernel 就是这样的等待命令，与上一份复核观察到的队头阻塞对应。
- **队列在提交时分配，优先空闲队列**：128 个 stream 时，被堵的总是 32、64、96，
  与长 kernel 放在哪个 stream、首次使用的顺序都无关。因此队列不是按创建次序固定
  映射的。31 个队列空闲时再提交 32 个 stream，前 31 个都落在空闲队列，只有第 32 个
  被堵。
- **补位**：32 个 stream 在途、其中一个完成后，在新 stream 或原 stream 上补位都在
  11.1 ms 完成（未被堵）；第 33 个并发 stream 在 1001 ms 完成（被堵）。

这些是本机驱动和所测提交形状下观察到的行为，不是 CUDA API 承诺。下一原型可先采用
最多 32 条在途 stream，继续实测是否有排队；数值正确性和资源生命周期不能依赖队列映射。
当前证据不要求先引入固定 stream 池，也不证明该并发数对其他驱动或工作负载始终足够。

## 5. 定向修改：同日 control 与 patched 对照

**构建**：用 `fastkpc/build` 中现有的生产目标文件重新链接，得到与生产完全相同的
native（8e457e81…），证明这些目标文件就是生产构建。control 与 patched 只替换
Small64 桶（L3/L5 窗口所用）的目标文件，二者从同一份源码副本编译：

| native | SHA-256 前缀 | 与生产的差异 |
|---|---|---|
| control | 0387d84d | device SASS 相同；host 指令在规范化 nvcc 路径哈希的匿名命名空间名后 0 行差异 |
| patched | 67baaf23 | device SASS 相同；host 只多出下述等待 |

patched 只在 `multi_penalty_gcv_cuda_optimize_batch` 的状态下载之前加入：在本 stream 上
记录一个 `cudaEventBlockingSync | cudaEventDisableTiming` event（每个 worker 线程一个），
`cudaEventSynchronize` 后再执行原 pageable D2H。数值计算、cohort、顺序、结果下标均不变。

**运行**：直接复用 Codex 的 `replay.cpp`、fixture、observer、比较器与捕获输入（均核对
SHA），环境同原文（仅 GPU0，CPU 0–19，32 connections，BLAS/OMP/MKL 线程数为1），
A/B/C 各预热一次后按 `ABC/BCA/CAB/CBA/BAC/ACB` 计时 6 轮。GPU1 全程 0% utilization。

| 输入 | native | A 中位数（范围） | B 中位数（范围） | C 中位数（范围） | 同轮胜 A（B / C） | 同轮 C/A 中位（范围） |
|---|---|---:|---:|---:|---:|---:|
| L5 W7+W8 | 生产（Codex） | 12.409（11.464–12.687） | 24.072 | 35.734 | 0/6 · 0/6 | — |
| L5 W7+W8 | control | 12.368（12.004–12.720） | 17.348（15.081–23.219） | 21.839（18.495–26.650） | 0/6 · 0/6 | 1.743（1.460–2.163） |
| L5 W7+W8 | patched | 12.228（11.504–15.269） | 9.851（8.193–11.381） | **6.705**（6.665–6.763） | 6/6 · 6/6 | **0.551**（0.437–0.582） |
| L3 W31+W32 | 生产（Codex） | 1.837（1.794–1.995） | 2.672 | 4.538 | 0/6 · 0/6 | — |
| L3 W31+W32 | control | 1.863（1.737–1.950） | 2.861（2.119–3.783） | 3.170（2.665–4.617） | 0/6 · 0/6 | 1.696（1.445–2.453） |
| L3 W31+W32 | patched | 1.903（1.847–2.412） | 1.519（1.351–1.738） | **1.305**（1.298–1.322） | 6/6 · 6/6 | **0.685**（0.546–0.716） |

- control 重现了生产的行为（B、C 均输）；差别只来自这一处等待。
- patched A 与 control A 的差在噪声内：该修改本身不会加快现在的 64/64 窗口形状，
  它的作用是让跨窗口补位成立。
- patched B（64 worker）也胜 A，但比 patched C 慢 47%（L5）与 16%（L3）：64 个在途
  stream 会共享队列（见下表的延迟 H2D）。在途数应为 32，而不是 64。
- 逐位：control 与 patched 的 campaign 各 48 次（含预热、计时与观测）、patched profile
  12 次，共 108 次完整 128-setup 重放、135,054 个 target 输出实例，全部通过 Codex 的完整状态比较器；
  输出 lease 全部关闭。两种 native 的显存峰值相同（L5 3511 MiB、L3 4169 MiB）。

**patched 的 Nsight 形状**（每档一次，只解释形状）：

| profile | wall s | 最后 kernel → D2H 开始 中位/最大 ms | D2H API 在本 kernel 结束前进入 | 延迟 H2D >5 ms | optimizer 峰值并发 | 跨窗口 overlap s |
|---|---:|---:|---:|---:|---:|---:|
| L5 A | 12.042 | 0.323 / 0.499 | 0/128 | 64 | 32 | 0 |
| L5 B | 9.050 | 0.337 / 0.816 | 0/128 | 96 | 32 | 6.201 |
| L5 C | 6.862 | 0.326 / 0.856 | 0/128 | 0 | 32 | 5.150 |
| L3 A | 1.952 | 0.281 / 0.392 | 0/128 | 27 | 32 | 0 |
| L3 B | 1.389 | 0.281 / 0.406 | 0/128 | 8 | 32 | 0.852 |
| L3 C | 1.419 | 0.319 / 1.064 | 0/128 | 0 | 32 | 0.396 |

生产 profile 中同一指标的中位数为 358–7769 ms。修改后，排在被占 D2H 之后、kernel
已完成的 worker·秒为 0。

## 6. 理想化模型与实测的对照

用生产 A profile 的逐 setup 服务时间（本 setup H2D 开始 → 最后 kernel 结束，加 D2H），
`schedule_model.py` 给出（不是预测，只用于对照）：

| 输入 | 模型 A（成对共享队列 + 屏障） | 实测 A profile | 模型：32 队列滚动，原顺序 | 实测 patched C | 模型：最长者先发 | 最长单 setup |
|---|---:|---:|---:|---:|---:|---:|
| L5 W7+W8 | 11.993 | 12.001 | 6.585 | 6.705 | 6.227 | 6.227 |
| L3 W31+W32 | 1.686 | 1.916 | 1.264 | 1.305 | 1.090 | 1.090 |

模型对 A 的 L5 几乎完全吻合；patched C 比“32 队列滚动”模型只多 1.8%（L5）与 3.2%（L3）。
保留窗口屏障、只把队列改成 32 个不共享的模型分别为 11.362/1.529 s：收益几乎全部来自
跨窗口补位，而不是减少队列共享本身。按时长排序（D 档）在这两组输入上最多再省
0.36/0.17 s，因为最长的 setup 在原顺序中开始得较早（L5 第 23 个、L3 第 74 个）；
对整层的作用要另测。

这只是两组相邻窗口的结果。上一份复核从两个窗口外推的整层区间（L5 约 104 → 22–33 s，
L3 约 48 → 13–15 s）仍未得到确认，本文不把它写成层级或方法级收益。

## 7. 对路线的建议（由 Codex 判定）

第 1 项现已完成，见 [原窗口 status-wait 资格报告](PHASE10_OPTIMIZER_STATUS_WAIT_20261023.md)。
prepared-owned、默认关闭的实现通过五桶局部检查及六次完整 canonical 硬门。
GPU1 共享主机下的四次 off/on/on/off 对照，中位数为 `543.6695/547.1705s`，
开启慢 `0.644%`，通过预定的 `<=3%` 开发无实质回退门，没有证明原窗口加速。
这不等于第 2 项已资格化：跨窗口补位、整层 prepared 常驻和 runtime rolling
仍未实现，runtime rolling 继续为 `NOT_YET_GO`。

1. **先做一个单独的生产候选**：只在 optimizer 首次状态下载前等待本 setup 完成。
   使用 prepared 持有的独立 blocking-sync event，创建时分配、销毁时释放，保留原 residual
   completion event。该等待已覆盖同 stream 前面的数值工作，后续 trace 下载不重复插入
   等待；fixed evaluator 等其他入口先不扩大修改。默认关闭，保持 64/64 调度，验证真实
   容量桶、逐位输出、异常取消、资源和完整 canonical 中性；不能沿用研究 patch 的资格。
2. **再做跨窗口补位原型**：在途 setup ≤32（32 worker，而不是 64），沿用现有的原子
   领取调度；先在 L3、L5 整层上测量，带逐位门。其余约束仍按 Codex 此前的要求：运行中
   不做 prepared 的 `cudaMalloc`/`cudaFree`（创建含 cudaMalloc 与 stream 同步，释放是
   cudaFree），以及 coefficient arena 登记、stage A 顺序、residual token 生命周期。
3. **设计约束**：不要在共享队列的情形下，把 blocking-sync event record 紧跟在长 kernel
   之后（它会堵队列）；有待执行工作的 stream 总数（含进程内其他组件）保持 ≤32。
   今后若让 stage B 或 consumer 与 optimizer 重叠，它们的 pageable D2H 同样会排在被占的
   D2H 之后，除非第 1 项已落实。
4. **按时长排序**放在第 2 项之后再评估，优先级低。

## 证据与复现

工具：[optimizer_d2h_review_20261023](../tools/research/optimizer_d2h_review_20261023/README.md)
（trace 分析脚本、四个独立探针、research native 构建、patch、replay 构建与 campaign 脚本）。
证据：`fastkpc/artifacts/full_cuda_ci/optimizer_d2h_review_20261023/`，包括 control/patched
的完整 campaign 与 profile 输出、探针输出、`analysis/` 下各脚本对生产与 patched trace 的
输出、目标文件等价性检查与 research native 构建日志。research native 位于
`/tmp/d2h_native`，其 SHA-256 写入 manifest，可按 README 在约 6 分钟内重建（重新链接
生产目标文件必须复现 8e457e81…）。这是 development diagnostic，不是 clean-producer
性能中位数或 promotion evidence。

manifest SHA-256：

```text
e1a70a0b5940c12d7463ed365fa62b4d2782815663be5a7fc24df6b291d4608e
```
