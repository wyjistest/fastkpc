# L4/L5 exact-primary optimizer 身份与阶段归因：复核

日期：2026-10-29。复核 [L4/L5 optimizer 身份与阶段归因](PHASE10_OPTIMIZER_L45_ATTRIBUTION_20261029.md)（证据
`fastkpc/artifacts/full_cuda_ci/optimizer_l45_attribution_20261029/attempt3/`）。只读复核，没有新的 GPU 运行或编译。
`PASS_L45_IDENTITY_AND_EXISTING_PHASE_ATTRIBUTION` 成立；accepted opt-in development median 仍为 311.440 s；默认关闭、
L8 optimizer STOP、Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。
同日按 Codex 核对意见修订，见“修订记录”。判定：**vector-postback 实现审计 GO，GPU 优化原型 NOT YET GO**。

## 修订记录（Codex 核对后）

Codex 重新核对了 267 项 manifest、51 项历史失败记录、22 个共享源码与 70 个硬门，并复现了 steepest-descent 比例、halving 统计
与时钟拟合，均一致。两条源码事实（reference DORMBR 只由一个线程执行；LWORK 使 L4 的块大小为 2、Q/P 各 19/18 块，L5 为 32、
两侧各 2 块）成立，后者必须保留，不能增加 workspace 或统一块大小。三处表述收紧，我复核后全部接受：

1. **低功耗不说明远未满载。** 第 3 节第 3 条只能说明功耗较低；某个执行资源或依赖链仍可能饱和，不能推出可兑现的并行余量。
2. **“整个 stage2 由一个线程完成”过宽。** 单线程的是 reference 反射变换核心（`dormbr_`）。stage2 计时还包含同步等待；direct
   路径的 stage2 另含 64 线程执行的 `left_u` 初始化（`lapack_312_small_dgesdd.cuh:1049`）。所以 58.9%/71.7% 不是纯单线程耗时
   占比。
3. **约 99% 是一致性估计。** 时钟采样与 cycles 相符，但没有逐 block 时间戳，不能认定候选 target 就是最后完成的 block；正文已有
   这层限定，结论部分同样沿用。

下一项（Codex 的建议，我同意）：先审计 DLARFB 内实际的 BLAS 调用与数据依赖，优先寻找独立输出可以并行的位置。DTRMM 原地覆盖
输入，不能把所有矩阵元素直接分给不同线程：我核对了 vendored 的 DTRMM，右乘 B := B·A（upper）按 j 从 n 到 1 处理各列，第 j 列
读取尚未更新的第 k < j 列，并有 `A(k,j) != 0` 的零值分支；各行彼此独立，各列之间有依赖。块划分、累加顺序、零值分支与
workspace 依赖都要保持。

## 结论

```text
完整性    closure.json、manifest.json、receipt.rds 的 SHA-256 吻合；manifest 267 项（56 个产物、104 个生产输入、10 个工具、
          97 个依赖）按当前文件重算一致；两个失败目录的索引 51 项一致；shared-dependencies 的 22 个源文件与 c38b04b 的
          blob、记录的 SHA 和当前文件三者一致，4 个数据文件一致；native 12acc3cd… 未变
门        canonical report 的 70 个 gate 全部为真；authority pMax 最大绝对误差 3.05425129631942e-12
身份      3,109 个 setup 的 geometry 地址都等于首项加请求序号 × 256 B，external ID 都等于 level × 10^6 + 请求序号。源码中
          geometry 在 reserve 之后按 work.requests 顺序 push_back，bind 传入 geometry_.at(i)，所以地址就是请求序号。
          GPU 开始序位与请求序号不同的 setup 为 35/45 个，最大位移 22/21
复现      最慢/最后结束四行、候选 target 倍数与每次评估 cycles、分布与相关系数、四个主阶段与 SVD 内部占比、全部计数，
          由 joined TSV 重算一致；计数之间的四个恒等式对每个 target 都成立
补充      一、一致性估计：关键候选 target 的四个主阶段 cycles 约等于 optimizer kernel 时长 × 2.72–2.75 GHz，L4/L5 期间
          GPU1 的 SM 时钟采样中位为 2,760 MHz，已记录阶段约覆盖 kernel 时长的 99%。没有逐 block 时间戳，不能认定候选
          target 就是最后完成的 block
          二、静态：L4/L5（q = 37/46 > 32）各路径中，stage2 的 reference 反射变换核心（dormbr）由 64 线程 block 中的
          1 个线程执行；stage2 计时另含同步与（direct 路径）64 线程初始化，不是纯单线程耗时。按 LWORK 合同，实际
          reflector 块大小 L4 为 2、L5 为 32
自我更正  我在 recover tile 复核中写“与长 setup 开始较晚形成的尾部相符（最长的一个 busy 10.4 s）”，括号部分错误：
          旧 trace 中该 setup 在 6.17–16.61 s 运行。我用旧 trace 独立复算，与 legacy.json 一致
建议      同意下一项只做 q=37/46 exact SVD vector-postback 实现审计（GO），先审 DLARFB 内的 BLAS 调用与数据依赖；
          GPU 优化原型 NOT YET GO。见第 4 节的几点输入
```

## 1. 完整性、门与身份

- `closure.json`（`fdc1cad1…`）、`manifest.json`（`b0edad9a…`）、`canonical/receipt.rds`（`277b2817…`）与报告一致。
- canonical `report.json`：70/70 gate 为真；`authority_pmax.maximum_absolute_error` 与 `authority_near_alpha` 均为
  3.05425129631942e-12；插桩运行 outer 313.765 s，只作诊断。
- 身份：`joined-setups.tsv` 中两层请求序号连续、无缺漏；geometry 步长与 external ID 规则对全部请求成立；每层 32 个
  thread、32 个 prepared、32 个 stream。生产代码（`full_cuda_ci_one_call.cpp` 的槽位层）先 `geometry.reserve`，再按
  `work.requests` 顺序 `push_back`，worker 领取 `i = next.fetch_add(1)` 后立即 `pool.bind(slot, i)`，后者传
  `geometry_.at(i)`，所以 geometry 地址确定请求序号。
- observer（`observer.cpp`）只包装 batch 入口、L4/L5 rebind 与 `cudaMemcpyAsync`：后者只记录 D2H 的 host 指针、
  大小与 stream，并要求每次 batch 恰好一次、大小为 3,848 B × target 数；状态在原函数返回后从 host buffer 复制，没有
  device 分配、拷贝或等待。逐 target 字段、各阶段之和与 native diagnostics 的核对都在其中。它在 worker 上的 host 检查
  合计 L4 0.154 s、L5 0.080 s（单次至多 0.22 ms），计入插桩运行。
- 跨域边界：optimizer kernel 结束晚于 host 返回的 L4 记录 11 条、最大 70,837 ns，L5 为 0，与报告一致。
- 层跨度：被关联 kernel 首尾 27.126446 / 31.899091 s；原生 host 边界（`work_ms`）27.128807 / 31.901292 s。
  被关联 kernel 合计 12,922 + 5,730 = 18,652 个。

## 2. 复现

由 `joined-setups.tsv` 与 `joined-targets.tsv` 重算，全部与报告一致：

- 最慢与最后结束：L4 请求 1308（11 个 target，候选 5，4.497783 s，16.159363–20.671533 s，迭代 140、评估 703、halving
  554）与 1898；L5 请求 251（4 个 target，候选 0，10.539326 s，6.175177–16.777423 s，187/710/509）与 861。四个候选都是
  所在 setup 中四主阶段 cycles 最大的 target；对 setup 内 target 中位数的倍数 23.92/20.98/21.97/20.17；每次评估
  17.52M/18.40M/40.76M/37.85M cycles，全层 target 中位 21.45M、46.74M。
- 分布与相关：迭代 p50/p99/max 12/32/149、13/36/187；评估 21/109/703、25/119.85/710；optimizer kernel 时长与 setup 内
  最大迭代、最大评估、最大阶段 cycles 的 Pearson r 为 0.963337/0.998695/0.999989 与 0.970471/0.998979/0.999991。
- 阶段：四个主阶段 1.481/96.389/0.503/1.627% 与 1.046/97.383/0.342/1.229%；只取候选 target 时 SVD 为 96.610%、97.511%。
  SVD 内部 stage0–3 为 17.139/23.133/58.928/0.829% 与 12.718/14.904/71.698/0.767%，四项之和 100.030%、100.087%；
  stage4–7 之和与 stage0 之比为 0.9998、0.9999。direct 路径把同一区间同时计入 stage2 与 stage3（stage0 与 stage7 同理），
  超出 100% 的部分来自这里，报告不强行归一化是对的。
- 计数：表中全部数值一致。另外以下恒等式对每个 target 都成立，说明状态布局提取正确：
  score calls = 迭代 + step halving；score calls = Newton trials + steepest trials + 1；
  objective calls = score calls + boundary probes + 1；objective calls = complete 请求 + score-only 请求，且 score-only =
  boundary probes。
- 旧 trace：我用自己的 `predict.py` 分段，从 `ktrace-run-cache-on` 独立复算：L5 最长 optimizer 10.412481 s、开始序位 251、
  4 个 target、6.167560–16.608051 s，最后结束的是序位 861（8 个 target，23.64–31.85 s）；L4 最长为序位 1308（11 个 target），
  最后结束为序位 1898（8 个 target）。与 `legacy.json` 一致，序位与 target 数也和本轮按 key 确认的四个请求相符；旧 trace 没有
  key，所以只说相符。

未复算：报告所说 kernel 活动并集与 optimizer 活动并集的两种算法对照（报告正文没有给出这四个数值）。

## 3. 补充观察

1. **已记录阶段覆盖 kernel 时长。** 对每个 setup，以最大 target 的四主阶段 cycles 拟合 optimizer kernel 时长：斜率对应
   2.737/2.746 GHz，截距 1.7/3.2 ms；逐 setup 的 cycles / 时长 p5–p95 为 2.710–2.735 与 2.724–2.743 GHz。L4/L5 期间
   GPU1 的 SM 时钟采样中位 2,760 MHz（最低 2,520 MHz）。由此估计，候选 target 的已记录阶段约占 kernel 时长的 99%，与候选
   target 就是最后完成的 block 相符，但没有逐 block 时间戳，不能认定。采样间隔约 0.2 s，这只是一致性检查；cycles 仍含 stall
   与 `__syncthreads` 等待。
2. **慢 target 的试探构成（描述）。** 按评估数取前 1% 的 target（L4 179 个、L5 69 个），steepest descent trial 占其 trial 的
   中位比例为 96.4%、96.6%，每次迭代的 step halving 中位为 2.84、2.66；全部 target 的对应中位数为 28.6%、30.4% 与 0。四个
   候选的 trial 中 steepest 为 683/694、573/575、684/696、581/583。这些 target 合计占评估数约 8%、阶段 cycles 约 7%，但决定了
   慢 setup 的时长。line-search 规则按 mgcv 逐位合同冻结，所以这只是描述；可调的是每次评估的成本，与下一项方向一致。
3. 运行期间 GPU1 功耗采样中位约 114–115 W（利用率读数 100%）。这只说明功耗较低：某个执行资源或依赖链仍可能饱和，不能据此
   推出可兑现的并行余量，也不能判断瓶颈。

## 4. 对下一项的意见

同意只做 q=37/46 exact SVD vector-postback 实现审计：覆盖常规与 direct 路径、左向量回变换与延迟右向量完成；先核对 blocked
reflector 中独立输出的工作划分、workspace 生命周期与真实依赖，确认能否保持每个 dot/update 的运算顺序；不预估节省秒数。
在源码中核对到的几点，供审计参考：

1. **stage2 的反射变换核心由一个线程执行。** 常规路径 `small_dgesdd_left` 的 stage2 在 `if (warp == 0)` 中调用
   `apply_reference_blocked_q_left` 与 `apply_reference_blocked_p_right_transpose`，两者只在 `lane == 0` 上运行 reference
   LAPACK 3.12 的 `dormbr_`，warp 1 在 `__syncthreads()` 等待。direct 路径（`small_dgesdd_left_direct_two_warp`）与延迟右向量
   完成（`complete_stable_right_vectors_after_score`）调用同样的函数。multi-penalty 调用传 `match_lapack_shape_path = true`，
   q = 37/46 大于 `kReferenceReflectorBlockSize`（32），所以 L4/L5 的评估都走这条 blocked 路径：stage2 的反射变换核心由
   64 个线程中的 1 个完成。stage2 计时还包含同步等待，direct 路径另含 64 线程执行的 `left_u` 初始化（第 1049 行），所以
   58.9%/71.7% 不是纯单线程耗时占比。
2. **块结构属于数值合同，且两层不同。** 源码注释写明 LWORK（3q² + 4q）是数值合同的一部分，因为 DORMQR/DORMLQ 由它决定
   reflector 块大小。按 vendored 的 DORMQR/DORMLQ（NBMAX 64、TSIZE 4,160、`ilaenv` 给 32、NW = q）计算：q = 37 时 LWORK 4,255
   小于所需的 5,344，块大小降为 (4,255 − 4,160) / 37 = 2，Q 侧 19 块、P 侧 18 块；q = 46 时 LWORK 6,532 足够，块大小 32，两侧
   各 2 块。所以 L4 实际以 2 个 reflector 为一块，L5 以 32 个为一块，审计须分别处理；不能增加 workspace 或统一块大小。dlarft
   的 T 矩阵、dlarfb 的步骤顺序与块划分都不能变；能改的只是把真正独立的输出分给多个线程，每个元素的运算序列与零值分支保持不变。
   DTRMM 原地覆盖输入，只有沿互不依赖的方向才能拆分（见修订记录）。optimizer 各编译单元都用 `--fmad=false` 编译，这有利于保持
   逐位一致，但仍须由逐位资格证明。
3. **左右两侧的调用次数需要先计。** 按代码，左向量回变换在每次 SVD 评估中执行；右向量只在需要时执行（`compute_right_vectors`
   为真时立即执行，或在延迟完成时执行）。建议审计先记录右向量完成的次数（立即与延迟分开），或把 stage2 左右分开计时，再决定
   先审哪一侧。
4. 范围与门按报告：不改原请求/cohort 顺序、32 槽、rank/tolerance/line-search 规则，不重开 stability replay、CPU fallback、L8 与
   recover tile；原型以 selected log-SP、全部 optimizer 输出与 trajectory 逐位一致为资格，以完整层跨度与 outer wall-time 判定。

## 5. 自我更正

我在 [recover tile 复核](PHASE10_STAGE_B_RECOVER_TILE_REVIEW_20261029.md) 第 3 节写 L5 约 5.1 s 的差距“与长 setup 开始较晚形成的
尾部相符（最长的一个 busy 10.4 s）”。括号部分错误：同一份旧 trace 中，最长的 setup 在 6.17–16.61 s 运行，最后结束的是另一项
（本轮确认为请求 861）。我当时没有查该 setup 在 trace 中的起止时间就把它和尾部联系起来；已在该复核中加注，指向本节。

## 证据

`attempt3/{closure,manifest,prior-attempts,shared-dependencies,analysis,legacy}.json`、`attempt3/canonical/{report.json,receipt.rds}`、
`attempt3/joined-{setups,targets}.tsv`、`attempt3/canonical-gpu.csv`、`tools/research/optimizer_l45_attribution_20261029/observer.cpp`；
源码 `src/full_cuda_ci_one_call.cpp`（槽位层的 geometry 与 worker）、`src/cuda/mgcv_multi_penalty_gcv.cu`（阶段计时与延迟完成）、
`src/cuda/lapack_312_small_dgesdd.cuh`（`small_dgesdd_left`、direct 路径、`apply_reference_blocked_*`、
`complete_stable_right_vectors_after_score`）、`src/cuda/third_party/lapack_3_12_dormbr_device.inc`（DORMQR/DORMLQ 的块大小）、
`tools/build_cuda_native.sh`；旧 trace
`optimizer_slot_line_review_20261025/ktrace-run-cache-on/`。计算为只读一次性脚本；没有修改生产源码、native 或默认配置，
没有 commit 或 push。
