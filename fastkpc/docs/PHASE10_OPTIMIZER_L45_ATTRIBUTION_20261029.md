# L4/L5 exact-primary optimizer：身份与阶段归因完成

日期：2026-10-29。结论：**`PASS_L45_IDENTITY_AND_EXISTING_PHASE_ATTRIBUTION`**。
本轮按 [recover tile 复核](PHASE10_STAGE_B_RECOVER_TILE_REVIEW_20261029.md) 做窄诊断，完成原请求身份、
慢 setup、关键候选 target 与已有内部阶段计数的对应。没有修改生产算术、原请求/cohort 顺序或32槽执行方式。

L4/L5 的 SVD 分别占已记录四个主阶段 cycles 的 **96.389% / 97.383%**；向量回变换又占 SVD cycles 的
**58.928% / 71.698%**。最慢 setup 的关键候选 target 有大量重复评估和 step halving，单次评估 cycles
并未异常升高。下一项建议仅审计 exact SVD 的向量回变换及延迟右向量完成实现；尚不进入优化原型或 runtime 集成。

accepted opt-in development median 仍为 **311.440 秒**。本轮插桩 canonical 耗时313.765秒，只用于诊断，
不产生新的性能中位数。默认关闭、L8 optimizer STOP、Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、
recommended=false 均不变。没有 commit 或 push。

## 运行身份与观察方法

- HEAD `c38b04b`，生产 runtime `1d8a11d`；104个生产输入在运行前后均未变。
  native SHA-256：`12acc3cd01cce2dcac5f2bfb5ec02f27056e92154cbab2534c7f03bdc22caa7d`。
- 只编译 host observer、CUPTI activity tracer 与薄 smoke runner，不编译或替换数值 CUDA kernel。
- GPU1独占，GPU0无计算任务；CPU affinity 0–19、单 math thread、CUDA connections=32。
  保持已接受的 L3/L4/L5/L6 slots、work-pair ON、properties cache、11 GiB component cache、bypass与八条 eig lanes。
- smoke覆盖每层首个与最多迭代的 setup，共4 setups /17 targets，包含149/187次迭代样本；随后运行完整 canonical。
  两个成功进程均 exit 0，监控未发现外部 GPU任务。
- observer包装原生 small64 batch入口与L4/L5 rebind入口。`cudaMemcpyAsync`包装只记录原有 status下载的host指针、
  大小和stream；原函数完成后读取该host buffer。observer不增加应用层 device allocation、copy或wait。
- `DeviceOptimizerState`、其evaluation及phase结构从当前已认证源码机械提取，以small64的64/7容量编译。
  每个状态为 **3,848 bytes**；本轮读取已存在的status下载合计94,595,384 bytes，没有增加residual/fitted payload D2H。
  每target字段与公开返回结果核对，再把每个phase/counter之和与native diagnostics核对，以验证布局。

完整canonical记录569,524个kernel activities、3,911,196个runtime API activities、65,306个external correlation
records；零dropped、零内部无效时间戳。L4/L5范围内 **18,652个kernel/API关联**全部通过。CUPTI本身有插桩开销，
即使数值kernel未变，仍不能将此次outer elapsed当成性能对照。

## 请求身份不再依赖GPU开始顺序

封存的L4/L5 fixtures提供原request index和每个setup完整、有序的target-key向量。当前运行逐项匹配所有key；
再利用原生 `OptimizerSlotPool::bind` 传入 `geometry_.at(request)` 引用的事实，记录geometry地址，并验证每层全部请求：

```text
geometry_address(request) = geometry_address(0) + request_index * 256
external_id = level * 1,000,000 + request_index
```

geometry数组在生产代码中按 `work.requests` 原顺序构造。CUPTI external ID与真实API/kernel correlation ID关联，
因此可把当前请求数组位置、prepared slot、stream和GPU工作对应，不再用观测开始顺序代替请求序号。
prepared指针/stream给出槽位身份；没有记录worker的原始数值slot编号。host时间点为bind与batch入口/返回，
不是 `next.fetch_add(1)` 的原子领取时刻。

| 观察项 | L4 | L5 |
| --- | ---: | ---: |
| q / penalty count | 37 / 4 | 46 / 5 |
| setups / targets | 2,154 / 17,767 | 955 / 6,816 |
| prepared slots / streams / workers | 32 / 32 / 32 | 32 / 32 / 32 |
| GPU开始序位与请求序号不同的setup | 35 | 45 |
| 最大序位位移 | 22 | 21 |
| 该层被关联kernel的首尾跨度，秒 | 27.126446 | 31.899091 |
| 原生optimizer host边界，秒 | 27.128807 | 31.901292 |

CUPTI threadId与Linux gettid是不同编号空间；验证的是每层32对32双射，不是整数相等。
跨层pthread编号可能复用，因此不要求两层合并后仍为全局双射。

旧trace没有canonical keys；`legacy.json`保留匿名stream/burst结果，不追认旧GPU开始序位为请求序号。
旧cache-on L5最长optimizer为10.412481秒，对应pipeline在该层6.167560至16.608051秒运行，而该层总跨度31.845836秒。
这已经说明旧的约10.4秒最长项不是最后结束的一项；新运行进一步确定了当前原请求身份。
其中一份旧JSON没有native SHA，该项明确记为unknown，不从另一份receipt推断。

## 最慢项与最后结束项是不同setup

以下request index和target index均从0开始。target取该setup中四主阶段cycles之和最大者，称为“关键候选target”；
没有target/block单独完成时间戳，不能声称已测得最后完成的block。
kernel秒为optimizer kernel时长；起止秒为该请求全部被关联kernel相对于本层首个kernel的范围。

| 层 / 类型 | request | targets | 候选target | optimizer秒 | pipeline起止秒 | 迭代 | 评估 | halving |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| L4 最慢 | 1308 | 11 | 5 | 4.497783 | 16.159363–20.671533 | 140 | 703 | 554 |
| L4 最后结束 | 1898 | 8 | 3 | 3.932136 | 23.177781–27.126446 | 149 | 584 | 426 |
| L5 最慢 | 251 | 4 | 0 | 10.539326 | 6.175177–16.777423 | 187 | 710 | 509 |
| L5 最后结束 | 861 | 8 | 6 | 8.167122 | 23.724245–31.899091 | 108 | 593 | 475 |

这四个候选target的cycles分别是各自setup内target中位数的23.92 /20.98 /21.97 /20.17倍。
其每次评估cycles为17.52M /18.40M /40.76M /37.85M；对应全层target中位数为L4 21.45M、L5 46.74M。
因此这些慢项体现的是长迭代/试探轨迹，不能归因为其单次分解特别慢。不同并发时刻、路径和stall仍会影响cycles，
不能把这个比值当作孤立矩阵性能。

| target分布 / setup相关性 | L4 | L5 |
| --- | ---: | ---: |
| target迭代数 p50 / p99 / max | 12 / 32 / 149 | 13 / 36 / 187 |
| target评估数 p50 / p99 / max | 21 / 109 / 703 | 25 / 119.85 / 710 |
| optimizer kernel时长 vs setup内最大迭代数，Pearson r | 0.963337 | 0.970471 |
| optimizer kernel时长 vs setup内最大评估数，Pearson r | 0.998695 | 0.998979 |
| optimizer kernel时长 vs setup内最大phase cycles，Pearson r | 0.999989 | 0.999991 |

p99采用线性插值，评估数分位点可以不是整数。相关性是当前工作负载的描述，不证明重排、削减迭代或改变line search可行。
关键候选target完整key、全部setup及target记录分别见 `analysis.json`、`joined-setups.tsv`、`joined-targets.tsv`；
setup身份由完整有序key向量确定，候选target的单个key不冒称独立setup key。

## 阶段归因与计数的准确含义

下面第一表的分母是所有target的四主阶段cycles之和。它们是现有 `clock64()` 计数，包含stall和调度影响；
不同target的时间重叠，cycles累加不能转换成独立可删的wall-time。

| 已记录主阶段cycles占比 | L4 | L5 |
| --- | ---: | ---: |
| penalty factor / augmentation | 1.481% | 1.046% |
| SVD | **96.389%** | **97.383%** |
| score construction | 0.503% | 0.342% |
| derivative / Hessian | 1.627% | 1.229% |

每setup仅取关键候选target后，SVD仍占96.610% /97.511%，主导方向一致。
此处SVD是 exact-primary 的 **GPU LAPACK-compatible `small_dgesdd_left` 路径**，不能沿用旧路线的
`cusolverDnDgesvdj` host-boundary解释，也不是structured full-eig Stage B的recorded DBDSQR。

| SVD内部已有计数 / 分母为SVD cycles | L4 | L5 |
| --- | ---: | ---: |
| stage0：QR / bidiagonal reduction | 17.139% | 12.718% |
| stage1：bidiagonal SVD | 23.133% | 14.904% |
| stage2：vector postback | **58.928%** | **71.698%** |
| stage3：left-vector product | 0.829% | 0.767% |

stage2包含初始向量回变换，也包含accepted deferred evaluation需要的右向量完成；本轮没有进一步拆开左右向量。
源码中常规路径涉及 `apply_reference_blocked_q_left` / `apply_reference_blocked_p_right_transpose`；
`complete_deferred_multi_penalty_target_device`把延迟右向量完成cycles也计入stage2及SVD主阶段。
stage4–7是stage0的细分；direct shape路径的stage2/3还有重叠，所以这八个分解计数不可相加，表内四行也不强行归一化。
bidiagonal求解含divide-and-conquer路径，不能把stage1统称为单个DBDSQR。

| 计数 | L4 | L5 |
| --- | ---: | ---: |
| iterations | 216,194 | 90,859 |
| score calls | 276,760 | 116,756 |
| objective calls = stable-SVD evaluations | 429,191 | 187,761 |
| complete-capability evaluation requests | 294,527 | 123,572 |
| score-only evaluation requests | 134,664 | 64,189 |
| step halving | 60,566 | 25,897 |
| Newton / steepest trials | 138,218 / 120,775 | 56,024 / 53,916 |
| boundary probes / accepted | 134,664 / 77,317 | 64,189 / 35,784 |
| selected reuse / forced refresh counter | 0 / 6,084 | 0 / 1,902 |

`complete_evaluation_count`记录完整能力请求，即使该请求带 `defer_complete_evaluation=true`；不能解释为每次都算了完整Hessian。
当前已有score-only/deferred机制，本轮没有发现可直接删除的一整批冗余Hessian工作。
forced refresh counter只计 `boundary_accepted_count==0` 且强制final evaluation的分支，不是所有selected-fit的数量。
所有本轮target均要求final selected evaluation；其cycles混在已有phase计数中，没有单独测时。
stability replay与terminal confirmation保持为零，不能再把它们列为当前长尾原因。

## 硬门与测量限制

完整canonical **70/70 gates PASS**：

- 全部 **240,498 CI** 与ON2 reference的p-value/pMax逐位一致；task及deletion顺序、adjacency、sepsets、
  n.edgetests与RNG一致。对封存authority的最大CI /pMax绝对误差均为 **3.05425129631942e-12**，小于1e-10。
- 80,751个multi-penalty targets仍全部exact-primary；stability replay、terminal confirmation为零。
  exact-fixed stable-SVD、原physical/Gram/cohort顺序、cache metadata与原L3/L4/L5/L6容量门全部通过。
- CPU数值authority、unknown fallback、residual/component large D2H均为零，CUDA资源计数最终归零。
- L4/L5全部 **3,109 setups /24,583 targets** 的11个浮点向量字段和16个整数向量字段与封存optimizer输出逐位一致。
  包括selected log-SP、coefficients、score、gradient/Hessian、rank/status及trajectory counters。
  本轮没有重新下载fitted/residual，也没有新增完整PDAG/orientation资格，不能把这些写成新测量。

另外保留以下限制：

1. **11条L4 optimizer记录的device end晚于host return时间戳，最大70,837 ns（70.837 us）。**
   全部差异保留于 `cross_domain_end_violations`，未裁剪。内部kernel时间合法，external IDs和key关联通过；
   不由这些跨域边界推断host wait、GPU空洞或同步收益，也不声称所有时间戳都满足跨域包含关系。
2. 两层kernel活动并集和optimizer活动并集，共四项均经独立endpoint sweep与区间合并算法核对。
   trace未记录全部memcpy device activities；不能把未见kernel的区间叫作已证实空闲。
3. 无target/block单独开始或结束时间；候选target不是已测定的最后完成block。
   未测实际occupancy、stall类别、DRAM/L2流量，不能把阶段偏高直接归因到带宽、stack或某种硬件资源。
4. 仅做一次插桩完整运行；不报告收益或重排机会预算。L5约5.1秒的旧模型差距仍只是诊断线索。

## 失败记录、离线修正与证据封存

- `attempt1`完成host build，launcher因smoke可执行文件与输出目录同名而未开始GPU运行；旧trace解析还遇到缺失native SHA字段。
  修正可执行名为`smoke-check`，旧身份缺失则显式标unknown。此attempt没有完整原工具source snapshot，不能声称完整历史源码闭包。
- `attempt2`的4-setup smoke通过；R canonical进程在数值campaign开始前SIGABRT（exit -6），没有外部GPU任务。
  检查发现R局部加载native/CUDA与standalone smoke的符号可见性不同。`cudaMemcpyAsync`包装随后增加从已打开native
  handle查找原函数的分支，替代仅依赖`RTLD_NEXT`；修改前的工具已保留于该attempt的`protocol-tools/`。
- `attempt3`完成全部GPU门。其原协议工具已保存，随后仅修正离线analyzer对thread ID和跨域时间戳的两项错误假设，
  不重跑或改写GPU结果。最终analyzer和原版本分别记录，失败记录不计入PASS。
- 补充的shared runner/fixture/R静态source依赖在封存时与Git `c38b04b` 的blob逐项相等；dataset/oracle哈希与已保存
  receipt/report绑定。此项明确为**运行后的归档闭包**，不追称全部依赖在运行前已预注册。原protocol保持原样。

工具见 [README](../tools/research/optimizer_l45_attribution_20261029/README.md)。成功证据目录：
`fastkpc/artifacts/full_cuda_ci/optimizer_l45_attribution_20261029/attempt3/`。
关键产物为 `protocol.json`、`build.json`、原始trace/TSV、`canonical/report.json`、`canonical/receipt.rds`、
`analysis.json`、两份joined TSV、`shared-dependencies.json`、`prior-attempts.json`、`closure.json`、`manifest.json`。

- receipt SHA-256：`277b2817ddd1aacfedd7d7fa80b3bf2671583e9cb4e9f6e1addf2ba79480cc79`。
- closure SHA-256：`fdc1cad11813b79479b7c484f317f2ef79100f6af5984aba6ab66ec70fa76f7f`。
- manifest SHA-256：`b0edad9a57233a82cb43382b331c986826cd26c103faa5471c8cd54e7872ef05`。
- manifest绑定56个产物、104个生产输入、10个工具、97个依赖，共267项；先前失败目录另由索引绑定。

封存后独立重算全部267项及两个失败目录的51项哈希，验证证据文件集合、两份原协议工具snapshot、native身份均通过；
另从原始target TSV独立核对全部24,583行的身份、计数和阶段汇总。全部自身GPU任务已结束。

## 下一项的边界

建议仅做 **q=37/46 exact SVD vector-postback实现审计**，覆盖常规与direct shape路径、左向量回变换及延迟右向量完成。
先核对blocked reflector中独立输出的工作划分、workspace生命周期与真实依赖，确认有无保持每个dot/update运算顺序的
实现空间。stage2尚未分解左右向量，因此不能直接指定哪一侧必然主导，也不能预先承诺节省多少秒。

之后若发现具体可行点，再另开captured-cohort GPU原型，验证selected log-SP、全部optimizer输出及trajectory逐位一致，
并以完整层跨度和outer wall-time作性能门。不得改变原请求/cohort顺序、32槽、rank/tolerance/line-search规则，
不得重开stability replay、CPU fallback、L8或已STOP的recover tile路线。
