# L2/L3 Stage B：活动时间线归因完成，硬件计数器未测

日期：2026-10-28。承接 [L3 runtime 复核](PHASE10_OPTIMIZER_L3_SLOT_RUNTIME_REVIEW_20261028.md)
第 5 节，只做原始 captured workload 的诊断，没有修改生产 kernel、native 或默认配置。

正式状态为 **`AUTHENTICATED_STAGE_B_ACTIVITY_DIAGNOSTIC_COMPLETE`**；硬件计数器状态为
**`UNAVAILABLE_PRIVILEGE_NOT_MEASURED`**。完成的是输入与输出身份验证、kernel/memcpy/memset/API
时间线和静态资源归因；实际 SM 占用率、eligible warps 与 warp stall 原因仍未测得。

最明确的后续研究对象是 **bidiagonal**：它的活动区间并集覆盖 L2/L3 各约 **92.4%** 的 Stage B。
两层没有记录到 device activity 的区间合计只有 **0.842006 s**；当前回放没有显示仅靠填补这些区间即可
兑现几十秒收益的空间。约 **54.649/60.483 s** 的 compact-status D2H API 边界几乎全程与 GPU 工作重叠，
不能当成独立的 host 开销或拷贝带宽收益池。这些结果也不证明 kernel 已达到硬件上限。

accepted opt-in development median 保持 **329.078 s**。本轮没有新的完整方法性能 GO、真实数据或
final-PDAG campaign，也没有 promotion。L8 STOP、选项默认关闭、Phase 10 ACTIVE、candidate NOT_FROZEN、
holdout SEALED、recommended=false 保持不变。

## 执行边界与身份

- 使用原 L3 runtime native `e8c39703…0ca045`；运行前后核对其 build receipt 的 **103 个生产输入**。
  本轮只编译单独的 host observer、原生 API replay 和 CUPTI activity tracer，没有重新编译生产 kernel。
- GPU1 独占，成功 campaign 期间 GPU0 空闲；fresh process，CPU affinity 0–19、CUDA connections=32、
  BLAS/OMP/MKL=1。保留每轮进程记录和 GPU 监控，所有自有测试、监控进程均已结束。
- observer 显式解析生产 native 的函数地址，转发原 begin/store/build 调用；保存全部 L2/L3 residual bytes、
  原始 cohorts、顺序 TargetKeys 及 rank/route/info，不重写数值实现。
- 原 Stage B 完全 join 后，分块下载缓存 payload，按原 key 顺序对每个 component 的三个已存字段
  （centered vectors、selected values、total）计算 SHA-256；不包含尚待计算的 self-moment。
  **这些额外诊断 D2H 与 hash 在 Stage B 计时之外**；不能宣称整个诊断没有 D2H。
- replay 直接调用未修改的 native cache API，保持 n=351、numCol=35、11 GiB cache、8 retained dense lanes、
  10 个宽度 64 的 structured contexts，以及原 residual cohorts、component 顺序和完整 wave joins。
  原 `begin_level` 清理缓存槽，回放没有跨层挪用旧 payload。
- replay 不保留 optimizer workspace，也没有穿插的 CI consumers，因此不代表完整方法容量或 outer 性能。
  显式释放额外输入 buffer/stream/event，native 受跟踪资源亦全部关闭。生产数值 authority 仍完全在 GPU。

## 资格与覆盖

canonical capture 的 **71/71** 硬门通过：**240,498 个 CI p-value 与全部 pMax 位值不变**，
changed CI、changed pMax、alpha flips 均为 0；相对保存的 authority，最大 CI/pMax 误差为
**`3.05425129631942e-12`**。task/deletion/skeleton/sepsets/count/RNG、原 cohorts/Gram order、
cache 元数据及资源门通过。生产 numerical-authority/fallback/large-D2H counters 仍为零；
observer 的显式诊断下载另行保留，不混入这个生产计数结论。

| 层 | components | 原 residual cohorts | 完整 waves | 尾部形状 |
|---|---:|---:|---:|---|
| L2 | 49,919 | 3,356 | 78 | 77 × 640 + 639 |
| L3 | 54,878 | 5,573 | 86 | 85 × 640 + 478 |

三次无 profiler 回放加一次 activity 回放，每次均与原 capture 的全部 **104,797** 个三字段 SHA 一致，
共 **419,188 次 component-hash 比较**。dense fallback=0，所有回放资源检查通过。
这些是对原输出的身份核对，没有新增 CPU 数值 producer。

| 运行 | L2 Stage B / s | L3 Stage B / s |
|---|---:|---:|
| 原 canonical capture 的 native 边界 | 54.484830 | 60.412542 |
| replay1 | 54.789477 | 60.245710 |
| replay2 | 54.906672 | 60.974943 |
| replay3 | 54.701438 | 60.300604 |
| 各层无 profiler 中位数 | **54.789477** | **60.300604** |
| activity replay 的 host 观测边界 | 55.341468 | 61.249255 |

各层中位数分别取值，不能把它们之和称为合计中位数。capture outer 为 **351.757 s**，包含额外诊断工作，
不进入 accepted 性能账本。activity 回放的对应阶段比无 profiler 中位数高约 **1.01%/1.57%**，同样不进入性能中位数。

## Activity 时间线

CUPTI 记录 concurrent kernel、runtime/driver API、memcpy、memset、synchronization，以及 stream/context、
correlation ID、字节数和 launch 资源。共 **1,000,301 条活动记录**，dropped=0、invalid timestamps=0。
**13,104 个 kernel** 全部按精确 correlation ID 匹配到 runtime API；memcpy 关联也全部解析成功。
本次 CUPTI 把这些活动关联到 runtime 域，不能只搜索 driver 域。

下表均使用区间并集；并发活动的 duration 之和不是 wall time。CUPTI marker 与 host 观测边界有微小差别，
此处只使用同一 CUPTI 时钟。

| 指标 / s | L2 | L3 |
|---|---:|---:|
| CUPTI Stage B 边界 | 55.341451 | 61.249252 |
| kernel 区间并集 | 54.802182 | 60.649809 |
| memcpy + memset 区间并集 | 0.167039 | 0.187613 |
| 全部已记录 device activity 并集 | 54.943562 | 60.805136 |
| 无 kernel 区间 | 0.539270 | 0.599443 |
| **无已记录 device activity 区间** | **0.397890** | **0.444116** |
| 其中单段超过 1 ms 的 gap 合计 | 0.006806 | 0.007702 |

“无已记录 activity”不等于已证明 GPU 空闲，也不等于可全部消除的时间。
这份回放没有显示几十秒的空洞机会，但不排除改变 kernel 调度、资源竞争或内部实现带来收益。

| kernel family | L2 活动并集 / s | L3 活动并集 / s | L2 仅该 family 活动 / s | L3 仅该 family 活动 / s |
|---|---:|---:|---:|---:|
| **bidiagonal** | **51.156629** | **56.588267** | 29.164904 | 32.417541 |
| recover | 23.885557 | 26.400796 | 2.632593 | 2.949349 |
| roots | 4.287537 | 4.644960 | 0.151492 | 0.131563 |

各 family 之间重叠，表中并集不能跨行相加；“仅该 family 活动”也不是可以直接减掉的独占收益。
完整八类 kernel 为 distance rows、reduce rows、prepare、bidiagonal、roots、recover、boundary check、center，
L2 每类 780 次，L3 每类 858 次。全部走 structured 路线，此处不能沿用早期的 cuSOLVER `gesvdj` 瓶颈判断。

原完整首/中/尾 wave 的 CUPTI 边界：

| 层 | 首 wave | 中 wave | 尾 wave |
|---|---:|---:|---:|
| L2（序号 0 / 39 / 77） | 0.873907 s | 0.690965 s | 0.710167 s |
| L3（序号 0 / 43 / 85） | 0.747590 s | 0.683214 s | 0.504446 s |

完整各 wave 及首/中/尾活动分类保存在 `attempt3/analysis.json`。

## Host API：status D2H 边界承载等待

生产 `StructuredComponentLane::read_status()` 在
[`full_cuda_ci_vertical.cu`](../src/cuda/full_cuda_ci_vertical.cu) 中将状态复制到普通
`std::vector<StructuredComponentStatus>`，随后 record/synchronize completion event。
本次 trace 的长边界主要出现在指向 pageable host buffer 的 `cudaMemcpyAsync` 内。

| compact-status D2H 指标 | L2 | L3 |
|---|---:|---:|
| 调用次数 | 780 | 858 |
| 字节总数 | 2,396,112 | 2,634,144 |
| host API 累计 / s | **54.649306** | **60.483250** |
| API 与已记录 device activity 重叠 / s | 54.646179 | 60.470386 |
| 实际 device copy duration 合计 / s | **0.001001** | **0.001104** |

因此不能把约 55/60 秒解释为传输这些字节所花的时间，也不能把它们与 kernel 时间重复累加。
观测与“API 内等待在前 GPU 工作完成”一致；单靠 API 名称无法进一步证明 driver 内部的具体等待机制。
把 buffer 改成 pinned 或把等待移到 event，并不自动缩短 dependency chain，仍须独立证明 outer 收益。
本轮未做这类改动。

相应 `cudaEventSynchronize` 累计仅 **0.003986/0.004389 s**，不能据此认为等待已经消失。
component cache 的 D2D payload store 为 **149,757/164,634** 次，搬运
**4,920,415,992/5,409,214,704 B**；host API 累计 **0.501072/0.558075 s**，
device copy duration 合计 **0.144248/0.158823 s**。这些 duration 也含并发重叠，不能直接当作可省的 wall time。

## 静态资源与未完成的硬件测量

RTX 4090 报告 128 SM、每 SM 最多 1,536 threads、102,400 B shared、65,536 registers，
warp=32，另有每 block 1,024 B reserved shared。CUPTI launch 资源记录为：

| kernel | threads/block | registers/thread | dynamic shared/block | local bytes/thread | 仅按 shared 推得 blocks/SM 上界 |
|---|---:|---:|---:|---:|---:|
| bidiagonal | 32 | 100 | 19,656 B | 0 | ≤4 |
| recover | 64 | 66 | 101,088 B | 0 | ≤1 |

常规 grid 为 64 blocks，L2/L3 尾部最后一次 launch 分别为 63/30 blocks。
上述上界包含 reserved shared，但不是 achieved occupancy，也没有体现并发 kernel 的实际驻留、调度或 stall。
它们足以把 shared scratch 和长依赖链列为后续核查对象，不能据此声称缩小 scratch 必然提速。

NCU 2024.2.1 已安装，但驱动 `RmProfilingAdminOnly=1`；当前 UID 1000 无有效 capabilities，
非交互 sudo 需要密码。没有更改驱动策略、没有采集硬件计数器，也没有用时间线冒充 occupancy/stall 测量。
因此原建议中的这部分测量保持 **未测**。

下一项建议只围绕 bidiagonal 的指令依赖与 shared scratch 生命周期做窄审计，recover 作为共存资源约束同时考虑。
若以后具备 profiler 权限，再对原 captured 输入补充 occupancy、eligible warps 和 stall 数据；
per-kernel NCU replay 会改变并发，不能把其耗时当成原 Stage B 性能，range 并发指标也需单独资格验证。
目前只支持确定研究优先级，不授予生产 kernel 改写或新的 speedup GO。

## 失败记录、修订与归档

没有删除任何完整成功计时，也没有因结果不好而重跑选择样本：

1. 初始 capture 在完成全部 L2 hash 后 exit 127：observer 直接引用 cache-info 符号，但 R 将 native 局部加载。
   修正为显式 native handle 的 `dlsym`。原日志、partial capture、工具与 binary 全部保留。
2. `attempt2` 的 activity smoke 因 injection library 局部符号不在 `RTLD_DEFAULT` 中而失败。
   改为从 `CUDA_INJECTION64_PATH` 显式取得已加载 handle。失败工具与日志保留；`attempt3` smoke 的
   135 条记录包含预期三次 copy 和一次 memset，dropped/invalid 均为 0。
3. 原 README/analyzer 把 L2 waves 写成 79；正确值为 `ceil(49919/640)=78`。
   修订在成功 canonical capture 尚在运行、任何 replay 计时尚未产生时记录；native observer 从始至终使用正确公式。
   冻结执行工具保存在 `attempt3/execution-source/`，完成 campaign 后只修正文档及 analyzer 的预期计数。
4. 首版离线分析仅搜索 driver API correlations，显式报告全部 kernel 未匹配。
   修正为 runtime/driver 两域的精确 ID 匹配并检查歧义；全部 kernel/memcpy 均成功关联。
   原 `analysis-v1.py/json` 保留，所有区间并集、gap、资源与 wave 结果核对不变；没有为此重跑 GPU。

修订详情见工具目录的 `CORRECTIONS.md`。build/protocol 绑定冻结执行版本，最终 manifest 同时绑定其快照和修订后工具。

工具：[`tools/research/stage_b_attribution_20261028/`](../tools/research/stage_b_attribution_20261028/)。
证据：[`artifacts/full_cuda_ci/stage_b_attribution_20261028/`](../artifacts/full_cuda_ci/stage_b_attribution_20261028/)。
成功 campaign 在 `attempt3/`，原失败证据保留在根目录及 `attempt2/`。

| 对象 | SHA-256 |
|---|---|
| unchanged production native | `e8c39703f01e09260d330167b49267c295aa2221167d5248d0980a97c80ca045` |
| closure.json | `717de47bed8b8e8014c122301d7bad8f84851fc2f959ce0b2316efa34b9291dc` |
| manifest.json | `82ca260d5b98b800b00aab533fb43fbab29ad709f0ef649178d449fb355416f3` |
| attempt3/analysis.json | `b4dbeebd2fd37e76ab64cbde7bbdd40f10f4ad3b287d919665dc18c28c4c6357` |
| attempt3/protocol.json | `484ee8c6f925805b0872ee77bc0a2a7d4ef3e5141da636a19885c758010291bd` |
| attempt3/capture/report.json | `10b3fa28a1dd0b5601bb5455b8b4fb03be733fc04b1291c86ee34e729b1fa5a9` |

归档后独立重算 manifest **275 个条目**（153 个 evidence 文件、103 个生产输入、11 个工具、8 个参考），
SHA 全部通过；证据文件集合与 manifest 完全一致（manifest 自身除外）。
本报告与路线图是归档后的说明文件，不属于上述封存的 evidence 文件集合。
当前 HEAD 仍为 `d27e59b5d7baaabd60b28a419fa0d0bd52c76b53`，证据绑定既有增量 source closure，
不是 clean-checkout producer。没有 commit 或 push。
