# q37/q46 vector postback：默认关闭的 runtime 集成资格

日期：2026-10-29。执行依据为
[隔离 GPU 原型复核](PHASE10_OPTIMIZER_POSTBACK_GPU_REVIEW_20261029.md)。
结论为 `GO_DEFAULT_OFF_RUNTIME_DEVELOPMENT`：GPU/cohort、编译资源与同一 native
的六次 canonical 配对全部通过，machine audit PASS。OFF/ON 中位数
**312.763 → 290.491 s**，节省 **22.272 s（7.1210%）**，三组均获益。
已安装通过资格的默认关闭 runtime，accepted opt-in development median 更新为 **290.491 s**。
这是 source-closure development evidence，不是 clean producer 或 promotion。

## 实现及隔离边界

新增 `FASTKPC_DCC_GAMMA_VECTOR_POSTBACK=0|1`，未设置或空值为 OFF，其他非空值拒绝。
capacity factory 仅在显式 ON 且 small64 的 q=37/46 时选择独立 `postback64` 后端；
其他 q 保留原后端。one-call 的 ON 入口仅允许 `dcc.gamma`。

新 CUDA 单元通过符号重命名编译现有 small64 实现，数值与 optimizer 控制逻辑共享同一源码。
只有 ON 单元包含新的 vector-postback helper。保留原始 small64 对象，未在原 kernel
调用图中添加运行时开关。prepared 对象创建后后端身份固定，rebind、残差 lease、fixed evaluator、
coefficient finalization 均按其固定类型分派。混合后端的 multi-setup 请求在 GPU 提交前拒绝；
不通过重新分组改变请求或执行顺序。

Q 以输出列、P 以输出行分配 warp0 lanes，各 lane 独占 WORK 行。保留 serial DLARFT、
reference DLARFB 函数体、LWORK/NB、倒序 block、WORK[0]、立即/延迟 P 条件与全部同步边界。
常数直接使用 reference 的 block size/leading dimension，并用 static assertion 绑定 32/65/4160。
未改变数学容差、秩判断、line search、cohort、槽数或 CPU/GPU authority。

新增 summary 的 `vector_postback_requested` 与 `vector_postback_execution`：后者分别记录
original/vector 的 prepared 创建、return 前销毁、optimizer batch/target、q37/q46/other targets、
fixed 和 coefficient finalization targets。这些是实际选择后端的执行计数，
不是 DLARFB 调用次数；wrapper 销毁计数也不代替原有 CUDA 资源闭环门。

## 编译身份和 GPU 资格

runtime native SHA-256：
`a1e16a9c0c8089ea90491426c56475394449efb03317be4a87f1925c7b73bd34`。
绑定 107 个生产输入；重编两个 host 对象及一个独立 ON CUDA 对象，复用 41 个已认证原对象，
共链接 44 个对象。此处是 incremental source closure，不是全量 clean producer rebuild。
标准 native build 脚本同时编译、链接 ON 独立对象，并保留 `--fmad=false`。

| 检查 | OFF | ON |
| --- | ---: | ---: |
| optimizer stack | 7,120 B | 7,232 B |
| fixed evaluator stack | 288 B | 320 B |
| 上述 kernel registers | 255 | 255 |
| 完整 L4 captured cohort | 2,154 setups / 17,767 targets，通过 | 同覆盖，通过 |
| 完整 L5 captured cohort | 955 setups / 6,816 targets，通过 | 同覆盖，通过 |

`cuobjdump` 证明全部原 device 函数的资源记录仍在新 native 中。
GPU 单元覆盖 20,628 用例，比较完整 C/WORK、只读区、guard 和 info，全部逐位通过；
memcheck/synccheck 分别覆盖全部 1,719 形状，零错误，memcheck 零泄漏。

新增 runtime 合同测试覆盖：默认 OFF、非法 flag 在 GPU 分配前拒绝、prepared 创建后切换 flag、
混合后端拒绝、live lease 下 rebind 拒绝、released token 拒绝、D2D 系数 detach 和 finalization、
独立 exact fixed evaluator、有效 multi-setup，以及 ON 请求下非目标 q28 仍走原后端。
普通运行、memcheck/synccheck 全部通过，GPU 资源与 wrapper 计数归零。
全部新增数值计算在 GPU 执行；host 读取既有 authority/captured receipts 作比较。

缩减 L4 15 setups/136 targets、L5 5/20 的 OFF/ON 比较与 ON sanitizer 均通过。
完整两层保留原 32 槽、请求顺序、target keys，比较 captured optimizer 状态/计数轨迹及
coefficients/fitted/residual 全部位，实际后端计数符合 q 范围，资源闭环通过。
资格运行的层秒数不是独立正式性能 campaign，收益以随后 canonical 为准。

第一次 harness 构建因 host `cudaMalloc` 缺少 `void**` 转型而失败，未执行 GPU。
原始源码/日志保留在 `qualification/`；修正后的完整资格在 `qualification2/`，没有覆写失败证据。

## Canonical 协议

同一 native 按 OFF1、ON1、ON2、OFF2、OFF3、ON3 执行。沿用 351×48、default-Inf、fresh R
和 fresh result/target caches，以及既有 retained stack、各层 slots、structured work-pair 配置。
独占 GPU1，CPU affinity 0–19，math threads=1，CUDA connections=32；同时监测两张 GPU，
发现外部计算则终止本轮自己的进程。压缩与大 receipt 持久化不计入 outer compute 边界。

保留全部既有 240,498 CI/pMax bits、authority、task/deletion/skeleton/sepset/count/RNG、
物理 cohort/Gram 顺序、资源和容量门。另要求实际 ON optimizer targets 为 q37=17,767、
q46=6,816、other=0，OFF 的全部 ON 计数为零；OFF 原始显存检查点须与旧实现一致。
收益门预先固定为 ON/OFF 中位比 ≤0.98、三组配对全部获益，保留 L8 波动，不做事后排除。

| 配对 | OFF | ON | 节省 | OFF/ON gates |
| --- | ---: | ---: | ---: | --- |
| 1 | 312.763 s | 285.085 s | 27.678 s | 76/76、74/74 |
| 2 | 312.515 s | 293.807 s | 18.708 s | 79/79、77/77 |
| 3 | 313.265 s | 290.491 s | 22.774 s | 79/79、77/77 |
| 三次中位数 | **312.763 s** | **290.491 s** | **22.272 s** | **全部通过** |

ON/OFF 中位比为 `0.9287895307309368`，满足 ≤0.98；3/3 配对获益。
全部六次 240,498 CI/pMax 与已接受参考逐位一致，最大 authority error
`3.05425129631942e-12`；authority p 值在 [0.05,0.15] 的 1,337 项没有新位差异。
重复 CI/pMax、task/deletion/skeleton/sepset/count/RNG、physical/cohort/Gram 顺序、
CPU authority/fallback、cache metadata、资源和原容量门全部通过。

每次 ON 创建并销毁 64 个 vector backend prepared，执行 3,109 optimizer batches、
24,583 targets（q37=17,767，q46=6,816，other=0）；同数 coefficient finalization targets。
其他 56,168 个 optimizer targets 保持 original。ON 的直接 fixed-evaluator targets 为零，
所以该入口的资格来自前述独立 runtime contract 测试，不能声称 canonical 覆盖了它。
三次 OFF 的全部 vector 计数均为零。

## 收益与容量解释

L4/L5 同次 optimizer 之和的中位数 **59.057851 → 31.129370 s**，节省 27.928481 s。
三个配对该项分别节省 27.869429、27.972528、27.977305 s，较稳定。
完整 optimizer boundary 中位数 **128.981366 → 106.641476 s**；
Stage B **127.816159 → 127.861051 s**，native setup **22.204608 → 22.205120 s**。
这些是各自 host boundary 的中位数，不能把重叠阶段直接相加当成总收益。

第二组 ON 的未改动 L7/L8 分别比对应 OFF 多 6.170683/3.020538 s；第三组 L8 多 5.299343 s。
按每对 outer 差值扣除 L4/L5、L7、L8 差值后，其余差额为 −0.181583、−0.073306、+0.153319 s。
全部波动保留，不将它们归因于 postback，也没有重跑择优。不能用隔离原型的 286.882 s
替代本次集成中位数 290.491 s。

三次 OFF 的原始容量检查点与原实现完全一致。ON 的 L3 建池、optimizer 和最低空闲保持原值；
从 L4 起，观测空闲比 OFF 少 **27,262,976 B（26 MiB）**。最紧 L6 检查点为
**2,230,452,224 B（2.077271 GiB）**，比 2 GiB 门高 **79.125 MiB**；
OFF 对应 2,257,715,200 B，余量 105.125 MiB。L5 创建后最低空闲为 2,335,309,824 B。
独立后端集成的 26 MiB 与先前替换 small64 的隔离原型 20 MiB 不同；
没有独立分离模块加载与 stack 预留成本，不能断言全部来自 stack。

本轮没有新增真实数据或 final-PDAG campaign。容量资格仅覆盖既有离散检查点，
不构成连续显存上界或更大输入资格。没有默认启用、candidate freeze、holdout 开封、commit 或 push。

## 证据位置

`fastkpc/artifacts/full_cuda_ci/optimizer_postback_runtime_20261029/`：

- `baseline/`：生产修改前封存的 104 个输入、43 个对象和原 native；历史 manifest 继续绑定该身份。
- `attempt1/build.json`、`device-build.json`：当前 source/native/object 身份。
- `attempt1/qualification2/`：预先冻结的测试源码/依赖、编译资源、12 项运行和 closure。
- `attempt1/canonical/`：预声明协议、同一 native 的 OFF/ON receipts 与完整 gates。
- `attempt1/analysis.json`：重算中位数、配对差额、阶段与容量；`installation.json` 绑定已安装的 native/44 个对象。
- `attempt1/closure.json`、`manifest.json`：最终 machine audit 与完整证据封存。

Closure SHA-256：`835244b1e40a5177fff43ca34af7e7b9fba956de0c1f8ec75cbcb19c2a01599e`。
Manifest SHA-256：`1c77c87d640be73b6fd849360d0624142a52829fe0c8b3014d98b67cee1f649d`。

复现命令见 `fastkpc/tools/research/optimizer_postback_runtime_20261029/README.md`。
默认关闭、L8 STOP、Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false。

## 仓库交接

本报告的运行后由 `c469ac563dcaf7b3e7c704b302cfbdd8a6282e10` 提交 runtime 与研究工具。
[交接索引](PHASE10_DEVELOPMENT_HANDOFF_20261029.md) 和
[Git source attestation](PHASE10_SOURCE_ATTESTATION_20261029.json) 将 107 个生产输入、
54 个研究文件、已安装 native 及原始证据绑定到提交身份。
[cubin 比较 receipt](PHASE10_POSTBACK_CUBIN_COMPARISON_20261029.json) 证明当前 native
保留全部 17 个原 cubin，并新增一个 postback cubin；它不证明完整源码重编等价。
原先“未提交/未 push”描述运行时状态。本次交接没有重编或重跑 GPU，原 receipt 的
incremental source-closure development 等级及所有默认/Phase 10 状态保持不变。
