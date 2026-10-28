# L3 槽位 runtime：复核

日期：2026-10-28。复核 [L3 槽位 runtime 资格](PHASE10_OPTIMIZER_L3_SLOT_RUNTIME_20261028.md)（未提交源码，
native `e8c39703…a045`，证据 `fastkpc/artifacts/full_cuda_ci/optimizer_l3_slot_runtime_20261028/`）。只读复核，
没有新的 GPU 运行。`GO_L3_BOUNDED_SLOTS_RUNTIME_DEVELOPMENT` 成立；accepted opt-in development median 改为
329.078 s；选项默认关闭；Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。
同日按 Codex 核对意见修订显存偏移的描述，见“修订记录”。

## 修订记录（Codex 核对后）

Codex 重算了 5 份 receipt 的 SHA、manifest 的 375 个条目、六次报告的性能数字与硬门计数，均一致；GO 与 329.078 s
作为受限 opt-in 基线成立，收益归因（L3 层三对 38.809/37.511/38.539 s，L7/L8 差额留在 outer）也认可。一处事实错误，
我重算后接受：

1. **显存偏移写错了方向，也把 ON 写成了统一偏移。** 首版写“OFF 与 ON 每个显存读数都比 L6 campaign 少
   7,012,352 B”。以 L6 campaign 的 ON2 为参照，实际是：已用显存（structured 采样）增加、空闲显存减少；OFF 在各层
   都是 6.6875 MiB，ON 则随阶段不同。见第 3 节的表。原因仍未查明。容量 PASS 不受影响：ON 的 L6 检查点仍比 2 GiB
   门高 105.125 MiB，但余量窄。
2. **L3 层内其余 3.5–3.7 s 只是实测差额。** 目前不能具体归于建池或调度。

Codex 建议下一项只做 L2/L3 Stage B 的 captured-workload 占用率与 stall 归因，暂不改 kernel。我同意，见第 5 节。

## 结论

```text
完整性    报告列出的 5 个 SHA-256 吻合；manifest 375 个条目（243 文件、103 生产输入、11 工具、6 依赖、
          12 参考）按当前文件重算一致；当前 native 与 103 个源文件与 build.json 一致。相对 L6 源码闭包改动
          4 个文件、新增 2 个 L3 头文件。六次运行都 exit 0、isolated、使用同一 native
代码      未发现缺陷。native 侧 rebind/drain/validate 与 L6 同构；容量头文件逐项对应分点合同；
          observe_capacity 按 建池 / optimizer 完成 / Stage A 三个阶段判定；配置门在 canonical S2 全套
          设置之外 fail closed；生命周期等式扩展到 L3
门        outer 中位 370.553/329.078 s，比值 0.8880726914；L3 optimizer 中位 50.160702/15.249473 s，比值
          0.3040123594；三对全胜（+42.388/+37.577/+48.465 s）。六次硬门 65/68/71/68/68/71 全过，
          changed CI 均为 0
归因      L3 层本身三对收益 38.809/37.511/38.539 s，其中 optimizer 35.114/34.055/34.911 s、层内其余
          3.696/3.455/3.628 s（两种模式都没有诊断 shadow，三对一致；只是实测差额，不能具体归于建池或调度）。outer 中位收益 41.475 s 中，
          L3 层 38.507 s，L8 +3.321 s，L7 −1.268 s，其余各层与层外 +0.915 s。累计账目里 L3 槽位按约
          38.5 s 计更准确
容量      三次 ON 的 L3 分点记录逐字节相同，最小 free 3,404,857,344 B（3.171 GiB）。相对 L6 campaign ON2：
          OFF 的 structured 已用显存各层 +6.6875 MiB、L4–L6 空闲 −6.6875 MiB；ON 的已用显存 L2 +6.6875、
          L3–L6 +2.6875、L7–L8 +4.6875 MiB，L4–L6 空闲 −2.6875 MiB；原因未查明。L6 最紧检查点比 2 GiB 门
          高 101.125 MiB（OFF）/ 105.125 MiB（ON），L6 campaign 为 107.8125 MiB
后续      新 profile 中 Stage B（144.3 s）超过 optimizer（128.9 s），L2/L3 合计约 114.8 s。同意 Codex 的下一项：
          只做 L2/L3 Stage B 的 captured-workload 占用率与 stall 归因，暂不改 kernel（第 5 节）
```

## 1. 代码复核

相对 L6 源码闭包（`optimizer_l6_slot_runtime_20261026/build-source/`，哈希与 L6 `build.json` 一致）逐文件 diff：

- **native 侧**（`mgcv_multi_penalty_gcv_capacity.{hpp,cpp}`、`mgcv_multi_penalty_gcv_small64.cu`、新增
  `mgcv_multi_penalty_gcv_l3.hpp`）：`validate_multi_penalty_gcv_l3_geometry` 要求 351×28、三个 penalty、pivot 为
  置换、offsets 连续累加、root rank 恰为 24、初始 log SP 有限。`rebind_..._l3` 在 lease 未释放时拒绝，形状或 root
  容量变化时拒绝，重传 10 个几何数组并递增 generation；`drain_..._l3` 同步槽位 stream。与 L6 版本逐项同构。
- **容量合同**（新增 `mgcv_l3_optimizer_slot_capacity.hpp`）：建池 ≤1,536 MiB、optimizer 新增 ≤1,280 MiB、
  总增量 ≤2,816 MiB、free ≥2,048 MiB；Stage A 只查 free。与 v2 共存合同一致。
- **调度层**（`full_cuda_ci_one_call.cpp`）：`L3OptimizerSlotPolicy` 只接受 2–33 个 target 的 cohort。
  `observe_capacity` 对 L3 按调用顺序区分阶段：第一次（建池后）为 creation，`after_work()` 为 optimizer_done，
  每次 Stage A bind 与关池前为 stage_a，各自按合同判定，越限即抛出。`pool_create_ms` 只计建池本身。几何打包沿用
  通用模板，要求 context 上没有保留的旧 handle。配置门要求 dcc.gamma、351×48、L4/L5/L6 槽位、属性缓存、L2
  component prefill、exact-primary 复用、status wait、关闭 setup-optimizer pipeline 与 decomposition trace，
  否则 fail closed。生命周期等式（建 = 销，bind = detach = Stage A bind，未启用时为 0）扩展到 L3。

## 2. campaign 复现

- 协议在 10:41:42 写入，OFF1 于 10:41:43 开始。与 L6 campaign 相同，协议由 campaign 脚本在启动时写入；阈值
  （0.98、0.90、三对全胜）与 L4/L6 相同，并固定在被哈希的工具中。六次按 OFF1/ON1/ON2/OFF2/OFF3/ON3 进行，
  均 exit 0、isolated，没有其他 GPU 进程。
- 由六个 `report.json` 重算，与报告一致：

| 运行 | outer | L0–L2 | L3 层 | L3 optimizer | L3 其余 | L4 | L5 | L6 | L7 | L8 | 层外 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF1 | 370.553 | 98.183 | 123.322 | 50.362 | 72.960 | 53.249 | 43.038 | 20.339 | 9.944 | 20.587 | 1.891 |
| ON1 | 328.165 | 98.579 | 84.513 | 15.249 | 69.264 | 52.915 | 43.044 | 20.395 | 9.928 | 16.975 | 1.817 |
| ON2 | 331.547 | 97.999 | 84.691 | 15.265 | 69.427 | 53.189 | 42.971 | 20.329 | 9.924 | 20.592 | 1.853 |
| OFF2 | 369.124 | 98.026 | 122.202 | 49.320 | 72.882 | 53.177 | 43.025 | 20.308 | 9.946 | 20.623 | 1.818 |
| OFF3 | 377.543 | 98.137 | 123.354 | 50.161 | 73.194 | 53.335 | 42.993 | 20.300 | 17.616 | 19.966 | 1.843 |
| ON3 | 329.078 | 97.703 | 84.815 | 15.249 | 69.566 | 53.059 | 42.823 | 20.377 | 11.212 | 17.266 | 1.824 |

逐对账目（秒，正数表示 ON 更快）：

| 对 | outer | L0–L2 | L3 层 | 其中 optimizer | 其中其余 | L4–L6 | L7 | L8 | 层外 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF1/ON1 | +42.388 | −0.395 | +38.809 | +35.114 | +3.696 | +0.272 | +0.016 | +3.612 | +0.075 |
| OFF2/ON2 | +37.577 | +0.027 | +37.511 | +34.055 | +3.455 | +0.021 | +0.022 | +0.032 | −0.035 |
| OFF3/ON3 | +48.465 | +0.434 | +38.539 | +34.911 | +3.628 | +0.369 | +6.404 | +2.700 | +0.019 |

- L3 层收益三对相差不到 1.3 s；层内 optimizer 以外的 3.5–3.7 s 三对一致。本轮两种模式都没有诊断 shadow，这部分
  不再受上一轮的计时不对称影响，但它只是实测差额，目前不能具体归于建池或调度。
- outer 的其余差额主要来自 L8（第一、三对）和 L7（第三对：OFF3 的 L7 为 17.616 s）。代码未变，按报告计入 outer，
  不归因。outer 中位数取自 OFF1 与 ON3，二者之差 41.475 s 中 L3 层占 38.507 s。

## 3. 容量与下游

- 三次 ON 的 L3 记录逐字节相同：基线 5,829,165,056 B，建池后 4,610,719,744 B，optimizer 后 3,404,857,344 B；
  建池 1,218,445,312 B、新增 1,205,862,400 B、总增量 2,424,307,712 B、Stage A 最大增量 2,424,307,712 B；
  4,068 个检查点（Stage A 组 4,066 个），32 建 / 32 销，4,065 次 optimizer bind、detach 与 Stage A bind。
  三项增量与 v2 共存一致。
- **相对 L6 campaign 的显存偏移**：以 L6 isolated completion 的 ON2 为参照（单位 MiB；已用为 structured 采样的
  `sampled_device_used_bytes`，正数表示用得更多；空闲为池检查点，负数表示更少）：

  | 指标 | 本轮 OFF（三次相同） | 本轮 ON（三次相同） |
  |---|---:|---:|
  | L2 structured 已用 | +6.6875 | +6.6875 |
  | L3–L6 structured 已用 | +6.6875 | +2.6875 |
  | L7–L8 structured 已用 | +6.6875 | +4.6875 |
  | L4 最小空闲、L5 建池后空闲、L6 最小空闲 | −6.6875 | −2.6875 |

  v2 共存的 W（旧 native，10-27）与这个参照逐字节相同，没有偏移。OFF 的 6.6875 MiB 从 L2 起各层一致；它随新 native
  或本轮运行环境出现，两者无法从现有数据区分，36 个 GPU 函数的指令一致，原因未查明。ON 在 L3 之后的偏移较小，
  同样只作描述。
- L6 最紧检查点比 2 GiB 门高：L6 campaign 107.8125 MiB；本轮 OFF 101.125 MiB、ON 105.125 MiB（ON 比 OFF 多
  4 MiB，与 v2 相同，只作描述）。L6 仍是全程最紧的位置，后续改动若再增加常驻显存，需在 L6 重测。

## 4. 新 profile（三次 ON 的中位数，各项分别取中位，不可相加）

| 项 | 秒 |
|---|---:|
| 完整方法 | 329.078 |
| optimizer 合计 | 128.915 |
| Stage B 合计（其中 L2 54.668） | 144.344 |
| Stage A | 14.294 |
| native setup | 22.445 |

| 层 | 层耗时 | optimizer | 其余 |
|---|---:|---:|---:|
| L1 | 7.325 | 0.134 | 7.187 |
| L2 | 90.491 | 5.373 | 85.118 |
| L3 | 84.691 | 15.249 | 69.427 |
| L4 | 53.059 | 27.249 | 25.781 |
| L5 | 42.971 | 31.821 | 11.197 |
| L6 | 20.377 | 18.000 | 2.377 |
| L7 | 9.928 | 9.355 | 0.572 |
| L8 | 17.266 | 16.903 | 0.363 |

- optimizer 以外的部分已占大头：L2 与 L3 的“其余”合计约 154.5 s，Stage B 合计 144.3 s。
- L3–L6 的 optimizer 都已改为槽位；L7、L8 仍是窗口模式。此前的 trace 显示，L7 的波动与同通道串行链相符，L8 的
  波动来自首次加载排空。本轮 L7 有一次 17.616 s，L8 在 16.9–20.6 s 之间。
- 这些只是描述，不构成新的实验预期；报告也写明本结果不授权新的 L8 实验。

## 5. 下一项：L2/L3 Stage B 归因（Codex 的建议，我同意）

- **为何是这两层**：六次运行的 Stage B（host 边界）中位约为 L2 54.7 s、L3 60.1 s、L4 20.0 s、L5 7.8 s、L6 1.2 s，L7/L8
  各约 0.3 s；L2 与 L3 合计约 114.8 s。六次运行各层的 structured `dense_fallback` 均为 0，实际执行的是 structured
  component kernel，早期“主要卡在 cuSOLVER gesvdj”的判断不再适用。
- **范围**：只做 captured-workload 的占用率与 stall 归因，不改 kernel；保持原有并发形状（eig lanes、structured
  contexts 与 wave）；覆盖 bidiagonal、recover、数据搬运和等待，区分实际驻留、stall 与提交空洞。144 s 是 host 边界，
  不能直接当作可消除的 device 时间。L8 STOP 保留。
- **工具上的两点**：我的 ktrace 只记录 kernel 与 runtime API，没有 memcpy/memset，因此只能给出“无 kernel 区间”，
  不能当作空闲；要覆盖数据搬运，需要加上 CUPTI 的 MEMCPY/MEMSET 活动。nsys 2024.2.3 曾在这一工作负载上崩溃（L3
  窗口 1，bundled CUPTI），per-kernel 的占用率与 stall 原因更适合用 ncu 在 captured workload 上取得。

## 证据

六个 `report.json` 与 `*-process.json`、`protocol.json`、`build.json`、`checks.json`（33 个请求、每次 2,272 次比较、
memcheck 0 错误 0 泄漏；首次尝试保留在 `checks/`）、`closure.json`、`manifest.json`；L6 源码闭包
`optimizer_l6_slot_runtime_20261026/build-source/`；显存对照：`optimizer_l6_slot_runtime_20261026_isolated_completion/on2/report.json`
与 `optimizer_l3_coexistence_v2_20261027/attempt1/W/report.json`。计算为只读一次性脚本；没有修改生产源码、native
或默认配置，没有 commit 或 push。
