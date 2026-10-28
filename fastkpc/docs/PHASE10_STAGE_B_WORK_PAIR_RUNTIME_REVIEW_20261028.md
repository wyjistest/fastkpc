# Stage B fixed-VT work pair 默认关闭集成：复核

日期：2026-10-28。复核 [Stage B work pair 默认关闭集成](PHASE10_STAGE_B_WORK_PAIR_RUNTIME_20261028.md)（未提交源码，
native `12acc3cd…a7d`，证据 `fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_runtime_20261028/`）。只读复核，没有新的
GPU 运行或编译。`GO_DEFAULT_OFF_WORK_PAIR_RUNTIME` 成立；accepted opt-in development median 改为 311.440 s；选项默认
关闭；Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false、L8 optimizer STOP 均不变。
同日按 Codex 核对意见修订，见“修订记录”。

## 修订记录（Codex 核对后）

Codex 重新核验了 runtime manifest 的 341 项哈希、当前 native、六次 receipt，并对照父 campaign 的容量记录，均吻合；
GO 成立，可以收口。三处表述收紧，我全部接受：

1. **L7/L8 波动的成因不能在本轮重新确诊。** 本轮没有新 trace，只能说明波动主要位于 L7/L8 optimizer 的 host 计时边界；
   首版写“首次加载排空”“同通道串行”偏强，改为“与既有诊断相符，本轮未重新验证机制”。“这两层代码未变”改为“这两层的
   optimizer 代码未变”：Stage B 确实改变了，前序执行时序也可能影响后续调度。
2. **16.549 s 与 17.024 s 分开记账。** 前者是 Stage B 的阶段归因，后者是预声明协议下的完整方法收益。首版“累计账目按约
   16.5 s 计更准确”容易被理解为可以替换正式结果，或把不同阶段、不同 campaign 的中位数相加；改为两项并列。
3. **父源码快照的位置**：Codex 确认 `runtime/build-source/` 保存本次源码，父源码应指向
   `optimizer_l3_slot_runtime_20261028/build-source/`，第 1 节的纠正保留。

## 结论

```text
完整性    deep/runtime 的 closure 与 manifest、verification.json 的 SHA-256 吻合；runtime manifest 341 个条目按当前
          文件重算一致；当前 native 与 104 个生产输入与 build.json 一致。相对 L3 runtime 闭包改动 5 个文件、新增
          1 个头文件
代码      未发现缺陷。generator 用带次数断言的文本替换生成 fixed-VT 入口；生产 recorded_dbdsqr_fixed_vt.inc
          （dff38bc3…）与隔离资格候选（work-pair.diff）只差入口名，我逐字节重建核对；原 include 不变。模式在 plan
          创建时固定，kernel 模板、动态 shared（7N/5N）与入口一起派发，提交前在 host 侧校验
资格      L4–L8 深层：25,873 个 component、2,669,450,530 个 rotation、17,061,684 个 sweep 零不一致；与 L2/L3 合计
          覆盖 canonical 全部 130,670 个 component。35 项错误配置被拒绝；548 个 target 的 OFF/ON 输出相同
计时      协议 16:34:35 写入，六次按 OFF1/ON1/ON2/OFF2/OFF3/ON3 运行，均 exit 0、isolated、同一 native。outer
          中位 328.464 → 311.440 s（−5.1829%）；Stage B 中位 144.350986 → 127.801689 s（−11.4646%）；最慢的 ON
          318.820 s 低于最快的 OFF 327.667 s。三个预声明门都满足
归因      三对的 Stage B 收益为 16.549/16.257/16.875 s，稳定；outer 收益 24.931/16.227/9.644 s 的差别主要位于
          L8 与 L7 的 optimizer 计时：OFF1 的 L8 慢 8.2 s，ON3 的 L8 慢 5.6 s、L7 慢 1.65 s。两项分开记：Stage B
          阶段归因（中位差 16.549 s）与协议下的完整方法收益（outer 中位差 17.024 s）
容量      六次运行的 L3 分点记录、L4/L6 池最小 free 与增量都相同，也与 L3 runtime campaign 的 ON 逐字节相同；
          新 native 没有带来额外的显存偏移。L6 最紧检查点仍比 2 GiB 门高 105.125 MiB
```

## 1. 完整性

- `deep/closure.json`、`deep/manifest.json`、`runtime/closure.json`、`runtime/manifest.json`、`verification.json` 的
  SHA-256 与报告一致。runtime manifest 341 个条目（215 个文件、104 个生产输入、12 个工具、10 个依赖）按当前文件重算一致。
- deep manifest 的 168 个条目封存的是集成之前的版本：其中 103 个源文件哈希与 L3 runtime 的 `build.json` 完全相同，
  对应的父源码快照在 `optimizer_l3_slot_runtime_20261028/build-source/`；6 个工具的历史版本在 `runtime/deep-tool-source/`。
  按这些版本核对全部一致。报告写“父 … 源快照保存在 runtime/build-source/”不够准确：`runtime/build-source/` 是本次的新源码。
- 相对 L3 runtime 闭包：改动 `full_cuda_ci_structured_eig.{cu,hpp}`、`full_cuda_ci_one_call.cpp`、`build_cuda_native.sh`、
  `generate_recorded_dbdsqr.py`，新增 `full_cuda_ci_structured_launch.hpp`。

## 2. 代码复核

- **generator**：`fixed_vt_work_pair()` 对生成的 DBDSQR 文本做替换，每处断言出现次数（四个写入块各一次、反向消费者两次、
  入口检查与改名各一次），只在给出第二个输出路径时生成。用封存的原 include 加上隔离资格的 `work-pair.diff`、再改入口名，
  结果与生产的 `recorded_dbdsqr_fixed_vt.inc` 逐字节相同。
- **launch 约束**（`full_cuda_ci_structured_launch.hpp`）：环境变量未设置、为空或为 `0` 时为原入口，`1` 为 work pair，
  其他值抛出；shared 字节由模式决定（7N/5N）；`validate_structured_dbdsqr_launch` 要求 plan 模式与入口模式相同、shared
  字节相符、配置为 L/1/0/0，否则在 host 侧抛出，不进入 dense fallback。
- **kernel 与 plan**：`bidiagonal` 按模式模板化，只在 work-pair 实例中调用 fixed-VT 入口；plan 的模式是 `const`，创建时
  为对应的模板实例设置最大动态 shared；每次提交按 plan 模式派发，在 `prepare` 之前校验。kernel 内的 d/e/q/work 布局
  不变，5N 时 work 为 2N。
- **one_call**：只增加按模式的计数输出。六次运行中，OFF 只用原入口、ON 只用 work-pair 入口，各为 10 个 plan、2,044 次
  提交、130,670 个 target。
- 这个开关没有像槽位选项那样绑定 canonical envelope。输出在结构上相同，作为默认关闭的 opt-in 可以接受；但 GPU 证据只
  覆盖 n=351 与已列的合成案例，报告已写明。将来若考虑默认开启，需要其他形状的覆盖。

## 3. 计时复现与归因

由六个 `report.json` 重算（秒）：

| 运行 | outer | Stage B | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF1 | 336.195 | 144.351 | 90.660 | 84.561 | 52.965 | 43.072 | 20.364 | 9.948 | 25.240 |
| ON1 | 311.264 | 127.802 | 84.181 | 77.799 | 50.666 | 42.056 | 20.259 | 9.942 | 17.027 |
| ON2 | 311.440 | 127.796 | 84.277 | 77.640 | 50.589 | 42.094 | 20.273 | 9.970 | 17.276 |
| OFF2 | 327.667 | 144.053 | 90.244 | 84.826 | 53.022 | 42.957 | 20.335 | 9.946 | 17.026 |
| OFF3 | 328.464 | 144.754 | 90.733 | 84.856 | 52.942 | 42.976 | 20.468 | 9.952 | 17.236 |
| ON3 | 318.820 | 127.879 | 84.299 | 77.754 | 50.688 | 42.095 | 20.204 | 11.602 | 22.868 |

（L2–L8 为层耗时。）各层 Stage B 中位数的降幅与报告一致：L2 11.50%、L3 11.61%、L4 11.37%、L5 11.49%、L6 9.77%，
L7/L8 约 0.5%。

| 对 | outer | Stage B | L7 层 | L8 层 |
|---|---:|---:|---:|---:|
| OFF1/ON1 | +24.931 | +16.549 | +0.006 | +8.212 |
| OFF2/ON2 | +16.227 | +16.257 | −0.024 | −0.249 |
| OFF3/ON3 | +9.644 | +16.875 | −1.651 | −5.633 |

- Stage B 的收益三对一致；outer 的差别几乎都来自 L8（OFF1 的 L8 optimizer 24.876 s、ON3 22.507 s，其余约 16.7–16.9 s）
  和 ON3 的 L7（optimizer 11.034 s，其余约 9.4 s）。这两层的 optimizer 代码未变，但 Stage B 已改变，前序执行时序也可能影响后续调度；按报告计入 outer、
  不归因。
- 两项分开记录：Stage B 的阶段归因为中位差 16.549 s；预声明协议下的完整方法收益为 outer 中位差 17.024 s。前者不替换
  后者，也不与其他阶段或其他 campaign 的中位数相加。

## 4. 容量

六次运行的 L3 建池 1,218,445,312 B、optimizer 新增 1,205,862,400 B、总增量 2,424,307,712 B、最小 free 3,404,857,344 B，
L4 池增量 809,500,672 B、最小 free 2,689,728,512 B，L6 池增量 473,956,352 B、最小 free 2,257,715,200 B 都相同，与 L3
runtime campaign 的 ON 逐字节相同。shared 是片上资源，work pair 不改变 device 显存；新 native `12acc3cd…` 也没有像上一次
那样带来恒定偏移。L6 最紧检查点仍比 2 GiB 门高 105.125 MiB。

## 5. 记录

- 新 profile 中 Stage B（中位 127.8 s）与 optimizer 规模相当。本轮的明显波动主要位于窗口模式的 L8 与 L7 optimizer 计时边界，
  与既有诊断（L8 首次加载排空、L7 同通道串行）相符，本轮未重新验证机制。它们会影响以后小改动的门；是否处理由后续选题决定，本复核不提出新实验。
- 报告中关于父源码快照位置的一句建议改为指向 `optimizer_l3_slot_runtime_20261028/build-source/`。

## 证据

`runtime/{build,protocol,closure,manifest,checks}.json`、`runtime/{off1,on1,on2,off2,off3,on3}/report.json` 与 `*-process.json`、
`deep/{closure,manifest}.json`、`verification.json`、`fastkpc/build/recorded_dbdsqr{,_fixed_vt}.inc`、
`stage_b_work_pair_20261028/attempt1/work-pair.diff`、`optimizer_l3_slot_runtime_20261028/build-source/`。计算为只读一次性
脚本；没有修改生产源码、generator、native 或默认配置，没有 commit 或 push。
