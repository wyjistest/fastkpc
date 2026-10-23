# Structured component prefill：22 列完整 skeleton 资格

后续 [351×48 canonical opt-in 资格与交错重复](PHASE10_STRUCTURED_PREFILL_CANONICAL_20261022.md)
已经完成。本文保留 22 列子资格当时的范围、native 和单次结果，不将其重标为后续证据。

日期：2026-10-22。执行
[修订评审](PHASE10_STRUCTURED_DEVICE_ADAPTER_REVIEW_20261022.md)
后的窄工作：先修正完成边界计时，再接入默认关闭的有界 structured producer，
最后运行同一 native 的 off / dense / structured 全深度对照。

```text
完整完成边界微基准                         PASS
structured producer 与已资格适配器逐位身份  PASS（640 分量）
351×22、default-Inf 完整 skeleton 硬门      PASS（24,264 CI）
单次完整方法性能                           development GO
canonical 351×48 structured 资格           NOT_RUN
生产默认 / Phase 10 / candidate / holdout   不变
```

当前只允许 `dcc.gamma`、351×22、numCol=35 的显式原型；不是生产默认或
promotion。完整 canonical 和新的最终 WAN-PDAG campaign 均未运行。

## 完成边界补测

原脚本在最后一次 `finish()` 返回时停表，未显式等待随后提交的 centering 与
D2D stores。新脚本在所有生产提交之后、停表之前统一执行最终 GPU join。
没有在每个 target 后增加等待，没有 shadow D2H；全部配置处理同样的 640
个真实 n=351 残差，单主机线程，预热一次、三次中位数。

| 配置 | 完成 640 分量中位数 | 完成吞吐 / s |
|---|---:|---:|
| dense，1×22 | 1.966589 s | 325.44 |
| dense，1×64 | 1.862834 s | 343.56 |
| structured，1×22 | 7.989723 s | 80.10 |
| structured，1×64 | 2.764059 s | 231.54 |
| structured，5×64 | 0.773075 s | 827.86 |
| structured，10×32 | 0.767996 s | 833.34 |
| structured，10×64 | 0.684232 s | 935.36 |

实测最终 completion tail 约 8–30 微秒；修正边界后并发池的优势仍成立。
5×64 与 10×32 在同样 320 个在途分量下很接近。这支持当前实现及 workload
增加在途数量的设计，不是“任何批宽都无关”的一般证明，也不是占用率剖析。
这些微基准不包含 context 创建时间；下面的完整方法计时包含 lazy 创建。

## 实现与容量

新增开关 `FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL`，默认 0；开启时
必须同时满足原 level-prefill、fresh caches、exact-primary reuse、component
cache/bypass、8 dense lanes 与原 scheduler 的要求。不支持的配置立即拒绝。

- 首次使用时按需求创建、最多保留 10 个 capacity=64 的精简 context；单主机
  线程先提交整个 wave，再读取 bounded-batch compact status。
- 每个 context 只有独立 workspace、stream 和 events，不复制 cache、cuBLAS/
  dense solver handles 或 pair consumer。所有 CUDA 资源进入原资源账本。
- structured 数学 kernels 从已资格实现原样移入独立 TU，保留 `--fmad=false`
  和 sm_89；vendored LAPACK 使用独立 namespace，防止与 optimizer 符号冲突。
  centering 的归约/减法语句逐字保持原实现；原 total 仍按原距离行归约生成。
- 固定 info / rotation overflow / gap<1e-8 门逐分量选择原 GPU dense；失败索引
  按原容量 22 分块。残差 stable-SVD route/status 与 component eig solver
  identity 分开保存、检查。没有 CPU numerical fallback。
- structured 与 dense fallback 的全部 D2D stores 汇合到原 context stream，
  完成后才 commit wave。未完成或失败的层禁止 lookup；最后 join 之后才释放
  residual arena。Gram 的原分组、key 顺序及 pair/gamma 实现保持不变。

10 个 context 的 solver workspace 为 **2,727,014,400 bytes**，约 2.54 GiB；
包括原距离矩阵、行和、居中向量等辅助存储后的 pool 总量为
**3,422,507,520 bytes**，约 **3.19 GiB**。pool 在层间复用；小层活跃 context
减少时，已分配容量仍保留，回执分别记录活跃宽度与保留字节。

外部 200-ms 采样的整卡显存峰值：off/dense 均为 **15,186 MiB**，structured
为 **18,498 MiB**（约 18.06 GiB）。这是包括 driver/display 等占用的采样峰值，
不是所有瞬间的严格最大值；单卡 24 GiB 容量通过本轮完整实际运行。

## 同 native 的完整运行

351×22、default-Inf、dcc.gamma；GPU1，另一张 GPU 无计算任务，connections=32，
数学库线程数为 1。每档为 fresh R process 与 fresh semantic caches，11 GiB
component cache、bypass、8 eig lanes。离线 authority 已完成后才运行计时；
回执序列化、压缩和对照审计在 skeleton timer 之外。

| 配置 | 完整 skeleton | Stage B 完成边界 |
|---|---:|---:|
| prefill off | 108.932 s | — |
| dense prefill | 94.408 s | 21.929169 s |
| structured prefill | **82.775 s** | **8.618157 s** |

structured 相对 dense 本次少 **11.633 s（12.32%）**；相对 off 少
26.157 s（24.01%）。Stage B 少 13.311 s；optimizer host 同时从 43.464 s
变为 45.098 s，不能把所有局部节省等同于 outer 收益，也不能仅凭此单次差异
判断 optimizer 回归。每档只有一次完整运行，不是重复中位数或 promotion。

| level | produced / resolved | dense Stage B | structured Stage B |
|---|---:|---:|---:|
| 3 | 4,662 / 4,148 | 14.064394 s | 5.384429 s |
| 4 | 2,093 / 1,981 | 6.307425 s | 2.452142 s |
| 5 | 473 / 461 | 1.430159 s | 0.505448 s |
| 6 | 42 / 42 | 0.127191 s | 0.276139 s |

小层变慢约 0.149 s，已计入总耗时。没有增加小层特殊策略。structured 一共
14 waves，最多 640 个分量在途，7,270 个全部 accepted，natural GPU dense
fallback 为 0；最小 gap 为 0.000370614422310563，gap<1e-4 为 0，最大
rotation count 为 114,830，低于固定容量 170,662。创建 pool 的 host 时间
约 4.305 ms，已计入 Stage B。重叠 lane 的 event 时长之和不作为总耗时。

## 全量正确性与独立 authority

历史 22 列 GPU 回执不能代替 kpcalg authority。现有 canonical authority
覆盖其中 19,623 个 logical CI；本轮将其余 4,641 个 logical CI（4,518 个
unique keys）补为独立离线 kpcalg/mgcv 测试基准，保留原始输入、源码与版本
身份。该基准仅在运行结束后比较，不参与任何 GPU 求解、结果替换或 graph
决策，不进入性能计时。

三档都通过全部 **24,264 / 24,264 CI** 和全部 pMax 的 abs error<=1e-10 门，
最大 CI/pMax 误差均为 **2.1049828546893e-13**。214 个 authority p 值处于
[0.05,0.15]，structured 最大误差约为 2.049e-13，无 alpha 翻转。

off 与 dense 的 p/pMax 逐位相同；structured 相对 dense 有 **6,000** 个 CI
及 **136** 个 pMax 单元发生低位变化，最大差异 **9.93649607039515e-15**。
因此不声称 structured 与 dense 全部逐位相同；独立 authority 的既定 1e-10
门与所有离散语义门分别通过。

task identity/order、deletion trace、adjacency、sepsets、n.edgetests、RNG endpoint、
物理 cohort 摘要和原 Gram key/order 摘要全部一致。7,270 produced、6,632
resolved、638 未消费 pending 的账本闭合。exact-primary、stable-SVD、route/status
检查通过，CPU authority/fallback/unknown fallback、residual/component 大 D2H
均为零，调用前后的资源计数相同。

## 资源与失败门

普通 source-shared 运行：全部 640 个真实分量的生产三字段，与此前已资格
adapter 的归档字节一致。独立 sanitizer 冒烟检查 4 个归档分量及 70 分量
混合批：30 个真实小 gap 分量按 22+8 回退，其余保持 structured；覆盖失败后
不发布、pending reset、重复 Gram 拒绝、真实 rotation overflow 和未 finish
提交的析构。结果 **0 errors、0 bytes leaked**。这不是整个 R 方法的全程
sanitizer 运行。新配置拒绝、既有 level-prefill 配置和 component-cache
回归均通过。

## 身份与下一门

```text
reference native = b1e7eab4297a13b7ff29f0bf40b2fc3ce4d6950542dff4318ee3e2dc76694ee9
new native       = 4b92e48d1ed505f9b3c33e59c52505717711aaa62f269eeba74a4abe40c8e1f4
```

本轮重编译 one-call、vertical CUDA、R API 和新的 structured TU；其余对象
继承此前 build。证据类别为 **source-closure development**，不是 clean producer。

工具：[structured_prefill_20261022](../tools/research/structured_prefill_20261022/README.md)。
证据：[closure.json](../artifacts/full_cuda_ci/structured_prefill_20261022/closure.json)、
[manifest.json](../artifacts/full_cuda_ci/structured_prefill_20261022/manifest.json)。

下一门是独立的 canonical opt-in 资格与同 binary 完整对照；当前 351×48
structured 入口仍拒绝运行。不能由 22 列结果直接承诺 canonical 约 140 秒
收益。optimizer overlap、level-2 prefill、生产默认启用均未进入本轮。
Phase 10 ACTIVE、candidate 未冻结、holdout SEALED、recommended=false 保持不变。
