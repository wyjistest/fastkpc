# numCol 参数通用化

日期：2026-11-06。本轮纠正 GPU 接口把历史基准参数 `numCol=35` 当成强制
条件的做法。`numCol` 是检验的谱截断/低秩参数，不是数据的变量数，也不是
mgcv 平滑基的维度。351 行的默认值恰为 `floor(351/10)=35`，不构成固定 35
的算法理由。改变参数可能改变 p-value、删边和图，必须按原参数重新比较 CPU。

candidate3 的最终参数矩阵、独立 CPU full eig／本地补丁 R rank=1 对照、其他方法、完整 canonical
以及默认 public WAN-PDAG 入口均通过，已更新本机 native，并验证普通 fresh R
loader 加载同一 SHA。所有中间结果继续绑定各自 native，不合并重标。

## Git 交接与原版参考边界

runtime/tool 提交为 `8d139b87a60dd2cd414b4e0528a7fbd142c37065`。
[Git 源码绑定索引](PHASE10_SOURCE_ATTESTATION_20261106.json)核对 107 个生产
构建输入、97 个 fastkpc R 依赖与提交逐项匹配，44 个本机对象与 candidate3
一致，331 个封存证据文件哈希全过。本次没有重跑 GPU 或生成新性能结果。

本次提交仅纳入 fastkpc 实现、测试工具和说明，保留 Git 中原版 `kpcalg/R`。
`kpcalg/R/dcovgamma.R` 的两处 scalar-diagonal 修改留在本地、未提交；未撤销
或改写已完成实验。它不是支持 500×50/default numCol=50 所必需的变更。

下述 11 月 6 日封存资格的 R 依赖包含这份本地补丁，因此不能把整个 closure
重标为未修改原版 R、或当前 Git checkout 上重新运行的结果。native 的 107 个
生产构建输入不包含该 R 文件，C++/CUDA 来源与本次 fastkpc 提交独立核对。
原始和补丁 R 的 SHA、快照均保留在下文的本地证据中。

rank=1 GPU 数值对照使用独立 CPU full eig；额外 R 对照使用补丁函数。原版 R
在这个参数下报错，故没有“与未修改原版 R 的 rank=1 数值一致”这一结论。
21 组 rank>=2 的旧／新 R 逐位比较只覆盖实际测过的用例。rank=1 的两个
WAN-PDAG 夹具都是空图，没有定向 CI，不能据此宣称原版 R 的 rank=1 定向
可用。保留原版参考时，这项上游边界需单独处理，不影响 GPU skeleton 的
独立数值资格。

## 实现与边界

- 公共入口继续采用用户指定的 `numCol`，省略时仍为 `floor(n/10)`；不截断、
  不夹紧到 35 或 64。R 与原生入口在整数转换前拒绝非有限、非整数、非标量、
  非数值或超出 `1 <= numCol < n` 的输入。其余 alpha/index/变量数/模型约束
  未在本轮扩大。
- one-call、prefill 与 legacy eig 请求移除 `==35` 检查。请求身份原本就包含
  numCol；component cache 保留固定 n/k 的形状校验，结果缓存键保留 numCol。
  与检验秩无关的 optimizer target 状态仍可按原身份复用。
- structured 能力条件仍为 `6<=n<=1024`、`1<=k<=64`、`k+1<n`、
  `2(k+1)+4<=n`。不满足时用 dense GPU，保持用户的 k。
- dense 选择器保留小形状的 shared 实现；k>64 或 mask 超出小型 shared
  存储策略时，分配实际 component 数 × n 的 mask 和 component 数 × k 的
  index scratch。选择顺序、绝对值排序 tie 规则、中心化运算和归约不变。
  新存储纳入原资源账本和 RAII 生命周期。
- HSIC 不再用 k² 的 shared 申请量限制公共参数。每次至多保存 4,096 个
  FP64 product（32 KiB），dot 内部运算和 thread 0 按原顺序相加不变；全局
  factor/triangular/cache 仍按实际 n/k 分配。dcc.perm 同步移除无必要的 k<=64
  接口限制。全部编译数值选项沿用父版本，method TU 保持 `--fmad=false`。
- 同进程跨 k 缓存测试发现一项既有校验错误：全部结果命中时，零组件 miss
  却被要求 active eig lanes>=1。现在零 miss 要求零 active lanes，有 miss
  仍要求正数；没有删除物理工作核算门。
- 本地实验中的 kpcalg `dcov.gamma` 的 `diag(P$values)` 在只有一个特征值时，将该值
  误作矩阵尺寸；给 P/Q 的 diagonal 指定实际长度后，rank=1 可以执行。
  这不是改用其他 numCol。21 组 rank>=2 的旧／新 R 结果逐位一致；该 R
  修改没有纳入本次 Git 提交。
- 空图停止条件补齐原 pcalg 的 `any(G)`：删除最后一条边后，不再追加一个
  零测试层。CUDA one-call 和 CPU legacy reference scheduler 都修正；
  标量数值、任务调度顺序和非空图的原停止条件不变。

## 身份与预注册验证

父 native 为 `476ad65e648775d52e7c58e69213269c2c7b05efd27c32ec2834baaaa5e5a37b`，
保留在 general_shapes_20261102/candidate4。候选先重编四个受影响 TU，再为
warm replay 核算修复单独重编 one-call。candidate3 为停止条件修复重编
one-call/r_api 两个 host TU，复用 candidate2 的全部 CUDA 对象。其 native 为
`2727ef51899688b54da697d0008f99a4908922dbcacec1c30ed58cef09198301`。
其余对象按 SHA 复用。所有来源、
编译命令、对象、native 和运行依赖在
`fastkpc/artifacts/full_cuda_ci/numcol_20261106/` 封存。
这仍是增量开发资格，不是 clean-checkout promotion evidence。

[工具与协议](../tools/research/numcol_20261106/README.md)冻结 16 组输入/参数：
三套完整真实数据默认 17/24/30，公开 500×50 和稠密 500×8 默认 50，1025×4
默认 102，140×6 的 1/2/7/35/50/64/65/139，以及 k64 的 133/134 行分派边界。
每组重新产生 CPU mgcv/native dCov 参考，比较普通 retained GPU 与完整 opt-in
GPU，k1/k65 另有 cache-disabled 运行。所有搜索为 `m.max=Inf`；默认参数用例
真实省略选项，不在测试中偷偷补上 35。每次 fresh R process 和 fresh cache，
固定 GPU1、CPU 0–19 和单线程数学库，并监测 GPU 任务隔离。最终 CPU 参考
使用 candidate3 中修正停止条件的 host scheduler；CPU mgcv、CPU Spectra 的
数值实现不变，仍独立于 CUDA 的 optimizer/full eig。CPU dCov 对象与父版本
SHA 一致。另一张 GPU 的外部任务另行记录，正确性测试不等待 GPU0 空闲。

每次必须通过所有 CI/有限 pMax 绝对误差 <=1e-10、任务身份与顺序、删边轨迹、
adjacency、sepsets、n.edgetests、RNG、GPU 数值 authority、无大型 residual/component
D2H、stable-SVD、cache metadata 和资源闭环。三套真实数据及公开 500×50
另查 CPU-authority 定向、最终 PDAG 与 RNG；单独调用公共 WAN-PDAG 默认入口。

其他方法使用原 kpcalg/mgcv/kernlab 和 R permutation 作为数值/RNG authority。
包含 k65/102、sig=1e-5 的 HSIC 用例，显式确认实际 CPU factor rank 为 65/102，
覆盖后续 product tile 中非零的项。小秩与旧 k35 同时保留比较。

## 最终参数矩阵（candidate3）

attempt4 全部重新运行，无旧运行复用：16 组输入/参数，16 次 CPU 参考和
34 次 GPU 对照，共 22,441 次 GPU CI 比较。最大 CI/pMax 绝对误差
`2.77605716192397e-12`，超 `1e-10` 项数为零；全部结构、顺序、数值 authority、
cache 和资源门通过。

下表是默认参数用例的 opt-in ON 数值结果；每组独立 CPU 与 retained OFF
也全部通过。耗时不列为性能验收。

| 输入 | 默认 numCol | 每次 logical CI | 最大 CI 绝对误差（ON） |
|---|---:|---:|---:|
| 完整真实数据 179×8 | 17 | 320 | 6.09512440519211e-14 |
| 完整真实数据 243×9 | 24 | 413 | 8.30446822419617e-14 |
| 完整真实数据 300×10 | 30 | 655 | 1.49769086021934e-13 |
| 公开 synthetic 500×50 | 50 | 2,789 | 1.82846793261859e-13 |
| 稠密 synthetic 500×8 | 50 | 3,559 | 7.79237785408782e-15 |
| synthetic 1025×4 | 102 | 48 | 3.35945371476414e-45 |

三套真实数据及公开 500×50 的 ON 运行另通过定向 CI、最终 PDAG 和 RNG。
其他用例涵盖 k1/2/7/35/50/64/65/139，以及 n=133/134、k64 的 structured
边界；k1/k65 的无 component cache 路径通过。稠密 500×8 自然搜索至第 6 层。
所有 50 个进程记录均显示 GPU1 独占、退出成功；GPU0 同时存在外部任务
PID 101845，故机器级隔离不成立。本轮只用于正确性，不登记性能提升。

## 其他方法与接口回归（candidate3）

supplement5 的五项接口/缓存/回归测试通过；24 个非法参数在 R 和原生入口
被拒绝，7→50→65→7 的结果缓存隔离与全部命中重放通过。CUDA memcheck
报告零错误、零泄漏。

dcc.perm、hsic.gamma、hsic.perm 共 28 组、792 次 CI 通过，最大绝对误差
`3.1863400806742e-14`；置换 p-value、RNG、任务/图以及原始 pcalg
n.edgetests 均通过。k35 的三方法与父 native task/pMax/graph/RNG 位模式相同。
HSIC 的 k65/102、sig=1e-5 用例确实产生 >64 的 CPU factor rank，验证了
product 后续 tile 中非零运算。此处方法资格的最大条件层为 2。

同一 final native 的完整 351×48/k35 canonical 回归通过 240,498 次 CI，
CI/pMax 与已验收父 GPU 结果逐位一致，与 CPU authority 的最大绝对误差
均为 `3.05425129631942e-12`。34 项门全部通过，包括任务与删边/图/计数、
sepsets、最终 PDAG、RNG、GPU authority、stable-SVD、slot 容量和资源闭环。
单次 elapsed `286.700s`，且 GPU0 有其他任务，仅记为开发正确性回归的运行
观测；不与 297.453s、291.429s 或历史重复中位数相减登记速度收益。

public2 在 fresh R 中调用 `fastkpc_compatible_cuda_wanpdag()`，真实省略
numCol，确认 500×50 使用 50。2,789 次 CI、全部有限 pMax、任务/图、最终
PDAG、RNG 和资源通过；最大 CI 误差 `1.82846793261859e-13`，最大 pMax
误差 `1.25455201782643e-13`。该运行绑定修正后的 legacy R source。

## 已完成的中间资格（candidate2）

补充资格绑定 candidate2 native
`eb5a32f0ac1f48de1a86bcd05bfe80af692cd99b14c11e0d3afb4ce98a5e4b45`：
五项接口/缓存/回归测试通过；24 个非法参数通过 R 与原生入口被拒绝，跨参数
结果缓存隔离、同参数全命中重放通过；CUDA memcheck 为零错误、零泄漏。
三种其他方法通过 28 组、792 次 CI（至多条件层 2），最大误差
`3.1863400806742e-14`，置换 p-value 和 RNG 精确一致。旧 k35 三方法与父 native
的 task/pMax/graph/RNG 位模式一致。

完整 351×48/k35 回归通过 240,498 次 CI，保持此前 GPU CI/pMax 位模式；CPU
authority 最大误差 `3.05425129631942e-12`。最终 PDAG、RNG、资源和 slot 容量门
全部通过。单次 elapsed `297.453s`，只记为开发回归观测，不是重复中位数。

参数矩阵 attempt3 为 16 组、50 次运行（16 CPU + 34 GPU），共 22,441 次
GPU CI 对照，最大误差 `2.77605716192397e-12`。这些记录在空图边界补充验证
之前完成，不能替代 candidate3 的最终运行。

## 本地补丁 R rank=1 与停止边界（candidate3）

rank-one2 通过 21 组旧／新 R 函数逐位比较（3 对数据，各取
2/7/35/50/64/65/139）。rank=1 的统计量、矩和 p-value 对照独立 CPU full eig，
最大统计量／矩误差 `1.22124532708767e-14`；不只比较饱和的 p-value。
另有两组完整 WAN-PDAG，分别在第 0 层和第 1 层删光所有边，共 27 次 CI，
与原 pcalg+修正后的 R dcov.gamma 比较最大误差 `2.78194134395449e-12`。
任务、删边、图、sepsets、原始 n.edgetests、RNG 和资源均通过。两组空图没有
定向 CI，因此不把它们计作非空图定向数值资格。

## 保留的失败与修正

- attempt1：k1 的搜索在第 1 层自然停止，数值/图/资源全过，但夹具错误要求
  每个名为 dense 的输入都执行第 2 层 prefill。改为到达该层才要求实际执行，
  记录最高层；不强制搜索深度、不放宽数值合同。
- supplement1：暴露真实的 warm replay active-lane 核算错误，已修复并重新构建。
- supplement2：新缓存测试通过，旧 one-call 测试的诊断列名表缺少已存在的三个
  worker/concurrency 列。显式补齐期望 schema；其余核算和数值断言不变。
- supplement3：numCol/one-call 通过，新 runner 的逐表达式 eval 提前执行了旧
  fixture 的 on.exit。改为一次 eval 整个表达式向量，恢复完整测试生命周期。
- attempt2：外部任务进入 GPU0 时，原双卡隔离监测停止了一个 CPU reference
  job；不是数值失败。attempt3 明确保留 12 个已完成运行的原报告、日志与进程
  身份，其余在独占 GPU1 上重跑；后续监测记录 GPU0 使用而不停止正确性验证。
- rank-one1：原 R scalar-diagonal 修复、21 组旧参数位比较和 CI/图/RNG 均过，
  但空图 GPU n.edgetests 为 `[6,0]`、pcalg 为 `[6]`。没有删除末尾 0 来放宽
  比较，而是修正两个生产 scheduler 的停止条件。三方法补充验证也改为直接
  核对原始 pcalg counts；之前的 trailing-zero 归一化仍保留作历史比较辅助，
  不再代替原始计数门。

## 复核与本地安装

最终审计使用 candidate3 / attempt4 / supplement5 / public2 / rank-one2，
核对 107 个 production source 输入、44 个链接对象、native、R 依赖与所有
进程退出记录。父 native `476ad65e…5e5a37b` 及其对象归档保留。安装工具持有
本地 build lock，逐文件校验并替换对象，最后替换 native；新 R 进程使用
`2727ef51899688b54da697d0008f99a4908922dbcacec1c30ed58cef09198301`。

legacy R 原文件 SHA 为
`d0620dd9b06f006170b5e142b892c202619296797b2bf9ca842a7c2b5b3b715f`，修正后为
`05a30eb59a7404ab24e3137380862d2840b389fc3b17415b9c6045a1fa0e4a69`。
旧 R 原文件、最终 R 文件和工具快照随 closure 封存。当前 closure 只采用
candidate3 的最终成功运行；candidate1/2 的失败与中间成功仍在原目录。
本地大体积 native、真实输入和 RDS 不随 Git 提交；工具缺少输入或 CUDA 时
失败，不将 SKIP 或旧 native 的 receipt 标成当前资格。

本地 artifact 根目录为 `fastkpc/artifacts/full_cuda_ci/numcol_20261106/`。
`installation.json` 状态为 `PASS_INSTALLED`；`closure.json` SHA 为
`f5969fc61d7d86fdbe5d3a2f6fa5c45031151449822cc4c260f9aa2003864870`，
`manifest.json` SHA 为
`b6ac295bbc415acb14c807b61d82682d575dde3a4cde5364a90d482c24645b08`。
资格与安装完成时尚未 commit/push。本次仓库交接补充 Git 与本地 R 补丁的
区别，提交 fastkpc 源码、工具与报告；不改写这些已封存记录。

Phase 10 仍 ACTIVE；candidate 未冻结、holdout 未打开、recommended=false。
本轮解决参数通用性；单次耗时不构成加速结论，不更新历史重复中位数。
