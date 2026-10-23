# GAM（mgcv magic）GPU 化的方法建议

> **状态（2026-10-20）：§6.1 判别实验已完成，结论 STOP**，见 [guarded-QR 诊断报告](PHASE10_GUARDED_QR_DIAGNOSTIC_20261020.md)。
> - 冻结守卫没有漏判，但标记了 51/52 个验证 cohort；整 cohort 重跑使 optimizer kernel 成本比 exact-primary 多约 32%。
> - 发现"离散决策序列完全相同，p 值仍偏 3.18e-9"的反例：逐点评估误差只有约 1e-15，但在 GCV 曲面平坦的方向上，Newton 迭代把它放大成 3e-7 的 log-SP 差。本文 §3.2"决策相同则迭代点差异与评估误差同量级"的假设不成立。
> - 生产保持 exact-primary。以下内容保留为历史提案。

> 日期：2026-10-20
> 基线：`main` @ `6a6a4b3`（runtime producer `dcb1749`）
> 性质：方法分析与实验提案。没有改动任何代码，没有实现，没有性能测量。
> 范围：kpcalg `regrXonS` 实际用到的 mgcv 子集（高斯分布、恒等连接、GCV.Cp），即 selected-SP 优化器这一段。canonical 中这一段的 host boundary 为 596.810 s。
> 与 `PHASE10_EXACT_SPEEDUP_FINDINGS.md` 的关系：那份记录测量结果和现有路线的问题；这份讨论"选平滑参数"这一段可以换一种做法。

## 0. 摘要

- kpcalg 实际调用的是 mgcv 的 C 函数 `magic()`，不是 `gam.fit3`/`newton()`（§1）。
- 建议：复刻 `magic()` 的**数学和每一个离散决策**，而不是复刻 LAPACK 的浮点运算顺序。具体分三层（§4）：
  - 快路径：语义复刻，GPU 原生执行
  - 决策守卫：记录每个决策的裕量
  - 精确兜底：裕量不足的 target 交给现有逐位内核
- 已有证据有正反两面（§5）：
  - 支持：项目里已有"守卫 + 精确重放"的先例；8 月 8 日的路线有 89% 的评估没走 SVD，canonical 也通过了契约。
  - 风险：那次的余量只有约 1.4 倍；有一个真实输入 witness 曾经失败；秩亏情况下，非 magic 的解在 p 值上可以偏差很大。
- 第一步是一个**判别实验**（§6.1），先确认守卫能否把超标任务全部标记出来，再决定是否写新内核。
- 本文不给耗时预测（§7）。

## 1. kpcalg 实际调用的是什么（已核实）

- **调用的模型。** `kpcalg:::regrXonS` 在 |S| > 2 时用加性平滑公式 `frml.additive.smooth`，否则用联合平滑公式 `frml.full.smooth`，都以默认参数调用 `gam(formx, data = data)`。
- **实际走的是 magic。** 在本机 mgcv 1.9-1 上用同样的公式验证：

  | 公式 | 系数数 q | 惩罚数 M | `gam()$optimizer` |
  |---|---:|---:|---|
  | `y ~ s(a)` | 10 | 1 | `"magic"` |
  | `y ~ s(a, b)` | 30 | 1 | `"magic"` |
  | `y ~ s(a) + s(b) + s(c)` | 28 | 3 | `"magic"` |

- **传给 magic 的控制参数。** `gam.fit` 用 `magic.control <- list(tol = G$conv.tol, step.half = G$max.half, rank.tol = control$rank.tol)` 调用 `magic`；`gam` 中 `G$conv.tol <- control$mgcv.tol`、`G$max.half <- control$mgcv.half`。`gam.control` 的默认值是 `mgcv.tol = 1e-7`、`mgcv.half = 15`、`rank.tol = .Machine$double.eps^0.5`。
- **版本说明。** 下文的源码行号基于仓库内 `mgcv_1.9-4.tar.gz` 中的 `src/magic.c`，本机安装的是 1.9-1。两版的 `magic` 是否完全一致需要核对（见 §9）。

## 2. magic 的算法结构（`mgcv/src/magic.c`，1.9-4）

### 2.1 每次评估：`fit_magic`（第 62–153 行）

1. 构造总惩罚 St = Σ_k e^{sp_k}·S_k。
2. `mroot(St)` 求平方根 rS，秩为 rank_S（第 114 行）。
3. 把 rS 堆在 R 下面形成增广矩阵 [R; rS]，用 `mgcv_svd_full` 做全 SVD（第 126 行）。
4. 截秩：rank 从 q 往下减，直到 `d[rank−1] ≥ d[0]·rank_tol`（第 128–129 行）。
5. 计算残差相关量：
   - y1 = U1ᵀ·y0，yAy = ‖y1‖²，yAAy = ‖U1·y1‖²
   - norm = yy − 2·yAy + yAAy，负数截为 0（第 140 行）
   - trA = ‖U1‖_F²，b = V·diag(1/d)·y1
6. GCV 分数：score = n·(norm + norm_const)/(n − γ·trA)²（第 150 行）。

### 2.2 外层循环（第 560–668 行）

- **试步**：每试一步就调用一次 `fit_magic`。
  - 只有 `score < min_score` 时才接受（第 582 行），否则步长减半。
  - 第 4 次尝试时，如果还没用最速下降，就改用最速下降（第 571 行）。
  - 尝试次数达到 max_half 就放弃（第 591 行）。
- **收敛**：从第 4 次迭代起检查（第 593 行）。
  - `d_score > tol·(1+min_score)` 或 `‖grad‖ > tol^{1/3}·(1+|min_score|)` 时判为未收敛（第 595、598 行）。
  - 如果这一轮试步达到 max_half，直接视为收敛并记为 step failure（第 599 行）。
- **求导与下一步**：
  - `magic_gH` 计算梯度和 Hessian（第 611 行），`mgcv_symeig` 对 Hessian 做特征分解（第 623 行）。
  - 有负特征值时，下一步用最速下降（第 624 行）；否则取 Newton 步，最大分量超过 5 时整体缩放到 5（第 629 行）。
- **边界探测**：对每个平滑参数，沿梯度的反方向每次移动 2，最多 5 次，只有分数下降才保留（第 639–660 行，接受判断在第 651 行）。
- **最终拟合**：在选定的 sp 上再调用一次 `fit_magic`（第 664 行）。

### 2.3 结构事实

- **与响应无关的部分可以共享。** X、R、惩罚矩阵、初始 sp（`initial.sp(X, S, off)` 不接收 y）都与响应无关，同一个条件集 S 的所有 target 共享这些数据。
- **每个 target 只有 q 维。** 优化只依赖 y0 = Q₁ᵀ·y（q 维）和标量 yy，与样本数 n 无关。
- **规模。** q ≤ 64，canonical 中多数是 28。8 月 8 日 canonical 回执的计数：
  - 多惩罚 target 80,751 个
  - 迭代共 890,171 次，单个 target 最多 187 次
  - 评估共 1,679,728 次（完整评估 1,167,021 次，只算分数 512,707 次）
- **分数本身带舍入噪声。** norm = yy − 2·yAy + yAAy 是大数相减，计算出的分数有约 ε·yy 量级的舍入噪声。分数差小于这个量级时，magic 自己的"接受/拒绝"已经由舍入决定；逐位复刻只是复现了同一份噪声。

## 3. 关键判断

### 3.1 必须复刻 magic 的停点

- magic 满足 §2.2 的两条阈值就停。tol = 1e-7 时，梯度阈值约为 4.6e-3·(1+|score|)。
- 停点离 GCV 的真最优还有一段距离，而残差对 log-sp 一阶敏感。如果直接求真最优，p 值偏差可能远超 1e-10。这只是方向性判断，没有定量。
- 所以必须复刻 magic 的迭代点和停止决策。

### 3.2 但不需要复刻每一位

- 契约只要求：p 值绝对误差 ≤ 1e-10，判定零翻转，`selected_log_sp_bit_identity_required = false`（`fastkpc/inst/contracts/full_cuda_ci/numerical_contract_v1.json:60`）。
- 如果每次评估的分数、梯度、Hessian 都足够准，并且每个离散决策都与 magic 相同，那么迭代点的差异与评估误差同一量级。
- **这个论断在病态或秩亏时不成立**（§5）。守卫的作用就是识别这些情况。

### 3.3 现有实现中与本提案相关的部分

- 每个 target 一个 64 线程 CTA：`kBlockSize = 64`（`fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu:37`），内核在 `fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu:2362`。
- 核心是 LAPACK 3.12 的 f2c 移植（`fastkpc/src/cuda/lapack_312_small_dgesdd.cuh:1227` 的 `small_dgesdd_left`），里面有 thread 0 单独执行的串行段；内核用满 255 个寄存器，工作区放在全局内存。
- exact-primary 下每次评估都做稳定 SVD。
- 主机侧有 pageable 拷贝锁和 64-setup 窗口，见 `PHASE10_EXACT_SPEEDUP_FINDINGS.md` §2。

## 4. 方案：三层

### 4.1 快路径：语义复刻 + GPU 原生执行

- **执行形态**：
  - 每个 target 一个 warp，q×q 量级的数据常驻 shared memory（q = 28 时约 20 KB，是估计值）。
  - 一个 target 的整条 magic 轨迹（试步、减半、收敛判断、求导、边界探测、最终拟合）都在一个常驻内核里完成。
  - 一层的所有 target 放进设备端队列，一次发射。
- **评估方法**：
  - 对增广矩阵 [R; rS] 做 warp 并行的 Householder QR，归约顺序固定，保证每次运行结果确定。
  - 条件数估计远离截秩阈值时，trA、yAy、yAAy、b、梯度、Hessian 直接从 QR 因子计算；满秩情况下这与截断 SVD 在数学上相同。
  - 接近截秩阈值时做 SVD，或者转兜底。
- **公式**：照用 magic 的公式，包括 norm 的相减形式，让舍入噪声同源。
- **单惩罚**：继续用现有的谱路径（已含端点守卫）。

### 4.2 决策守卫

对 magic 的每一个离散决策记录裕量：

| 决策 | magic.c 位置 | 裕量 |
|---|---|---|
| 截秩：`d[i]` 对比 `d[0]·rank_tol` | 第 129 行 | 最接近阈值的 `d[i]/(d[0]·rank_tol)` |
| `mroot` 的秩 | 第 114 行 | 数值秩是否等于结构秩（每个加性块秩为 8） |
| 接受这一步：`score < min_score` | 第 582 行 | \|score − min_score\| 是分数噪声 ε·yy·n/(n−γ·trA)² 的多少倍 |
| 减半次数、第 4 次改最速下降 | 第 571、591 行 | 由"接受这一步"派生 |
| 两条收敛阈值 | 第 595、598 行 | 与阈值的相对差 |
| Hessian 是否有负特征值 | 第 624 行 | 最小特征值 / 最大 \|特征值\| |
| Newton 步是否超过 5 | 第 629 行 | \|最大分量 − 5\| |
| 边界探测是否接受 | 第 651 行 | 同"接受这一步" |

- 任何一个裕量低于守卫带，就标记这个 target。
- **守卫带怎么定**要靠误差模型加实验标定（§6.2）。这是本方案最关键、也最没有把握的部分。

### 4.3 精确兜底

- 被标记的 target 交给现有的逐位 exact-primary 内核，它已经通过全量资格验证。
- **一个 target 内不混用两条路径**：整条轨迹要么全走快路径，要么全部兜底，避免拼接问题。兜底时最终残差由现有的 exact fixed-SP 路径给出。

## 5. 已有证据：正反两面

| 证据 | 来源 | 含义 |
|---|---|---|
| 单惩罚谱路径曾在一个真实输入 witness 上失败（task 14591，`8 ⊥ 21 \| {3,38}`）。修复方法：优化端点 ≤ 2×收敛容差时改走精确重放 | `MGCV_GPU_SEMANTIC_CLOSURE.md:48–51`、`:335` | "守卫 + 精确重放"在项目里已有先例；失败集中在接近收敛容差的端点 |
| 8 月 8 日的路线：89% 的评估走 guarded QR（1,495,277 / 1,679,728），只重放了 199 个 target；canonical 最大 p 误差 6.96e-11 | `fastkpc/artifacts/full_cuda_ci/phase10_pvalue_contract_repair_v1/canonical_clean_audit.rds`；`PHASE10_PVALUE_CONTRACT_REPAIR.md` | 非逐位轨迹在 canonical 上能通过契约，但余量只有约 1.4 倍。两次运行用的是同一条 dCov full eig 路线，所以推断这部分误差主要来自 GAM 部分 |
| 语义闭合之后（exact primary + 端点守卫）：canonical 3.05e-12，三组真实数据 ≤ 4.86e-12 | `MGCV_GPU_SEMANTIC_CLOSURE.md` | 逐位路线的余量大得多 |
| KKT/refinement：14 个真实 fixed-SP witness 中，1 个证书通过但 p 误差为 1.45e-10；最大 p 误差 0.0146 | `MGCV_GPU_SEMANTIC_CLOSURE.md:353–375` | 秩亏时"数学上满足方程"不等于选中了 magic 的截断伪逆解。截秩决策必须原样复刻并守卫 |
| guarded QR 已有实现：`min\|R_ii\|/max\|R_ii\| > 1e-6` 时用 QR；但只要轨迹里出现过 guarded QR，整条轨迹就会重放 | `fastkpc/src/cuda/lapack_312_small_dgesdd.cuh:498`；`fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu:886` | 快路径的评估部分已经有了，缺的是"按决策裕量判断"而不是"整条重放"的守卫 |

## 6. 建议的实验（按顺序）

### 6.1 判别实验（先做，不写新内核）

- **改动**：加一个诊断开关，保留 guarded-QR 主路径，关掉"出现过 guarded QR 就整条重放"的规则（`fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu:886`）；同时在现有内核里导出每个 target 的 §4.2 裕量。
- **数据**：30 列子集、canonical、三组真实数据，都与 legacy oracle 对比。
- **观测**：
  - 最大 p 误差及其分布
  - 判定翻转
  - 超过 1e-10 的任务清单，以及这些任务的裕量分布
- **判定**：如果超标任务全部能被"低裕量"标记出来，并且标记比例很小，就继续；否则停止这条路线。

### 6.2 守卫标定

用 6.1 的数据定守卫带。要求所有未被标记的任务 p 误差 ≤ 一个安全值（例如 1e-11，留 10 倍余量）。

### 6.3 GPU 原型（6.1、6.2 都通过后再做）

warp-per-target 的常驻 magic 内核，加上守卫和兜底。先在子集上对比选出的 log-sp、残差、p 值、守卫触发率和吞吐。

### 6.4 接入与资格验证

口径同 `PHASE10_EXACT_SPEEDUP_FINDINGS.md` §4.7：
- 全部 canonical p 值和 pMax ≤ 1e-10，零翻转
- 任务顺序、deletion trace、邻接、sepset、`n.edgetests` 一致
- 三组真实数据通过
- 资源释放、结果确定、全程 GPU 执行

## 7. 为什么不给性能预测

- 目前只有一个算术量级：canonical 多惩罚部分约 168 万次评估，q 多为 28，粗算 1–3 Tflop FP64。4090 的 FP64 吞吐只有 FP32 的 1/64。这只说明当前的 597 s 主要不是算术量造成的，不能用来预测能做到多快。
- 能做到多快取决于守卫触发率、兜底占比、warp 内核的实际效率和长尾，都要等 6.1–6.3 的数据。

## 8. 范围之外但相关

- **硬件**：大量小型 FP64 稠密问题加分支，是 GeForce 显卡最吃亏的负载；A100/H100 的 FP64 吞吐是 4090 的数倍到数十倍。是否换硬件、是否允许 CPU 做数值计算，是项目层面的决定，与 full-CUDA 目标冲突。本提案不依赖这两点。
- **setup**：单变量 TPRS 块已经按变量复用（commit `358f164`）；每个 S 的 QR 可以批量放到 GPU 上。native setup 共 22.8 s，不是本提案的重点。

## 9. 请 Codex 重点核查

1. mgcv 1.9-1（本机安装版）与 1.9-4（本文阅读的源码）的 `magic`/`fit_magic` 是否一致；本文行号基于 1.9-4。
2. 8 月 8 日运行的 6.96e-11 误差是否确实主要来自 GAM 部分。
3. task 14591 的失败细节：所在数据集、误差大小、当时走的是哪条路径。
4. 能否在现有内核里低成本记录 §4.2 的全部裕量。
5. "同一 target 整条轨迹只走一条路径"这个约束是否足够，会不会漏掉跨评估的误差累积。
6. 6.1 的判别实验是否足以否定或支持这条路线，还缺哪些对照。
