# q=37/46 exact vector postback：隔离 GPU 原型

日期：2026-10-29。执行范围来自 [vector-postback 复核](PHASE10_OPTIMIZER_POSTBACK_AUDIT_REVIEW_20261029.md)。
结论为 `GO_ISOLATED_POSTBACK_DEVELOPMENT`：基础 GPU、sanitizer、全量 captured-cohort、三组完整层及三组 canonical 配对门全部通过。
同期 outer 中位数 **315.067 → 286.882 s**，节省 **28.185 s（8.9457%）**；全部 240,498 CI 和 pMax 与已接受参考逐位一致。
这是隔离研究原型的完整 canonical development evidence。生产源码、native 和默认配置保持不变，runtime 集成仍为 NOT YET GO。

## 实现边界

研究目录内生成独立源码副本，只修改其中 `lapack_312_small_dgesdd.cuh` 的 q=37/46 Q/P 回乘入口。
保留原 LWORK/NB、倒序分块、串行 reference DLARFT、原 reference DLARFB 与 BLAS 函数体、64 线程 block、
原 workspace、原请求顺序/cohort 和 32 槽。Q 按输出列、P 按输出行分配给 warp0，各 lane 独占 WORK 行，
leading dimensions 不变。所有 32 lanes 在 T 构造后及当前块全部向量完成后同步，包括没有第二个向量的 lane。
L5 的 Q 尾块为 14、P 尾块为 13，随后均为 32；WORK[0] 的最后写入也保留。
其他 q 的回乘仍使用原入口。未改变 rank/tolerance、line search、立即/延迟 P 的调用条件或数值运算顺序。

单元测试中参考和候选计算均在 GPU 执行；host 只生成输入位模式和读取比较计数。
完整 optimizer 候选重新编译 small64 CUDA 单元，与其余 42 个已认证对象链接成研究目录内的独立 native。
生产 `fastkpc/build/fastkpc_cuda.so` 没有被替换。

## 复核工具与口径修复

- `slice_check` 现在校验参数、文件打开、封存 inventory/TSV SHA、逐行解析及完整计数。
  有效输入要求 1,553 行；数值模式要求 18,636 用例、709,404 次向量调用。
- 本轮用 `--validate-only` 验证有效输入，并确认缺失文件、无参数、空文件、截断、篡改五种负例均退出 2。
  未重跑原 host 浮点求解；旧 18,636 用例仍属于先前复核的辅助证据。
- NaN 检查只表示有效输出未观察到哨兵污染，不证明从未发生读取。未检出的敏感性扰动也不全部归因于舍入。
- 复核中的“L5 每块 k=32”已改为 NB=32，并明确 Q/P 尾块。

## GPU 数值与同步资格

| 检查 | 覆盖 | 结果 |
| --- | ---: | --- |
| 原 DLARFB vs 按向量调用 | 1,553 块形状 × 4 模式 × 3 seeds | 逐位通过 |
| 原 DORMBR vs 保留原分块的候选完整 Q/P | 83 augmented shapes × 2 侧 × 4 模式 × 3 seeds | 逐位通过 |
| 合计 | 20,628 用例 | C/W、只读区/guard、info 全部通过 |
| 基础 memcheck | 全部 1,719 形状，单模式/seed | 零错误、零泄漏 |
| 基础 synccheck | 全部 1,719 形状，单模式/seed | 零错误 |
| optimizer 缩减资格 | L4 15 setups/136 targets；L5 5/20 | 逐位通过 |
| optimizer memcheck / synccheck | 同一缩减集合 | 零错误；memcheck 零泄漏 |
| 完整 L4 | 2,154 setups / 17,767 targets | 逐位通过，资源闭环 |
| 完整 L5 | 955 setups / 6,816 targets | 逐位通过，资源闭环 |

四种输入模式覆盖稠密有限值、signed zero、T 对角 0/1，以及 C 中的 Inf/NaN；比较包含 NaN payload 位。
只读区/guard 与初始副本分别比较，不能只靠参考/候选相互一致掩盖共同越界写入。
global workspace 的并发正确性依靠独立地址审计、明确同步边界和 GPU 资格共同支撑；synccheck 不证明所有 global memory 竞争不存在。

完整 cohort 比较捕获中的 11 类 optimizer 浮点向量、16 类整数/已记录轨迹字段，以及 coefficients/fitted/residual 和输出 route/rank/info。
这不是新增逐次 trial 全部中间矩阵的轨迹导出。计时模式单独运行，诊断 coefficients/residual 在 GPU 上 D2D 保存，
计时结束后才 D2H 和核对；fitted 的完整资格由前述独立 qualification 提供。
测试 arena、准备、packet packing 和销毁成本分别记录，不能把其独立显存用量当成生产 cache/arena 共驻资格。

## 独立细分归因

另编译 serial/sliced 两个 profile 库，各自通过全量 captured-cohort 逐位门。
它们的 Q、立即 P、延迟 P 调用及块数一致：

| 层 | Q calls | 立即 P calls | 延迟 P calls |
| --- | ---: | ---: | ---: |
| L4 | 429,191 | 35,534 | 198,427 |
| L5 | 187,761 | 13,632 | 84,043 |

| 层 | serial DLARFT 周期占比 | serial DLARFB 周期占比 | sliced DLARFT 周期占比 | sliced DLARFB 周期占比 |
| --- | ---: | ---: | ---: | ---: |
| L4 | 1.713% | 98.224% | 13.505% | 86.019% |
| L5 | 12.472% | 87.524% | 57.081% | 42.900% |

分母是对应 profile 库内累计 Q/P 回乘边界周期；包含 stalls，跨 target 相互重叠，不等于 wall time。
剩余小份额是循环、metadata 和同步边界。立即/延迟 P 分开计数，不把接受前 ambiguity completion 忽略掉。
L5 的串行 DLARFT 在候选中占比上升，不据此自动启动新优化。

## 编译资源及性能门

未加 profile 的主 optimizer 寄存器数保持 255，stack 从 7,120 增至 7,232 bytes；
fixed evaluator 的 stack 从 288 增至 320 bytes。不存在额外 device 矩阵 workspace。
编译资源变化必须计入真实层跨度与 outer 验证，不能由单元测试提速替代。

预先固定的层门：三组交替配对、每侧一次预热；L4/L5 合计 work 与计入准备/销毁的生命周期中位数均至少下降 5%，
单层回归不超过 3%，三组配对全部获益。通过后才做三组完整 canonical 对比，要求 outer 中位数至少下降 2%、三组配对获益，
并保留全部既有 CI/pMax、authority、任务/图/RNG、物理 cohort/Gram 顺序和资源硬门。
各运行独立进程；预热不等于后续进程保留同一 CUDA context，不据此称为 promotion compute-warm campaign。

不带 profile 的三组正式计时：

| 项目 | 参考中位数 | 候选中位数 | 下降 |
| --- | ---: | ---: | ---: |
| L4 work | 38.007 s | 21.460 s | 43.54% |
| L5 work | 31.829 s | 15.741 s | 50.54% |
| 同次运行两层 work 之和 | 69.826 s | 37.198 s | 46.73% |
| 两层 charged lifecycle 之和 | 70.156 s | 37.534 s | 46.50% |

合计中位数取各次运行的两层之和，不是两个单层中位数相加。参考三次合计为 69.826/66.295/69.836 s，
候选三组均获益，达到 `GO_ISOLATED_OUTER_QUALIFICATION`。该层结果尚不能代替完整 canonical 的容量与收益判断。

## 完整 canonical 配对结果

沿用 canonical 351×48、default-Inf、fresh R process 和 fresh result/target caches，
在独占 GPU1 上依次运行 reference1、candidate1、candidate2、reference2、reference3、candidate3。
CPU affinity 为 0–19，math threads=1，CUDA connections=32；原有 11 GiB component cache、bypass、8 eig lanes、
properties cache、各层 slot 和 structured work-pair 设置全部保留。两张 GPU 的进程监测没有发现外部计算任务。
没有并行编译或其他本轮 GPU 作业；全部六次运行正常结束。

| 配对 | 原实现 outer | 候选 outer | 节省 | 下降 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 317.128 s | 286.882 s | 30.246 s | 9.5375% |
| 2 | 315.067 s | 286.560 s | 28.507 s | 9.0479% |
| 3 | 312.680 s | 292.207 s | 20.473 s | 6.5476% |
| 各侧三次中位数 | **315.067 s** | **286.882 s** | **28.185 s** | **8.9457%** |

候选/参考中位数比为 `0.910542836920401`，三组全部获益，每次候选也均快于每次参考，
通过预先固定的比值 ≤0.98 和 3/3 配对获益门。比较使用本轮同期参考，不使用历史 311.440 s 来计算本轮收益。
这不是五次 compute-warm promotion campaign。

每侧首轮 71/71 硬门通过，后两轮加上重复运行检查，均为 74/74。结果包括：

- 全部 240,498 CI、pMax 与已接受参考逐位一致，新增 CI/pMax 差异和 decision flips 均为零。
- 相对既有 authority 的最大 CI/pMax 绝对误差仍为 `3.05425129631942e-12`，低于 `1e-10`。
  authority p∈[0.05,0.15] 的 1,337 个 CI 全部保持原 p 值位模式。
- task identity、deletion decisions、skeleton adjacency、sepsets、n.edgetests、RNG endpoint、
  已有 optimizer 迭代计数、physical cohort 和 Gram key/order 检查通过。
- exact-primary、stable-SVD、无 approximate gamma、cache route/status、CPU authority/fallback、
  大 payload D2H 和原有资源/arena 关闭门全部通过。
- 104 个生产输入和原 native 在运行前后保持一致；研究 candidate 单独认证，未将它伪装成原 native。

### 收益归因与波动

| 已有 host 边界 | 参考三次中位数 | 候选三次中位数 |
| --- | ---: | ---: |
| L4 optimizer | 27.284600 s | 15.219153 s |
| L5 optimizer | 31.772292 s | 15.702290 s |
| 同次 L4+L5 optimizer 之和 | 59.044824 s | 30.926986 s |
| 全部 optimizer | 130.990653 s | 102.596742 s |
| structured Stage B | 127.856272 s | 127.848582 s |
| native setup | 22.359313 s | 22.330502 s |

L4/L5 合计中位数节省 **28.117839 s**，与 outer 节省 **28.185 s** 一致，支持收益来自目标回乘工作。
合计中位数按同次两层之和计算；各阶段中位数不构成可直接相加的总账，也不把独立 cohort 中的节省秒数直接移入 outer 预算。
第三组 L4/L5 仍节省 28.081144 s，但 L8 optimizer 的候选/参考分别为 24.921575/16.556331 s，
是该组 outer 收益较小的主要观测差别。L8 使用原算法，这种非目标层波动不作为新优化收益或原因证明。

### 容量与覆盖边界

原有 L3 原始增量门（创建 ≤1.5 GiB、optimizer 增长 ≤1.25 GiB、合计 ≤2.75 GiB）、
L4/L6 pool 增量 ≤1 GiB、各相应检查点 ≥2 GiB 空闲，以及 L5 创建后空闲门均原样通过。
候选 L3 的创建/增长/合计最大观测值为 1,218,445,312 / 1,226,833,920 / 2,445,279,232 bytes；
原实现合计为 2,424,307,712 bytes。因此“没有额外矩阵 workspace”不表示零显存增量：本轮原始记录增加 **20 MiB**。
候选 L3/L4/L6 容量检查点中的最低空闲为 **2.083130 GiB**，原实现为 2.102661 GiB；
候选距 2 GiB 门约 **85.125 MiB**。这些是既有创建、optimizer 完成、Stage A 等检查点及进程采样，
不是连续显存上界，也不证明更大真实数据的容量资格。

没有新增三套大型真实数据、orientation/final-PDAG campaign 或所有逐 trial 中间状态导出。
下一步可单独启动默认关闭的最小 runtime 集成资格，保留原入口、q 范围和原有容量门，再验证集成后的 native；
本轮不自动修改 production default，不重开 DLARFT、L8、调度或其他已 STOP 路线。

## 证据和当前状态

工具：[README](../tools/research/optimizer_postback_gpu_20261029/README.md)。
证据目录：`fastkpc/artifacts/full_cuda_ci/optimizer_postback_gpu_20261029/attempt2/`。
首轮 attempt1 在 GPU 执行前因命名空间及 char-pointer 参数编译错误退出；日志和源码副本保留，修正后使用独立 attempt2。

生产 native SHA-256：`12acc3cd01cce2dcac5f2bfb5ec02f27056e92154cbab2534c7f03bdc22caa7d`。
候选 native SHA-256：`06916ca1dd84d61780623f16b9461d6554da4f30bd94c02bcad892ef3b417792`。
machine audit 为 PASS，独立复算 outer 中位数、配对门、完整 CI 计数及 receipt/native 身份均通过。

- closure SHA-256：`f2de10d89f3a29cb59eb78d7dc01d73b9327234b8518415867e9fea4de494ece`。
- manifest SHA-256：`f5f4dd074f3929d013fdf088130da9a7ad6f2457855b593acf1da2614cea26f1`。
- `canonical-analysis.json` 记录复算后的时间、硬门和容量摘要；`profile-analysis.json` 验证 Q/P 调用/块计数与周期分区。
- `shared-source-attestation.json` 将七个共享 fixture/audit/runner 辅助源码与 HEAD `c38b04b` 的 Git blobs 核对。
  这是明确的事后 attestation，不追溯称为运行前冻结；各次预先冻结的依赖仍由其原 protocol 逐项记载。
- 原静态审计的 133 项保持不变；本轮工具快照、source overlay、编译产物和所有运行记录由新 manifest 绑定。
  独立复核通过 310 个证据文件的精确文件集和 SHA，以及 11 个本轮工具、7 个共享工具和 104 个生产输入；
  当前工具与封存快照一致，新增/修订文件的语法和空白检查通过。

accepted opt-in development median 仍为 311.440 s；默认关闭、L8 STOP、Phase 10 ACTIVE、candidate NOT_FROZEN、
holdout SEALED、recommended=false。此研究原型不是 production integration、clean producer 或 promotion evidence。
全部本轮 GPU 作业已结束；未提交或 push。
