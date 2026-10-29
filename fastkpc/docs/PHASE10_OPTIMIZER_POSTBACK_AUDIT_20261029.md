# Exact optimizer vector-postback：独立向量边界审计

日期：2026-10-29。结论：**`STATIC_VECTOR_SLICE_AUDIT_COMPLETE`**。
q=37/46 的现有reference `DLARFB` 存在保持标量运算顺序的独立向量切分：Q侧分配不同输出列，P侧分配不同输出行，
两者都独占WORK的一行。可以据此进入**单独的隔离GPU资格原型**；尚未实现数值原型，不批准runtime集成或宣称性能收益。

本轮执行 [L4/L5归因复核](PHASE10_OPTIMIZER_L45_ATTRIBUTION_REVIEW_20261029.md) 后建议的实现审计。
生产源码、native、原请求/cohort顺序和32槽不变；没有GPU运行、CUDA编译或CPU浮点数值求解。
仅编译并执行了使用整数编号、依赖位集和符号表达式的host审计工具。
accepted opt-in development median仍为 **311.440秒**；默认关闭、L8 STOP、Phase 10 ACTIVE、candidate NOT_FROZEN、
holdout SEALED、recommended=false均不变。没有commit或push。

## 绑定的实现与可测结论

- HEAD `c38b04b`，runtime `1d8a11d`，104个生产输入在审计前后逐项相等。
- native SHA-256：`12acc3cd01cce2dcac5f2bfb5ec02f27056e92154cbab2534c7f03bdc22caa7d`。
- 归因依据仍为前一轮的L4/L5 SVD cycles占比96.389% /97.383%，其中stage2占58.928% /71.698%。
  stage2包含reference反射变换、同步、direct路径初始化及延迟右向量完成，不能当成纯DLARFB时间。
- reference `DORMBR` 当前由warp0的lane0执行；本次审计的具体对象是其 `DLARFT → DLARFB` blocked循环。
  未测DLARFT/DLARFB或左右两侧各占多少时间，也未测新增线程执行对register、stack、stall与occupancy的影响。

## 调用形状与分块必须固定

本optimizer的SVD输入是已压缩系统，`m=q+root_rank`，不是原dataset行数351加q。
有效producer要求 `root_rank>0`；pivoted factor的rank至多q。`m < floor(11q/6)` 走direct路径，其余先QR归约。

| 项目 | L4 | L5 |
| --- | ---: | ---: |
| q | 37 | 46 |
| 合法augmented rows | 38–74 | 47–92 |
| direct rows | 38–66 | 47–83 |
| 先QR归约的rows | 67–74 | 84–92 |
| DORMBR收到的LWORK=`3q²+4q` | 4,255 | 6,532 |
| DORMQR/DORMLQ的NW | 37 | 46 |
| 实际NB | **2** | **32** |
| Q块数 / P块数 | 19 / 18 | 2 / 2 |

实际调用链：

| 侧 | DORMBR | 下一层 | DLARFB |
| --- | --- | --- | --- |
| Q | Q, L, N | DORMQR(L,N)，反射数q | L, N, F, C |
| P | P, R, T | DORMLQ(R,N)，反射数q−1 | R, T, F, R |

P侧有两次transpose解释，不能把DLARFB的T误改成N。Q/P均按原倒序blocked循环执行：
L4 Q offsets为36,34,…,0，首块IB=1，其余为2；P offsets为34,32,…,0，IB均为2。
L5两侧offsets均为32,0；Q的IB为14,32，P为13,32。
P的DORMBR先跳过首列，内部C/V视图的列偏移为 `1+offset`；首列不应被此调用改写。

**不能把DORMBR整体按单列/单行调用。** 这样NW会从q变为1；L4的同一LWORK随即允许NB从2变成32，改变原数值合同。
独立切分必须发生在原分块和T已经确定的 **DLARFB内部边界**。

## DLARFB真正可以独立的方向

两个实际分支都执行下列步骤，所有DTRMM均为右乘。因此不同WORK行没有相互依赖；同一行内部的列依赖必须保留。

| 顺序 | Q：L/N/F/C | P：R/T/F/R |
| --- | --- | --- |
| 1 | W ← C1ᵀ | W ← C1 |
| 2 | TRMM(R,L,N,U) | TRMM(R,U,T,U) |
| 3，存在尾部时 | GEMM(T,N)，alpha=1,beta=1 | GEMM(N,T)，alpha=1,beta=1 |
| 4 | TRMM(R,U,T,N)，乘T | TRMM(R,U,T,N)，乘T |
| 5，存在尾部时 | GEMM(N,T)，alpha=−1,beta=1 | GEMM(N,N)，alpha=−1,beta=1 |
| 6 | TRMM(R,L,T,U) | TRMM(R,U,N,U) |
| 7 | C1 ← C1−Wᵀ | C1 ← C1−W |

TRMM标记依次为side/uplo/transpose/diag，U/N表示unit/nonunit。
T是固定的upper triangular matrix，V和T在DLARFB期间只读。

可供后续原型采用的最窄形式是：保留原reference `DLARFB` 及其BLAS代码，用warp0的各lane调用单个向量视图。
对于独立向量r，参数映射为：

| 侧 | 调整后的DLARFB m,n | C指针偏移 | WORK指针偏移 |
| --- | --- | --- | --- |
| Q | 原m, 1 | `r * ldc` | r |
| P | 1, 原n | r | r |

`ldc/ldv/ldt/ldwork`保持原值，尤其 **ldwork仍为q**。每lane处理 `r=lane, lane+32` 中有效的向量；
V/T和当前块IB保持一致。这样不必先重写DGEMM、DTRMM或引入新的归约顺序。
这是候选调用形状，不是已经生成或资格验证的CUDA代码。

原地DTRMM只能按独立行切。例：R/U/N的两列运算中，原列1必须先读取列0的旧值，随后才能更新列0；
若两列被独立线程任意排序，列1可能读到已更新的列0。符号工具包含这一反例。
DGEMM的TN分支先累加dot再做alpha/beta组合，NN/NT分支则保留逐项update形式；不能统一成另一种“等价”公式。

## Workspace生命周期与同步边界

每target有独立的 `DeviceEvaluationWorkspace`，本次不跨target共享scratch。
`decomposition.work`先用于bidiagonal求解，再用于blocked reflector；同一块内划为W和T：

| q | W元素区间，0-based | T首元素 | T leading dimension | 最大实际T元素索引 | LWORK |
| --- | --- | ---: | ---: | ---: | ---: |
| 37 | 0–73 | 74 | 65 | 140 | 4,255 |
| 46 | 0–1471 | 1472 | 65 | 3518 | 6,532 |

上述上界涵盖完整NB，尾块更小；T只访问当前IB的upper triangle。W与T不重叠，向量切分继续使用原scratch，
源码层面不要求额外device矩阵。但此表不允许缩小整个work数组：前面的bidiagonal求解仍使用它。

`DLARFT`中T的当前列依赖前序T列，包含DTRMV及与零值相关的扫描；它继续由lane0按原顺序构建。
每个block至少需要明确的完成关系：

```text
lane0 构建本块T / 写必要metadata
→ warp0同步
→ 各lane处理自己的完整DLARFB向量
→ warp0同步
→ 下一块可覆盖T/W
```

所有warp0 lanes都必须到达同步点，包括没有第二个向量的lane。现有warp1等待与外层block同步保持。
reference DORMBR/DORMQR/DORMLQ对 `WORK[0]` 的query/status写入也只能发生在其正确生命周期，不能由各lane共同调用外层函数。
本轮没有实现这些CUDA同步，因此符号检查不提供race/synccheck资格。

常规Q使用r中的reflectors，direct Q使用a中的reflectors；Q完成后score把 `qr_tau` 复用为projection、把 `tau_q`
复用为fitted coordinates。因此Q不能延迟到score之后。延迟P使用 `tau_p`，并按path选择r/a；score-only阶段保留这些输入。
P必须在后续derivative重用a等workspace之前完成。立即P与延迟P都受 `right_vectors_complete` 约束。

延迟完成不只发生在已接受trial：现有ambiguity判断也可能要求在最终接受决定前补完整evaluation。
本原型候选不得改变这一控制规则。当前完整能力计数不能推出立即/延迟P的准确调用次数；本次未新增device计数器。

## 非数值检查结果

工具将封存源码中的 **DCOPY、DGEMM、DTRMM、DLARFB四个函数体原样提取**，以符号类型替代 `doublereal`。
host编译器执行的是表达式建图和整数地址计算，不计算SVD、reflector浮点结果或CPU authority。
每个表达式节点以完整 `(opcode,left_id,right_id)` 精确匹配；哈希表碰撞由键相等检查解决，不以概率hash代替表达式相等。

| 检查 | 结果 |
| --- | ---: |
| q=37/46全部合法root-rank对应augmented shapes | 83 |
| 全部Q/P blocked调用地址案例 | 1,553 |
| C/W整数地址检查 | 1,875,069 |
| 符号形状：direct两端与QR路径的全部块 | 123 |
| dense、sparse-zero、mixed-unit三组模式 | **369 /369 PASS** |
| 比较的C/W单元表达式 | 418,692 |
| 独立向量检查 | 13,977 |
| 符号运算次数，包含两种调用形状 | 16,705,059 |
| 非法三角列重排反例 | 检出 |

每组比较完整DLARFB与逆序lane调度的单向量调用：C/W最终表达式一致，依赖位集不跨向量，V/T不被修改，guard保持，
未定义输入不被用于算术或判断。V1的隐式unit diagonal及无效三角部分、T的无效下三角故意不提供有效符号。
三组标签覆盖本次分支的零/非零、unit/nonunit输入判断，但不是IEEE特殊值数值资格。

整数检查枚举全部合法尺寸；符号执行选择direct尺寸端点及QR路径，不冒称全部真实cohort或全部浮点输入已覆盖。
单block表达式独立加上原block顺序和必要同步，支持源码层面的向量切分论证；编译后的GPU舍入、错误传播和资源行为仍须另测。

准备阶段曾因生成源码缩进与预期不同而未匹配到P分支终点，在创建证据目录和执行符号工具之前失败。
修正为匹配明确的后续分支注释后完成本次审计；没有数值执行失败被隐去或改写为PASS。

## 下一项：隔离资格原型GO，生产集成NOT YET GO

建议后续仅做 **原分块 + 串行DLARFT + 按向量调用原DLARFB** 的隔离GPU原型，保留64线程外层block、warp0内部32lane、
全部原workspace布局、`--fmad=false`、q=37/46、原请求/cohort顺序和32槽。
reference和candidate的数值计算都在GPU上执行；不改变算法、rank/tolerance、line search或CPU fallback策略。

同一窄原型中记录Q/P、DLARFT/DLARFB以及立即/延迟P的计数和device时间，避免把整个stage2当成收益池。
先比较常规/direct、NB=2/32及尾块的GPU输出、guard、info与资源闭环，再覆盖封存L4/L5全部optimizer cohorts：
selected log-SP、coefficients、score、gradient/Hessian、rank/status和全部trajectory必须逐位一致。
然后才用完整层跨度、资源上限和重复outer wall-time决定是否保留。若编译资源或同步代价吞掉收益，明确STOP。

本次不对速度、实际occupancy或GPU利用率作预测，也不重开L8、recover tile或调度重排路线。

工具：[README](../tools/research/optimizer_postback_audit_20261029/README.md)。证据：
`fastkpc/artifacts/full_cuda_ci/optimizer_postback_audit_20261029/attempt1/`，包含原源码与工具snapshot、
`protocol.json`、`inventory.json`、`shapes.tsv`、原样提取的函数、host编译记录、`symbolic.json`、`audit.json`、`manifest.json`。

- audit SHA-256：`f88aa37cb5e33093ca0f8d0da477aaa907b9b4fa9cbaecb5ff6f3c723cea2f22`。
- manifest SHA-256：`0695f2192a90a33bb7505ad00f6fccf22ef5936d3057cbfb5a4eaf5ebfa66076`。
