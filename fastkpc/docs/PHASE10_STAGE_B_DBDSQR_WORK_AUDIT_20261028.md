# Stage B DBDSQR work 生命周期审计：支持隔离验证两段 scratch

日期：2026-10-28。承接 [Stage B 复核](PHASE10_STAGE_B_ATTRIBUTION_REVIEW_20261028.md)，
完成固定 `UPLO=L、NCVT=1、NRU=NCC=0` 调用的只读源码审计。

正式状态：**`STATIC_WORK_LIFETIME_AUDIT_COMPLETE`**。
建议下一项为 **`ISOLATED_FIXED_VT_WORK_PAIR_COMPACTION_QUALIFICATION`**，尚未实施。

结论是：四段 work 在每次 QR sweep 中只有一对被后续消费。正向用前一对，反向用后一对；
删去另一对的无效存储，再将保留的一对映射到相同的两段物理 scratch，有清楚的源码数据流依据。
保守使用 `d/e/q` 各 N 个 double、work 共 2N 个 double，可把 canonical 动态 shared 从
**19,656 B 降到 14,040 B**。在原硬件参数下，shared 资源上界从 **4 提高到 6 blocks/SM**。

这不是 compiled GPU 逐位资格或性能结果：没有修改生产源码/generator/native，没有 CUDA 编译或 GPU 运行，
也没有执行 CPU QR/SVD 数值替代。静态检查只处理地址、符号标签和消费顺序。
accepted opt-in median **329.078 s**、L8 STOP、默认配置及 Phase 10 ACTIVE / candidate NOT_FROZEN /
holdout SEALED / recommended=false 均不变。

## 身份与范围

生产 native 保持 `e8c39703…0ca045`，L3 runtime build 绑定的 **103 个生产输入**全部核对一致。
当前 `fastkpc/build/recorded_dbdsqr.inc` 与该 build 的 generator-output SHA 相同；
从 vendored DBDSQR 按生产 generator 的机械规则在内存中重建，文本也逐字节一致。
没有运行 generator 覆盖生产 include。

本审计区分四个维度，避免把 component DBDSQR 与 optimizer 的 q 混淆：

| 符号 | 含义 | canonical 范围 |
|---|---|---:|
| N | residual 样本行数、外层 shared 布局步长 | 351 |
| G | prepare 得到的不同 residual 值数量，源码 `z.m` | 2–351；不合格输入提前返回 |
| r | DBDSQR 的矩阵维度，调用中的 `r=z.m-1`，函数内的 `*n` | 1–350 |
| p | 每段旋转 scratch 的逻辑长度，函数内 `nm1=r-1` | 0–349 |

DBDSQR 中活动子矩阵记为 `[ll,M]`，以区别外层 G。一次 sweep 写入、随后消费的长度为
`h=M-ll <= p`。scratch 每次从各段的开头使用，与 ll 的绝对位置无关；VT 的目标位置仍按原 ll 偏移。

生产调用见 [`full_cuda_ci_structured_eig.cu`](../src/cuda/full_cuda_ci_structured_eig.cu)：
171–214 行是 bidiagonal，21–64 行是 rotation wrappers，284–347 行是 recover。
下文 DBDSQR 行号对应封存的 `input-source/recorded_dbdsqr.inc`，而非重新生成的未来版本。

## 所有 work 读写

将逻辑四段记为 `W0/W1/W2/W3`，分别对应原一基索引：

```text
W0 = work[1       ... p]
W1 = work[p+1     ... 2p]
W2 = work[2p+1    ... 3p]
W3 = work[3p+1    ... 4p]
```

源码中一共 **18 个 work 存储语句、29 个 work 指针引用**，覆盖全部 **47 个 work[] 引用**。
指针引用来自 14 个 `record_dlasr_` 调用（各两个指针）和一个 `dlasq1_` 调用。
在固定配置下，后者不可达，14 个旋转调用中只有以下四个可达。

| 路径 | 原 W0/W1 存储值 | 原 W2/W3 存储值 | 固定配置真正消费 | 消费顺序 | 源码行号 |
|---|---|---|---|---|---|
| 下双对角转上双对角 | `cs, sn` | 无 | 无；仅 U/C 更新会读，二者均关闭 | 无 | 142、149、156、160 |
| zero shift，正向 | `cs, sn` | `oldcs, oldsn` | W0/W1 | F，按原顺序应用旋转 | 471、483 |
| zero shift，反向 | `cs, -sn` | `oldcs, -oldsn` | W2/W3 | B，按原逆序应用旋转 | 524、536 |
| nonzero shift，正向 | `cosr, sinr` | `cosl, sinl` | W0/W1 | F | 589、599 |
| nonzero shift，反向 | `cosr, -sinr` | `cosl, -sinl` | W2/W3 | B | 649、665 |

`NRU=0、NCC=0` 使另一对的全部消费者不可达；`NCVT=1` 保证 `rotate=true`，不进入需要其他 work
语义的 dqds 路线。正反方向会随活动子块改变，不能永久删除“后半段”，也不能只验证正向 sweep。

每个保留的前缀在本次 sweep 中全部写入，之后同步调用 `record_dlasr_` 消费，才回到下一次 QR 迭代。
该函数不保存 work 指针：`record_drot_` 把 c/s **值**追加到 global rotation log，随后更新 q。
因此不存在上一 sweep 的 work 值在下一 sweep 中继续被读取的依赖，前缀以外的旧值不被消费。

下双对角转换的两项 work 存储没有消费者，但生成 cs/sn 和更新 d/e 的算术仍必须保留。
同样，sweep 中“未消费的 scratch 存储”不等于产生它们的标量计算没有用：
`oldcs/oldsn/cosl/sinl` 等还参与 d/e 的递推，不允许随存储一起删除计算。

## 最窄候选：两段物理 scratch，按原方向消费

保留 `d=local、e=d+N、q=e+N、work=q+N`；仅将 work 可用容量从 4N 改为 2N。
在 DBDSQR 的固定配置专用实现中：

- 正向 sweep 保留原 W0/W1 写入，省去 W2/W3 存储；消费者地址保持 `work[1]、work[r]`。
- 反向 sweep 保留原 W2/W3 的值和符号，写到物理 W0/W1；消费者也映射到 `work[1]、work[r]`。
  `F/B` 参数和旋转应用次序保持原样。
- 下双对角转换的 work 存储可保留，仍落在这两段内；它们不必为了容量一起删除。
  第一版宜避免叠加不必要的清理。
- d/e/q 与 work 始终分离；q 的基址与 `RotationLog::base` 保持一致，记录的索引差不变。
  不引入 cooperative QR，不调整共享常量、shift、收敛判据或排序。

**直接别名四段是错误候选。** 例如 zero-forward 的原写入顺序是 `cs、sn、oldcs、oldsn`。
如果 W2/W3 直接覆盖 W0/W1，消费者读到的将是 `oldcs/oldsn`，而不是要求的 `cs/sn`。
shifted-forward 同样会把 `cosr/sinr` 错换为 `cosl/sinl`。必须同时省去不被消费的存储并重映射正确读写，
不能只改指针或 launch 的 shared 字节数。

单独压缩 work 尾部的 padding 不足以越过五 block 门槛：保留 d/e/q 原布局时，最大逻辑 work 为
`4*(N-2)`，只比现在少 **64 B**。四个逻辑段在不同方向中都可能被使用，不能随意裁掉更多尾部。

## 数值顺序与异常语义

两段映射的静态论证是逐 sweep 的归纳：相同输入 d/e/q 和控制状态，执行相同的标量计算与分支，
向 scratch 写入相同的保留 c/s 位值；同一个 wrapper 按相同顺序读取这些值，对相同 q 位置施加相同旋转。
下一 sweep 的 d/e/q、分支状态及已记录 rotation 序列因此具有相同的数据流。这个论证以原语句及 helper
不变为前提；它不能代替改变源码后对编译器生成代码的 GPU 验证。

后续若做原型，下列内容必须逐项保持并验证：

| 边界 | 本审计结论 / 保留要求 |
|---|---|
| `r=1` | 直接进入符号/排序终结，不访问 work；不得把 p=0 当普通 sweep |
| 二阶活动块 | `dlasv2_` 与直接 `record_drot_`，不使用四段 work；保持原特例 |
| 正反、零/非零 shift | 保留所有四类；反向 sine 的负号及 signed-zero 行为不能简化 |
| 分裂、收敛、非收敛 | 原 `tol/thresh/iter/iterdivn` 与 goto 顺序不变；失败 `info` 仍由原 e 计算 |
| sorting / sign | 保留 `<=` 的 tie 选择、`record_dscal_` 与 `record_dswap_` 次序 |
| rotation capacity / overflow | wrapper、追加顺序、count、overflow 和继续更新 q 的既有行为均不变 |
| 外层 finite / scale / deflation | `w.a/w.b`、norm、scale、live 列表与 `z.info` 的原计算次序不变 |
| 不支持的调用形状 | 这个证明不能扩展到 `NRU>0`、`NCC>0` 或 `NCVT=0`；专用入口必须固定或检查配置 |

保留当前 `--fmad=false` 和相同 toolchain；不得以数值相近替代 d/e/q、rotation log、输出和状态的逐位比较。
删除无效存储可能改变寄存器分配、调度和编译器优化，所以即使源码算术顺序不变，GPU 资格仍是必需的。

## 资源推导与 recover 共存

使用既有 trace 的 102,400 B shared/SM、每 block 1,024 B reserved shared，以及本机 CUDA occupancy
模型的 128 B shared 分配粒度。表中“已分配”是模型计算值：
`round_up(dynamic + reserved + static, 128)`，本 kernel static=0。

| 形状 | dynamic / B | 模型分配 / B | shared 限定 blocks/SM 上界 |
|---|---:|---:|---:|
| 当前 7N | 19,656 | 20,736 | 4 |
| 只去 work padding | 19,592 | 20,736 | 4 |
| 五 block 的临界值 | 19,456 | 20,480 | 5 |
| 临界值再加 1 B | 19,457 | 20,608 | 4 |
| 候选 5N | **14,040** | **15,104** | **6** |
| 原 recover，不变 | 101,088 | 102,144 | 1 |

候选减少 **5,616 B / 28.57%** 的动态 shared。按原寄存器使用量，寄存器限制 16 blocks/SM 不主导；
但候选编译后的寄存器、spill、shared carveout 及实际驻留仍未知，表中不是实际 occupancy。
全卡理论容量从 512 提高到 768 blocks，并不保证 640-component wave 同时全部驻留或自动缩短到某个时间。

recover 的 101,088 B 来自 36 列各自拥有 351 个 FP64 元素。每列由一个线程维护自己的向量，
这些列缓冲同时活跃，不能简单覆盖复用。旋转 log 在 recover 中按逆序应用，各向量内部的顺序必须保留。
本轮不改变 tile、recover block 数或 launch 次序；recover 仍无法与一个 bidiagonal block 同驻一个 SM，
所以提高 bidiagonal 上界后可能改变原重叠与资源竞争。

前一轮 trace 的正确读法继续保留：普通 wave 是 **8 个早启动、2 个晚启动**，所有这些 wave 都出现过
10 个 bidiagonal kernel 区间重叠；不是两轮互不重叠的执行。普通晚启动约 223–347 ms，
540 ms 属于 L2 首次执行。launch 区间不能证明 block 的驻留位置，不能将某段晚启动直接作为可兑现收益。
本审计没有新增 trace，也未采集受权限限制的 occupancy/stall 计数器。

## 机器检查与限制

工具仅做源码专用的 inventory 和整数/符号验证，不是通用 C++ 证明器。人工阅读完整的 762 行 generated
DBDSQR，检查 early return、两个方向、零/非零 shift、非收敛和终结路径后，再用固定源码断言绑定模型。

- **244,300** 个地址界限组合：r=2–350、每个活动前缀长度、四种分支；r=1 单独按不访问 work 处理。
  长度 1 也纳入保守索引检查，不宣称二阶特例实际执行了 sweep。
- **1,396** 个符号 sweep、**488,600** 次符号操作数读取：原四段与候选两段的值标签、符号和消费次序一致。
- 六次连续方向/宽度切换使用同一未清空 scratch，确认模型的消费前缀由当前 sweep 覆盖，未读旧值。
- 两个 forward 分支均给出盲目 alias 的反例，阻止将“可复用”误写成“可直接共享地址”。
- 生产 native、103 个输入、generated include、生产 generator 与 vendored 源码身份检查通过。

这些计数不是 GPU 数值样本数，不证明 rounding、SASS、编译器行为、内存安全运行结果或性能。
没有调用 CPU 数值 authority，也没有替换 GPU 路线。

## 下一步边界

审计支持一项单独的隔离资格实验：固定 VT 的 **work 两段压缩**，保留 d/e/q、scalar QR、rotation wrappers、
recover、lane 数、wave、status 同步及 cache 策略。先完成 GPU d/e/q、rotation log 全字段/顺序/count、
info、a/b、scale/live 与最终 component 三字段逐位比较，并覆盖小维度、重复值、正反方向、零/非零 shift、
符号/tie、错误与 rotation-capacity 边界；既有 104,797 captured components 用于真实回放覆盖。
没有自然触达的分支要显式报告，不能用整体 p-value 一致替代中间语义资格。

只有正确性、边界和资源通过后，才按预先固定协议比较完整 L2/L3 Stage B 的重复时间，包含 recover、
status/checkpoint 和 cache stores；单个 bidiagonal 变快或理论 blocks 增加都不构成完整收益。
之后才讨论生产 opt-in 和完整方法 hard gates/重复 outer 性能。所有原型都必须通过逐位资格，
“需要逐位资格”不是禁止隔离验证的理由；本轮仍止于静态审计，没有实现或运行原型。

## 复现与归档

工具：[`stage_b_dbdsqr_work_audit_20261028/`](../tools/research/stage_b_dbdsqr_work_audit_20261028/)。
证据：[`stage_b_dbdsqr_work_audit_20261028/`](../artifacts/full_cuda_ci/stage_b_dbdsqr_work_audit_20261028/)。
`work-accesses.json` 保存每条 store/consumer 的源码行与参数；`audit.json` 保存模型结果；
`input-source/` 保存当前 kernel、generated DBDSQR、generator 和 CUDA occupancy header。
旧 activity artifact 不变，新的 manifest 不会重标记旧数据为新的 GPU 运行。

```bash
python3 fastkpc/tools/research/stage_b_dbdsqr_work_audit_20261028/audit.py NEW_OUTPUT_DIRECTORY
```

| 对象 | SHA-256 |
|---|---|
| production native | `e8c39703f01e09260d330167b49267c295aa2221167d5248d0980a97c80ca045` |
| generated DBDSQR | `be76c5e06ce8bce4013a9759a3a54fc63feed9e421a1d3d44af77e11f32f53b9` |
| audit.json | `2b9a2ff10e595596dd98934253c88997f7647cc8c0898d1edf104bb59a1a5ca8` |
| manifest.json | `f4611c32be6b9a7bea669d7f84193ea95108587dc434edd325a781ef6044d9da` |

本报告和路线图在归档后编写，不属于上表 manifest 的封存文件集合。没有 commit 或 push。
