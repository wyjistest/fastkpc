# 输入尺寸通用化与开发方向纠正

日期：2026-11-02。用户要求纠正以 351×48 为生产优化适用条件的做法。
该尺寸保留为一个回归样本；后续生产调度不使用数据集身份、固定行列数或从
单个样本观察到的批次白名单决定优化资格。历史性能记录保留原输入与 native
身份，不外推为任意输入的性能保证。

2026-11-06 后续：[numCol 参数通用化](PHASE10_NUMCOL_GENERALIZATION_20261106.md)
另行移除固定 35 的接口限制。本文件的历史运行、参数和 native 身份保持原样，
不将这里使用 k35 的 500 行结果重标为默认 k50 的验证。

## 本轮实现

- L3–L6 reusable slot 的 `n` 来自实际 setup，target capacity 来自当前层请求。
  X/QR 拷贝、prepared 创建与 rebind 校验使用这些尺寸。移除历史目标数白名单。
- 保留由 additive 模型决定的 `q=1+9|S|`、penalty/rank 合同和 small64 后端容量；
  它们是数值模型与内核限制，不是输入矩阵必须等于某个 benchmark 的限制。
- structured 能力条件与内核共享：`6<=n<=1024`、`1<=k<=64`、
  `2(k+1)+4<=n`。当前 `k=35` 时为 `76<=n<=1024`；范围外使用 dense GPU
  component producer。requested/effective 状态分别记录。
- 整层 component cache 或 residual arena 预算不足时，保留普通 exact GPU
  消费路线，并记录 capacity skip。L2 也支持 dense GPU prefill。
- eig lanes 使用实际配置 `min(requested,p)`；移除 prefill 必须恰好有八条
  effective lanes 的隐藏少列限制。
- optimizer pool 根据实测分配量、既有创建预算与 2 GiB 余量限制 slot 数量，
  上限仍为 32。structured rotation log 使用 `ceil(1.5*n*n)` 条记录的容量策略，
  pool 上限 3 GiB/10 contexts。这是存储预算，不是算法收敛上界；overflow
  继续使用既有 dense GPU fallback。第一版固定 4 MiB 配额已被跨尺寸测试否定。

统计模型没有改变。严格接口目前仍要求 `alpha=0.1`、`index=1`、`numCol=35`，
有限 double 矩阵、`2<=p<=64`，以及实际到达的 additive 模型 `q<n`。
特别是 n=500 时，不能把默认 `numCol=floor(n/10)=50` 悄悄改成 35 后宣称
与用户原参数等价。本轮处理尺寸绑定，未扩展这些参数的数值资格。

## 验收协议

[工具](../tools/research/general_shapes_20261102/)在运行前冻结输入生成种子、
模式顺序和判据。覆盖三套完整真实输入、公开 500×50 合成输入，以及
500×8 深层搜索、75/76 行 structured 边界、1025 行 dense 路线、256×12 分块输入。
另增加 75×6、76×6、1025×4 三个稠密边界输入，要求实际 produced components>0，
避免只有 dispatch 标记但搜索过早停止、未执行 producer 的空覆盖。
全部完整搜索使用 `m.max=Inf`，保持相同统计参数。

每个输入 fresh process、空 dataset cache，顺序为 A1/C1/C2/A2/A3/C3。
A 为同一 candidate native 的通用 exact/retained 路线，C 启用参数化后的
structured/slots/postback 组合。每个输入分别报告中位数；不汇总成单一加速比。
预先声明不回退门为 C/A<=1.05，速度门为 C/A<=0.95；数值和行为门独立判定。

每轮比较全部逻辑 CI 和有限 pMax（绝对误差 <=1e-10）、任务/删边顺序、
adjacency、sepsets、n.edgetests、RNG、数值 authority、cache metadata 和资源释放。
三套真实输入及公开输入复用已认证 CPU oracle，新增输入现场生成独立 CPU
reference。前三套真实输入及公开 500×50 另在 C1 比较一次完整定向 trace、
p-value、最终 PDAG 和 RNG；其余新增合成输入仅声明 skeleton 资格。
定向代码未改变，仍使用 CPU kpcalg，定向耗时不计入 skeleton 对照。

另用多尺寸 GPU slot 资格检查较大的 target capacity、两份 geometry 交替 rebind、
selected log-SP/coefficients/fitted/residual 位模式、route/rank/info、live-lease
拒绝及失败后 generation 不变。主机测试覆盖 192 个合法 geometry、1,152 个
非法 geometry 与 structured 能力边界。

## 当前结果

已完成审计并安装本机开发版本；native 为
`476ad65e648775d52e7c58e69213269c2c7b05efd27c32ec2834baaaa5e5a37b`。
候选复用已认证父对象，按原 TU 数值编译参数重编七个受影响对象；第二版
再重编涉及 rotation 配额与 structured 公共头的三个对象，第三版修正少列入口，
第四版仅重编 one-call scheduler，修正 capacity skip 时 slots 的选择。
本轮不重标历史 receipts，不解封 holdout，不提升 Phase 10 推荐状态。

`campaign4` 的 **12 个输入、72 次 fresh-process 运行全部通过**，合计
**50,610 次 CI 比较**；最大 CI/pMax 绝对误差 **8.68440042101071e-12**。
每轮 skeleton 数值、图、顺序、RNG、authority、cache metadata 和资源门均通过。
前三套真实输入及公开 500×50 的最终 PDAG/定向检查通过。其余合成输入只声明
skeleton 资格。下面是每个输入自己的 OFF/ON 三次中位数，单位秒：

| 输入 | 每轮 CI | OFF | ON | 耗时变化 | 5% 速度门 |
| --- | ---: | ---: | ---: | ---: | --- |
| 179×8 真实 | 338 | 1.081 | 1.086 | +0.46% | 未通过 |
| 243×9 真实 | 413 | 1.779 | 1.692 | −4.89% | 未通过 |
| 300×10 真实 | 657 | 3.086 | 2.573 | −16.62% | 通过 |
| 500×50 公开合成 | 2,722 | 10.010 | 10.390 | +3.80% | 未通过 |
| 500×8 稠密合成，至 L6 | 3,559 | 12.906 | 10.088 | −21.83% | 通过 |
| 75×8 稀疏合成 | 45 | 0.249 | 0.242 | −2.81% | 未通过 |
| 76×8 稀疏合成 | 50 | 0.255 | 0.248 | −2.75% | 未通过 |
| 1025×6 稀疏合成 | 38 | 0.998 | 0.988 | −1.00% | 未通过 |
| 256×12 分块合成 | 98 | 0.379 | 0.373 | −1.58% | 未通过 |
| 75×6 稠密边界 | 226 | 0.475 | 0.457 | −3.79% | 未通过 |
| 76×6 稠密边界 | 241 | 0.457 | 0.440 | −3.72% | 未通过 |
| 1025×4 稠密边界 | 48 | 1.404 | 1.389 | −1.07% | 未通过 |

12 个输入均满足事先声明的 C/A<=1.05 不回退门，但这不等于各输入都更快。
公开 500×50 的 ON 三次均比 OFF 慢，观察到约 3.8% 回退；不为它登记速度收益。
亚秒输入的百分比保留原值，不据此宣称稳定提速。300×10 和稠密 500×8 通过
速度门；其余输入未通过。优化组合继续显式开启，默认配置未改变。

实际 producer 覆盖：75×6 以 dense GPU 产生 60 个 components；76×6 的 60 个
components 全部由 structured 成功产生；1025×4 以 dense GPU 产生 12 个。
500×50 的 160 个 structured components 全通过，rotation overflow 为零。

机器复核为 `PASS_GENERAL_SHAPE_QUALIFICATION`；107 个生产文件、44 个 linked
objects 与安装 native 已绑定。旧 `b2b79fc6…e135b12b` 及对象保留于原封存目录。
原始数据和大 receipts 留在本地；这仍是 incremental source-closure 开发证据，
不是 clean-checkout promotion，也没有形成任意输入均更快的保证。

本地证据：[审计](../artifacts/full_cuda_ci/general_shapes_20261102/audit.json)、
[安装](../artifacts/full_cuda_ci/general_shapes_20261102/installation.json)、
[封存索引](../artifacts/full_cuda_ci/general_shapes_20261102/manifest.json)。

已完成的最终版本补充资格：

- 351×48 单次完整回归 **291.429 秒**：240,498 CI/pMax 与原合格 GPU 结果逐位
  一致，CPU authority 最大误差 **3.05425129631942e-12**；任务/图/sepsets/counts、
  最终 PDAG/定向 trace/RNG、数值 authority 与资源门均通过。该耗时不是重复中位数。
- L3/L4/L5/L6 各创建/释放 32 slots；最低空闲显存检查点分别为
  3,824,287,744 / 3,081,895,936 / 2,754,740,224 / 2,649,882,624 bytes。
  全部高于 2 GiB；这不是连续峰值证明。容量策略变化后不沿用旧版 699.125 MiB 余量。
- slot 直接资格 **80 组、7,808 个目标逐位一致、640 个非法复用拒绝通过**。
  原始 receipt 保留 `a0db10dc…ec2ef708`。最终版本与它的 43 个非 one-call linked
  objects 完全一致，单独封存 object bridge；改变的 scheduler 在最终版本重新验收。
- 7 组配置测试文件通过，覆盖 p=2/6/21/23、n=350、实际 lanes、默认关闭、
  非法依赖与资源释放。另有 192 个合法/1,152 个非法 geometry 主机检查通过。
- `dcc.perm`、`hsic.gamma`、`hsic.perm` 的 15 组跨 native 回归通过，包含 p-value、
  pMax、图、route/status、RNG 和资源检查；不是新增这些方法的完整性能 campaign。
- 1 MiB component cache 的 300×10 输入触发 **3 次 prefill capacity skip**，
  L3 slot 创建数为 0；657 CI 全通过，最大误差 **4.76063632959267e-13**。

以上对象映射及原始日志位于 `general_shapes_20261102/supplement6/`。

本轮作用于公开 `full_cuda` one-call 路由。历史 prepared-DTO schema 与捕获回放
夹具保留其原资格范围，不作为新的通用输入接口；其旧 benchmark 常量不参与
本轮 one-call 的优化选路。统计参数范围仍按前述合同，后续扩展需独立资格。

`campaign1` 的首轮 OFF 在数值、trace、sepsets、counts、PDAG 和资源门通过后，
因 CPU scheduler reference 未携带输入 gene labels 而被 name-aligning 图比较器
拒绝。夹具按原始输入列名补齐 reference dimnames；未改变顶点排列、生产计算
或门槛。失败记录保留，`campaign2` 从头执行新冻结协议。

`campaign2` 的前三套真实输入全部通过，但 public500 的 ON 中位数
10.995 秒高于 OFF 的 10.031 秒，超过预声明 5% 不回退门。160 个 structured
component 全部触发 174,762 条记录的日志容量上限，然后重新执行 dense GPU。
数值门通过，性能失败保留；在 dense500-A1 已完成 skeleton 后停止余下运行，
未将其未完成定向声明为 PASS。第二版按输入规模分配日志，并以独立
`campaign3` 协议重跑。公开 500×50 首轮已确认 160/160 structured 成功、
零 rotation overflow；实际日志最大 220,410 条，证明原固定配额不足。

`campaign3` 在七个输入完成六次重复后，于 large1025-C1 被旧的八条 effective
lanes 检查拒绝，未发生该轮数值计算。cache 已正确将请求八条限制为 p=6，
错误发生在 prefill 入口验证；第三版修正该限制并重跑独立协议。
同时补充真正进入 component prefill 的三个稠密边界输入。

小缓存测试最初使用低于原合同下限的 256 KiB，被正常拒绝；改用合法的 1 MiB
后发现实际 scheduler 问题：整层 prefill 暂停但仍使用要求 Stage A 再绑定的
slots，最终触发生命周期检查。第四版把 slot 使用与当前层 prefill 绑定，保留
原生命周期检查，并通过上述容量跳过测试。诊断夹具的编译 include、参数路径
及未执行检查中的字段修正也保留独立失败日志；未修改既有数值容差。
