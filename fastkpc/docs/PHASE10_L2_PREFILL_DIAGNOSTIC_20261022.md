# L2 single-penalty prefill：隔离资格诊断

日期：2026-10-22。按
[canonical 复核建议](PHASE10_STRUCTURED_PREFILL_CANONICAL_REVIEW_20261022.md)
执行真实 L2 captured-cohort 诊断。生产源码、native、scheduler、默认开关均未修改。

```text
原 cohort 的 exact fixed 重放                  PASS（87,110 target occurrences）
重新分组的 coefficient / route / rank / info   EXACT
L2 全部 CI 与冻结 full-trace pMax 数值门        PASS（125,293 个 L2 CI）
dense / structured 各两次 p-value 确定性       BITWISE
隔离 structured component 收益               GO
下一步受限 opt-in 完整 runtime 资格            GO
完整 runtime L2 集成 / outer 收益              尚未验证
Phase 10 / candidate / holdout / recommended   ACTIVE / NOT_FROZEN / SEALED / false
```

本轮支持继续做受限集成资格，不代表 L2 已进入生产路线。阶段 B 约 55 秒的设想
得到支持；阶段 A 的 5–10 秒假设应修正为本次隔离重放循环约 **14.8 秒**。
文档中的完整方法约节省 125 秒仍是情景推算。

## 真实输入与数值归因

外部 observer 执行 canonical 351×48 的 L0–L2 前缀，并与原 canonical off 回执
核对。全部 180,165 个 CI、pMax、任务/删除身份、前缀 adjacency、counts、RNG
及生产 GPU-only 门通过。捕获增加了显式诊断 D2H，故捕获耗时不作性能证据。

L2 实际有 1,126 个 S、49,919 个 planned targets；42,685 个 unique components
最终消费，7,234 个未消费。原路径执行 9,668 个 residual cohorts、87,110 次 target
求解。这里的 single-penalty shape 全部是 **q=30**。

首先按原 cohort 调用生产 exact fixed evaluator：87,110 次求解的 coefficients、
fitted、residual 均逐位重现，route/rank/info 完全相同。然后固定原 selected log-SP
位模式，每个 S 的 distinct targets 均衡分成最多 16 列的 cohort，prepared handle
在该 S 内复用，得到 3,356 个 cohorts、49,919 次求解，并 D2D detach 到有界 arena。

| 对原始 target occurrences 的比较 | 数量 | fitted/residual 变化 | 最大绝对差 |
|---|---:|---:|---:|
| 原宽度 ≤16 | 63,005 | 0 | 0 |
| 原宽度 ≥17 | 24,105 | 24,105 | 1.06581410364015e-14 |

所有已消费 target 的 coefficients 及 route/rank/info 均保持 exact。
按首次消费的 unique component 计，变化的是 **18,539 / 42,685**；24,105 是
求解次数，不能当作 unique component 数。未消费的 7,234 个 planned targets
没有原 consumer 输出可比，不把它们算作原输出逐位资格。

输入 X、selected-SP 和 coefficients 均不变，差异从生产
`cublasDgemm(n, target_count, q)` 的 fitted materialization 开始，并完全落在原宽
cohort；residual 随之发生低位变化。该证据定位了 materialization 边界，没有
证明任意其他 shape 都能按同样门限重分组。

## component、原 Gram 与 GPU p-value

probe 使用原生产 component lifecycle、structured 对象及提取的原 pair/gamma
GPU 调用，保留 22,720 个 consumer groups、9,668 个 Gram groups 的原键和顺序。
为了隔离数值工作，研究工具对 L2 fixtures 调用既有 level-3 lifecycle；没有放开
生产 L2 入口。原 residual + dense 的 125,293 个 physical p 全部逐位重现原 native。

| 模式 | 相对原路线变化的 L2 p 数量 | 最大变化 | 对独立 authority 最大误差 |
|---|---:|---:|---:|
| 原 residual + dense | 0 | 0 | 3.05475089668050e-12 |
| 新 residual + dense | 86,175 | 3.41393580072236e-15 | 3.05475089668050e-12 |
| 新 residual + structured | 125,078 | 1.05471187339390e-14 | 3.05425129631942e-12 |

全部 125,293 个 L2 CI 通过 `1e-10`，决策翻转为零。authority p∈[0.05,0.15]
的 527 个测试全部通过；structured 下它们都发生了低位变化，最大 authority
误差仍为 3.05425129631942e-12。structured 对新 dense 的最大差为
1.06581410364015e-14。同模式的两次 logical p 各自逐位一致。

将新 L2 p 放回已归档的完整 structured trace、保持其他层 p 冻结后，pMax 也通过
`1e-10`。dense/structured 各改变 622/732 个 pMax 单元，最大 authority 误差分别
为 3.05475089668050e-12 / 3.05425129631942e-12。完整冻结 trace 的决定不变。
这是控制重放，不是新的完整 skeleton、orientation 或 PDAG 运行。

任务映射必须保留 `(x,y,S)` 的方向。初版 post-check 把 reciprocal CI 合并，
在原始 anchor 上就失败；原路线这类计算自身存在最大约 2.637e-16 的差异。
现已从原 consumer 恢复全部 125,293 个唯一有向身份，再与 logical trace 对齐。
另一处 post-check 修正了 pMax finite-mask 的 dimnames 比较；矩阵形状、非有限值
位置/位模式及有限 pMax 门均保留。GPU 输出没有重跑或改写，拒绝日志一并归档。

## 隔离计时

GPU0 顺序运行，GPU1 空闲；32 CUDA connections、数学库线程数 1、同一 native。
component 顺序为 original → dense1 → structured1 → structured2 → dense2。

阶段 A 另做两次无 shadow 重放，driver wall 为 14.817704 / 14.768355 秒，
中位数 14.793030 秒。API 分项中位数如下，单位秒：

| prepared create | exact evaluate | residual detach | prepared release | 分项合计 |
|---:|---:|---:|---:|---:|
| 1.747274 | 12.493646 | 0.028424 | 0.439372 | 14.708716 |

driver wall 包含 Python 循环与 arena 销毁，不含输入读取及 arena 初次分配。
原 cohort 诊断的 prepared create / evaluate 合计分别为 14.554 / 34.593 秒；
原诊断未记录释放，不能将这两项之和称为原完整阶段 A。

| component 重放 | 生产分量数 | Stage B 秒 | 原 Gram/pair 重放秒 | 两项合计秒 |
|---|---:|---:|---:|---:|
| original dense anchor | 42,685 | 123.286786 | 4.025401 | 127.312187 |
| regrouped dense1 | 49,919 | 144.279717 | 4.014299 | 148.294016 |
| regrouped structured1 | 49,919 | 54.295481 | 4.011121 | 58.306602 |
| regrouped structured2 | 49,919 | 54.622801 | 4.043307 | 58.666108 |
| regrouped dense2 | 49,919 | 144.400632 | 4.008917 | 148.409549 |
| regrouped dense 两次中位数 | 49,919 | **144.340175** | 4.011608 | **148.351783** |
| regrouped structured 两次中位数 | 49,919 | **54.459141** | 4.027214 | **58.486355** |

相同 49,919 个分量的 Stage B 约 **2.65×**、减少 **62.3%**，两次差距均远大于
同模式波动。多算 7,234 个未消费分量的成本已经包含在候选里；original 行分量数
不同，不能当作相同 workload 的 dense/structured 对照。

Stage B 是完成后的 host wall；Gram/pair 重放包含 Python 调用成本。
两项合计不含 fixture 上传、pool 建立/残差存储与 teardown。不同阶段不能直接
相加冒充完整 L2 或 outer wall time。本轮没有重跑约 691 秒的完整方法。

## 回退、容量和资源

两次 structured 均为 49,919 accepted、0 natural dense fallback、78 waves、最大
640 在途。minimum gap 为 4.118713192092871e-4；maximum rotations 为 147,167，
容量 170,662，余量约 13.8%，小于此前仅深层 canonical 的约 22%。
pool bytes 为 3,422,507,520。

两次 structured 整卡 200-ms 采样峰值均为 15,701 MiB；dense 为 12,393 MiB。
GPU1 全程采样利用率 0%，显存 425 MiB。这是隔离 probe 峰值，既不是瞬时上界，
也不能替代包含其他层 cache、optimizer/workspace 的完整运行容量资格。

每次重放的 vertical resource counters 闭合。compute-sanitizer 覆盖 3 个原 fixed
cohort、3 个候选 S，以及 137 个 structured 分量/828 个 pairs，均为
`0 errors / 0 bytes leaked`；不是全 campaign memcheck。所有数值生产都在 GPU，
CPU 仅索引、比较已归档 authority 和重放图/pMax 控制；诊断的 fitted/residual
大 D2H 明确存在，不能把它标为生产零 D2H 性能运行。

## 后续门与证据

下一步限于受限、默认关闭的 L2 runtime opt-in：保留当前 exact producer、原 Gram
和 canonical 消费顺序，有界生命周期，测量阶段 A 所有成本。随后做完整 default-Inf
off/on 资格及交错重复，逐项验证 240,498 个 CI、pMax、任务/删除/图/sepsets/counts、
RNG、生产 authority/fallback/大 D2H、资源和整轮显存。只有完整 outer 收益明确才
接受集成；不能提前把本次 isolated 差值写入生产预算。

- native SHA：`89a2fab0bd02b695878556a74694fc522cddcbe2c5c2f22f74c7e4da6d5d56c3`。
- 生产源码逐项 SHA 与上一轮完整 source snapshot 相同；本轮没有新的 runtime build。
- 工具：`fastkpc/tools/research/l2_prefill_20261022/`，含复现 README。
- 证据：`fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022/`。
- `closure.json`、`qualification.json`、`residual-attribution.json`、`timing.csv` 保存机器门与归因。
- `manifest.json` 绑定 raw capture、原/新 fixed 输出、GPU pair 输出、回执、工具与历史 authority。
- manifest SHA：`1af2e8c18a97483753a830d4379fa45b73440f0421023333c35e70e406440d53`。

observer recipe 在捕获后补齐 probe 构建分支；重新构建的 capture.so 与实际捕获
二进制 SHA 完全相同。fixed 数值资格之后补加释放/driver 计时，分别保留在
`fixed.json` / `fixed-timing.json`，不伪装为同一运行或 clean producer evidence。
没有新的真实数据、WAN-PDAG、promotion 或 sealed holdout campaign。
