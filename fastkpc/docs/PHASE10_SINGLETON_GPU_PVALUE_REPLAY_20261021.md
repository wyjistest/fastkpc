# singleton self_moment：完整受影响 CI 的 GPU 差分重放

日期：2026-10-21。承接
[singleton 分析](PHASE10_SINGLETON_SELF_MOMENT_ANALYSIS_20261021.md)的评审，
执行范围仅为：保存候选 scalar 比特，复现原 GPU pair 输出，保持原 pair 分组，
只替换 self_moment，测量所有受影响 CI。生产实现和验收门均未修改。

## 决策

```text
原 accepted-native 捕获的完整方法回归              PASS
原 payload / pair-cross / compact / canonical p    全部逐位复现
只替换 self_moment 的 GPU 差分诊断                  PASS_GPU_DIAGNOSTIC
受影响 CI                                         1,186
其中 p 比特改变                                    1,144
最大 |Δp|                                         3.0531133177191805e-16
alpha = 0.1 decision flip                          0
原四字段 payload 门                               STOP_PAYLOAD_BITS，保持
LSCP / 新 Gram / 验收例外                          未集成，未采用
```

本次测量证明，在此 351×22 default-Inf 输入的实际消费序列上，singleton-miss
self_moment 的差异只造成极小的 GPU p 低位变化。该结果支持后续方案评审，
不能自动解除四字段逐位门，也不是对 kpcalg 的新资格或完整 LSCP 运行。

## 身份与覆盖

accepted runtime：`dcb1749`。native 保持：

```text
fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50
```

仍使用 GPU 0、connections=32、11 GiB component cache、residual bypass、8 eig
lanes，数学库线程数 1。独立探针构建于临时目录；`fastkpc/src` 和 `fastkpc/R`
没有本次修改。所有 residual、component、pair、gamma 数值均由 GPU 生成。
CPU 仅做身份映射、字节比较、误差统计和归档。

1. 复用前一次经过原 miss cohort 逐位资格的 GPU payload probe，按自然序、
   反序、自然序重复生成 22 宽 payload，共 19,896 次 component 比较。其余
   三字段始终逐位相同，候选 self_moment 三次逐位一致；保存全部 6,632 个
   原值与候选值的 binary64 比特和 key，改变集合仍为原来的 1,002 个条目。
2. 对 accepted native 的共同 legacy-eig 消费入口进行诊断捕获，覆盖 exact
   residual 与后续 all-hit 路径。共记录 4,949 个实际批次、24,264 个物理 pair；
   本轮逻辑 CI 也是 24,264，compact-result cache hit 为 0。
3. 捕获其中 1,083 个包含改变分量的批次，共 1,276 个 pair；包含 75 个 all-hit
   批次。保留各自原 pair count（1–5）、顺序、component keys、实际四字段
   payload、pair-cross 矩阵和 72-byte compact 输出。
4. 通过原 hash 输入中的 prepared / target / residual 身份，把物理 pair 精确
   映射到 canonical `(S, x, y)`。不依赖 p 值相似度推断对应关系。

捕获运行与已有未插桩基线的全部 p/pMax、task/deletion、adjacency、sepsets、
n.edgetests、RNG endpoint 一致；native authority/fallback 与资源门均通过。
探针执行的 payload/pair-cross D2H 明确属于诊断，未计入原生 D2H counters；
此次运行不能作为生产无大块 D2H 或性能证据。

## GPU 对照方法

独立 replay 直接包含生产 CUDA 源码。构建时从原函数逐字提取 pair GEMM 与
pair/gamma 调用段，记录源码、提取段、binary 与 native SHA。

每个实际批次先上传原四字段 payload，使用同一 pedantic / no-atomics BLAS
配置和原 pair count/order 执行。要求：

- pair-cross 的每个字节等于原 native 捕获；
- 全部 compact 字段，包括 p、statistic、mean、variance、shape、scale、status、
  independent、iterations、flags 与 sequence id，逐位相同；
- p 同时与 canonical task receipt 的原 binary64 比特相同。

只有以上通过，才把 payload 最后一个 scalar 替换为对应候选 self_moment。
variance 继续由生产核按 `factor × sm_left × sm_right / n²` 计算，gamma 使用
同一 GPU `regularized_gamma_q`；没有 CPU gamma 或比例缩放替代。

候选每批执行三次，全部输出逐位重复。pair-cross、statistic、mean 均与基线
逐位相同；90 个同批但未引用改变分量的 pair，其全部 compact 字段也不变。

## 实测结果

| 范围 | CI 数 | p 比特改变 | 最大 \|Δp\| | alpha 翻转 |
|---|---:|---:|---:|---:|
| 全部受影响 | 1,186 | 1,144 | 3.0531133177191805e-16 | 0 |
| 基线 p ∈ [0.08, 0.12] | 6 | 5 | 3.0531133177191805e-16 | 0 |
| 基线 p ∈ [0.05, 0.15] | 12 | 11 | 3.0531133177191805e-16 | 0 |
| 后续 all-hit 消费中的受影响 CI | 83 | 81 | 4.163336342344337e-17 | 0 |
| 同批未受影响对照 | 90 | 0 | 0 | 0 |

受影响 CI 按 conditioning level 为 3/4/5：863 / 309 / 14。所有 1,002 个改变
条目均被覆盖。GPU gamma status 失败为 0；10 个 CI 的 gamma 迭代次数发生
变化，但均正常收敛。最接近 alpha 的受影响基线 p 距离 0.1 仍为
`0.0030326051105633994`，因此本次没有几乎贴着阈值的样本。

最大差 witness：

```text
canonical order   18887
physical id       21316
x / y             18 / 2
S                 {4, 6, 9}
baseline p        0.09305598689761131
candidate p       0.09305598689761100
signed Δp         -3.0531133177191805e-16
status            OK / OK
```

补齐全部候选 scalar 后，逐条配对的最大 self_moment 相对差为
`3.4668370342193376e-15`，最大 27 ULP。之前 `1.73e-14` 是用全局最大绝对差
除以最小原值形成的保守界；20 个旧 witness 的范围为 1–14 ULP。

## 边界与后续

本次是固定原 task 序列的 scalar-only GPU 诊断。归档的全 task p 数组中，
未受影响位置沿用基线；没有声称用新实现重新执行全部 24,264 个 CI。
尚未执行新的 LSCP 调度、生产 Gram、pMax 或最终 PDAG 资格，也没有新增
kpcalg authority 比较。因此该样本上的极小差分不是任意输入的误差上界。

原 payload 位差仍然存在。若继续设计，可评审：保留原消费 miss 分组、只提前
生产 eig 并延后 Gram/self_moment；或者独立资格新的批宽无关 Gram；或者明确
修订局部验收门。不能仅凭本次结果自动选择其中任一项。Phase 10、candidate、
recommended route 和 sealed holdout 的状态保持不变。

## 资源与复现

原 native 捕获、候选 scalar 导出、完整 GPU pair replay 的资源计数均闭合。
Compute Sanitizer memcheck（含 full leak check）覆盖 19 个批次、51 个 pair，
包含所有实际 pair count 1–5 与 all-hit 消费：0 errors、0 bytes leaked / 0
allocations，退出码 0。

工具：[singleton_pvalue_20261021/README.md](../tools/research/singleton_pvalue_20261021/README.md)。
证据目录：`fastkpc/artifacts/full_cuda_ci/singleton_pvalue_20261021/`。
依赖上一轮已归档 payload campaign；完整比特、逐 pair 结果、source/binary SHA
与实际消费请求均保留。所有计时仅为诊断耗时，不作性能或 promotion 证据。

归档审计 `PASS`，原始捕获逐文件 SHA 与压缩包逐项回读校验通过。
[归档 manifest](../artifacts/full_cuda_ci/singleton_pvalue_20261021/manifest.json)
SHA-256：

```text
352c47bee2ee0505ab6c53fd8e00ff847bb15b059241cbc571efa0fc96964d60
```
