# Structured eig：device 三字段适配器与原 GPU consumer 资格

日期：2026-10-22。执行
[canonical prefill 评审](PHASE10_LEVEL_COMPONENT_PREFILL_CANONICAL_REVIEW_20261022.md)
之后的窄工作。范围为 level≥3 的 captured residual 重放，保持既有 delayed
Gram 分组；尚未接入完整 one-call runtime。

## 状态

```text
device residual → structured 三字段 → 原 GPU consumer     PASS（本次样本）
批宽 / 位置 / 顺序 / 十 context 并发逐位重现              PASS
逐分量 GPU dense fallback / 错误后拒绝发布                 PASS
sanitizer 资源门                                           PASS（0 errors / 0 leaked bytes）
22 列 / canonical 完整 structured 方法                    NOT_RUN
production default / Phase 10 / candidate / holdout         不变
```

数值生产全部在 GPU 上。Python 只做调用、字节比较及误差计算；R 只读取已经归档
的 kpcalg authority。没有新的 CPU 数值求解或 CPU fallback。

生产源码与 `fastkpc/build/fastkpc_cuda.so` 未改变。当前 native SHA 为：

```text
b1e7eab4297a13b7ff29f0bf40b2fc3ce4d6950542dff4318ee3e2dc76694ee9
```

这是前次 opt-in dense prefill 的 source-closure binary，不应改称 clean producer
或 promotion candidate。本轮另编译独立 research library，未新增生产开关。

## 实际接通的边界

```text
已认证的 GPU residual 字节，一次载入 fixture device arena
→ 仅传 target indices，在 device 上 gather
→ structured prepare / bidiagonal QR / roots / recover，求 k+1 个候选
→ 每个有界 batch 一次 compact status checkpoint
→ 原 GPU centering 归约语句
→ D2D 三字段 cache payload
→ completion event
→ 固定的原消费分组 / delayed Gram / self_moment
→ 原 GPU pair GEMM / gamma
→ compact result
```

`total` 使用原 distance-row / total reduction，未由 structured mean 乘 n²
重建。centering 和 pair-stage 语句由构建脚本从当前生产源码原样提取；dense
eig、delayed Gram、自矩归约和 gamma kernel 直接共享生产 CUDA 源码。
structured TU 保留原研究实现的 `--fmad=false`，consumer TU 保持生产编译设置。

正常数据流没有 residual / component 大量 D2H。资格工具另有明确的 payload
shadow D2H；这些下载只用于审计，不供 pair/gamma 数值生产使用。本原型包含
bounded-batch metadata host checkpoint，不宣称完全无 host wait。

research entry 单独记录 `component_solver_identity`：

- `structured-device-three-field-v1`；
- `retained-dense-three-field-v1`。

这与 residual 的 stable-SVD route/rank/info 分开。本轮只复用生产 cache 的
payload 存储布局，并在 research wrapper 保存身份与 pending/resolved 状态。
真正 runtime 集成仍需扩展生产 entry metadata，不能假定现有 cache lookup
已经识别新 eig solver。

## 数值结果

| 输入 | 分量 | GPU 配对 | 对归档 authority 最大 p 误差 | 对 GPU dense 最大 p 差异 |
|---|---:|---:|---:|---:|
| 22 列 real640，独立 exact-fixed captured cohorts | 640 | 320 | 1.0975942377200454e-13 | 7.049916206369744e-15 |
| canonical 实际 prefill residual | 951 | 902 | 2.015054789694659e-14 | 2.733924198139448e-15 |
| n=242 深层真实 witness | 20 | 10 | 2.733924198139448e-15 | 1.6028844918025698e-15 |
| n=253 深层真实 witness | 24 | 12 | 5.113287325171303e-17 | 1.5720931501039814e-17 |
| n=351 hard/deep witness | 80 | 40 | 2.65898414397725e-14 | 5.523359547510154e-15 |
| n=364 深层真实 witness | 20 | 10 | 5.065392549852277e-16 | 5.84601811404184e-16 |
| 合计 | **1,735** | **1,294** | **1.0975942377200454e-13** | **7.049916206369744e-15** |

所有 p-value 满足绝对误差≤1e-10，alpha=0.1 翻转为零。26 个配对位于
[0.05,0.15]，该区间最大误差也是 1.0975942377200454e-13。
statistic/mean/variance 对 GPU dense 的既有绝对及相对门全部通过；所有输入
的 total 与 dense 逐位相同。

real640 与 72 个 witness 来自此前 `fe8286f7…` 的独立 exact-fixed capture；
951 个 canonical residual 来自 `b1e7eab4…` 实际 on-run 的 54 个 first-use
cohorts。本轮没有重新优化 selected-SP 或重算 residual。每份输入都核验原
manifest 与 residual/payload hash；两类 producer epoch 没有混称。

canonical 的三字段 dense 重放全部逐位复现归档；按 request identity 和原
dataset target 顺序重建的 54 个 Gram groups 也逐位复现全部 951 个归档
self_moment。actual residual 的 route/rank/info 全部为 2/28/0。
这只是所选 cohorts 的验证，不等于 240,498 个完整 CI、pMax 或图资格。

## 批形与并发

全部 1,735 个分量覆盖 producer 批宽 1、2、17、31、64，以及正序、逆序、
固定种子乱序和重复运行。三字段逐位不变；保持固定 delayed Gram groups
后，完整 compact GPU result 也逐位不变。不能据此声称任意未来布局均已证明。

另对 640 个 real components 使用 **10 个独立 context × capacity 64**，
每个 context 有独立 stream/workspace，实际执行 D2D payload stores。三次
并发输出与各自串行输出逐位相同。仅 structured workspace 合计
`2,728,811,520 bytes`（约 2.54 GiB），包括额外 k+1 候选的成本。

本轮没有 11 GiB 实际生产 cache / Prepared-S / optimizer 同时运行的容量资格。
JSON 中的并发 wall 包含 diagnostic D2H 和 host thread 开销，无同边界 dense
性能对照；不作为性能证据，不据此推算 canonical 会减少 140 秒。

## GPU dense fallback 与发布规则

测试前固定门为：info非零、rotation overflow，或

```text
gap = (abs(lambda[k-1]) - abs(lambda[k]))
      / max(abs(lambda[k-1]), abs(lambda[k]), DBL_MIN)
gap 非有限或 gap < 1e-8 → 原 GPU dense
```

k+1 候选实际由 device solver 求出。所有真实样本自然回退为零；最小实测
gap 为 `0.0003706144223105628`，最大 rotation count 为 118,193，低于固定
容量 170,662。足够的 gap 与 info=0 都不是 p 误差的数学证明；authority 门
仍独立强制执行。

四分量混合批逐一验证：

- 注入一个非零 info：只回退该分量；
- 将一个分量实际 QR rotation capacity 限为 1：真实触发 overflow，只回退该分量；
- 注入一个小 gap：只回退该分量；
- 二值 residual 的真实零边界 gap：自然进入 dense，无阈值注入。

回退分量的三字段与独立 dense 逐位相同，其余 structured 分量保持逐位不变。
consumer 拒绝未完成三字段与尚未 resolve 的 self_moment；重复 resolve 被拒绝。
dense failure 注入导致整批不发布，reset 后能够恢复并重现输出；未 finalize
的 in-flight submit 也完成清理后销毁。

compute-sanitizer memcheck 覆盖四种 n 的真实 witness、混合 fallback、dense
失败后拒绝发布、reset 恢复及 abandoned submit teardown，结果为 **0 errors、
0 bytes leaked**。source-shared 资源计数在各 campaign 前后闭合；这不等价于
完整 R runtime 或全部 1,735 分量同时运行的 sanitizer 审计。

## 证据与下一步

工具：[structured_adapter_20261022](../tools/research/structured_adapter_20261022/README.md)。
证据目录：`fastkpc/artifacts/full_cuda_ci/structured_adapter_20261022/`。
其中 `qualification.json`、`smoke.json`、`build.json`、canonical 全身份索引、
GPU payload/compact arrays、源码快照和日志由 manifest 绑定。

数值与资源门支持下一步 **opt-in 22 列完整方法资格**：生产 metadata 接入 eig solver
身份，保持调度与 delayed Gram，完成全部 CI/pMax authority、task/deletion、
图、sepsets、counts、RNG、资源与同 binary off/dense/structured 比较。
该门通过后才进入完整 canonical。optimizer overlap 与 level-2 prefill 后置。

此前 review 的 L2 34.4 秒 residual solve 不能因移动到阶段 A 就全部计为节省，
约 495 秒组合场景继续不进入正式预算。当前没有新的 full-method speedup，
Phase 10 ACTIVE、candidate 未冻结、holdout SEALED、recommended=false 均不变。
