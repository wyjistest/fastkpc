# q=37/46 exact vector postback 隔离 GPU 原型：复核

日期：2026-10-29。复核 [隔离 GPU 原型](PHASE10_OPTIMIZER_POSTBACK_GPU_20261029.md)（证据
`fastkpc/artifacts/full_cuda_ci/optimizer_postback_gpu_20261029/attempt2/`）。只读复核：没有 GPU 运行；用 `cuobjdump` 读了候选
native 的资源，并按 README 重跑了我的 host 交叉检查。`GO_ISOLATED_POSTBACK_DEVELOPMENT` 成立；同意下一项做默认关闭的最小 runtime
集成资格。accepted opt-in development median 仍为 311.440 s（本轮的 286.882 s 是隔离原型对同期参考的结果，不替换它）；默认关闭、
L8 optimizer STOP、Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。
同日按 Codex 核对意见修订，见“修订记录”。

## 修订记录（Codex 核对后）

Codex 重新核对了 432 项封存记录、静态审计 133 项与二进制资源，均一致；同意隔离原型 GO，下一步做默认关闭的最小 runtime 集成资格。
关于 OFF 路径的 stack 隔离，Codex 补充了必须实测的几项（已写入第 3 节第 1 条）：OFF 的 optimizer 保持 7,120 B stack、fixed
evaluator 保持 288 B；OFF 的数值与容量与原实现一致；ON 重新通过容量门；记录实际执行模式与 q 范围，证明开关选到了预期路径。单纯
加模板不能代替这些验证。另有三处收紧，我复核后全部接受：

1. **不是“所有空闲检查点”都少 20 MiB。** 两侧 L3 的 `free_before_create`（5,829,165,056 B）与 `free_after_create`
   （4,610,719,744 B）都相同，差异从 `free_after_optimizer`（3,404,857,344 → 3,383,885,824 B）开始出现。这与 stack 预留在首次
   optimizer launch 时增长相符，但原因没有独立证明。
2. **outer 的差别“主要对应” L8 的波动。** 每对扣除 L4/L5 与 L8 的变化后，其余部分仍有 −0.956、−0.989、+0.757 s（负值表示候选
   在其余部分较慢）。这不改变 8.95% 的完整方法收益。
3. **LWORK 已由 `dgesdd_bdspac(q)` 给出。** 需要与 reference 绑定的是 NB 规则、32/4160/65 这几个常数与适用形状，并保留完整
   DORMBR/workspace 逐位测试。

执行顺序（Codex 的建议，我同意）：独立的 OFF/ON 双入口与 OFF 资源核对 → GPU/cohort 资格 → 同一 native 的 OFF/ON 完整配对测试；
通过后才更新 accepted development median。

## 结论

```text
完整性    closure.json（f2de10d8…）、manifest.json（f5f4dd07…）的 SHA-256 吻合；manifest 432 项（104 个生产输入、11 个工具、
          7 个共享工具、310 个文件）按当前文件重算一致；静态审计的 133 项仍一致；生产 native 12acc3cd… 未变，候选
          native 06916ca1… 只在研究目录；7 个共享源码与 c38b04b 的 blob 及当前文件一致
代码      overlay 只改 lapack_312_small_dgesdd.cuh：n = 37/46 时，Q/P 回乘不再调用 reference DORMBR，改由 postback_apply
          重写 DORMQR/DORMLQ 的 blocked 外层循环，lane 0 串行 DLARFT，各 lane 按向量调用原 DLARFB。这比静态审计多走了
          一步（审计保留 DORMBR/DORMQR/DORMLQ 外层）。我逐项对照 reference 核对，等价成立；GPU 的 DORMBR 级用例也对整个
          缓冲区（含 3q²+4q 的全部 WORK、work[0] 与 T 区）逐位比较
资格      20,628 个 GPU 用例零差异；1,719 例 memcheck/synccheck 零错误、零泄漏；缩减 cohort 的 sanitizer 零错误；完整 L4
          2,154 setups/17,767 targets、L5 955/6,816 全部 shadow 逐位通过
计时      协议 13:24:15 写入（门：中位比 ≤ 0.98、3/3 配对获益），六次按 R C C R R C 运行。重算：outer 中位 315.067 →
          286.882 s（−28.185 s，−8.9457%），配对 30.246/28.507/20.473 s，每次候选都快于每次参考；同次 L4+L5 optimizer
          之和中位 59.045 → 30.927 s（−28.118 s）；Stage B 与 native setup 不变；第三组收益较小，主要对应 L8 16.56 →
          24.92 s 的波动
容量      L3 建池前后的空闲两侧相同；L3 optimizer 增长与总增量多 20 MiB，自 L3 optimizer 完成起各检查点的空闲少 20 MiB；
          最紧的 L6 为 2.083130 GiB，比 2 GiB 门高 85.125 MiB（原 105.125）。与 optimizer kernel 每线程 stack 7,120 →
          7,232 B 相符，原因未独立证明（见第 2 节）
意见      同意下一步；集成须用独立的 OFF/ON 入口，并实测 OFF 资源、数值、容量与执行路径，见第 3 节
```

## 1. 完整性与代码

- manifest 的 432 项全部吻合；静态审计 manifest（`0695f219…`，含我封存的 L4/L5 复核）133 项仍一致。
- `shared-source-attestation.json`：7 个共享源码的 SHA 与 `c38b04b` 的 blob、当前文件一致；报告已注明这是事后 attestation。
- 源码 overlay 与生产源码逐文件比较，只有 `lapack_312_small_dgesdd.cuh` 不同。`postback_apply` 与 reference 的对应：
  - NB 按 LWORK = 3q² + 4q、TSIZE 4,160、`ilaenv` 32、NW = q 推出（q=37 为 2，q=46 为 32），与 DORMQR/DORMLQ 一致；
  - Q 侧 K = q、P 侧 K = q − 1，按原倒序分块，IB = min(NB, K − offset)；
  - DLARFT 的 order 为 m − offset（Q）或 q − 1 − offset（P），V 取 `a + offset + lda·(base + offset)`，P 的 base 为 1，对应 DORMBR
    跳过首列；
  - DLARFB 的 TRANS（Q 为 N，P 经两次转换为 T）、`ldwork = q`、T 位于 `work + q·nb`，C 的输出偏移（Q 为行 offset，P 为列 1 + offset）
    均一致；
  - WORK(1) 依次写 q·32、1、q·32 + 4160，循环后写 q·32 + 4160、q·32，重现 DORMBR/DORMQR/DORMLQ 的写入次序；
  - T 写好后、各向量完成后、结束时各一次 `__syncwarp()`，32 个 lane 都参与；Q 的入口要求 m ≥ n 且 K = n，其他 q 仍走原入口。
- 单元测试的 `compare` 对每个用例的整个缓冲区逐位比较，非 C/W 区还要求与输入相同；所以外层循环重写后的 workspace 状态也被直接比较，
  不只是 C。
- 资源（`cuobjdump -res-usage`，候选 native）：small64 的 optimizer kernel REG 255、STACK 7,232（生产为 7,120），evaluator STACK
  320（生产为 288）；Base80 与 ext 版本不变。
- 我的 host 交叉检查按加固后的 README 重跑：`--validate-only` 通过，数值模式 18,636 例零差异，空文件退出 2。

## 2. 复现

由六个 `report.json` 重算（秒）：

| 运行 | outer | L4 opt | L5 opt | 全部 optimizer | Stage B | L8 opt | 门 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 参考 1 | 317.128 | 27.210 | 31.772 | 132.723 | 127.856 | 22.136 | 71/71 |
| 候选 1 | 286.882 | 15.219 | 15.659 | 102.190 | 127.947 | 19.038 | 71/71 |
| 候选 2 | 286.560 | 15.225 | 15.702 | 102.597 | 127.849 | 18.867 | 74/74 |
| 参考 2 | 315.067 | 27.293 | 31.786 | 130.991 | 127.859 | 20.211 | 74/74 |
| 参考 3 | 312.680 | 27.285 | 31.760 | 128.841 | 127.736 | 16.556 | 74/74 |
| 候选 3 | 292.207 | 15.197 | 15.767 | 108.217 | 127.768 | 24.922 | 74/74 |

- 中位数：outer 315.067 → 286.882；L4 27.2846 → 15.2192；L5 31.7723 → 15.7023；全部 optimizer 130.9907 → 102.5967；Stage B
  127.8563 → 127.8486；native setup 22.3593 → 22.3305。各配对的 L4+L5 节省为 28.104/28.151/28.081 s，很稳定；outer 的差别主要
  对应 L8 的波动（L8 变化 +3.098/+1.344/−8.365 s），扣除 L4/L5 与 L8 后其余部分为 −0.956/−0.989/+0.757 s。
- 层门（隔离 replay，无 profile）：中位 L4 38.007 → 21.460、L5 31.829 → 15.741，同次两层之和 69.826 → 37.198，charged
  lifecycle 70.156 → 37.534；参考三次之和 69.826/66.295/69.836。注意 replay 的 L4 参考（34.5–38.0 s）比 canonical 的 L4
  optimizer（27.2–27.3 s）长 7–11 s，候选同样偏长，L5 则一致；相对降幅相近（L4 43.5% 对 44.2%，L5 50.5% 对 50.6%）。原因未查，
  replay 的 L4 秒数不宜当作 canonical 的预算。
- profile 分解：调用数 Q 429,191/187,761 等于 objective calls；立即 P 与延迟 P 之和 233,961/97,675，正好等于迭代数加 target 数
  （216,194 + 17,767、90,859 + 6,816）。DLARFB 周期合计 L4 5,061.7G → 562.5G、L5 5,268.5G → 567.0G，DLARFT 基本不变
  （88.3G；750.7G → 754.4G）；份额与报告一致（候选 L5 的 DLARFT 占 57.08%）。
- 容量：L3 optimizer 增长 1,205,862,400 → 1,226,833,920 B，总增量 2,424,307,712 → 2,445,279,232 B，都是 +20,971,520 B（20 MiB）；
  L3 的 `free_before_create`、`free_after_create` 两侧相同（5,829,165,056 与 4,610,719,744 B）；从 `free_after_optimizer` 起，
  L3/L4/L6 的最低空闲与 L5 创建后空闲都少同样的 20 MiB。L6 的 2,236,743,680 B 比 2 GiB 高 85.125 MiB。
  这与 stack 相符：设备 stack 预留按已 launch 的 kernel 中最大的每线程 stack 设置，small64 optimizer kernel 在 L3 首次 launch，晚于
  建池检查点；(7,232 − 7,120) B × 196,608 个常驻线程 ≈ 21 MiB，实测增量为 2 MiB 的整数倍（10 个）。这只是相符的估计，原因没有
  独立证明。

## 3. 对默认关闭 runtime 集成的意见

同意下一步只做默认关闭的最小集成资格：保留原入口与 q 范围，沿用全部容量门。几点需要在设计时先定：

1. **stack 属于编译后的 kernel，不随分支开关。** 若 ON 只是同一个 small64 optimizer kernel 内的运行时分支，OFF 运行也会带着
   7,232 B/线程，设备 stack 预留同样多约 20 MiB，OFF 就不再与当前生产相同。所以要保留独立的 OFF/ON 实例（例如像 work-pair 的
   `bidiagonal<Mode>` 那样用模板），并实测：OFF 的 optimizer 仍为 7,120 B stack、fixed evaluator 仍为 288 B；OFF 的数值与容量与原
   实现一致；ON 重新通过容量门（本轮 L6 余量 85.125 MiB）；记录实际执行的模式与 q 范围，证明开关选到了预期路径。单纯加模板不能
   代替这些验证。
2. **需要与 reference 绑定的常数。** LWORK 已由 `dgesdd_bdspac(q)` 给出；写死的是 NB 规则（`lwork < q·32 + 4160` 时取
   `(lwork − 4160)/q`）、32/4160/65 这几个常数、WORK(1) 写入序列与适用形状（q = 37/46，Q 侧 m ≥ n 且 K = n）。它们只对当前 DORMBR
   合同成立，生产实现应与 reference 的推导绑定并加断言；本轮完整 DORMBR/workspace 逐位的 GPU 用例应保留为集成资格的一部分。
3. **收益口径。** 本轮 −28.185 s 是对同期参考的结果（参考中位 315.067 s，高于 311.440 s，主要因 L8/L7 的波动）。development
   median 只能由集成后的同一 native OFF/ON 预声明 campaign 更新；L8 的波动（本轮 16.6–24.9 s）仍会影响配对，门的设计应考虑。
4. 收益之后，L5 回乘中串行 DLARFT 占 57%，L4 为 13.5%。报告不自动开新优化，我同意；以后选题时可以参考。

## 4. 对我上一份复核的修改

Codex 修改了 [postback 审计复核](PHASE10_OPTIMIZER_POSTBACK_AUDIT_REVIEW_20261029.md) 与我的交叉检查工具，我复核后全部接受：
NaN 检查改为“未观察到哨兵污染”，不证明没有读取；敏感性对照中未检出的扰动不再全归于舍入；“L5 每块 k=32”改为 NB=32 并写明尾块
14/13；工具对缺失、空、截断或篡改的输入原先会以零用例、退出码 0 返回，这是真实缺陷，现已改为校验 SHA 与完整计数并退出 2。
已在该复核中加修订记录。

## 证据

`attempt2/{closure,manifest,canonical-analysis,profile-analysis,shared-source-attestation,unit-full,unit-memcheck,unit-synccheck}.json`、
`attempt2/canonical/{protocol.json,*/report.json}`、`attempt2/cohorts-{qualification,timing,profile}/*.json`、`attempt2/source/cuda/lapack_312_small_dgesdd.cuh`、
`attempt2/candidate/fastkpc_cuda.so`（`cuobjdump -res-usage`）、`tools/research/optimizer_postback_gpu_20261029/unit.cu`。计算为只读一次性脚本；
没有修改生产源码、native、默认配置或封存输入，没有 commit 或 push。
