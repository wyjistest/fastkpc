# L3 首次执行 stack 归因：复核

日期：2026-10-27。复核 [L3 stack 归因诊断](PHASE10_OPTIMIZER_L3_STACK_ATTRIBUTION_20261027.md)（证据
`fastkpc/artifacts/full_cuda_ci/optimizer_l3_stack_20261027/attempt2/`）。只读复核，没有新的 GPU 运行。
`DIAGNOSTIC_COMPLETE_ORIGINAL_CAPACITY_STOP_RETAINED` 成立；原 `STOP_L3_REPLACEMENT_POOL_INCREMENT` 与
`GO_CAPTURED_L3_SLOTS` 保留；L3 runtime 仍为 `NOT_YET_GO`；accepted development median 368.851 s。
Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。

## 结论

```text
完整性    报告列出的 protocol/build/closure/manifest 与 tap.so 的 SHA-256 吻合；manifest 的 33 个条目
          按当前文件重算一致；生产源码与 native 未变。W exit 0，S 按预期在容量守卫 exit 1；GPU1 独占，
          GPU0 有外部进程 1918904（运行前声明的共享主机诊断，不是性能证据）
观测      W/S 的 stack limit 都在首批 L3 launch 中从 1,024 B 变为 7,120 B；最早 enter 到最早 return 之间
          free 都恰好减少 1,205,862,400 B（1,150.0 MiB），应用账本不变。263/201 条记录中没有一次查询
          前后应用 bytes 或分配次数变化，API 状态全为成功，worker 资源建销为 0。W 结束时 limit 为 7,136 B
正确性    W 的 67 个 canonical 门全部为 true；Stage A 参考与共存轮的 SHA 相同；W/S 的 4,065 setups /
          54,878 targets optimizer 输出逐位一致；W 保留的非应用增长 1,158.679443–1,164.024658 MiB、
          L3 最小 free 3.048645 GiB，与共存轮一致
归因      我在共存复核中留下的两种读法已有定论：L3 之前的 context limit 是默认的 1,024 B，800 B 的单
          penalty kernel 没有提高它。模型为 1,143 MiB，实测 1,150 MiB，7 MiB 未解释并计入容量。我依据
          CUPTI 260 MiB 推出的 1,184 MiB 读法撤回：那是该 kernel 自身 launch 的需求，不是 context 的预留
合同      同意下一轮合同的结构（原始账本、分列三项增量、整个 optimizer + Stage A 的联合余量）。补充四点，
          见第 3 节
```

## 1. 复现

由 `W-observations.jsonl`、`S-observations.jsonl` 按时间排序重算：

| 模式 | limit 首次为 7,120 B 的记录 | 最早 L3 enter（limit / free / 应用） | 最早 L3 return（limit / free / 应用） |
|---|---|---|---|
| W | 第 11 个 launch enter，之前已有 11 个 enter、0 个 return | 1,024 B / 6,205,276,160 / 17,113,798,688 | 7,120 B / 4,999,413,760 / 17,113,798,688 |
| S | 第 2 个 launch enter，之前已有 2 个 enter、0 个 return | 1,024 B / 4,617,732,096 / 19,019,989,752 | 7,120 B / 3,411,869,696 / 19,019,989,752 |

- 两者的 free 差都是 1,205,862,400 B（1,150.0 MiB），应用 bytes 差为 0。并发记录不是原子快照，这只说明增长
  发生在首批并发 launch 的区间内，与报告的限定一致。
- S 在最早 return 时的 free（3,411,869,696 B）与全部 optimizer 结束时相同：增长一次完成，之后没有继续。
- 第一次单 penalty launch 前后，W/S 的 limit 都保持 1,024 B，应用账本不变，free 都减少 4.000 MiB。首次 launch
  也会带来几 MiB 的非应用分配；这与把 7 MiB 保留为未解释、计入容量的做法一致，但不能据此确定 7 MiB 的来源。
- 1,205,862,400 B 已在三次运行中出现：共存轮的日志重放 S、本轮 W、本轮 S。共存轮第一次 S 在守卫处退出，
  没有记录这一数值。

## 2. 对此前读法的更正

共存复核第 3 节给出两种读法：L3 之前的预留若等于此前最大 kernel 的需求，增长为 1,184 MiB；若等于默认 limit
1,024 B，增长为 1,143 MiB。本轮直接读到 L3 之前的 limit 为 1,024 B，单 penalty kernel 的 800 B 需求没有提高
它，所以后一种成立。CUPTI 的 `localMemoryTotal` 在单 penalty kernel 上报 260 MiB，是该 launch 自身的需求，
不是 context 当时的预留；据此推出的 1,184 MiB 读法撤回。

## 3. 对下一轮合同的意见

报告建议的四项原始测量门：建池物理增量 ≤1.5 GiB（本次 1.134766 GiB）、建池后至 optimizer 完成 ≤1.25 GiB
（本次 1.123047 GiB）、总增量 ≤2.75 GiB（本次 2.257813 GiB）、整个 L3 optimizer + Stage A 的 free ≥2 GiB。
结构我同意，补充四点：

1. **门槛是看过数据后提出的。** 这只能作为新协议、新证据目录在运行前声明，旧 receipt 继续是 STOP；报告已
   这样写明。三项增量的余量分别为 0.365/0.127/0.492 GiB。首次执行增长在三次运行中逐字节相同，增量门基本
   是记账；真正未知的是 S 的 Stage A 显存与完整语义门，S 从未到达 Stage A。
2. **与 W 的对照要预先说明用途。** 报告写“与同配置 W 对照”。若下一轮要支持“S 不降低生产路径的显存余量”
   这类结论，应预先声明它是门以及容差；否则只作报告，结论中不使用。
3. **1.25 GiB 只适用于 small64 envelope。** 它对应 (7,120 − 1,024) × 196,608 加未解释部分与余量。其他 bucket
   的 stack 更大：ext559 为 8,016 B，按同一模型增长约 1,311 MiB（约 1.28 GiB），会超过 1.25 GiB。L3 槽位路径
   限定 n=351、q=28、三个 penalty、cohort 2–33，在此之外 fail closed，这一门才成立。
4. **runtime 守卫要按合同分点测量。** 现有 L4/L6 的 `observe_capacity` 只记录一个相对建池前的最大增量
   （`full_cuda_ci_one_call.cpp:5144–5157`）。L3 runtime 需要在建池完成、任何 launch 之前单独记录建池增量，
   再记录 optimizer 后与 Stage A 期间的总增量和 free。L4/L6 的 1 GiB 守卫不受影响：两种模式下扩栈都发生在
   L3，早于 L4/L6 建池。

## 证据

`attempt2/{W,S}-observations.jsonl`、`attempt2/{W,S}-closure.json`、`attempt2/S/stack-closure.json`、
`attempt2/W/checkpoints.csv`、`attempt2/closure.json`（W `canonical_gates` 67 项）、`attempt2/{W,S}-process.json`、
`attempt2/manifest.json`。计算为只读一次性脚本；没有修改生产源码、native 或默认配置，没有 commit 或 push。
