# L3 首次执行 stack 归因：直接观测完成，原容量 STOP 保留

日期：2026-10-27。承接 [共存复核修订版](PHASE10_OPTIMIZER_L3_COEXISTENCE_REVIEW_20261027.md)。
本次完成一项窄诊断：在原 W/S 执行中观测 stack limit、物理显存和应用账本，不改原容量门。

正式状态：**`DIAGNOSTIC_COMPLETE_ORIGINAL_CAPACITY_STOP_RETAINED`**。
W/S 均直接观测到 L3 首批持久 kernel 提交附近的 **1,024 → 7,120 B** 扩栈，
对应物理占用均增加 **1,150 MiB**，应用分配字节不变。公共扩栈是这笔首次执行成本的
有力主体解释；按 limit 变化计算的模型仍比实测少 **7 MiB**，不能把全部增长精确归为 stack。

原 **`STOP_L3_REPLACEMENT_POOL_INCREMENT`** 保留；`GO_CAPTURED_L3_SLOTS` 保留；
完整 L3 runtime 为 `NOT_YET_GO`。本次没有新的方法性能结果，accepted development median
继续为 **368.851 s**。Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 不变。

## 实验边界

- 完整复用冻结的 coexistence `attempt2/probe.so` 和生产 native；没有修改或重编译生产源码/native。
- 单独编译 host 日志库。通过 CUDA 函数注册识别原 kernel，不调用 function-attribute 查询、不预加载函数。
  原 launch 参数逐项原样转发一次；small64 batch 的透明包装只记录线程局部请求身份和既有返回边界。
  没有替换 GPU 数学实现或修改数值输出。
- 记录首个 single-penalty 持久 launch、前 64 个 L3 持久 launch 的 API enter/return，以及这些请求的
  原生完成返回。另记录 level、建池、worker join、原窗口关闭和异常清理后的检查点。
- 每条记录包含时间区间、Linux thread、原 target key、stream、launch 序号、stack limit、free bytes、
  查询前后应用 bytes/allocation calls 及资源计数。并发记录不是原子快照，不能把一次区间内的所有变化
  都归属于某个单独请求。
- 没有新增 warmup、SetLimit、GPU allocation、device/stream/event wait、worker barrier 或跨原 launch 的锁。
  原探针的 Stage A/optimizer shadow 与 10 ms 采样保持不变。观测 API 会扰动调度，所有耗时只作诊断。
- **GPU1 独占、主机共享**：GPU0 的外部 Python PID `1918904` 在运行；两轮均无外部 GPU1 compute 进程。
  这是运行前声明的显存/语义诊断条件，不是两卡空闲的性能 campaign。GPU0 任务未被操作。
- fresh R、affinity 0–19、connections=32、BLAS/OMP/MKL=1、真实 S2、11 GiB component cache、bypass、
  8 eig lanes、原 L4/L5/L6 slots 和 status wait 均保持原配置。

原门保持：池创建及 optimizer 完成时，相对“真实 arena reserve 后、建池前”的原始物理增量
**≤1.5 GiB**；记录的 L3 free **≥2 GiB**。不扣 stack 模型，不重置基准。
预期 S 在原 guard 停止；数值预期不是验收门，任何不同观测都应保留分析。

## 实测归因

| 路径/边界 | stack limit | free bytes | 应用 bytes |
|---|---:|---:|---:|
| W：L3 起点 | 1,024 B | 7,878,803,456 | 16,099,634,392 |
| W：首批 L3 launch 的最早 enter 观测 | 1,024 B | 6,205,276,160 | 17,113,798,688 |
| W：首批 L3 launch 的最早 return 观测 | 7,120 B | 4,999,413,760 | 17,113,798,688 |
| S：arena reserve 后、建池前 | 1,024 B | 5,836,177,408 | 18,139,784,184 |
| S：建池后 / workers 启动前 | 1,024 B | 4,617,732,096 | 19,019,989,752 |
| S：首个 L3 launch 的 enter 观测 | 1,024 B | 4,617,732,096 | 19,019,989,752 |
| S：最早 L3 launch return 观测 | 7,120 B | 3,411,869,696 | 19,019,989,752 |
| S：全部 optimizer workers 结束 | 7,120 B | 3,411,869,696 | 19,019,989,752 |

W 的最早 enter 观测结束时，已有 64 个线程进入日志包装；最早 return 观测对应另一个请求。
因此这是首批并发 launch 的区间对照，不是单请求的分配因果证明。S 的最早 return 观测时已有第二个
launch 进入包装，同样保留并发限制。所有观测的应用 bytes/allocation calls 在各自查询前后均未变化。

W 在全部 64 个窗口关闭后保持 **7,120 B**；扣除应用 bytes 变化后的保留物理增长为
**1,158.679443–1,164.024658 MiB**，与前一轮完全一致。完整 W 结束后 limit 为 **7,136 B**，
与后续 base80 kernel 的更大 stack 需求一致；不能把 7,120 B 写成整个运行不变的上限。

S 的完整账本为：

```text
建池应用分配                  880,205,568 B =   839.429443 MiB
建池物理增量                1,218,445,312 B = 1,162 MiB = 1.134765625 GiB
首次执行物理增长            1,205,862,400 B = 1,150 MiB = 1.123046875 GiB
optimizer 后原始总增量      2,424,307,712 B = 2,312 MiB = 2.257812500 GiB
optimizer 后 free           3,411,869,696 B = 3.177551270 GiB
worker 期间应用 bytes 增长 / 受审计资源建销 = 0 / 0
```

根据实际 limit 变化而非默认值猜测，模型为：

```text
(7,120 − 1,024) × 196,608 = 1,198,522,368 B = 1,143 MiB
实测首次执行增长 − 模型      =     7,340,032 B =     7 MiB
```

196,608 是该 RTX 4090 的 128 SM × 每 SM 最多 1,536 线程的模型参数，不是该 kernel 的实际活跃线程数。
limit 与物理占用的变化支持公共扩栈解释，但没有观察 driver 内部分配。
**7 MiB 不标成已证明的 rounding、module 或 allocator 成本；保持未解释并计入容量。**
建池阶段应用账本以外的 322.570557 MiB 同样没有被进一步精确归为某种库分配。

## 正确性与资源结果

W 完整 canonical **67/67** 硬门通过：

- **240,498 个 CI 和 pMax 位值不变**；最大 CI/pMax authority 误差 **`3.05425129631942e-12`**。
- task/deletion/adjacency/sepsets/count/RNG、cohort/Gram order、authority/fallback 和生产 D2H 门通过。
- **4,065 setups / 54,878 targets** optimizer 输出与冻结 capture 逐位一致。
- **5,573 Stage A cohorts / 54,878 targets** 保持原消费顺序；新生成的 Stage A 二进制与原 W 的 SHA 完全一致。
- L3 最低观测 free **3.048645 GiB**；应用 resources 全部关闭。

S 的全部 **4,065 setups / 54,878 targets** optimizer 输出也逐位一致。
之后仍报 `optimizer slots exceeded the qualified memory budget`，exit 1。
总增量 **2.2578125 GiB >1.5 GiB**，不是 OOM，也不表示相对原生产路径净增了 2.258 GiB。
S 没有进入 Stage A，没有发布完整结果。异常后 allocations/bytes/streams/events/BLAS handles 全为 0。
两个模式均记录 4,065 个原生 optimizer 请求和 4,065 个持久 kernel，所有观测 API 状态成功。

## 下一项：新容量合同下的实际共存，尚未执行

此次直接观测已经回答了先前的窄问题，不必继续依靠 stack 源码形状猜测，也不必立即缩小 32 槽或修改 kernel。
下一轮应使用新的独立协议和证据目录，保持原 receipt 的 STOP。

建议预声明以下**原始测量门**，不作任何模型扣减：

| 项目 | 建议下一轮门 | 理由 |
|---|---:|---|
| 建池物理增量 | ≤1.5 GiB | 保留此前 L3 独立池额度；当前实测 1.134766 GiB |
| 建池后至 optimizer 完成的物理增长 | ≤1.25 GiB | 单列本次 W/S 共有的首次执行增长，并保留未解释部分和余量；当前实测 1.123047 GiB |
| 相对原建池前基准的总增量 | ≤2.75 GiB | 前两项预算之和，始终同时报告原始总数 |
| 整个 L3 optimizer + Stage A 的 free | ≥2 GiB | 检查点和 10 ms 采样；与同配置 W 对照，不能宣称覆盖未观测瞬间峰值 |

这些是**下一轮建议值，未据此运行或授予 PASS**。1.25 GiB 也不是精确 stack 分配量。
若采用该合同，原 L4/L6 的 1 GiB 增量守卫保持不变；S 仍需通过真实 coefficient gather、
原 Stage A cohorts、完整 CI/pMax/图/RNG/authority 和资源门，才能讨论默认关闭的生产接入。
完整方法性能仍须另做重复、隔离的 outer 计时。

## 复现与身份

工具：`fastkpc/tools/research/optimizer_l3_stack_20261027/`。
证据：`fastkpc/artifacts/full_cuda_ci/optimizer_l3_stack_20261027/attempt2/`。
W/S 的 `*-observations.jsonl`、进程/GPU 监控、原检查点、完整 W receipt、S 异常清理和汇总均已保留。
`attempt1/` 为 CUDA launch C++ 重载地址解析导致的编译失败，没有 GPU 运行；修正函数指针类型后另建 attempt2，
没有覆盖失败目录。生产 source closure 的 **101 项**和全部原 coexistence manifest 输入在运行前后核对一致。

| 身份 | SHA-256 |
|---|---|
| production native | `313d9f3cc6dc8512b5624aae84cc459c6083ff04fd7883c32812fdb081510a1b` |
| original coexistence overlay | `14f6a8b1d7ae152b969c3e98056bd34e93f49906fad559b7fd82011ab800eef9` |
| logging tap | `22331a3e20cc7bf9e14fe6c50ee6ec799f526e69343cb73bd62b7ad35b72113c` |
| protocol.json | `4c9ed15a26e8edb3a688e11b3b35e1467315c08799998c8b0aa4e5a2ff3e4870` |
| build.json | `56a2007ad73681ad2ef9246ed8cac292445d4c6f6fe79c3fbb86544b666d1507` |
| closure.json | `b08051ffd38c7bb8777059a61f5f3f82696bd757543fa08f40dc22a90fcd439b` |
| manifest.json | `ba6c17215f2d96fe80c1aeeef8d038b023d8707829385b7e49cc64c48dcbb1ad` |
| W/stage-a.bin | `dd7ac0fb31d278c763f9a7780cae0e1fbc800364fc456312d63e3a532f200a17` |

所有自有测试和监控进程已结束。未修改生产默认配置，没有 commit 或 push。
