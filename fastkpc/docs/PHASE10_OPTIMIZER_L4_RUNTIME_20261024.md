# L4 原窗口 32-worker opt-in runtime 资格

日期：2026-10-24。承接 [L4 captured diagnostic](PHASE10_OPTIMIZER_L4_DIAGNOSTIC_20261023.md)。
实现和构建开始于 10 月 23 日，完整 canonical campaign 于 10 月 24 日执行；
工具和 artifact 目录保留 `optimizer_l4_runtime_20261023` 名称。

最终结果：`GO_L4_ORIGINAL_WINDOW_WORKERS_DEVELOPMENT`。六次完整 canonical
运行全部通过；OFF/ON 三次中位数为 **475.439 / 461.663 秒**，下降 **2.90%**，
三组配对均改善。L4 optimizer host 中位数 **90.617233 / 80.233098 秒**，下降
**11.46%**。保留该默认关闭的窄选项。预声明协议和复现入口见
[runtime README](../tools/research/optimizer_l4_runtime_20261023/README.md)。

## 实现范围

新增默认关闭选项：

```sh
FASTKPC_DCC_GAMMA_L4_OPTIMIZER_WORKERS32=1
```

仅限制 L4 原窗口内的 worker 数；34 个窗口（33×64+42）、2,154 个 prepared setups、
17,767 个原 target 请求、全部数值 control 和 prepared 生命周期保持原样。
没有 L4 slot pool、跨窗口补位、新 stream/workspace 或 CUDA kernel 修改。
scope 检查要求 canonical S2、exact-primary coefficient reuse、status-wait=1、
原 setup scheduling，L4 请求限定 n=351、q=37、四 penalty、2–18 targets。

`prefill_batches` 记录实际 multi-penalty 请求的 concurrency、worker 数和 peak host
calls；single-penalty 行保持零，不能把它们解释为 single-penalty 无并发。
host calls 的峰值也不代表实测 GPU overlap。

## 构建与资格边界

本轮唯一改变的生产源文件为 `full_cuda_ci_one_call.cpp`。其余 97 个源输入与
L5 全量构建闭包完全相同。使用该次全量构建的 hash-verified 对象，在独立源码快照中
重编 one-call、按原 build script 重编 structured eig，再 relink。
41 个保留对象逐字节一致；structured eig 的规范化 SASS 完全一致；DBDSQR 生成内容一致。
这是有完整来源链的增量构建，不是本轮重新执行了全量 CUDA build 或 clean-checkout run。
`build.py` 的起点要求为已归档的父 native `b0d35232…cdf58`；从头复现需先在独立
工作环境恢复这一 binary 起点，再使用当前源码和新的输出目录。现有 evidence 目录
不可覆盖。父对象、98 个源输入、43 个 link 对象及生成器身份均写入 `build.json`。

```text
parent native
b0d3523285bc8098286d00350c5d4204edec31aab13f6336090fab2d203cdf58

qualified native
2c8dd8af4ea021d8a784588717476f091a0132b9415067e4a3e115ca65bcd0ee
```

六种非法/不支持配置拒绝检查及资源关闭通过。已有 L4 diagnostic 的全部 14 种容量、
249,010 个 target 输出比较实例及 reduced memcheck 通过证据，由相同 CUDA 对象绑定；
本轮没有声称重新运行 memcheck，也不把该诊断当作本轮完整方法性能。

## 预声明完整运行门

两组均启用既有 L5 slots、11 GiB component cache、residual bypass、八 eig lanes、
status-wait=1、S2 prefill。GPU1、CPU affinity 0–19、connections=32、BLAS/OMP/MKL
各一线程。GPU0 外部训练保留，因此证据为 shared-host uncommitted source-closure
development；不称为隔离、clean producer 或 promotion evidence。

六次 fresh-process 顺序为 OFF/ON/ON/OFF/OFF/ON。每次均比较完整 240,498 个 CI，
要求 CI/pMax 位值、task/deletion/skeleton/sepset/count/RNG、physical cohort 与 Gram
顺序 exact，CI/pMax 对 mgcv authority 误差 <=1e-10。CPU 数值 authority、fallback、
large D2H、cache route/status mismatch 均为零，application resources 归零。

所有原窗口和请求必须一致。OFF 的 L4 workers 为 64（最后窗口 42），ON 全部为 32；
其他层实际 worker 配置不变。L5 pool 继续要求创建后至少 2 GiB 显存余量。
大型 artifact 写入和离线 authority 比对均在 outer elapsed 之外。

开发性能 GO 要求 ON/OFF 总时间中位数比 <=0.99、三组相邻配对均改善，且 L4
optimizer host 中位数比 <=0.95。正确但收益较小或配对不稳定为 CONDITIONAL；
L4 没有改善或 outer 回退则 STOP。任何硬门失败立即停止，不调整验收门。

## 状态约束

本轮只资格化 canonical S2、L5 slots 开启时的 L4 原窗口 32-worker。
其他层 worker 数、L4 槽位/跨窗口调度、最终 PDAG 和三套真实数据均未新增资格。
默认关闭；Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false。
本轮未提交或 push；此前用户的其他工作区改动继续保留。

## 六次完整结果

| 执行顺序 | L4 配置 | 完整方法（秒） | L4 optimizer host（秒） |
| --- | --- | ---: | ---: |
| 1 | OFF1 | 475.299 | 90.617233 |
| 2 | ON1 | 461.663 | 80.306486 |
| 3 | ON2 | 466.051 | 80.233098 |
| 4 | OFF2 | 475.439 | 90.190468 |
| 5 | OFF3 | 479.427 | 90.835671 |
| 6 | ON3 | 458.615 | 80.139421 |
| OFF 中位数 | 64 / 最后窗口 42 workers | **475.439** | **90.617233** |
| ON 中位数 | 32 workers | **461.663** | **80.233098** |

outer ON/OFF 比率 `0.971024674`，通过预声明 `<=0.99` 门；三个配对分别节省
13.636、9.388、20.812 秒。L4 比率 `0.885406623` 通过 `<=0.95` 门，中位数差为
10.384135 秒。两组都使用同一新 native；历史 L5 中位数 472.080 秒不作为本轮对照。

| 其他边界的独立中位数（秒） | OFF | ON |
| --- | ---: | ---: |
| total optimizer host | 270.736995 | 257.080644 |
| L5 optimizer host | 31.850360 | 31.845513 |
| Stage A | 16.577465 | 16.454420 |
| Stage B | 144.539034 | 144.874990 |
| native setup | 23.560880 | 23.493113 |
| prepared build | 12.790891 | 12.664323 |

这些是各边界独立取中位数，不能相加当作单次可加分账。L6/L8 在 worker 配置完全
相同的情况下仍有数秒波动；不能把 outer 的全部 13.776 秒差都解释成 L4 device
加速。最直接的证据是三组 L4 host 边界各减少约十秒，以及完整 outer 三组均改善。
本轮未采集 device overlap，也未证明这一比例可外推到其他输入或隔离性能 campaign。

六次每次 240,498 个 CI，CI/pMax 与既有 L5-qualified S2 及同 native 对照逐位一致。
对 mgcv authority 的最大 CI/pMax 绝对误差均为 `3.05425129631942e-12`。
task identity/order、deletion、skeleton、sepsets、n.edgetests、RNG、physical cohort
和 Gram order 全部 exact。34 个 L4 窗口/2,154 setups/17,767 targets 完全保留；
OFF/ON 实际 workers 为 64/42 与 32，其他层 worker 配置完全相同。
exact-primary、stable-SVD、component cache metadata、GPU-only authority 和资源硬门均通过。

六次显存采样峰值均为 **22,366 MiB**。L5 slot 创建后 free memory 均为
**2,372,730,880 bytes（约 2.21 GiB）**，继续满足既有 2 GiB 门。
每次 prepared build/release 均为 **7,691/7,691**，coefficient arena 和暴露的
application allocations/bytes/streams/events 全部关闭；无 CUDA OOM 或 GPU1 外部进程。
GPU0 的训练进程未被干预。采样峰值不等于瞬时完整显存上界。

## 证据与收口

artifact：`fastkpc/artifacts/full_cuda_ci/optimizer_l4_runtime_20261023/`。
`protocol.json` 在首个运行前冻结工具和依赖身份、六次顺序及性能门；`build.json`
绑定父全量构建和本轮增量闭包。`checks.json`、六份 `report.json`/`receipt.rds`、
两卡采样、进程日志及 `closure.json`/`manifest.json` 绑定完整结果。

```text
manifest SHA256
ff4b04b2fa077f0043da02f3695a10d38a4ad398dc383238341a6fe5049ced5a
```

本次窄工作完成。新开发配置的三次中位数为 461.663 秒；保持 opt-in，并将该结果
与 clean-checkout、隔离重复、候选冻结及 performance promotion 严格区分。
既有 `STOP_FULL_LAYER_CAPACITY` 不变，不据此扩展为其他层或跨窗口调度的 GO。
