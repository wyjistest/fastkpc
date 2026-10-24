# L5 有界 prepared 槽位池隔离原型

日期：2026-10-23。承接 [整层常驻诊断](PHASE10_OPTIMIZER_LAYER_REFILL_20261023.md)
和 [复核及执行前勘误](PHASE10_OPTIMIZER_LAYER_REFILL_REVIEW_20261023.md)。

结论：`GO_L5_BOUNDED_SLOT_PROTOTYPE`。32 槽保留了 955 个对象全层常驻的工作速度，
六轮 R/S 中位数为 `31.808618/31.828590s`（S +0.0628%）；计入建销与打包后为
`33.582750/31.932535s`，S 六轮均胜对应 R。独立创建显存增量为 564 MiB。
所有输出与生命周期硬门通过，允许进入后续 L5 opt-in 集成资格工作；runtime rolling
本身仍为 `NOT_YET_GO`，本次没有改生产调度或默认值。

## 范围与预声明门

仅验证同一批 L5 的 955 setups / 6,816 targets，n=351、q=46、五个 penalties。
原顺序、实际 target cohort、optimizer 控制和数值实现不变。第二次复核指出的模型外推、
显存表漏项和生命周期措辞已修正；原 review 的 manifest/工具/输出保持原样。

生产 native 保持 `ba8f806bfc1ebbf8ec90b28dcb095a57d163c986836ac33ddaf8fb0a3ed9298f`。
原型独立库直接 include 现有 small64 CUDA 源码，以隐藏可见性编译，不导出或替换生产
entrypoint；仅公开研究用的 bind/execute/detach 接口。完整 GPU 指令字必须与已资格化
small64 object 相同，再与冻结输出逐位比较。

- R：生产 native，955 个 prepared 全层常驻，32 workers，原顺序连续领取。
- S：32 个固定容量 8-target 槽位，仍按原实际 2/3/4/8 target 数执行，原顺序补位。

每次重绑定上传 X、QR/tau/R、pivot、penalty roots/matrices/ranks/offsets 和 initial SP，
并更新 host geometry metadata、绑定 identity 和 generation。所有 host payload 由
runner 持有到 stream 完成，不引入 pinned I/O、prefetch 或额外上传 stream。

S 把完整 coefficients/fitted/residual D2D 保存到预分配的 40,786,944-byte 整层设备区，
在该槽 stream 上记录预创建的 blocking event，确认复制完成后才解除旧 lease。
所有旧 token 保留到运行后的核验阶段，证明它们不能因槽位重用重新获得访问权限。
诊断 D2H 与逐位比较均位于工作计时之外；D2D 保存和完成等待计入 S 净工作时间。
整层 host packet 打包另行测量，保守地给每次 S 生命周期估计计入一次完整打包成本；
R 的几何打包已在 prepared 构建中。预热和正式循环均销毁前一档的对象，再创建后一档，
没有在 S 运行时额外保留 955 个 prepared。
该整层输出区用于完整诊断，生产只保留系数等方案不在本轮范围内。

预声明门：完整位值/route/rank/info/轨迹与 identity 通过；GPU authority/fallback 合同
保持；busy/invalid bind 不改变状态，旧 token/旧析构不影响新 lease，注入取消后资源关闭。
缩小范围 CUDA memcheck 为零错误/零泄漏；整层普通运行另做全量和资源检查。
worker 活动期间所有受审计的 CUDA 分配/释放、stream/event、cuBLAS handle 建销为零。

容量先在独立进程实测，32 槽加输出区的创建显存增量不超过 1 GiB，且剩余至少 2 GiB。
这不等于与生产 component cache/arena 的联合容量资格。
每档预热一次，再交错六轮 RS/SR；S 工作中位数不超过 R 的 1.15 倍，且计入完整
pool/arena 建销后仍快于 R 生命周期，才继续推进。否则按实测 CONDITIONAL/STOP。
不把旧 P 的约 108 秒直接当成本次同时运行的对照。

GPU1、CPU affinity 0–19、connections=32、status-wait=1、BLAS/OMP/MKL 单线程。
GPU0 的外部任务继续保留；证据类别为 shared-host development。

## 完整验证结果

独立数值对象完整编译，14 个 GPU 函数的全部指令字与已资格化 small64 object 相同；
匿名 translation-unit 名称只用于符号归一化，不忽略指令。host runner 的构建修正仅
重新链接该对象；`build.json` 绑定 CUDA 输入、最终对象与可执行文件，并记录 link-only。
独立 DSO 不导出生产 prepared/optimizer entrypoint，因此 R 仍调用原 native。

同一槽按实际 `8→2→4→3` target 请求反复重绑定，并用 1/2 workers 验证。
busy bind、错误形状 bind 均被拒绝且不改变 generation/identity；旧 token 不能访问
后续输出，其析构不能解除新 lease。在 execute 后注入错误，所有 workers 取消、
drain、释放 lease 并关闭资源。普通检查和缩小范围 memcheck 均通过，后者为
`0 errors / 0 bytes leaked`。memcheck 范围是 8 个请求实例、四种 target 数；不是全层 sanitizer。

普通完整运行共 15 次：独立容量 1 次，配对 campaign 预热 2 次和正式计时 12 次。
每次覆盖全部 955 setups、6,816 targets、90,859 optimizer iterations；完整 selected-SP
状态、route/rank/info、coefficients、fitted、residual 和轨迹全部逐位一致。
加上普通和 sanitizer 的缩小范围重放，共比较 `102,376` 个 target 输出实例。
失败注入的部分执行不计入完整输出通过次数。

所有完成运行的 application-visible allocations/bytes/streams/events/cuBLAS handles
及输出 leases 关闭，worker 活动期间受审计资源建销为零。库/driver 内部缓存不在该
拦截器的零资源声明内，实际 GPU 占用另行采样。

## 容量与净耗时

独立 S 容量运行实测：

| 项目 | 结果 |
|---|---:|
| prepared 槽位 / streams / cuBLAS handles | 32 / 32 / 32 |
| events（producer / status wait / detach） | 96 |
| application-visible allocations | 33（32 个槽位 buffer + 输出 arena） |
| application-visible bytes | 264,796,928（252.53 MiB） |
| 其中完整输出 arena | 40,786,944 bytes |
| 创建时 free-memory 增量 | 591,396,864（564 MiB） |
| 创建后可用显存 | 23,834,984,448 bytes |
| 该独立进程运行期间 GPU1 采样峰值 | 2,536 MiB |

564 MiB 是创建增量，不是进程总占用或执行峰值；2,536 MiB 是整张 GPU1 的采样值，
含 context/library 及既存非 compute 占用。二者不能混用。R 首次创建的 free-memory
增量仍为 `15,487,467,520 bytes`（约 14.42 GiB），与此前常驻容量检查一致。
本轮没有同时分配生产 11 GiB component cache/arena，联合容量仍未资格化。

六轮正式计时，顺序为 `RS / SR / RS / SR / RS / SR`，单位秒：

| 档 | 工作中位数 | 工作范围 | 创建中位数 | 销毁中位数 | 打包计入 | 完整生命周期中位数 |
|---|---:|---:|---:|---:|---:|---:|
| R：955 常驻 / 32 workers | 31.808618 | 31.611535–31.880818 | 1.463665 | 0.318565 | 已在创建内 | 33.582750 |
| S：32 槽 / 32 workers | 31.828590 | 31.683100–31.889018 | 0.049263 | 0.013724 | 0.037594 | 31.932535 |

各列分别取中位数；最后一列先对每次运行的工作、创建、销毁和打包求和，再取中位数。
S/R 工作中位数比为 `1.000628`，完整生命周期比为 `0.950861`，后者下降约 4.91%，
六轮 S 均胜对应 R。工作时间内 S 的重绑定 host 时间总和中位数为 `0.272248s`，
输出保存与等待 host 时间总和为 `0.080866s`；它们跨 worker 相加，不是可直接相加到
wall time 的独立阶段。对照已经包含这些净成本，不另用模型扣除。

独立未预热容量运行的 S 工作时间是 `37.301301s`；campaign 的 R/S 预热分别为
`36.615148/32.218283s`，均不计入正式六轮中位数。
生命周期是排除事后诊断 D2H/位值比较的边界之和，不是完整 one-call outer wall time。

## 决策与下一步

预声明的数值、资源、容量和性能门全部通过：32 槽把全层常驻实验变成了显存有界的
隔离实现，保留工作速度，并降低了生命周期成本。
这没有测量新的完整 canonical p-value、图或真实数据结果；已有约 541 秒的 S2 完整
方法证据保持原样，不能用本轮时间直接更新它，也不能把先前 L5 的 3.34× 当成方法加速。

下一步只建议 L5 opt-in runtime 集成资格：保持原 physical cohorts 和 Gram 顺序，
处理 coefficient arena/Stage A 的实际消费与 lease 释放，实测与生产 cache/arena 的
联合峰值，并跑完整 canonical 的 p-value/task/graph/RNG/authority/资源门和重复 outer
wall-time。其他层先做形状、容量与服务时间归因，不能直接套用本层 8-target 槽位。
排序、更高并发和 Stage B overlap 继续后置。

## 证据与状态

工具：`fastkpc/tools/research/optimizer_slot_pool_20261023/`。
artifact：`fastkpc/artifacts/full_cuda_ci/optimizer_slot_pool_20261023/`。
机器 `closure.json` 为 `GO_L5_BOUNDED_SLOT_PROTOTYPE`；manifest 绑定完整工具、
source/native/companion 身份、fixtures 上游索引、计时、telemetry、检查和修订后的 review。
prototype DSO SHA-256：

```text
35eaada4876ee6befcad60497a469c6c0517179351c95ae7a1380859528bafd8
```

最终 manifest SHA-256：

```text
d557adf991290bd02037f3e854e0324a5a6de9e10a86e51e9c11ba8249be4f93
```

无 runtime rolling 集成、排序、更高并发或 Stage B overlap。Phase 10 ACTIVE、
candidate NOT_FROZEN、holdout SEALED、recommended=false、runtime rolling NOT_YET_GO。
