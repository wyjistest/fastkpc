# Stage B fixed-VT work pair：默认关闭的生产集成

日期：2026-10-28。承接 [隔离资格](PHASE10_STAGE_B_WORK_PAIR_20261028.md) 和
[复核](PHASE10_STAGE_B_WORK_PAIR_REVIEW_20261028.md)。结论为 **GO_DEFAULT_OFF_WORK_PAIR_RUNTIME**：
深层逐位资格、默认关闭的生产集成、完整 canonical 正确性/容量与六次隔离 OFF/ON 全部通过。
同 native 的 outer 中位数 **328.464 → 311.440 秒**，减少 **17.024 秒（5.1829%）**；
accepted opt-in development median 更新为 **311.440 秒**。不是 clean-producer 或 promotion evidence。

## 已完成的资格与实现

- 未改动的 `e8c39703…0ca045` native 捕获 L4–L8，canonical 的 71 项原有硬门全部通过。
- 深层 25,873 个 component、2,669,450,530 条 rotation、17,061,684 个 sweep：GPU 中间状态、
  rotation 全字段与顺序、d/e/q、root/recover 输出、status 均逐位相同，shared 末端 canary 完整。
  433 项合成/边界案例再次通过；诊断资源归零。
- 与此前封存的 L2/L3 资格合并，覆盖 canonical 全部 130,670 个 component、
  13,184,762,608 条 rotation。L4–L8 并非按浅层加速比例外推。
- generator 保留原 `recorded_dbdsqr.inc`，另生成 `recorded_dbdsqr_fixed_vt.inc`。
  原文件 SHA 仍是 `be76c5e06ce8bce4013a9759a3a54fc63feed9e421a1d3d44af77e11f32f53b9`；
  新文件除入口名称外，与已资格通过的隔离候选完全相同。
- `FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR=1` 显式启用；未设置、空值或 `0` 保持原入口。
  其他值抛出异常。plan 的 `const` 模式在创建时固定，模板入口与 7N/5N shared 一起派发，
  所有配置校验均在 prepare 和数值 kernel launch 之前。没有把配置错误转成 dense fallback。
- 35 项错误配置/提交被拒绝，包含危险的“原入口 + 5N shared”组合；测试没有真的 launch 该组合。
  更改环境变量不改变已有 plan 的模式；548 个真实 target 的生产 OFF/ON GPU 输出相同，资源归零。
- 每种模式有 host plan/submit/target 计数，以认证完整运行实际使用的入口；没有增加 GPU 数值计算或 payload D2H。

两种生产 bidiagonal 实例的 REG 为 100/82，STACK/LOCAL 均为 0；n=351 时动态 shared 为
19,656/14,040 B。此前 occupancy API 上限为 4/6 blocks per SM，实际 occupancy 仍未测量。
标量 QR、旋转符号和顺序、recover、lane/wave/cache 策略均保持不变。

## 构建身份

生产 native：`12acc3cd01cce2dcac5f2bfb5ec02f27056e92154cbab2534c7f03bdc22caa7d`。
本次是 104 项生产输入绑定的 incremental source closure，仅重建 structured eig 与 one-call 两个目标文件；
其余目标文件逐个与父构建哈希一致。不是全量 rebuild、clean producer 或 promotion evidence。

新增专用 include SHA：`dff38bc3e3d3daa5352bd0ef056b61e2ab701e4d9c05f7f4448599f31f94006f`。
本次生产输入与对象哈希记录在 `runtime/build.json`，本次源码快照位于 `runtime/build-source/`。
父构建身份由其中的 `parent_build_sha256` 绑定到
`optimizer_l3_slot_runtime_20261028/build.json`；父源码快照位于
`optimizer_l3_slot_runtime_20261028/build-source/`，父 native 保存为 `runtime/previous-native.so.gz`。

提交后源码身份见 [2026-10-28 开发交接](PHASE10_DEVELOPMENT_HANDOFF_20261028.md) 与
[Git attestation](PHASE10_SOURCE_ATTESTATION_20261028.json)。原运行保留提交前 source-closure 身份。

## 完整方法验收协议

同一 native，GPU1，CPU affinity 0–19，单 math thread，CUDA connections=32，另一张 GPU 无计算任务。
固定其余已接受的 optimizer/cache 选项；fresh R process / dataset caches；大型保存与离线 authority 比较在计时外。
执行顺序 OFF1、ON1、ON2、OFF2、OFF3、ON3。性能运行没有 observer/interposer 或额外 payload D2H。

预声明 GO 门：完整正确性及原有容量硬门全过；ON outer 中位数至少下降 2%；最慢 ON 快于最快 OFF；
Stage B 总时间中位数至少下降 5%。分列 L2–L8，不能把隔离 replay 的 13.327 秒直接从历史总时间扣除。
L3 的 creation/growth/raw-total/free 容量合同及 L4/L6 守卫不变，不做 stack 扣除、额外 warmup 或 baseline 重置。

计数期望为每次 10 个指定模式 plan、2,044 次提交、130,670 个 target；另一个模式为零。
`destroyed_before_return` 在局部 cache 仍持有 plan 时采样，最终资源门在 one-call 返回并销毁 cache 后检查。

## 六次完整结果

| 顺序 | 模式 | outer 秒 | Stage B 秒 | 硬门 |
|---|---|---:|---:|---|
| 1 | OFF1 | 336.195 | 144.350986 | 70/70 |
| 2 | ON1 | 311.264 | 127.801689 | 70/70 |
| 3 | ON2 | 311.440 | 127.795989 | 73/73 |
| 4 | OFF2 | 327.667 | 144.053423 | 73/73 |
| 5 | OFF3 | 328.464 | 144.753917 | 73/73 |
| 6 | ON3 | 318.820 | 127.878532 | 73/73 |
| 中位数 | OFF | **328.464** | **144.350986** | PASS |
| 中位数 | ON | **311.440** | **127.801689** | PASS |

重复轮增加三个 repeat 门。OFF1 另外与已接受的父 native 输出逐位对齐；ON1–ON3 与本次 OFF1 同 native 对照。
六次各覆盖 240,498 个 CI，CI/pMax 位模式、task/order/deletion、adjacency、sepsets、n.edgetests、
RNG endpoint、physical groups 和 Gram 顺序都相同。独立保存的 mgcv authority 最大 CI/pMax 绝对误差
均为 **3.05425129631942e-12**，低于 `1e-10`。CPU 数值 authority/fallback、生产 residual/component
payload D2H、route/status mismatch 和 dense fallback 均为零。

每次实际执行计数均为指定模式 **10 个 plan、2,044 次 submit、130,670 个 target**，另一模式为零。
三次 ON 全部快于三次 OFF：最慢 ON **318.820 秒** < 最快 OFF **327.667 秒**。三个预声明性能门全部成立。
Stage B 中位数减少 **16.549297 秒（11.4646%）**，与 outer 收益规模相符；各阶段中位数不能作可加的精确账本。
OFF1 与 ON3 的 Stage B 以外时间有约 8 秒波动，不能把第一组 24.931 秒的差作为典型收益。

### 各层 Stage B

| level | component | active contexts | OFF 中位数秒 | ON 中位数秒 | 观察降幅 |
|---|---:|---:|---:|---:|---:|
| L2 | 49,919 | 10 | 54.789273 | 48.486387 | 11.50% |
| L3 | 54,878 | 10 | 60.257073 | 53.260696 | 11.61% |
| L4 | 17,767 | 10 | 19.798211 | 17.547667 | 11.37% |
| L5 | 6,816 | 10 | 7.754200 | 6.862959 | 11.49% |
| L6 | 1,148 | 10 | 1.273936 | 1.149412 | 9.77% |
| L7 | 124 | 2 | 0.283935 | 0.282481 | 0.51% |
| L8 | 18 | 1 | 0.266220 | 0.264827 | 0.52% |

上表是各层分别取中位数；总 Stage B 的中位数从每次的总和取，不能把两者混用。
L7/L8 只有约 1.4 毫秒差，不据此宣布独立性能收益或重开 L8 optimizer。shared/寄存器变化没有改变原 cohort、
lane 数、wave 数与缓存策略。当前数值覆盖限定于 canonical 351×48 及已列合成案例，不外推到任意输入性能。

### 容量与生命周期

六次以下原始数值完全一致，且均通过原门：

| 指标 | bytes | 既定门 |
|---|---:|---|
| L3 creation | 1,218,445,312 | ≤1.5 GiB |
| L3 optimizer growth | 1,205,862,400 | ≤1.25 GiB |
| L3 optimizer raw total | 2,424,307,712 | ≤2.75 GiB |
| L3 minimum free | 3,404,857,344 | ≥2 GiB |
| L4 pool increment | 809,500,672 | ≤1 GiB |
| L4 minimum free | 2,689,728,512 | ≥2 GiB |
| L6 pool increment | 473,956,352 | ≤1 GiB |
| L6 minimum free | 2,257,715,200 | ≥2 GiB |

L3 每次覆盖 4,068 个容量 checkpoint 与 4,066 个 Stage A checkpoint。外部 200 ms 请求间隔采样的 GPU1
峰值均为 22,081 MiB；这是采样峰值，不是连续最高值证明。最终 application device allocation/bytes、stream、
event 计数全部为零。没有外部 GPU 任务干扰，各进程 exit 0；本任务 GPU 进程均已结束。

## 证据位置

工具：`fastkpc/tools/research/stage_b_work_pair_runtime_20261028/`。
证据：`fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_runtime_20261028/`。

- `deep/`：未改动 native 的 capture、逐位资格、原协议与封存 manifest。
- `runtime/`：集成构建、配置/生命周期检查、同 native 完整 campaign 与 closure。
- `runtime/deep-tool-source/`：深层阶段使用的工具精确版本。深层结束后 README 仅增加后续执行命令；
  历史版本按 deep manifest 的哈希保存，执行代码未改变。
- `runtime/checks-build.log` 保留第一次诊断 harness 编译的 nvcc rpath 参数错误；修正为 `-Xlinker` 后通过，
  该次错误未启动 GPU 工作，也未改变生产 native。

封存后独立重算：deep **46** 个 artifact 文件与 runtime **215** 个 artifact 文件集合及 SHA 全部一致；
runtime 的 **104** 个生产输入、**12** 个工具、**10** 个依赖一致；deep 的 **103** 个父源码快照与
**6** 个历史工具版本一致。外层 `verification.json` 记录复核结果。

| receipt | SHA-256 |
|---|---|
| deep closure | `ae7e6262432653e4afc4a526e2c412b424ff54367116327bd3cca6a06b5d5ae8` |
| deep manifest | `f05ef59778b3cec7a7670455e42b623b6d67ca046182c41ee8a40cb739f309e8` |
| runtime closure | `1532a0e19a9a8e02ab56b21ca9e53f306e7c4ebf5a34386e7960c34c46ef87a0` |
| runtime manifest | `1390844de99fb06b580dccb562ad521279ecafa2bdfb638119893aebb2dfa0ef` |
| independent verification | `e7bbededa77a6ae94c033b347bd60408af8d545af6ad7e836248d2016fe49b3e` |

本项按原计划完成并保留默认关闭实现，不再追加微基准或改动参数。Phase 10 ACTIVE、candidate NOT_FROZEN、
holdout SEALED、recommended=false、L8 optimizer STOP 与默认关闭政策保持不变。没有 commit 或 push。
