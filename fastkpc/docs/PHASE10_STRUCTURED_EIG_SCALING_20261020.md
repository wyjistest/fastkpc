# 结构化 eig 原型的大批次吞吐补测

> 日期：2026-10-20
> 性质：对 [执行记录](PHASE10_EXACT_SPEEDUP_EXECUTION_20261020.md) §3 的补充测量。没有改动生产代码、原型源码或 native；只在 GPU1 上运行研究原型。
> 结论边界：这是隔离的分量级吞吐测量，不是完整方法的耗时或资格证据。
>
> 后续：[真实残差与有界批次诊断](PHASE10_STRUCTURED_EIG_REAL_REPLAY_20261020.md)
> 已完成256/640分量、11 GiB容量共驻留、深层/三真实输入witness及独立剖析。
> 分量级推进GO，完整生产集成仍NOT YET GO；整层预计算的缓存容量需按新增
> 未消费分量与逐层生命周期重算，不能直接引用本页的已消费分量容量证明。

## 0. 摘要

- 执行记录测的是 9、16、48 个分量的小批次。那个规模下结构化原型比 dense 慢 2–9 倍，所以没有通过性能推进门。
- 原型每个分量只用一个 32 线程 block。单个分量的串行双对角 QR 约需 250 ms，小批次时 GPU 大部分空闲，所以 9→48 个分量的耗时几乎不变（277.8→285.5 ms）。这是单分量延迟主导的特征。
- 补测更大的批次：
  - dense 的吞吐固定在约 355 个分量/秒（设备已饱和）。
  - 结构化随批次增大，在 2,560 个分量时达到约 1,000 个/秒，耗时为 dense 的 0.36 倍。
- 结构化要占优，前提是分量能大批量计算。生产路线目前每批平均只有约 3 个分量，直接接入一定更慢。可行的方向是"按层预先计算分量"（§4）。
- 按实测吞吐换算，canonical 的 eig-only 工作量大约是：dense ≈ 310 s，结构化 ≈ 111–133 s。这只是吞吐换算，不是流水线耗时预测。
- 即使如此，120 s 门槛仍然达不到（§5）。

## 1. 测量方法

- **原型**：研究目录中未修改的 `structured_eig_gpu.cu` 和 `generate_recorded_dbdsqr.py`，按研究目录 README 的命令编译到临时目录。
- **dense 对照**：`fastkpc/build/phase10_cuda_full_eig_lane_benchmark`，8 条 lane，取 eig-only 的设备时间中位数。48 个分量时得到 138.6 ms，与执行记录的 139.041 ms 一致。
- **大批次的模拟方式**：原型的单个 context 最多接受 64 个分量（`structured_eig_create` 要求 `count ≤ 64`）。所以更大的批次是用多个 64 分量 context 并发运行模拟的：每个 context 一个主机线程、一个 CUDA stream，`CUDA_DEVICE_MAX_CONNECTIONS=32`。
- **计时口径**：结构化计的是墙钟，从启动全部线程到全部 join，包含 H2D、四个设备阶段、全部 values/vectors/means 的诊断性 D2H，以及主机同步。dense 只计 eig 的设备时间，不含距离矩阵构造和中心化。**两种口径都偏向 dense。**
- **rotation log 容量**：默认的 8·n² 条/分量装不下几十个 context，所以先用一个 context 实测每个分量的旋转数（最大 113,092，均值 105,471），再按 1.5 倍加 1024 取 170,662。所有运行都检查了没有超容量；超容量时原型会 fail closed。
- **输入**：与 `benchmark_structured_eig.py` 相同的合成序列，不是真实残差。
- **重复与确定性**：每个规模先预热一次作为参照，再取 3 次的中位数；每次输出都与参照逐位一致。

## 2. 结果

n = 351，k = 35，GPU1（RTX 4090）：

| 分量数 | context 数 | 结构化墙钟 | 结构化吞吐 | dense eig-only | dense 吞吐 | 结构化 / dense 耗时 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 288.0 ms | 222/s | 182.6 ms | 350/s | 1.58 |
| 256 | 4 | 355.3 ms | 721/s | 719.5 ms | 356/s | 0.49 |
| 640 | 10 | 733.6 ms | 872/s | 1,797.0 ms | 356/s | 0.41 |
| 1,280 | 20 | 1,349.6 ms | 948/s | 3,596.5 ms | 356/s | 0.38 |
| 2,560 | 40 | 2,560.5 ms | 1,000/s | 7,179.1 ms | 357/s | 0.36 |

单个 64 分量 context 首次运行的阶段耗时（prepare / 双对角 QR / roots / recover）为 0.9 / 268.6 / 10.4 / 31.2 ms，QR 占绝大部分，与执行记录中"48 个分量里约 247 ms 在串行 QR"一致。

## 3. 解读

- **dense 是设备饱和的。** 吞吐在 64 到 2,560 个分量之间基本不变，与执行记录"`sytrd4_gpu` 占满设备、扩 lane 无收益"的结论一致。
- **结构化在小批次时受单分量延迟限制，大批次时由吞吐决定。** 吞吐在约 1,000 个/秒附近趋于饱和。饱和点的限制因素没有剖析，可能是 QR kernel 的 shared memory 占用（每个 block 7·n 个 double，约 19.7 KB）、recover 阶段，或者主机线程和诊断性 D2H。
- **执行记录的停止决定针对的是当前接法。** 它停止的是"按现有批次结构集成当前实现"，并说明不外推为所有结构化算法都不可行。本补测说明：同一个实现，在大批次下已经快于 dense。

## 4. 前提：按层预先计算分量

- **为什么需要。** 生产路线是按 frontier 小批次计算分量的。canonical 中有 110,683 个 unique miss，分在 38,622 个 component batch 里，平均每批约 2.9 个（旧收口文档）；22 列子集的 retained 剖析是每批 3.63 个。在这个批次结构下，结构化 eig 每批都要付约 280 ms 的延迟，会比 dense 慢得多。
- **为什么可行。**
  - whole-level prefill 已经把一层的所有 target 都优化完了：8 月 8 日回执中 prefill target 为 132,926 个，其中实际被消费的 110,635 个。
  - 分量只取决于残差，与检验顺序无关。所以可以在一层的优化和残差物化完成后，整层成批计算全部分量，检验阶段直接读缓存。
  - retained 的 component cache 容量是 119,817 条，能装下 canonical 全部 110,683 个已消费 unique 分量。它不能直接证明可容纳约132,974个预计算分量：当前缓存满后停止插入，没有自动淘汰，须先验证按层回收/有界驻留。exact-primary 的系数 arena 本来就按层组织，但系数驻留也不等于整层residual物化已通过资格。
- **代价与未知。**
  - 未被消费的 target 也会算分量，数量约为被消费分量的 20%。
  - 残差物化要覆盖整层，不再按被消费的批次逐批进行。
  - 结构化 solver 的身份必须写进 cache key。
  - 分量的低位比特会变化（约 1e-13），必须按完整方法重新资格验证。
- **吞吐换算（不是流水线预测）：**
  - dense：110,683 / 357 ≈ 310 s。作为对照，旧收口文档中 canonical full eig 为 407.7 s，理想 packing 下为 323.6 s。
  - 结构化：110,683 / 1,000 ≈ 111 s；如果把未消费的 target 也算上（约 132,974 个分量），约 133 s。
  - 这两个数都没有计入残差物化、调度和流水开销，也没有用真实残差的旋转数。

## 5. 与门槛的关系

在 32 connections 下，canonical 880.659 s 中 optimizer 占 335.361 s，exact residual 34.431 s，native setup 22.592 s，另有约 50 s 的控制与打包。即使 eig 降到上面的吞吐换算值，总时间也远高于 120 s。这条路线最多只能帮助接近相对门槛（cold ≤ 601 s、compute-warm ≤ 481 s），而且前提是流水线开销小、optimizer 不回退。本文不给耗时承诺。

## 6. 建议的下一步（按顺序）

1. **剖析饱和点**：对 ≥ 640 个分量的并发运行做 nsys，看 QR kernel 的实际驻留、recover 阶段占比和主机侧开销，判断约 1,000/s 的上限从哪里来。
2. **换成真实残差**：从 22 或 30 列子集导出真实残差作为输入，检查旋转数、容量和 fail-closed 行为，以及吞吐是否保持。真实数据里有近乎并列的值，旋转数可能不同。
3. **子集流水线原型**：在 22 或 30 列子集上实现"按层预先计算分量 + 结构化 eig"，与 32 connections 的 retained 基线比较各阶段耗时，并检查完整方法的门槛：p 值对 oracle ≤ 1e-10，图、trace、sepset、`n.edgetests` 一致，资源释放。
4. **全量资格验证**：前三步都通过后，再做 canonical 和三组真实数据，以及 `PHASE10_EXACT_SPEEDUP_FINDINGS.md` §4.7 的全部资格项。

## 7. 复现

在仓库根目录运行。结构化库的编译命令见研究目录 README 的 "Structured eig isolation" 一节。

```sh
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -I fastkpc/tools/research/exact_speedup_20261020/benchmark_structured_eig_concurrency.py \
  "$structured_build_dir/structured-eig.so" \
  fastkpc/build/phase10_cuda_full_eig_lane_benchmark \
  OUTPUT.json
```

脚本先用一个 context 测旋转数并据此设容量，然后依次测 64、256、640、1,280、2,560 个分量，打印上表并写出 JSON。本次的 JSON 在 `/tmp/claude-structured-dcov/structured_concurrency.json`，机器重启后会丢失；表中数字已完整记录在本文。
