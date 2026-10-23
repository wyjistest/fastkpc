# Structured device 适配器：复核与在途宽度测量

日期：2026-10-22。复核 [适配器资格](PHASE10_STRUCTURED_DEVICE_ADAPTER_20261022.md)。
本文不改生产代码、native 或状态；新增一项只读的三字段生产计时（GPU1）。

执行补充：下表保留原 host-return 计时，最后的 D2D 完成未显式纳入边界。
后续补测加入最终 completion join，10×64 完成吞吐仍为约 935/s；默认关闭的
22 列完整 skeleton 已通过，见[执行结果](PHASE10_STRUCTURED_PREFILL_RESULT_20261022.md)。
下述约 14/140 秒仅为样本吞吐外推情景，不是严格上界或 canonical 性能证据。

## 结论

```text
适配器资格           成立；qualification.json 与文档数字一致，代码路径未见错误
新测的设计风险       当前实现/workload 的 structured 吞吐受同时在途分量数显著影响：
                     单 context 22 宽 80/s、64 宽 232/s（dense 为 326/348/s）；
                     10×64 约 935–945/s，20×64 不再增加
对 22 列完整方法门   阶段 B 不能沿用按 p 容量（22/48）逐批同步的循环；
                     需要 320–640 个分量同时在途，例如 10 个 64 宽 context，
                     由同一主机线程先全部提交、再逐批 checkpoint
接受的修正           L2 的 residual solve 不能整体计为节省；约 495 s 的组合场景不进入预算
```

## 复核要点

- **数字**：六组输入的 authority 误差、对 dense 差异、fallback 计数、最小 gap
  （canonical951 为 3.706e-4）和最大 rotation 数（118,193）均与文档一致。
  mean 对 dense 的差为 0，即 total 逐位不变。
- **候选与选择**：`roots` 求出 Perron 根、最低极点以下的根、最低 k+1 个区间根，
  以及 deflated 极点，按 |λ| 降序排序，并列时取有符号值较大者。这与 dense 的
  “并列取较大升序下标”在相反符号、同幅值时对应；重复同号特征值不能据此
  保证向量基逐位相同，仍须边界 gap 与最终数值资格。按幅值排在前 k+1 的根
  必在候选中。适配器以
  k+1 创建求解器，只取前 k 个值与向量；`check_boundary` 以 k+1 为步长读取第
  k−1、k 位，索引一致。
- **退化输入**：按实际第 k、k+1 个幅值计算边界 gap。若恰好有 k 个非零值，
  第 k+1 个为 0，归一化 gap 为 1；不能笼统声称少于 k+1 个非零值就回退。
  两个边界值均为 0 时 gap 为 0，进入 dense；真实二值测试已覆盖该情况。
- **流序**：structured 提交等待 `input_ready`；`finish` 等待 `view.complete`；
  structured 分量的 D2D 存储先于 dense 回退覆盖本地槽位入队；下一批的 gather
  排在上一批存储之后；consumer 等待 `payload_ready`。未见读写冲突。
- **total**：沿用 `build_distance_rows_batch_kernel` 与行归约，逐位不变。代价是
  每个分量物化一份 n×n 距离矩阵（n = 351 时约 1 MB）。生产版若改为只算行和，
  须保持相同的求和次序并单独做逐位检查；这一项可选。
- **编译合同**：structured TU 使用 `--fmad=false`，consumer TU 沿用生产编译选项。
  集成时须把求解器保留在独立 TU 中，带同样的 `--fmad=false` 与 `-arch`，否则
  本次资格不能沿用；建议在 `build_cuda_native.sh` 中加构建期断言，并写入源码闭包。
- **gap 门**：1e-8 是求解器稳健性的保护，不是 p 误差的证明，同意 Codex 的表述。
  建议回执记录逐层最小 gap 和 gap < 1e-4 的分量数，仅作监测。

## 在途宽度测量

范围：只测三字段生产，即 gather、原 total、structured 或 dense eig、status
checkpoint、centering 和 D2D 存储；不含 pair，计时区内没有 shadow D2H。
输入为 640 个真实 n = 351 残差。GPU1，connections = 32；预热一次后取三次中位数。

| 生产方式 | 同时在途 | 时间 | 分量/s |
|---|---:|---:|---:|
| structured，1 context，22 宽（30 批） | 22 | 7.989 s / 640 | 80 |
| dense，1 context，22 宽 | 22 | 1.966 s / 640 | 326 |
| structured，1 context，64 宽（10 批） | 64 | 2.764 s / 640 | 232 |
| dense，1 context，64 宽 | 64 | 1.837 s / 640 | 348 |
| structured，2×64，线程池 | 128 | 1.509 s / 640 | 424 |
| structured，5×64，线程池 | 320 | 0.760 s / 640 | 842 |
| structured，10×64，线程池 | 640 | 0.684 s / 640 | 935 |
| structured，5×64，单主机线程 | 320 | 0.396 s / 320 | 809 |
| structured，10×64，单主机线程 | 640 | 0.677 s / 640 | 945 |
| structured，20×64，单主机线程 | 1,280 | 1.450 s / 1,280 | 883 |

“单主机线程”指先对所有 context 调用 `begin`，再依次 `finish`。

- 每个 structured 批耗时约 0.27 s，基本不随宽度变化（22 宽 0.266 s，64 宽
  0.276 s）。这只是小宽度串行批的近似；增大并发后整组延迟也增长，不能普遍
  用“在途分量数 / 0.27 s”推导吞吐。当前样本在 320–640 个附近达到高吞吐。
- 机制推测，未做剖析：bidiagonal QR 是单线程 FP64 代码，每个 block 占
  7n 个 double 的 shared memory（n = 351 时 19.7 KB），每个 SM 约可驻留 4 个
  block；recover 在 k+1 = 36 时每个 block 约 99 KB，每个 SM 只能驻留 1 个。
  这与饱和点的量级一致。需要 ncu 才能确认。
- 一个主机线程就够：所有批提交之后再做 checkpoint，即可达到线程池的吞吐。

## 对 22 列完整方法门的建议

1. **阶段 B 的 structured producer** 保持 320–640 个分量同时在途，例如 10 个
   64 宽 context 先全部提交、再逐批读 status 与做 dense 回退。若放开 64 的上限，
   也可以改用单个更宽的 context。回退走现有 p 容量的 dense context。
2. **显存**：10×64 的 workspace 约 2.54 GiB（n = 351 时 rotation 容量 170,662），
   另有 11 GiB cache；峰值需实测。
3. **样本吞吐外推情景**：22 列 level ≥ 3 计划生产 7,270 个分量。按 900/s
   约 8 s，对照 dense 阶段 B 约 21.9 s（22 宽），即每次运行约 −14 s。canonical 的
   80,751 个约 90 s，对照 231 s eig。若用单 context 实现，22 列上 structured 会
   比 dense 慢约 4 倍；这是实现方式造成的，不是方法本身的性质。
4. **小层**：一个 structured 批至少约 0.27 s，L8 的 dense 只要 0.055 s。差别可以
   忽略，不需要特殊规则。
5. **回执**：记录 solver identity 计数、回退原因、逐层最小 gap、在途 context 数
   与宽度、structured workspace 字节和显存峰值。

## 证据

工具：`fastkpc/tools/research/structured_width_20261022/`（`width_timing.py`、
`single_thread_pipeline.py`，复用 Codex 的 `adapter.so` 与 `run.py` 中的 `Probe`）。
输出与 manifest：`fastkpc/artifacts/full_cuda_ci/structured_width_20261022/`
（manifest SHA `c8b24bf1aaf16502962a272c09724ad8bd11177d5ee0a3bb16c4f4c78b3ed2cc`，
绑定 native `b1e7eab4…`、`adapter.so`、`run.py` 与 real640 输入）。这是三字段生产的
开发测量，不是完整方法性能证据。

复现方法：在仓库根目录，GPU 空闲时执行：

```sh
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_MAX_CONNECTIONS=32 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/cuda/lib64 \
python3 fastkpc/tools/research/structured_width_20261022/width_timing.py OUT.json
```

`single_thread_pipeline.py` 的用法相同。
