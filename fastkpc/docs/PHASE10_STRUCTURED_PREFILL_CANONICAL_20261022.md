# Structured component prefill：canonical 资格

日期：2026-10-22。按
[22 列结果复核](PHASE10_STRUCTURED_PREFILL_REVIEW_20261022.md)
执行受限的 351×48 opt-in 扩展。off 兼容性、完整 canonical 硬门及
dense/structured/structured/dense 交错重复全部通过。

```text
完整生产源码快照、空目录全量构建       PASS
351×48 default-Inf 全部数值/离散硬门  PASS（240,498 CI / run）
同模式两次 CI/pMax 逐位确定性         PASS
canonical 容量和资源闭合             PASS
canonical opt-in development 性能   GO（两次中位数下降 17.46%）
生产默认 / promotion / holdout       不变
```

证据为同一 native 上 dense/structured 各两次的 source-closure development
运行；不是 clean producer commit、五次中位数或 promotion。

## 完整结果

| 模式 | 完整 skeleton 秒 | Stage B 秒 | optimizer host 秒 |
|---|---:|---:|---:|
| off | 895.417 | — | 335.658 |
| dense1 | 828.918 | 233.288 | 327.697 |
| structured1 | 693.783 | 89.402 | 336.124 |
| structured2 | 688.537 | 89.503 | 330.688 |
| dense2 | 845.749 | 233.339 | 344.399 |

| 各两次中位数 | dense | structured | structured − dense |
|---|---:|---:|---:|
| 完整 skeleton 秒 | 837.3335 | 691.1600 | −146.1735 |
| Stage B 秒 | 233.313767 | 89.452639 | −143.861128 |
| optimizer host 秒 | 336.048381 | 333.406280 | −2.642101 |
| consumer dCov host 秒 | 181.209745 | 181.309881 | +0.100136 |

完整耗时下降 **17.457%**。两次 structured 都快于两次 dense，最小间隔也有
135.135 秒；Stage B 的 143.861 秒下降解释了几乎全部整体收益。两次中位数
只描述这四次运行，不是正式性能 campaign 的中位数。

optimizer host 的顺序是 327.697 → 336.124 → 330.688 → 344.399 秒，迭代总数
始终为 890,171、最大迭代数为 187。此次没有看到可重复的 optimizer 回退；
也不能从四次结果确定其波动原因或认定 structured 加速了 optimizer。Stage B
两档各自仅相差约 0.051/0.101 秒，收益判断主要依赖这项稳定差异。各层和其他
阶段存在包含关系，不能把所有 receipt 计时直接求和。

## 数值、账本与容量

五次运行均通过全部 240,498 个 CI/pMax authority 门，最大误差
`3.0547508966805e-12`。off 对历史 canonical off 的 CI/pMax 逐位相同。
两次 dense 之间、两次 structured 之间的 CI/pMax 均逐位相同。
structured 对 dense 有 60,239 个 CI 和 344 个 pMax 单元发生低位变化，
最大差异 `1.12132525487141e-14`；alpha flips 为零。task/deletion、图、sepsets、
counts、RNG、physical cohort/Gram 摘要、GPU-only 和资源门全部通过。
authority p∈[0.05,0.15] 的 1,337 个 CI 也全部通过；其中 179 个相对 dense
有低位变化，最大 authority 误差仍为 `3.0547508966805e-12`。

两次 structured 的 80,751 个分量均全部 accepted，natural dense fallback 为零；
每次 resolved 65,694、pending unused 15,057，账本一致。
共 129 waves，最大 640 在途。最小边界 gap 为 `1.2048292179898e-4`，
最大 rotations 为 133,473（容量 170,662）。pool 保留 3,422,507,520 bytes。
两次 dense/两次 structured 的整卡采样峰值均分别为 18,681/21,991 MiB；
structured 峰值约 21.48 GiB，后者余量为 2,573 MiB。off 峰值为 18,677 MiB。
这个输入完成了整轮容量检查，采样峰值仍不是瞬时严格上界。无需调整 cache
预算或 pool 生命周期。GPU1 在全部五次运行的采样利用率均为 0%，显存为 425 MiB。

## 构建与实现边界

- native SHA：`89a2fab0bd02b695878556a74694fc522cddcbe2c5c2f22f74c7e4da6d5d56c3`。
- 完整生产源码快照、空 build 目录，从头编译并链接全部 43 个生产对象；没有继承
  旧对象。快照中没有 `tools/research` 目录。证据类别仍为未提交的 source-closure
  development，不是 clean producer commit 或 promotion。
- DBDSQR 生成器放到 `fastkpc/tools/generate_recorded_dbdsqr.py`，从脚本位置定位
  vendored 输入。生产构建不再依赖研究目录；在仓库以外的 cwd 运行时，生成结果
  与前一轮归档逐字节一致。
- 相对 22 列实现，运行时只扩展为 `n=351, p∈{22,48}, numCol=35`。
  structured kernel、consumer、selected-SP、residual、原 Gram 分组、scheduler、
  回退阈值、11 GiB cache 和八条 dense eig lanes 均保持原样。开关仍默认关闭。

## 已完成的局部检查

- 新旧配置拒绝及资源闭合、retained component-cache regression：PASS。
- 640 个真实分量的三字段与归档适配器逐位相同：PASS。
- 70 个混合输入中，56 个真实小 gap 分量触发 GPU dense 回退，按 48+8 分块；
  回退三字段与原 dense 逐位相同：PASS。
- 原 Gram、pending reset、失败不发布、rotation overflow、abandoned submit：PASS。
- compute-sanitizer 覆盖 4 个归档分量和上述混合/生命周期用例：
  `0 errors / 0 bytes leaked`。这不是全 R 流程的 sanitizer 运行。

## 完整运行协议

顺序为 `off → dense1 → structured1 → structured2 → dense2`。GPU0 顺序执行，
GPU1 不运行计算任务；每次 fresh R process、fresh dataset caches，固定 32 CUDA
connections、数学库线程数 1、同一 native。每次启动前检查两卡无其他计算进程。

off 对前一份 canonical off 回执要求 CI/pMax 逐位相同。所有运行的全部
240,498 个 CI 和所有 pMax 值必须对独立 kpcalg authority 满足 `1e-10`，并保持
task/deletion、adjacency、sepsets、counts、RNG 及原 physical cohort/Gram 摘要。
structured 对 dense 的低位变化单列；同模式重复要求 CI/pMax 逐位一致。
CPU 数值 authority/fallback、生产 residual/component 大 D2H 均须为零，CUDA
资源最终闭合。任一硬门失败即停止后续运行。

完整 skeleton 调用是唯一性能计时边界，校验、压缩与落盘不在其中。Stage B
采用完成后的 host wall time，不把并发 lane event 时长相加冒充 makespan。
显存使用两卡 200-ms 外部采样，结果仅为采样峰值。

证据目录：`fastkpc/artifacts/full_cuda_ci/structured_prefill_canonical_20261022/`。
复现工具：`fastkpc/tools/research/structured_prefill_canonical_20261022/`。

- `closure.json`：机器资格结论、两次中位数、数值差异、容量及账本。
- `manifest.json`：完整源码快照、构建对象、native、回执、工具及输入 SHA。
- manifest SHA：`54b9e45568c880c1f7709188b72db0e00aa9738d51e89f16fe3b2da72df1e6d3`。
- `build.json` / `checks.json`：全量构建和局部资源/回退资格。
- `timing.csv`、各运行 `report.json` / `receipt.rds` 及 `*-gpu.csv`：原始分解与证据。

Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false。
本轮是完整 skeleton 子资格，不包含新的三真实数据或 WAN-PDAG/orientation campaign。
