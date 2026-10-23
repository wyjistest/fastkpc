# Structured prefill canonical：复核与下一步

日期：2026-10-22。复核 [canonical opt-in 资格](PHASE10_STRUCTURED_PREFILL_CANONICAL_20261022.md)
（native `89a2fab0…`）。本文不改代码、不改状态；数字取自 Codex 归档的回执。
下文的节省量都是情景换算，不是承诺。

> **2026-10-22 执行回执：** [L2 窄诊断已完成](PHASE10_L2_PREFILL_DIAGNOSTIC_20261022.md)。
> 固定真实 selected-SP 后，125,293 个 L2 CI 与冻结完整 trace 的 pMax 均通过 `1e-10`，
> dense/structured 各两次逐位确定。阶段 A 隔离循环约 14.79 秒，故下文 5–10 秒假设
> 偏低；相同 49,919 个分量的 Stage B 两次中位数为 144.34 / 54.46 秒，structured
> 全部 accepted。下一步受限 runtime opt-in 完整资格为 GO；完整 L2/outer 收益尚未
> 实测，默认与 promotion 状态不变。下面保留原复核与情景推算。

## 结论

```text
资格结论          成立；源码 diff、生成器迁移、计时表与路线图条目复核无误
收益              dense → structured 两次中位数 −146.2 s，其中阶段 B −143.9 s，与此前约 −140 s 的情景一致；
                  off → dense 两次中位数 −58.1 s，与此前“结构性收益约 −58 s”的判断一致
显存              采样峰值 21,991 MiB，与推算的 21,990 MiB 一致
optimizer 噪声    同模式两次最多相差 16.7 s（dense 327.7 / 344.4 s）；比较应以阶段 B 这类稳定项为主
当前最大的两块    optimizer 约 333 s（48%）；level 2 约 250 s（36%，未经 prefill）
建议下一步        level 2 single-penalty prefill + structured（情景约 −120 s 量级）；
                  阶段 B 与 optimizer 重叠放在其后，并先实测 FP64 争用
提交              现在是自然的检查点；按明确路径提交，排除根目录大文件（由用户决定）
```

## 复核

- **源码**：相对 22 列快照，one-call 只放宽形状门（p ∈ {22, 48}）；vertical.{cu,hpp}
  与 structured TU 的 SHA 未变；`build_cuda_native.sh` 只改了生成器路径。
- **生成器**：`fastkpc/tools/generate_recorded_dbdsqr.py` 的变换与研究版相同，输入按
  脚本位置定位。我在 `/tmp` 下运行它，输出与归档 `generated-dbdsqr.inc` 逐字节相同。
- **计时**：两次中位数、阶段 B 与 `timing.csv`、`closure.json` 一致；运行顺序为
  off → dense1 → structured1 → structured2 → dense2。
- **路线图**：`goal-5.6.md` 新增的条目与回执一致。

## 逐层（dense1 与 structured1）

单位为秒。

| level | dense1 | structured1 | optimizer（s1） | 阶段 B（s1） | 阶段 B 吞吐 |
|---|---:|---:|---:|---:|---:|
| 0–1 | 7.66 | 7.69 | 0.26 | — | — |
| 2 | 249.50 | 249.75 | 7.30 | — | — |
| 3 | 224.41 | 125.35 | 48.10 | 60.25 | 911/s |
| 4 | 146.55 | 117.71 | 88.23 | 19.58 | 907/s |
| 5 | 126.09 | 113.79 | 100.85 | 7.76 | 878/s |
| 6 | 45.93 | 45.34 | 42.56 | 1.26 | 911/s |
| 7 | 10.04 | 10.08 | 9.46 | 0.28 | 单批延迟 |
| 8 | 16.97 | 22.23 | 21.85 | 0.26 | 单批延迟 |

- canonical 上每层吞吐约 880–910/s，高于 22 列的 844/s，因为 L3 有 86 个满 wave。
- L8 的 +5.3 s 来自 optimizer 单个 setup（16.8 → 21.9 s；structured2 为 16.5 s），
  与此前观察到的单 setup 波动同类。
- level 2 现在是最大的一层，约 250 s，没有经过 prefill。

## 下一步候选

### A. level 2 single-penalty prefill + structured（建议先做）

- L2 计划生产 49,919 个分量，实际消费 42,685 个（未消费 14.5%）；按需 dense eig
  约 157 s；single-penalty residual solve 34.4 s（L1–L2 合计，其中 mixed 29.9 s，
  all-miss 4.6 s）；optimizer 7.3 s。
- 情景换算：structured 阶段 B 约 49,919 / 900 ≈ 55 s，consumer eig 降为 0。residual
  solve 移到阶段 A，按 Codex 的意见不当作白省；若阶段 A 花 5–10 s，L2 约
  250 − 157 − 33 + 55 + (5–10) ≈ 120–125 s，即约 −125 s。这只是量级，需要实测。
- 新工作：
  - single-penalty 阶段 A：从 single-penalty prefill 窗口物化残差；
  - begin/build/finish 扩到 level 2；
  - 容量核对：L2 的 49,919 个，加上 L0–L1 的 2,304 个，低于 119,817；
  - 按合同资格，并对残差 cohort 改变造成的低位差异逐项归因，与 canonical dense
    对 951 个宽 cohort 分量的做法相同。

### B. 阶段 B 与 optimizer 窗口重叠

- 理想重叠上界 Σ_L min(阶段 B_L, optimizer_L) ≈ 48.1 + 19.6 + 7.8 + 1.3 + 0.3 + 0.3
  ≈ 77 s。做完 A 之后，L2 最多再加 7.3 s。
- 风险：structured QR 是单线程 FP64 代码，到 512–640 个在途时饱和；optimizer 的
  device LAPACK 移植很可能也是单线程 FP64 代码。二者同时运行，可能拖慢 optimizer
  的关键路径。建议先在 22 列上按窗口提交阶段 B（2、5、10 个 lane），对照 optimizer
  host 时间，再决定是否实现。

### C. optimizer（约 333 s）

这是最大的一块，单次波动约 ±8 s。任何接近 120 s 的路径都必须经过它；不属于
prefill 这条线。若要继续，应先在 32 connections 下重新剖析 optimizer 的占用。

做完 A 和 B 且都接近上界时，约为 691 − 125 − 77 ≈ 490 s。

## 推广到其他输入时要注意

rotation 容量固定为 170,662。canonical 的最大实测值为 133,473，余量约 22%；
n = 364 时为 118,193。rotation 数约随 n² 增长：n = 1000 时每个分量约需 26 MB 日志，
640 个在途约 17 GB，无法容纳。推广前应按 n 计算容量，并按显存预算确定 pool
宽度。溢出时已有 dense 回退，结果正确，只是变慢。

## 提交建议

现在是自然的检查点：空目录下全量构建了 43 个对象，canonical opt-in 资格通过，
开关默认关闭。建议用明确路径提交：

- 已修改：vertical.{cu,hpp}、one-call、`build_cuda_native.sh`、lane benchmark、`goal-5.6.md`；
- 新增：`full_cuda_ci_structured_eig.{cu,hpp}`、`fastkpc/tools/generate_recorded_dbdsqr.py`、
  两个新测试、`with_phase10_exact_development_env.sh`、本系列 `PHASE10_*` 文档、
  `fastkpc/tools/research/`（约 1.2 MB）。

排除：

- 根目录的 `causalDiscovery.zip`（738 MB）、`causalDiscovery/`、`claude-21303`、
  `claude-21305`（各约 286 MB）；
- 各处 `__pycache__`（建议加入 `.gitignore`）。

不要用 `git add -A`。`fastkpc/artifacts/` 大部分已被忽略，文档按既有惯例引用
本地证据。是否提交、是否 push，由用户决定。
