# Guarded-QR 窄诊断：当前守卫不进入生产

2026-10-20。执行
[GPU design proposal](PHASE10_MAGIC_GPU_DESIGN_PROPOSAL.md)
的判别实验，范围限于现有 optimizer 的 captured cohorts。

**结论：当前这组裕量守卫与双 target cohort 重跑策略 STOP 集成。**
未经整条 exact 重跑的 QR 确实更快，但存在超过数值合同的误差；
冻结守卫捕获了本次独立验证中的全部失败，却几乎消除了快路径的收益。
继续保留 exact-primary，不进入 warp/shared-memory runtime 改写。
这不是所有 QR 算法或更精确守卫的不可行性证明。

## 样本与隔离方式

- 72 个固定双 target cohort，144 个 targets，q 为 28/37/46/55/64/73。
- 20 个标定 cohort、52 个独立开发验证 cohort；相同 dataset/conditioning set
  不跨分区。含历史困难 witness、canonical 任务和三套大型真实输入。
- 历史真实数据回执只提供任务身份；其旧 p 值不作参考，也不参与选样。
- GPU 0，RTX 4090，`CUDA_DEVICE_MAX_CONNECTIONS=32`，BLAS/OMP/MKL 各一线程。
- 独立、改名后的诊断库编译生产 optimizer 的副本；关闭整条 stability replay
  和边界 exact-shadow confirmation。保留并记录既有逐点评估的 QR→SVD reroute。
- 两条轨迹均把 selected log-SP 交给同一生产 GPU exact fixed evaluator，
  再进入相同的 GPU full-eig/gamma consumer。没有 CPU 数值回退。
- 先抽查相同 SP 点上的 QR/SVD score、gradient、Hessian、coefficients，
  再比较独立完整轨迹及最终 residual/p。共得到 826 个共享 SP 点对照。
- 守卫只读快路径特征。标定后冻结阈值；验证阶段不调参，不读取 sealed holdout。

q=82、更大 shape、完整 skeleton/PDAG 和正式外层性能 campaign 不在本次范围。
参考是已经接受的 exact GPU producer；本次不替代全量 mgcv authority 资格验证。

## 结果

| 指标 | 标定 | 独立开发验证 |
|---|---:|---:|
| cohorts / targets | 20 / 40 | 52 / 104 |
| 原始 QR 的 p 误差超过 1e-10 | 3 | 3 |
| 原始 QR 任一数值门失败的 cohorts | 3 | 12 |
| 最大 p 误差 | 3.184342e-9 | 2.448516e-5 |
| alpha=0.1 决策翻转 | 0 | 0 |
| 冻结守卫标记的 targets | 30 / 40 | 92 / 104 |
| 至少一个 target 被标记的 cohorts | 19 / 20 | 51 / 52 |
| 守卫漏掉的数值失败 | 0 | 0 |
| 守卫漏掉的 1e-11 p 安全裕量失败 | 0 | 0 |

数值门同时检查 fitted/residual 最大绝对误差及相对 L2、RSS、GCV、EDF、
rank/info、dCov statistic、moments 和 p 值。全部 72 对 CI 无决策翻转，
仍然有 6 对 p 值超标，说明只检查决定或图结构不足以验收。

冻结尺度为 16，对应 score/noise ≤512、收敛相对裕量 ≤0.0016、
Hessian 最小绝对特征值/最大绝对特征值 ≤1.6e-7、Newton 限幅相对裕量
≤0.0016、轨迹最大 condition ≥62500 等触发条件；rank/root/status/mixed-SVD
风险也会触发。该规则是经验筛选，没有构成误差证书。

在冻结阈值下，仅移除 mixed-SVD 触发项的诊断对照没有改变两组标记结果；
高标记率不能单归因于这一项保守规则。该对照未改变验证时使用的守卫。

## 离散决策一致仍然超标

`formal-formal-partition-dense-boundary-edf-tolerance` 中，两个 target
记录的接受/拒绝、收敛、Hessian 符号、Newton 限幅及边界探测决策序列一致：

```text
exact p                        0.30626887624698551
QR-selected + exact-final p     0.30626887306264394
absolute p difference          3.18434156820757e-9
maximum selected log-SP error  3.047941e-7
maximum residual error         6.373352e-9
maximum EDF error              7.305555e-8
```

在该 witness 抽查的共享 SP 点上，score 最大差约 8.22e-15，gradient/Hessian
最大差约 6.77e-16 / 9.18e-16。小的逐点评估误差与相同的决策序列，
没有阻止连续状态漂移及最终 p 超标。selected-SP 的 1e-6 诊断阈值也不能
替代 CI 数值门。

所有样本中 141/144 个 target 的已记录离散决策序列一致；仍有三个
完整 cohort 在两 target 的决策序列均一致时出现 p 误差超过 1e-10。

## 大型真实输入中的反例

`20240102191119-L5-1`，n=242、q=46、五个 penalty：

```text
targets                        3, 1
conditioning set               2, 4, 6, 7, 35
historical capture task ID     110709
exact p                        0.0017250642217815301
QR-selected + exact-final p     0.0017005790592597284
absolute p difference          2.448516252180166e-5
maximum residual error         0.023816712140171781
```

这是隔离 QR 候选的误差。冻结守卫通过 rank/root/condition 等特征捕获了它。
当前生产 exact-primary 没有被替换。

## 重跑成本

每条路径先跑一次带 trace 的预热，再计时三次无 trace 的完整 optimizer
kernel。下表是各 cohort 三次中位数之和，不是完整方法的 outer elapsed。

| GPU kernel 成本 | 标定 | 独立开发验证 |
|---|---:|---:|
| exact-primary 基线 | 21.740 s | 39.953 s |
| 原始 QR 轨迹 | 4.230 s | 13.234 s |
| 被标记 cohort 从初始 SP 重跑 exact | 21.258 s | 39.632 s |
| QR + exact 重跑加和 | 25.487 s | 52.866 s |
| 相对 exact 基线 | 慢 17.2% | 慢 32.3% |

验证集中，需重跑的 cohort 覆盖基线 exact kernel 成本的 **99.2%**。
本次成本模型采用整双 target cohort 重跑，并未实现仅收集被标记 target
的压缩队列；guard/dispatch/downstream 成本也未计入。因此不能把这些加和
称作集成 runtime 的实测，更不能据此预测 canonical wall-time。

单 target 压缩重跑的净收益仍未测得。当前证据支持停止这版集成，
没有支持先花成本重写 warp/shared-memory 内核。

## 对照与证据身份

- 72/72 cohort 的诊断 exact selected-SP 与已安装生产库逐位一致。
- 刷新的 exact coefficients 与生产 exact fixed evaluator 逐位一致。
- 另做 72/72 cohort 的生产 small64/base80 capacity 分派对照：selected-SP、
  coefficients、fitted、residual 全部逐位一致，rank/info/route 通过。
- 带 trace 与三次无 trace 运行的 selected-SP/coefficients/status 一致。
- GPU full eig/gamma 的 CPU authority 计数为零；生产资源计数恢复基线。
  residual shadow D2H 仅用于诊断比较，full-eig 消费设备端 residual。
- 所有 runtime 源码、默认值及生产 native SHA 保持原样。

```text
HEAD at fixture capture
6a6a4b31c6f814d2a15c127236c23e10a00b0a27

production native SHA256
fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50

isolated probe SHA256
c10553fa0f159afe06efe5f5231501aec8b7fdef455047369092362fe2f88af2

artifact manifest SHA256
8e38f389805ec3f375225a9b3a482592ad7aec2e89bbcd3e767cc88d315065f2
```

复现：[工具说明](../tools/research/guarded_qr_20261020/README.md)。
本地证据：[summary](../artifacts/full_cuda_ci/guarded_qr_20261020/summary.json)、
[manifest](../artifacts/full_cuda_ci/guarded_qr_20261020/manifest.json)、
[逐 cohort 表](../artifacts/full_cuda_ci/guarded_qr_20261020/cohorts.csv)。
artifact 目录按仓库现有规则忽略；它们是本地 development evidence。

生产 `exact-primary → exact fixed/reuse → full eig` 继续保留。
Phase 10 为 ACTIVE，candidate 未冻结，recommended=false，sealed holdout 未打开。
若继续研究守卫，需要新的误差预测依据、独立开发样本和真实 target 重跑成本；
不能把本次零漏判解释成所有输入均满足 1e-10 的保证。
