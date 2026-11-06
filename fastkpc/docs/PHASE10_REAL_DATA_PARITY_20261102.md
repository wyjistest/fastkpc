# 四套真实输入的 GPU/CPU 一致性复核

日期：2026-11-02。结论：**PASS_REAL_DATA_CORRECTNESS**，限以下输入与配置。
本轮没有修改生产源码、重编 native 或使用 sealed holdout。

当前 Git HEAD 为 `cc123c54db0762b8fabe65db4514543a594cd38e`，安装 native：

```text
b2b79fc6219841a7b9de193e312de27bbdb8bca77a9d3a9d4a626aa1e135b12b
```

107 个生产输入与 10 月 30 日 attestation 一致。使用物理 GPU1，另一张卡无
compute 任务；每个输入使用 fresh R process 和空 dataset cache，CPU affinity
0–19，数学库单线程。全部输入保持原始行列，不裁剪。

## 输入与结果

统一参数：`dcc.gamma`、`alpha=0.1`、`index=1`、`numCol=35`、`m.max=Inf`。
CI 和有限 pMax 的绝对误差门均为 `1e-10`，pMax 特殊值单独精确检查。

| 真实输入 | 尺寸 | CI 数 | 最大 CI 误差 | 最大 pMax 误差 | 最终 PDAG / RNG |
| --- | ---: | ---: | ---: | ---: | --- |
| `20221022181603` / CD8.c11.Tex.PDCD1 | 179×8 | 338 | 5.82867087928207e-14 | 5.82867087928207e-14 | exact / exact |
| `20220418113957` / logged-liver-exhausted-allData | 243×9 | 413 | 9.22595333463505e-14 | 9.22595333463505e-14 | exact / exact |
| `20240315113222` / CD8.c11.Tex.PDCD1 | 300×10 | 657 | 4.76840789076505e-13 | 4.76840789076505e-13 | exact / exact |
| `cancer_RD` | 351×48 | 240,498 | 3.05425129631942e-12 | 3.05425129631942e-12 | exact / exact |

合计 **241,906 个 CI**，无超容差值、无决策翻转。四套输入的 skeleton adjacency、
无向边 sepset union、`n.edgetests` 均一致。CPU 权威定向分别执行 1、7、2、3 个 CI；
定向 CI 身份/顺序、决策和最终 PDAG 通过。RNG endpoint 一致。

GPU skeleton 的 CPU 数值权威、unknown/approximate fallback、residual/component
large D2H 计数均为零。exact-primary、stable-SVD、cache route/status 检查通过；
最终 device allocations/bytes、streams/events 均归零。

## 两种配置与参考来源

**351×48** 使用当前已验收的完整 opt-in 配置：structured rows、5N fixed-VT work-pair、
L3/L4/L5/L6 slots、q37/q46 vector postback，以及 retained cache/bypass 等。
复用未修改的 canonical 验证入口，全部 **74 项门通过**；CI/pMax 位模式与既有
GPU 对照一致，同时直接满足 CPU oracle 的 `1e-10` 合同。

大输入的 CPU skeleton 来自已封存的 `oracle_351x48_default_inf_v2`，本轮没有重算
240,498 次 CPU CI。另现场分别对新 GPU skeleton 与封存 CPU skeleton 执行
kpcalg 权威定向，比较完整最终结果。封存 oracle 保留原身份。

**另三套输入** 使用当前通用 exact GPU 路线：exact-primary、coefficient reuse、
11 GiB component cache、all-hit residual bypass、8 eig lanes。由于生产入口的
形状限定，不启用 canonical 专用 structured/slots/postback。每套现场独立运行
`pcalg::skeleton(method="stable", m.max=Inf)`，CI 采用仓库 kpcalg/mgcv 实现；
非法 p 值直接失败。然后由原 kpcalg 路线完成 WAN-PDAG 定向。

因此，本轮既检查最新完整优化组合，也检查不同形状的通用 exact 路线；没有证明
canonical 专用优化已可用于任意输入。最终定向仍由 CPU kpcalg 执行。

## 顺序口径及保留的工具失败

直接 pcalg 的原始调用按节点遍历，native 调度按无向边组织两种方向；三套小输入
的原始调用顺序均不相同。不能把这一点报告为 raw pcalg call-order exact。

对照分为两项：独立 pcalg 搜索按 CI 身份逐项验证全部 p 值及最终图；再将这些
独立 CPU p 值交给既有 CPU native scheduler，验证 GPU 的 **native 合同顺序**和
删边轨迹。额外 speculative CI 仍由 CPU oracle 计算，不使用 GPU p 值补齐。
该 CPU 回放也必须复现独立 pcalg 的 pMax、图、sepsets 和计数。

新验证工具的前三次尝试保留在 artifact 中，未重写为 PASS：

- `attempt1`：数值/图/定向/RNG/资源通过，但错误地将 raw pcalg 顺序与 native 顺序直接比较。
- `attempt2`：新增 CPU 回放 provider 没有处理零行尾批，R 的 `paste()` 回收标量 level 导致越界。
- `attempt3`：逻辑轨迹已通过，但旧 generic CPU 回放入口将空层计数补到请求上限。
- `attempt4`：修正比较对象与空层处理；辅助回放止于独立完整 CPU 搜索实际结束的层。
  两条原始搜索仍为 `m.max=Inf`，全部严格数值/图/计数门保持，并全部通过。

这些修改只涉及新验证工具，没有改动生产算法，也没有放宽数值门槛。

## 耗时与证据边界

三套小输入的 GPU 完整 WAN-PDAG 观察耗时为 **2.705 / 3.776 / 4.885 秒**，
对应现场 CPU 完整参考为 **10.441 / 20.779 / 40.108 秒**；辅助顺序回放另行执行。
351×48 的 **skeleton** 为 **288.813 秒**，定向另行验证。不同边界不拼接比较。

这些均为单次 correctness 运行，没有构成性能对照或新的重复中位数。
283.328 秒开发基线、Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、
recommended false 均保持不变。此次未验证另外三种 CI 方法或更大数据范围。

## 复核材料

- [工具与协议](../tools/research/real_data_parity_20261102/README.md)
- [汇总](../artifacts/full_cuda_ci/real_data_parity_20261102/summary.json)
- [文件哈希清单](../artifacts/full_cuda_ci/real_data_parity_20261102/manifest.json)
- [canonical 74 项门](../artifacts/full_cuda_ci/real_data_parity_20261102/canonical/run/report.json)
- [三套现场 CPU/GPU 与 canonical 定向](../artifacts/full_cuda_ci/real_data_parity_20261102/attempt4/summary.json)

原始输入、native、RDS 回执和监控日志留在本机；无需修改或上传 Shiny 备份。
