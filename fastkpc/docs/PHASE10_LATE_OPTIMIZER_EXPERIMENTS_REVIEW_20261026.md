# 晚层 optimizer 实验复核（L8 预加载、L6 槽位）

日期：2026-10-26。复核 [晚层 optimizer 实验](PHASE10_LATE_OPTIMIZER_EXPERIMENTS_20261025.md)
（native `994c43d3…20df9`，cache ON；共享主机开发证据）。只读复核：数字取自其 receipts，
另用我此前两次生产 trace（`optimizer_slot_line_review_20261025/` 下的 `ktrace-run`、
`ktrace-run-cache-on`）做对照。本次没有新的 GPU 运行。正式开发基线仍为 407.531 s；设备
属性缓存的隔离第三对仍未完成；Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、
recommended=false 均不变。

## 结论

```text
L6 槽位原型   复现一致：W/S 全生命周期中位 43.740/18.040 s（比 0.4125），六对全胜；
              S 六次极差仅 0.12 s。S 的逐 setup 时间之和 359.8 s，与我 trace 中 L6 的
              kernel busy 之和 360.7 s 相当：这一层在 32 槽持续负载下服务时间没有变长。
              实测比 cache-ON 请求顺序模型（18.388 s）低 1.9%
容量进程 30.1 s  多出的时间集中在三段，其中两段与 cuBLAS 按 target 数选用的 gemm 实例
              首次使用对齐：3-target 首见于请求 56，2-target 首见于 217。campaign 进程
              先跑了 W，已加载全部实例。生产 trace 中 L4–L7 的 optimizer 阶段没有任何
              kernel 函数首次启动，生产 L6 预计不会遇到这种阻塞
共存          生产 trace 中 L6 是全程显存最紧的一层：nvidia-smi 已用峰值 22,378 MiB，
              余量约 2.1 GiB，且已含 64 个窗口 prepared 对象。直接叠加 472 MiB 会低于
              2 GiB；但槽位会替代窗口对象（W 增量 924 MiB，S 468 MiB）。共存应像 L4 那样
              做替换探针，不做叠加探针
L8 预加载     机制成立：ON 的 L8 为 15.28–15.31 s，OFF 为 16.9–18.7 s，中位少 3.32 s，
              首次使用阻塞消失。STOP 符合预声明门；但该门对约 3 s 的效应几乎无法通过：
              最小收益取 OFF 极差 12.55 s，其中约 9.2 s 来自 OFF2 一次 L7=19.20 s
同 campaign 噪声  预加载在 L8 之前不起作用，L0–L7 不可能因它改变。三对中 ON 的 L6 却都更慢
              （3.36/5.95/3.89 s），L0–L5 也慢 0.66–2.22 s。三对方向一致的晚层差异并不需要
              因果；这也支持 Codex 对缓存“晚层收益”的谨慎
建议          1. L6：先做生产 cache/arena 替换共存探针，再做 opt-in runtime 资格
              2. L8 预加载：不因本轮放弃。L6 槽位落地后（L6 极差从约 6 s 降到约 0.1 s），
                 用预先声明的 L8 目标层门加 outer 不回退门重新资格化
              3. 可选的稳健性措施：槽位池启动前，对该层用到的 target 数各做一次 cuBLAS
                 预热；canonical 不需要，但能防止换数据集后出现首次加载阻塞
```

## 1. L6 槽位原型

由 `campaign-runs.csv` 重算：W 中位 43.739627 s（42.32–48.88），S 中位 18.040455 s
（18.00–18.12）；六对 W/S 为 (43.04, 18.02)、(44.23, 18.06)、(48.88, 18.00)、(42.32, 18.00)、
(44.93, 18.11)、(43.25, 18.12)，全部 S 胜；中位比 0.412451。与报告一致。

逐 setup 的 host 时间之和：W 为 759.7 s，S 为 359.8 s。W 的数值约为两倍，因为 64 个 stream
共享 32 个 channel 时，各 setup 的时间包含等待。S 的数值与我 cache 关闭 trace 中 L6 的
kernel busy 之和（360.7 s）几乎相同。修订版槽位复核的前提是“窗口模式 busy 在持续补位下
不变”；在 L6 这一层（32 槽在途约 150 个 block），这一前提成立。L3 的在途 block 约为 L6 的
三倍，不能由此推断。

## 2. 容量进程中首次 S 执行的 30.1 s

`capacity-setups.csv` 与 campaign 的 S timing 逐请求对比，容量进程多出 462 s（逐 setup 之和）。
多出的时间集中在请求 0–59（约 218 s）、90–139（约 28 s）、170–219（约 216 s）；后一段中
请求 191–212 各多约 9–10 s。campaign 的 S warmup 只在请求 0–19 多出约 7.8 s。

生产 trace 显示 cuBLAS 对 L6 的 fitted 值 gemm 按 target 数使用不同实例：
`gemmSN_NN_kernel<…, 2|3|5, …>`。按请求顺序，5-target 首见于请求 0，3-target 首见于 56，
2-target 首见于 217。两个大集中区分别覆盖这两个首次出现的位置。lazy loading 下，函数首次
启动会让设备等正在运行的 kernel 结束（[槽位线复核](PHASE10_OPTIMIZER_SLOT_LINE_REVIEW_20261025.md)
第 6 节的探针），此时池中有长达 10 s 左右的 optimizer 在运行，后续 setup 会一起被拖住。
容量进程只用生产 DSO 做过一次 5-target warmup，而槽位使用另行编译的数值副本，所以开头还有
副本自身 kernel 的首次使用。campaign 进程先跑完整的 W，因此三个 gemm 实例都已加载。
90–139 的小集中区尚未解释。

生产路径的情况：在 cache-ON trace 中，逐层检查 optimizer 阶段内是否有 kernel 函数首次启动，
结果为 L3 有 2 个（2/3-target 的 gemm 实例）、L8 有 2 个（base80 的 gather 与 residual），
L4、L5、L6、L7 均为 0。因此在 canonical 顺序下，生产 L6 槽位不会遇到这种阻塞。
如果换数据集后某层首次出现新的 target 数，槽位池中途仍可能被拖住；第 3 条建议针对这种情况。

## 3. 与生产 cache/arena 共存

我的 cache 关闭 trace 有 500 ms 一次的 GPU1 显存采样（nvidia-smi，总量 24,564 MiB）。
层边界按各层耗时从方法结束时间倒推，属近似：

| 层 | 已用峰值 MiB | 最小余量 MiB |
|---:|---:|---:|
| L3 | 21,000 | 3,564 |
| L4 | 21,554 | 3,010 |
| L5 | 22,274 | 2,290 |
| L6 | 22,378 | 2,186 |
| L7 | 22,082 | 2,482 |
| L8 | 21,744 | 2,820 |

L6 是全程余量最小的一层，而且这个数字已包含当时 64 个窗口 prepared 对象。原型 warmup 中，
W 的物理增量为 968,884,224 B（924 MiB），S 为 490,733,568 B（468 MiB），timed S 为 462 MiB。
若在窗口对象之外再叠加 472 MiB，余量约 1.7 GiB，低于 2 GiB 门；若槽位替代窗口对象，
余量约可回升到 2.6 GiB 左右，但这只是推算。L4 当时的保守叠加探针就因重复计入窗口对象
而失败，随后改用替换探针通过。L6 应直接按替换口径声明探针，在实际 cache/arena 状态下测量。

## 4. L8 预加载

由六个 `report.json` 重算（秒；正数表示 OFF 更慢）：

| 对 | outer | L0–L5 | L6 | L7 | L8 |
|---|---:|---:|---:|---:|---:|
| OFF1/ON1 | −4.00 | −2.22 | −3.36 | −0.02 | +1.63 |
| OFF2/ON2 | +4.59 | −0.66 | −5.95 | +7.74 | +3.44 |
| OFF3/ON3 | −1.51 | −1.33 | −3.89 | +0.35 | +3.31 |

L8：OFF 16.912/18.741/18.624 s，ON 15.283/15.304/15.310 s。ON 稳定，与 trace 中首次使用阻塞
消失一致（报告：最晚 setup 的启动延迟由 1.703 s 降为 0.016 s）。预加载成本 36.8 ms、6 MiB。

**为什么 outer 门无法判定**：预加载在第一个 L8 prepared 创建之后才执行，L0–L7 不可能因它
改变。所以 L0–L7 的差就是本 campaign 中与处理无关的变化：L6 三对都偏向 OFF（3.4–6.0 s），
L0–L5 也偏向 OFF（0.7–2.2 s），L7 在 OFF2 出现一次 19.20 s。预声明的最小收益取
`max(2 s, 0.5%, OFF 极差, ON 极差)`，即 12.551 s，其中约 9.2 s 来自那一次 L7。期望约 3 s 的
L8 收益，在这一设计下几乎不可能通过。按协议判 STOP 是对的，且报告已写明这不说明预加载
无效。

这组数据还给出一个旁证：三对 L6 的差方向一致、量级 3–6 s，却不可能由处理引起。缓存复核中
“晚层改善”同样只有三对，因此 Codex 要求单独记账、不作结论，是必要的。

## 5. 建议

1. **L6**：先按替换口径做生产 cache/arena 共存探针，在实际 L6 时刻测量余量与增量；
   再做 opt-in runtime 资格（完整 canonical CI/图/RNG 门与重复 outer 计时）。
2. **L8 预加载**：等 L6 槽位落地后再重新资格化。届时 L6 不再是窗口模式，原型中 S 的极差
   只有 0.12 s，outer 噪声会明显下降。这类只作用于 L8 的改动，其门可以考虑由两部分组成：
   一是 L8 目标层门，例如三对 L8 全胜且中位收益 ≥1 s；二是 outer 不回退门，容差按不受
   影响的层的实测波动预先声明。是否采用由 Codex 决定。
3. **稳健性（可选，低优先级）**：槽位池启动前，对该层出现的每个 target 数各做一次 cuBLAS
   gemm 预热（或同等的首次加载处理），并计入成本。canonical 不需要，但能避免数据集变化后
   出现第 2 节那样的池内阻塞。
4. **L7**：OFF2 的 19.20 s 超出历史 24 次 L7 层耗时的范围（10.0–14.9 s），约为本轮其余
   五次（9.95–11.46 s）的两倍，成因未查。若再次出现，值得像 L6 一样做一次窗口诊断；
   在此之前不作判断。

## 证据与工具

数据来源：`fastkpc/artifacts/full_cuda_ci/late_optimizer_20261025/{off,on}{1,2,3}/report.json`；
`fastkpc/artifacts/full_cuda_ci/optimizer_l6_slots_20261025/qualified/` 的
`campaign-runs.csv`、`campaign-setups.csv`、`capacity-setups.csv`。生产 trace 与显存采样：
`fastkpc/artifacts/full_cuda_ci/optimizer_slot_line_review_20261025/`（`ktrace-run/`、
`ktrace-run-cache-on/`、`gpu1.csv`）。计算均为一次性只读脚本，未修改任何生产源码、native 或
默认配置，没有 commit 或 push。
