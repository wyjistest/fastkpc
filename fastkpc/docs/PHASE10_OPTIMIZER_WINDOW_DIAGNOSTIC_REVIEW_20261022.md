# Optimizer 窗口诊断：复核与队头阻塞证据

日期：2026-10-22。复核 [固定窗口并发诊断](PHASE10_OPTIMIZER_WINDOW_DIAGNOSTIC_20261022.md)。
本文只读分析 Codex 归档的 Nsight 导出（`profile-L5-W7.sqlite`、`profile-L3-W31.sqlite`），
不改代码、不改状态。下文的收益都是理想化估计，不是承诺。

## 结论

```text
诊断结论          成立；64 worker 最快、逐位门全部通过
新证据            64 个 stream 共用 32 个硬件队列，造成队头阻塞，在两个窗口中均可直接观察到：
                  64 worker 时，超出 32 的 setup 中，其首个操作（Y 的 H2D）均延迟到另一 stream
                  的最后一个 kernel 结束后 ≤10 µs 才开始（L5 W7：32/32；L3 W31：24/24）；
                  32 与 16 worker 时没有这种延迟
单 setup 耗时     不随并发变化：L5 W7 的 optimizer kernel 总时长在 64/32/16 worker 下为
                  45.9 / 45.7 / 45.5 s，最长者均为 6.2 s；≤32 并发时 GPU 未饱和
窗口时长          约等于最长 setup 加上它的排队延迟：L5 W7 为 0.85 + 6.22 ≈ 7.06 s；
                  device 工作总量除以 32 只有 1.44 s
对设计的含义       在途 setup 不超过 32 个（每个硬件队列一个），跨窗口立即补位；
                  运行中不调用 cudaMalloc / cudaFree
建议实验           在 Codex 的双窗口对照中加入 32 worker 与按时长排序两档
```

## 证据

Codex 归档的 profile 进程在每个窗口上依次运行 64、32、16 个 worker（见
`profile-*-runs.csv`）。按 64 个 optimizer kernel 为一组切分后：

| 窗口 | worker | wall | 峰值并发 kernel | H2D 延迟 >5 ms | 其中紧接另一 stream 结束 | kernel 总时长 | 最长 kernel |
|---|---:|---:|---:|---:|---:|---:|---:|
| L5 W7 | 64 | 7.062 s | 32 | 32 | 32 | 45.9 s | 6.22 s |
| L5 W7 | 32 | 7.384 s | 32 | 0 | — | 45.7 s | 6.21 s |
| L5 W7 | 16 | 12.322 s | 16 | 0 | — | 45.5 s | 6.20 s |
| L3 W31 | 64 | 0.516 s | 32 | 24 | 24 | 6.5 s | 0.44 s |
| L3 W31 | 32 | 0.960 s | 16 | 0 | — | 5.9 s | 0.44 s |
| L3 W31 | 16 | 1.402 s | 11 | 0 | — | 5.8 s | 0.43 s |

- **机制**：在 64 worker 档，被延迟的 H2D 都在另一个 stream 的
  `form_multi_penalty_residuals`（20 个）、cuBLAS `gemmSN_NN`（9 个）等最后一批 kernel
  结束后 ≤10 µs 开始（中位差 −0.2 µs，即测量精度内同时发生）。这正是共享硬件队列的
  特征：B 的整条链排在 A 链之后，A 的后续 kernel 依赖 A 的长 optimizer kernel，于是
  B 要等 A 全部完成。延迟的数量恰好是超出 32 的那部分（L5 为 32/64）。L3 W31 只有 24 个
  超过 5 ms，可能是其余队列伙伴的链短于 5 ms。
- **峰值并发**：64 worker 时并发 optimizer kernel 的峰值恰为 32，与
  `CUDA_DEVICE_MAX_CONNECTIONS=32` 一致。Codex 原文“不单独证明因果”的保留由上面的
  逐事件对应补足。
- **GPU 余量**：单 setup 的 kernel 时长在 16/32/64 worker 下几乎不变，说明在 ≤32 个
  并发 setup 时，各 setup 互不拖慢。L5 每个 setup 约 7 个 target，即约 7 个 64 线程
  block；32 个并发只占用数百个 block，远低于 128 个 SM 的容量。
- **窗口时长的构成**：L5 W7 的 7.06 s 中，最长的 setup 跑了 6.22 s，且它是所在队列的
  第二个，晚了 0.85 s 才开始。其余 63 个 setup 的 kernel 时长中位数只有 0.56 s；全部
  device 工作除以 32 仅 1.44 s。所以窗口时长由单个长 setup 加上排队决定，而不是由
  总工作量决定。

## 理想化估计（假定单 setup 耗时不变，须实测）

去掉窗口屏障后，若一层的 setup 由 32 个队列连续领取，列表调度的完工时间满足：

max( Σ t / 32 , max t )  ≤  T  ≤  Σ t / 32 + max t

若 L5 的 15 个窗口都与 W7 相似（每窗口 kernel 总时长约 46 s，全层最长 setup 约 11 s）：
Σ t / 32 ≈ 21.6 s，T 约在 22–33 s 之间，对照实测约 104 s。若 L3 的 64 个窗口都与 W31
相似：Σ t / 32 ≈ 12.5 s，最长约 1.4–1.9 s，T 约 13–15 s，对照约 48 s。

这是从两个窗口外推的理想化区间。它忽略 host 侧开销（见下文 32 worker 的方差）
与 kernel 以外的每 setup 工作，只说明这个方向值得精确测量，不能写成方法级收益。
L4、L6 需要各自的数据；L7（单窗口）与 L8（单个 17–19 s setup）基本不受影响。

## 对滚动设计的含义

1. **在途 setup ≤ 32 个，每个硬件队列一个**。64 个 stream 只会在 32 个队列上排成
   队头阻塞的对子；滚动调度应保持 32 个 worker 或 stream 立即补位，而不是增加 stream。
2. **跨窗口连续领取**。一层内的 setup 形成单一队列，长 setup 不再按窗口串行。
   若能预估时长，长者先发，可以进一步收紧完工时间。
3. **运行中不分配或释放显存**。按 CUDA 编程指南，`cudaMalloc`（device memory
   allocation）属于会造成跨 stream 隐式同步的操作；`cudaFree` 同样会同步设备。
   滚动调度需要可复用的句柄池，或 stream-ordered 分配，并单独测量。
4. **host 提交路径**：结果 D2H 是 pageable 拷贝，会阻塞 worker 线程直到 kernel 完成。
   32 worker 的主计时方差很大（L5 W8 为 7.46–19.78 s），而 profile 中 32 worker 的
   kernel 总时长不变，说明方差来自 host 侧或领取次序，原因尚未查明，应在下一实验中
   记录逐 setup 时间线。

## 建议的下一项实验

在 Codex 提议的“两个相邻窗口合成 128 个 setup”隔离重放中，同时比较四档：

| 档位 | 内容 |
|---|---|
| A | 两次 64 setup / 64 worker（现状基线） |
| B | 128 setup / 64 worker（Codex 原提议） |
| C | 128 setup / 32 worker（每个硬件队列一个） |
| D | 128 setup / 32 worker，按捕获的 setup 时长降序领取（oracle 排序，给出排序收益的上界） |

要求：

- 128 个 prepared 句柄预先常驻，把调度与生命周期分开测量；
- 逐位门与 Codex 现有门相同；
- 每档记录逐 setup 的 host 与 device 时间线、峰值并发、H2D 延迟，以及 kernel 时长
  是否在持续 32 并发下保持不变；
- 用 L5 W7+W8 与 L3 W31+W32 两组。

若 C 或 D 明显优于 A 与 B，再进入有界的跨窗口滚动原型，同时设计句柄池与登记的所有权。

## 证据

分析脚本：`fastkpc/tools/research/optimizer_hol_review_20261022/`，包括
`hol_per_window.py`、`hol_h2d_coincidence.py`、`own_stream_chain.py`。
输出：`fastkpc/artifacts/full_cuda_ci/optimizer_hol_review_20261022/L5-W7.txt`、`L3-W31.txt`。
manifest SHA `15dc823b195a41caa0ba09ac187ad9488aec271bd9566b2505478960e4797f61`，
绑定两份 sqlite 与 Codex 的 manifest。

复现：在仓库根目录运行 `python3 -I <脚本> <sqlite>`。
