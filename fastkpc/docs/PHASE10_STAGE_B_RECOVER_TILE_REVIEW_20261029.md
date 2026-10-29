# Stage B recover tile 隔离原型：复核

日期：2026-10-29。复核 [recover tile 隔离原型](PHASE10_STAGE_B_RECOVER_TILE_20261029.md)（证据
`fastkpc/artifacts/full_cuda_ci/stage_b_recover_tile_20261029/attempt3/`）。只读复核，没有新的 GPU 运行或编译。
tile16、tile8 的 `STOP_NO_STAGE_GAIN` 成立；生产保留现有 tile 策略；accepted opt-in development median 仍为 311.440 s；
默认关闭、L8 optimizer STOP、Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。
同日按 Codex 核对意见修订，见“修订记录”。

## 修订记录（Codex 核对后）

Codex 核对了当前 native、104 个生产输入与 closure/manifest 哈希，以及 255 个寄存器、7,120 B STACK 的静态资源，均一致；
同意 recover tile 收口、下一项转向 L4/L5 optimizer 本体归因。第 3 节的新增分析有两处需要修正，我复核后全部接受：

1. **“请求顺序模型”实际按观测到的 GPU 开始顺序。** `predict.py:102` 在槽位模式下按每个 setup 首个 kernel 的开始时间
   排序。生产中 32 个 worker 先以 `next.fetch_add(1)` 领取请求序号，再各自 `pool.bind`、上传并提交
   （`full_cuda_ci_one_call.cpp` 的槽位 worker），GPU 开始顺序不保证等于领取顺序。表中该列改称“观测开始顺序模型”。
   我 10-25 的 README 与 `predict.json` 把它标作 “burst start (fetch) order”，把两者等同了，同样应读作观测开始顺序。
   要讨论长 setup 在原请求序列中的位置，需要先建立 setup key、请求序号与 trace 的对应。
2. **删去 stability replay。** 当前 exact-primary 资格门（`stage_b_work_pair_runtime_20261028/run.R:89–93`）要求 80,751 个
   multi-penalty target 全部走 exact primary，stability replay 的 target 与 selected 计数、terminal confirmation 计数都为零；
   ON2 的 `gates.exact_primary` 为真。长 setup 的待查因素改为迭代数、step halving、objective/SVD 次数、导数与 Hessian，
   以及最终 selected-state evaluation。

另一处保持谨慎（Codex 的意见，我同意）：L5 的约 5.1 s 是实测跨度与模型下限之差。模型由同一份 trace 校准，并假定改变
顺序后各 setup 的耗时不变；拟合接近实测不能证明这部分可以兑现，只作诊断线索。

后续更正（同日，L4/L5 归因之后）：第 3 节“（最长的一个 busy 10.4 s）”错误。同一份旧 trace 中该 setup 在 6.17–16.61 s
运行，最后结束的是另一项；见 [L4/L5 归因复核](PHASE10_OPTIMIZER_L45_ATTRIBUTION_REVIEW_20261029.md) 第 5 节。

下一项（Codex 的建议，我同意）：只做 L4/L5 慢 setup 与关键 target 归因。先用旧 trace 定位最长的 setup 并核实身份，再结合
target 级计数，区分“迭代多”“单次求解慢”和“少数 target 拖长整个 setup”。现有 trace 只有 kernel 级时间，不能直接回答哪个
target、哪个内部阶段主导。补充一点：旧 trace 中每个 setup 的 optimizer kernel grid 等于其 target 数，可作身份的旁证；
确认身份仍需新运行在 host 侧记录请求序号、setup key、slot 与提交时刻，再与 trace 对齐。

## 结论

```text
完整性    closure.json、manifest.json 的 SHA-256 吻合；manifest 381 个条目（240 个产物、104 个生产输入、9 个工具、
          28 个依赖）按当前文件重算一致；native 12acc3cd… 与 104 个生产输入未变
原型      只在生产 tile 计算之后加 c->tile = min(c->tile, RT_TILE)；tile 本是 kernel 的运行时参数，三档的 6 个 device
          函数指令字相同。两档都通过 104,797 个 component、10,515,312,078 个 rotation 的逐位比较
计时      协议 00:38:21 写入，九次于 00:54–01:11 按 A1 B1 C1 / B2 C2 A2 / C3 A3 B3 运行，均 exit 0、isolated。重算：
          各层中位数之和 tile36 101.684859 s、tile16 108.627164 s（+6.8273%）、tile8 104.664768 s（+2.9305%）；
          每次 tile16/8 都慢于最慢的 tile36
trace     三份原始 trace 重算与报告一致：recover 并集 29.88 → 36.97/33.00 s，post-bidiagonal recover 24.15 →
          31.31/25.83 s，recover kernel 中位 39.1 → 49.2/47.5 ms；tile16 的 roots→recover 间隔变长（22.8 → 27.1 ms），
          tile8 的交集略增（5.71 → 7.15 s），但都不足以抵消 recover 变慢
范围      STOP 只覆盖固定 64 线程的 tile16/8；tile12、32 线程 launch 与其他 recover 算法未测，报告已写明
建议      同意转向 L4/L5 optimizer 本体归因，先做慢 setup 与关键 target 归因。我 10-25 的两份 trace 与当前 native
          的 optimizer 指令字相同，可先离线定位最长的 setup；L5 实测跨度比同一 trace 校准的模型下限多约 5.1 s，
          只作诊断线索，见第 3 节
```

## 1. 复现

- `timing.json` 与九个 `*-process.json`：数值与报告逐项一致；三档的总时间范围分别为 101.632–101.733 s、108.594–108.685 s、
  104.627–104.763 s，互不重叠。
- 用三份 `trace{36,16,8}/events.bin` 独立重算 L2+L3（秒）：

| 指标 | tile36 | tile16 | tile8 |
|---|---:|---:|---:|
| Stage B 边界 | 101.958784 | 109.000041 | 105.086747 |
| bidiagonal 并集 | 76.584415 | 76.465083 | 77.958616 |
| recover 并集 | 29.883891 | 36.969280 | 32.999291 |
| bidiagonal ∩ recover | 5.711446 | 5.635449 | 7.147751 |
| 只有 recover 的区间 | 18.507937 | 28.182657 | 21.529809 |
| 最后一个 bidiagonal 结束后的 recover | 24.145139 | 31.308573 | 25.825785 |
| 最后一个 bidiagonal 结束后的完整 wave 尾部 | 24.872300 | 32.038724 | 26.626356 |
| 未记录到 device 活动 | 0.545840 | 0.547359 | 0.592978 |

  两层合并的中位数：recover kernel 39.062/49.151/47.496 ms，roots→recover 间隔 22.807/27.080/23.131 ms，与报告按层给出的
  数值一致。三档的普通 wave 都是 10 个 bidiagonal 在首个开始后 50 ms 内开始（L2 76/76、L3 84/84）；普通 wave 中位数
  （L2/L3）为 619.1/620.8、663.7/663.7、638.4/640.7 ms。
- `binary-audit.json`：三档 timing 库的 6 个 device 函数指令字相同（只在原型库之间比较）。

## 2. 解读

- 两档都没有兑现“降低 shared、允许同驻”的预期：tile16 的 recover 没有更早开始，tile8 的交集略增，但 recover kernel
  中位数分别变长约 26% 与 22%（39.06 → 49.15/47.50 ms），尾部随之增加。原因未测（没有驻留、stall 或流量计数），报告不作
  归因，我同意。
- 判定来自九次无插桩的完整阶段计时，trace 只作解释，符合预先的约定。
- 前两次 attempt 的失败都在诊断夹具：attempt1 异常退出时 raw-pointer lane 未释放（sanitizer 报 240 个 allocation），
  attempt2 的 allocation 清零与 guard 初始化在默认 stream 上、计算在 nonblocking stream 上，中间缺少完成依赖。修正
  （RAII 清理、lane 构造结束时同步默认 stream）只在夹具中，两次都未开始正式计时，失败证据另行保留，处理得当。

## 3. 对 L4/L5 optimizer 归因的输入

同意下一项回到 L4/L5 selected-SP optimizer 本体。以下是已有数据与静态事实，标为模型的部分未经测量。

1. **规模。** 311.440 s 那次运行（work-pair runtime ON2）各层的 optimizer wall（`optimizer_by_level.host_sec`，即每批
   `wall_host_ms` 之和）：L5 31.870 s、L4 27.141 s、L6 18.017 s、L8 16.914 s、L3 15.245 s、L7 9.399 s、L2 5.377 s、
   L1 0.138 s，合计 124.101 s。`stages.optimizer_sec` 的 127.579 s 边界更宽（每批另含准备与完成处理，槽模式另含池销毁），
   两者不混用。L4/L5 都以 32 个槽运行，分别有 2,154 与 955 个 setup。
2. **kernel 结构与静态资源。** optimizer kernel 以 `<<<target 数, 64>>>` launch，每个 target 一个 64 线程的 block，kernel
   在最慢的 target 结束时才结束。q ≤ 64 且 penalty ≤ 7 的形状（包括 L3–L6 的 q = 28/37/46/55）用 small64 版本的
   `optimize_multi_penalty_targets_kernel`：每线程 255 个寄存器、STACK 7,120 B、LOCAL 0；L8 有 8 个 penalty，用 Base80
   版本（STACK 7,136 B；旧 trace 中也只出现这两个版本）。按寄存器计每 SM 至多 4 个 block。按平均每个 setup 的 target
   数（L4 8.25、L5 7.14）计，32 个槽都在 kernel 中时约有 264/228 个 block，平均每 SM 约 2 个（模型，不是实测驻留）。RTX 4090 的 FP64 吞吐为 FP32 的 1/64，STACK 位于 local memory；哪一项限制
   单个 target 的时长需要测量。
3. **已有 trace 可以先用。** 我 10-25 采集的两份 ktrace（`optimizer_slot_line_review_20261025/ktrace-run{,-cache-on}/`，
   native 037b92f1、994c43d3）含 L4/L5 槽位模式的全部 kernel。比较 SASS：五个版本的 `optimize_multi_penalty_targets_kernel`
   在这两个 native 与当前 12acc3cd 中指令字相同；small64 编译单元的 14 个 device 函数在 994c43d3 与当前之间也全部相同。
   host 代码与前序层已经改变，下面是当时的观察，本轮未重新验证。cache-on 那份的已有分析（`predict.json`，单位秒；实测跨度
   为该层第一个 kernel 开始到最后一个结束，busy 为每个 setup 在其 stream 上的 kernel 时长之和）：

   | 层 | setup | 实测跨度 | busy 之和 / 32 | 最长 busy | 观测开始顺序模型 | 下限模型 | 最长优先模型 | 随机顺序中位 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|
   | L4 | 2,154 | 27.163 | 25.721 | 4.564 | 27.233 | 26.226 | 26.313 | 27.776 |
   | L5 | 955 | 31.846 | 26.247 | 10.421 | 31.967 | 26.736 | 26.968 | 31.849 |

   模型把每个 setup 的服务时间取为 busy × 1.0179（L4/L5 各自的 setup 跨度之和与 busy 之和之比的平均）加 0.672 ms（同一槽
   上相邻 setup 间隔的中位数 0.657、0.687 ms 的平均），按观测到的 GPU 开始顺序在 32 个槽上排程，与实测跨度相差
   0.07/0.12 s；下限模型为 max(服务时间之和 / 32, 最长服务时间)。这些模型由同一份 trace 校准，并假定改变顺序后各 setup
   的耗时不变。按模型，L4 的跨度只比下限多约 0.9 s；L5 多约 5.1 s，最长优先排程为 26.968 s，与长 setup 开始较晚形成的
   尾部相符（最长的一个 busy 10.4 s）。拟合接近实测不能证明这部分可以兑现，只作诊断线索。观测开始顺序的结果与随机顺序的
   中位数相近（L4 好 0.54 s，L5 差 0.12 s）；改变领取顺序会触及现有的槽位合同，这里不作建议。L5 的归因宜单列最长的几个
   setup，查明它们为什么长：迭代数、step halving、objective/SVD 次数、导数与 Hessian，以及最终 selected-state evaluation
   （exact-primary 门已确认 stability replay 与 terminal confirmation 为零）。它们在原请求序列中的位置，需要先建立
   setup key、请求序号与 trace 的对应才能讨论。
4. **计数器** 仍需管理员权限。没有计数器时，归因只能依靠时间线、静态资源与受控重放（例如同一批 captured setup 在不同
   并发下的 kernel 时长），结论应相应保守。

## 证据

`attempt3/{timing,protocol,binary-audit,closure,manifest}.json`、`attempt3/{A,B,C}{1,2,3}-process.json`、
`attempt3/trace{36,16,8}/{events.bin,names.tsv,markers.tsv}`、`attempt3/{qualify,memcheck}{16,8}.json`、`resources.json`；
原型构建见 `tools/research/stage_b_recover_tile_20261029/build.py`。第 3 节：`stage_b_work_pair_runtime_20261028/runtime/on2/report.json`、
`src/cuda/mgcv_multi_penalty_gcv{.cu,_capacity.cpp,_small64.cu}`、对 `build/fastkpc_cuda.so` 与两份保存的旧 native
（`device_properties_cache_20261025/`、`optimizer_l6_slot_runtime_20261026/` 下的 `previous-native.so.gz`）运行
`cuobjdump -res-usage/-sass`、`optimizer_slot_line_review_20261025/ktrace-run{,-cache-on}/predict.json`。计算为只读一次性
脚本；没有修改生产源码、native 或默认配置，没有 commit 或 push。
