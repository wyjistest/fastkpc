# 当前 work-pair OFF/ON Stage B 时序诊断：复核

日期：2026-10-28。复核 [work-pair OFF/ON 时序诊断](PHASE10_STAGE_B_WORK_PAIR_TRACE_20261028.md)（证据
`fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_trace_20261028/attempt1/`）。只读复核，没有新的 GPU 运行或编译。
`AUTHENTICATED_CURRENT_WORK_PAIR_ACTIVITY_COMPLETE` 成立；accepted opt-in development median 仍为 311.440 s；默认关闭、
L8 STOP、Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。2026-10-29 按 Codex 核对意见修订，见“修订记录”。

## 修订记录（Codex 核对后）

Codex 重新核验了 168 项封存哈希，并从原始 events 重算启动数量、kernel 时长与前序间隔中位数，均一致；诊断数据可靠，支持继续做
隔离的 recover tile 原型。首版第 2 节的性能解释过强，四点全部接受：

1. **“每 SM 吞吐几乎不变”只是模型。** 4/365 ms 与 5/448 ms 依赖均匀驻留的假设；10 个 kernel 同时开始不等于 640 个 block
   同时驻留，kernel 时长也含调度影响。只能说数据与“减少分批启动”的解释相符，不能据此证明收益主要来自这一机制；work-pair
   同时减少了存储操作和寄存器用量。首版“再提高驻留未必有用”的推论一并撤回。
2. **“收益只能来自更早同驻”过强。** 36、32、24 只是 shared 约束下的列槽位上限，实际吞吐还受活跃 warp、缓存和尾部负载影响。
   tile 16 的最后一个 block 只有 4 列，tile 12 恰好分成三个满块，不能仅凭 32 > 24 认定 tile 16 更优。
3. **“rotation log 读取倍数 1/3/5”不是流量倍率。** 那首先是每个 component 的 block 数。按执行 rotation 的活跃 warp 计，tile 36
   现在已有 2 个（36 个线程分在两个 warp），tile 16/8 分别为 3 个、5 个。实际内存流量取决于合并访问与缓存，表中该列删去，改列活跃
   warp 数并注明流量待测。
4. **尾部估算的两处口径。** 列容量从 36 降到 32 是下降 11.11%，若按吞吐反比估时，耗时增加 12.5%，首版写成约 11%。18.52 s 是只有
   recover 活跃的区间总和，不是 wave 尾部；post-bidiagonal recover 是 24.08 s。两者重叠，也都不能换算成可节省的时间。

下一步（Codex 的建议，我同意）：以 work-pair ON、tile 36 为对照，先试 16/8，保持每列算术顺序；通过 payload、中间状态与错误边界
资格后，做交错重复、无插桩的完整 L2/L3 Stage B 计时。trace 用来解释，完整阶段计时决定 GO/STOP。

## 结论

```text
完整性    closure.json、manifest.json 的 SHA-256 吻合；manifest 168 个条目（43 个产物、104 个生产输入、6 个工具、
          15 个依赖）按当前文件重算一致；native 12acc3cd… 未变
复现      用 off/on 两份原始 events.bin 独立重算，L2/L3 的阶段边界、bidiagonal 与 recover 并集及交集、只记录到
          recover 的区间、各 wave 最后一个 bidiagonal 结束后的 recover、无活动区间、early start（OFF 8、ON 10，
          普通 wave 全部如此）、开始跨度中位数与普通 wave 中位数，全部与报告一致
观察      一、数据与“减少分批启动”的解释相符：ON 时 10 个 bidiagonal kernel 同时开始，wave 缩短。按均匀驻留的模型，
          每 SM 的 bidiagonal 吞吐前后相近（约 10.96 与 11.16 个 block/s），但这依赖未测的驻留假设，不能证明收益主要
          来自这一机制；work-pair 同时减少了存储和寄存器
          二、shared 约束下 recover 每 SM 至多 36 列，tile 36 已用满；tile 16/8 为 32、tile 12 为 24。这只是列槽位上限，
          实际吞吐还取决于活跃 warp、缓存与尾部负载，哪种 tile 更好只能实测
建议      同意做 captured recover tile 原型，按完整阶段判定。预期应保守；报告需单列 roots→recover 间隔、
          bidiagonal/recover 交集、只有 recover 的区间、wave 尾部和 recover kernel 时长
```

## 1. 复现

由 `attempt1/{off,on}/{events.bin,names.tsv,markers.tsv}` 重算，与报告逐项一致（秒，L2+L3）：

| 指标 | OFF | ON |
|---|---:|---:|
| Stage B 边界 | 116.373601 | 101.866657 |
| bidiagonal 并集 | 107.657591 | 76.559478 |
| recover 并集 | 50.347732 | 29.883314 |
| bidiagonal ∩ recover | 43.583596 | 5.772404 |
| 只记录到 recover 的区间 | 5.476437 | 18.518707 |
| 最后一个 bidiagonal 结束后的 recover | 6.736606 | 24.083704 |
| 无活动区间 | 0.887601 | 0.540982 |

普通 wave 的 early start 在 OFF 都是 8（L2 76/76、L3 84/84），在 ON 都是 10；开始跨度中位数 324.2/324.5 ms → 0.490/0.493 ms；
普通 wave 中位数 702.4/710.5 ms → 616.6/622.8 ms。两层合并的 kernel 时长中位数：bidiagonal 365.1 → 448.1 ms，recover
50.2 → 38.8 ms；同 stream roots 结束到 recover 开始的间隔中位数 83.1 → 23.1 ms（报告按层给出，数值一致）。recover 的 launch
配置在两档相同：每个 64 个 block、64 个线程、101,088 B 动态 shared。

## 2. 两项观察（launch 级，模型而非实测）

**bidiagonal 的分批与吞吐。** OFF 时每个 wave 先有 8 个 lane、512 个 block 开始，kernel 中位 365 ms；ON 时 10 个 kernel 同时开始，
中位 448 ms。若假设 block 均匀驻留（OFF 约每 SM 4 个、ON 约 5 个），折算每 SM 约 10.96 与 11.16 个 block/s。这与“减少分批启动”
的解释相符，但 10 个 kernel 同时开始不等于 640 个 block 同时驻留，kernel 时长也含调度影响；work-pair 还减少了存储操作和寄存器
用量（100 → 82）。因此不能据此证明收益主要来自减少分批，也不据此推断提高驻留的价值。

**recover 的列槽位上限。** recover 每个线程负责一列，对本列逐条逆序施加 component 的全部 rotation，所以每个线程的工作量与 tile
无关。每列的 y 向量占 351 × 8 = 2,808 B shared，加上每个 block 的 1,024 B 保留与 128 B 分配粒度：

| tile | 每 component block 数 | 每 SM recover block | 每 SM 列槽位上限 | 每 component 执行 rotation 的活跃 warp | 与一个 recover 同驻的 bidiagonal |
|---:|---:|---:|---:|---:|---:|
| 36（当前） | 1 | 1 | 36 | 2（32 + 4 个线程） | 0 |
| 16 | 3（16+16+4） | 2 | 32 | 3 | 3 |
| 12 | 3（满块） | 2 | 24 | 3 | 4 |
| 8 | 5（8×4+4） | 4 | 32 | 5 | 5 |

- 列槽位只是 shared 约束下的上限；实际吞吐还取决于活跃 warp、缓存命中与尾部负载。tile 16 的最后一个 block 只有 4 列，tile 12
  恰好三个满块，不能仅凭 32 > 24 认定 tile 16 更优。
- rotation log 的读取由各活跃 warp 分别进行；实际内存流量取决于合并访问与缓存，需在原型中测量，不按 warp 数作倍率。
- 较小的 tile 允许 recover 与 bidiagonal 同驻，可能让 recover 更早开始（ON 的 roots→recover 间隔中位 23.1 ms，两者交集 5.77 s）；
  也可能因列槽位减少而在只有 recover 的时段变慢。若按吞吐反比粗估，36 → 32 列（−11.11%）对应耗时 +12.5%。这些都只是模型。
- 时段口径：只有 recover 活跃的区间合计 18.52 s，各 wave 最后一个 bidiagonal 结束后的 recover 为 24.08 s；两者重叠，都不是可节省的时间。
- tile ≤ 32 时仍以 64 个线程 launch，会多出一个空闲 warp；改为 ⌈tile/32⌉ 个 warp 只是 launch 配置，不改变算术。

哪种 tile 更好，净效果如何，只能由原型测量。

## 3. 对原型的意见

同意 Codex 的建议：以当前 work-pair ON、tile 36 为基线，在 captured 数据上先试 tile 16/8，保持每列算术顺序；先通过全部 payload/
中间状态、错误与边界资格，再做交错重复、无插桩的完整 L2/L3 Stage B 计时；trace 用于解释，完整阶段计时决定 GO/STOP；重复显示
净收益后才考虑 runtime opt-in。补充：

1. 报告除完整阶段外，单列 roots→recover 间隔、bidiagonal/recover 交集、只有 recover 的区间、各 wave 的 post-bidiagonal 尾部和
   recover kernel 时长，以判断收益（或损失）来自哪一侧。
2. 预期应保守：较小的 tile 不能提高 shared 约束下的列槽位上限；能否通过同驻、warp 或缓存行为得到净收益，只能实测。
3. 若原型不显示净收益，按 Codex 的计划转向 L4/L5 optimizer 本体归因。

## 证据

`attempt1/{off,on}/{events.bin,names.tsv,markers.tsv}`、`attempt1/{closure,manifest}.json`；recover 结构见
`fastkpc/src/cuda/full_cuda_ci_structured_eig.cu` 的 `recover` 与 plan 的 tile 计算。计算为只读一次性脚本；没有修改生产源码、
native 或默认配置，没有 commit 或 push。
