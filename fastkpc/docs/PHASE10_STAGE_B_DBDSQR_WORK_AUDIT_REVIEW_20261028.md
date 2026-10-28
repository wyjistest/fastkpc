# Stage B DBDSQR work 生命周期审计：复核

日期：2026-10-28。复核 [DBDSQR work 生命周期审计](PHASE10_STAGE_B_DBDSQR_WORK_AUDIT_20261028.md)（证据
`fastkpc/artifacts/full_cuda_ci/stage_b_dbdsqr_work_audit_20261028/`）。只读复核，没有新的 GPU 运行或编译。
`STATIC_WORK_LIFETIME_AUDIT_COMPLETE` 成立；同意下一项 `ISOLATED_FIXED_VT_WORK_PAIR_COMPACTION_QUALIFICATION`。
accepted opt-in development median 仍为 329.078 s；L8 STOP、默认关闭、Phase 10 ACTIVE、candidate NOT_FROZEN、
holdout SEALED、recommended=false 均不变。同日按 Codex 核对意见修订，见“修订记录”。

## 修订记录（Codex 核对后）

Codex 同意核心论证（固定 NCVT=1、NRU=NCC=0 时正向只消费前两段、反向只消费后两段，19,656 → 14,040 B，上限 4 → 6）
和两项新增检查，并复查当前 native 的 bidiagonal 为 REG=100、STACK=0、LOCAL=0。另指出三处，我复核后全部接受：

1. **manifest 的完整性声明已不成立。** 我核对 manifest 时 123 个条目全部一致；之后我修订了其中作为引用输入封存的
   `PHASE10_STAGE_B_ATTRIBUTION_REVIEW_20261028.md`（封存 SHA `0442b4b6…`，现为 `90aadd3a…`），所以现在是 122/123。
   native、103 个生产输入、工具和封存文件都一致。差异来自文档修订，不影响源码审计结论；原 manifest 保留不改。
2. **旧归因复核的遗留措辞。** “需要逐位资格的不进入原型”改为“未通过逐位资格的改动不得进入生产集成”；正文的
   “第一轮/第二轮”“全部同时开始”统一为早启动、晚启动和近乎同时启动。已在该文档中修订。
3. **“上一 wave 已 join”要限定范围。** 源码只等待本流水线上一 wave 的全部任务（包括 cache stores）完成，这不保证整张
   GPU 空闲，也不保证下一 wave 的 640 个 block 同时驻留。第 3 节已改。

## 结论

```text
完整性    native e8c39703… 与 103 个生产输入未变；build 目录与封存副本的 recorded_dbdsqr.inc 的 SHA-256 都为
          be76c5e0…；audit.json、manifest.json 的 SHA-256 吻合。manifest 123 个条目现为 122/123 一致：唯一
          不一致的是我事后修订的 Stage B 归因复核（文档版本差异，不影响源码审计；原 manifest 保留）
核心论点  在封存的 generated DBDSQR 中独立核对成立。生产调用为 UPLO=L、NCVT=1、NRU=NCC=0，唯一活跃的消费者是
          对 q（VT）的 record_dlasr_：正向 sweep 读第 1、2 段，反向 sweep 读第 3、4 段；另一对只被 U/C 更新读取，
          两者都关闭。下双对角转上双对角时写入的 cs/sn 同样只被 U/C 读取
候选      两段物理 scratch：正向只写第 1、2 段，反向把原第 3、4 段的值（含负号）写到这两段，消费者改指向它们。
          直接别名四段是错的；只去尾部 padding 只省 64 B。动态 shared 19,656 → 14,040 B，按 128 B 粒度
          的分配为 20,736 → 15,104 B，shared 上限每 SM 4 → 6 个 block，均复算一致
时间      能否缩短 wave 仍是未知数，只能由隔离原型测量
建议      同意只做两段 work 的隔离 GPU 原型，先逐位资格，再按预声明协议测完整 L2/L3 Stage B。另补三点，见第 3 节
```

## 1. 在源码中核对的数据流

在封存的 `input-source/recorded_dbdsqr.inc` 中（f2c 风格，`work` 为一基；`nm1 = n−1`，`nm12 = 2·nm1`，
`nm13 = 3·nm1`，因此 `work[*n]` 是第 2 段的开头）：

| 路径 | 写入 work（四段） | NCVT>0 时的消费者 | 行号 |
|---|---|---|---|
| 下双对角转上双对角 | `cs`、`sn`（两段） | 无；只有 `nru>0`、`ncc>0` 读取 | 140–162 |
| zero shift，正向 | `cs`、`sn`、`oldcs`、`oldsn` | `&work[1]`、`&work[*n]` → 第 1、2 段 | 465–490 |
| zero shift，反向 | `cs`、`−sn`、`oldcs`、`−oldsn` | `&work[nm12+1]`、`&work[nm13+1]` → 第 3、4 段 | 518–540 |
| shift，正向 | `cosr`、`sinr`、`cosl`、`sinl` | `&work[1]`、`&work[*n]` → 第 1、2 段 | 583–602 |
| shift，反向 | `cosr`、`−sinr`、`cosl`、`−sinl` | `&work[nm12+1]`、`&work[nm13+1]` → 第 3、4 段 | 643–668 |

- 消费者都在各自的循环之后执行，读取的正是本次 sweep 刚写入的前缀（`m−ll` 个旋转）；没有跨 sweep 读取旧值。
- 生产调用 `recorded_dbdsqr_("L", &r, &one, &zero, &zero, d, e, q, &r, …)` 即 UPLO=L、N=r、NCVT=1、NRU=0、NCC=0，
  与审计的固定配置一致。
- 别名反例成立：若第 3、4 段与第 1、2 段共用地址，正向循环先写 `cs`、再写 `oldcs` 到同一位置，消费者读到的是
  `oldcs`。候选必须同时去掉未被消费的写入并重映射，审计报告写明了这一点。
- 不被消费的写入可以删，但产生这些值的计算不能删：`oldcs/oldsn`、`cosl/sinl` 仍参与 d/e 的递推，审计报告也写明了。

## 2. 资源数字

按 102,400 B shared/SM、每个 block 1,024 B 保留、128 B 分配粒度复算，与报告一致：

| 形状 | 动态 shared | 分配 | 每 SM 上限 |
|---|---:|---:|---:|
| 当前 7N | 19,656 | 20,736 | 4 |
| 只去尾部 padding（work 为 4(N−2)） | 19,592 | 20,736 | 4 |
| 五个 block 的临界值 | 19,456 | 20,480 | 5 |
| 候选 5N（d/e/q 各 N、work 2N） | 14,040 | 15,104 | 6 |

全卡 shared 上限从 512 个提高到 768 个 block，可以容纳一个 wave 的 640 个 component。这只是上限，不是实际占用率。

## 3. 对隔离原型的补充意见

1. **用 CUDA occupancy API 确认上限。** 对编译后的原型 kernel 与 14,040 B 动态 shared 调用
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor`，记录结果，不只依赖手算模型。这一步不需要 profiler 权限；它给出的
   仍是理论驻留上限，不能证明实际并发或提速。
2. **对照编译资源。** 用 `cuobjdump -res-usage` 对比原型与生产 kernel 的 REG/STACK/LOCAL：当前为 REG=100、
   STACK=0、LOCAL=0。删除写入或重映射后若寄存器、stack 或 spill 变化，要单列说明。
3. **wave 是否缩短不能预先估计。** 当前每个普通 wave 都是 8 个 lane 的 bidiagonal 先开始、2 个在 223–347 ms 才开始，
   10 个 launch 区间有重叠（L2 77/77、L3 85/85），与 512 个 block 的上限相符。上限提高到 768 后，640 个 block 有机会
   同时开始。源码在下一 wave 之前只等待本流水线上一 wave 的全部任务（包括 cache stores）完成，这不保证整张 GPU 空闲，
   也不保证 640 个 block 同时驻留。此外，目前早启动 lane 的 recover 与晚启动 lane 的 bidiagonal 重叠；改变之后，recover
   （每 SM 1 个、与 bidiagonal 互斥）可能更集中地排队，同一 SM 上的 block 增多也可能改变每个 block 的时长。所以只能以
   完整 L2/L3 Stage B 的重复计时判断。

逐位资格的范围（d/e/q、rotation log 全字段与顺序与 count、info、a/b、scale/live、最终 component 三字段；r=1、二阶块、
正反方向、零/非零 shift、tie、错误与 rotation-capacity 边界；未自然触达的分支单独报告）按审计报告执行，我同意。

下一项保持很窄（Codex 的建议，我同意）：只改固定 VT 入口的 work 存储与地址映射，保留 QR 算术、recover、lane 数和
同步拓扑；先在现有 104,797 个 captured component 上验证中间状态、完整 rotation log、异常边界与最终输出逐位一致；
通过后，重复比较包含 recover、status 与 cache stores 的完整 L2/L3 Stage B；完整阶段有稳定净收益，才考虑生产 opt-in。

## 证据

`input-source/recorded_dbdsqr.inc`（第 140–162、465–490、518–540、583–602、643–668 行）、`audit.json`、`manifest.json`、
`fastkpc/build/recorded_dbdsqr.inc`；lane 时序由 `stage_b_attribution_20261028/attempt3/trace/` 重算。计算为只读一次性
脚本；没有修改生产源码、generator、native 或默认配置，没有 commit 或 push。
