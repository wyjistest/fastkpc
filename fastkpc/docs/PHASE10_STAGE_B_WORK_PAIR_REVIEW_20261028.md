# Stage B fixed-VT work-pair 隔离资格：复核

日期：2026-10-28。复核 [Stage B work-pair 隔离资格](PHASE10_STAGE_B_WORK_PAIR_20261028.md)（证据
`fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_20261028/attempt1/`）。只读复核，没有新的 GPU 运行或编译。
`GO_ISOLATED_STAGE_B` 成立；生产集成与完整方法性能资格仍待完成；accepted opt-in development median 仍为
329.078 s；L8 STOP、默认关闭、Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。

## 结论

```text
完整性    closure.json、manifest.json 的 SHA-256 吻合；manifest 218 个条目按当前文件重算一致；生产 native、
          103 个生产输入与生产 recorded_dbdsqr.inc（be76c5e0…）未变
代码      work-pair.diff 与静态审计的候选逐行对应：入口拒绝 UPLO=L、NCVT=1、NRU=NCC=0 以外的配置；正向两处删去
          第 3、4 段写入；反向两处把 oldcs/−oldsn、cosl/−sinl 写到第 1、2 段，VT 消费者改指 &work[1]、&work[*n]；
          下双对角转换与全部标量算术不变。未发现缺陷
资格      104,797 个 captured component、10,515,312,078 个 rotation、68,297,253 个 sweep 零不一致；433 个
          合成案例、19,513,248 个 rotation 零不一致；4 个 capacity 案例、1 个注入的不收敛、8 个 prepare 边界通过；
          资源归零。自然数据从未触达 r=1、zero shift 反向、排序 tie 与不收敛，靠合成案例覆盖，报告已写明
资源      我上一轮建议的两项都已执行：occupancy API 每 SM 4 → 6 个 block；寄存器 100 → 82，LOCAL 0；动态 shared
          19,656 → 14,040 B
计时      协议 15:27:49 声明，六次于 15:33–15:44 按 A1/C1/C2/A2/A3/C3 运行，均 exit 0、isolated。重算：中位
          L2 54.697374 → 48.434411 s、L3 60.260344 → 53.196172 s，两者之和 114.957718 → 101.630583 s
          （−11.5931%）；最慢的候选 101.690455 s 低于最快的基线 114.824857 s。每个 wave 约 0.701 → 0.620 s
补充      L4–L8 的 Stage B 用同一 kernel，有 25,873 个 component（约 29.3 s），不在本轮逐位回放中，集成时应补
```

## 1. 代码复核

`work-pair.diff`（封存的 generated DBDSQR → 隔离的 fixed-VT work pair）只有以下改动：

| 位置 | 改动 | 与审计的对应 |
|---|---|---|
| 入口 | `uplo != 'L' \|\| ncvt != 1 \|\| nru != 0 \|\| ncc != 0` 时 `info = −101` 并返回 | 固定配置专用、其余 fail closed |
| zero shift 正向（L120） | 删去 `oldcs`、`oldsn` 两处写入 | 第 3、4 段无消费者 |
| zero shift 反向（L130） | 四处写入改为 `work[i−ll] = oldcs`、`work[i−ll+nm1] = −oldsn`；消费者改为 `&work[1]`、`&work[*n]` | 原第 3、4 段的值与符号移到第 1、2 段 |
| shift 正向（L140） | 删去 `cosl`、`sinl` 两处写入 | 第 3、4 段无消费者 |
| shift 反向（L150） | 改为 `work[i−ll] = cosl`、`work[i−ll+nm1] = −sinl`；消费者同上 | 同上 |

- 负号表达式与原文相同（`−oldsn`、`−sinl`），带符号零的行为不变。反向的写入下标为 1…m−ll 与 nm1+1…nm1+m−ll，
  不超过 2(N−1)，落在 2N 的 work 内。
- 下双对角转上双对角的两段写入保留；NRU/NCC 的消费者代码仍在，但入口检查使其不可达。
- 入口检查在生产调用中不会触发（实参固定为 "L"、1、0、0）；若触发，`info = −101` 会使该 component 走既有的失败路径。

## 2. 资格与计时的复现

- `qualification.json`：状态 `PASS_GPU_BITWISE_QUALIFICATION`。captured 的 branch 计数为
  `[0,104797,104797,21163,0,104797,100556,100556,104796,0,0]`，合成为 `[9,420,399,8,5,230,379,222,370,173,1]`，
  与报告一致。
- `resources.json`：基线 occupancy 4、寄存器 100；候选 occupancy 6、寄存器 82；两者 LOCAL 与 static shared 为 0。
- `campaign.json` 的六次数据与报告一致；预声明的门（两层都改善、和至少降 5%、候选总时间全部低于基线）都满足。

寄存器从 100 降到 82、shared 写入减少、驻留上限从 4 升到 6 同时发生，收益不能单独归于驻留上限，报告已写明。
本轮没有 trace，不知道 wave 内“8 个 lane 早启动、2 个晚启动”的形态是否改变，也不知道 recover 的排队如何变化。

## 3. 对默认关闭集成的意见

同意下一项做默认关闭的生产集成，再验证完整正确性与重复总耗时。补充四点：

1. **L4–L8 的覆盖。** canonical 运行中 structured component 共 130,670 个：L2 49,919、L3 54,878、L4 17,767、L5 6,816、
   L6 1,148、L7 124、L8 18；L4–L8 的 Stage B 约 29.3 s。本轮逐位回放只覆盖 L2/L3（约 80%）。若开关作用于共用的
   kernel，应在集成资格中把 captured 回放扩展到 L4–L8 的 component，或先把开关限定在 L2/L3。完整 canonical 的 CI/pMax
   逐位门是端到端的覆盖，但不能代替 component 级的中间状态比较。
2. **生成与配置的一致性。** 两段 work 版本宜由生产 generator 生成为单独的 fixed-VT 入口，保留原 DBDSQR，并纳入源码
   闭包与 include 哈希。开关必须同时决定 kernel 的动态 shared 大小（5N）与所用的入口：原四段入口配 5N 的 shared 会越界。
   建议在资格中加入这一组合的拒绝测试，并保留 shared 末端 canary 的诊断构建。
3. **容量。** shared 是片上资源，不改变 device 显存分配；但完整 canonical 仍应照常检查 L3 分点合同与 L4/L6 守卫。
4. **计时。** 按预声明协议做同一 native 的隔离 OFF/ON 完整方法 campaign；报告分列 L2–L8 各层 Stage B 的变化。
   13.327 s 是隔离回放中两层中位数之和的差，不能直接从 329.078 s 中扣除，报告已写明。可选：做一次带 CUPTI trace 的
   候选回放，只作诊断，查看 lane 时序与 recover 排队的变化，供以后选题参考。

## 证据

`attempt1/{work-pair.diff,qualification.json,resources.json,protocol.json,campaign.json,closure.json,manifest.json}`、
`attempt1/{A1,C1,C2,A2,A3,C3}-process.json`；L4–L8 的 component 数来自
`optimizer_l3_slot_runtime_20261028/on3/report.json`。计算为只读一次性脚本；没有修改生产源码、generator、native 或
默认配置，没有 commit 或 push。
