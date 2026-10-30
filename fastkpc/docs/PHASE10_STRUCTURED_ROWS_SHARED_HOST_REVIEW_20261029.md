# Structured 行和原型共享主机补充对照：复核

2026-10-30 执行状态补记：[原协议隔离验收](PHASE10_STRUCTURED_ROWS_ISOLATED_20261030.md)
已完成，容量 GO、速度 NOT_PROVEN，ON native 已安装。下文保留本次复核及修订时的
判断，未安装/待执行语句是当时状态；原共享主机证据和复核意见不变。

日期：2026-10-29。复核 [共享主机补充对照](PHASE10_STRUCTURED_ROWS_SHARED_HOST_20261029.md)（证据
`fastkpc/artifacts/full_cuda_ci/structured_rows_shared_host_20261029/`）。只读复核，没有 GPU 运行或编译。
`PASS_SHARED_HOST_DEVELOPMENT` 成立：数值、容量与共享主机不回退门通过，速度门 NOT_MET。原隔离验收仍待执行，原型未安装；
accepted opt-in development median 仍为 290.491 s，安装的 native 仍为 `a1e16a9c…`；默认关闭、Phase 10 ACTIVE、candidate
NOT_FROZEN、holdout SEALED、recommended=false 均不变。

次日按 Codex 核对意见修订，见“修订记录”。

## 修订记录（Codex 核对后）

Codex 重新核对了 419 项哈希、六轮统计与分层差值，主要数据吻合；结论保持“容量收益成立、共享主机不回退通过、速度收益未证明”。四处
收紧，我复核后全部接受：

1. **“都来自 L7/L8 的窗口波动”过强。** 扣除 L7/L8、L4/L5 与 Stage B 后余数为 0.179/0.035/0.057 s，这支持差值的位置集中在 L7/L8，
   但没有证明波动的原因。改为“差值主要集中在 L7/L8 optimizer，未识别出矩阵删除的稳定速度收益”。
2. **撤回“把原型改动移回研究目录”的建议。** 隔离脚本会校验工作区与已认证的 ON 源码一致，移动会破坏这项校验；保留固定源码与两份独立
   native，并写明“尚未安装”即可。我核对了加载器：`R/cuda_native.R:926` 只在 `rebuild = TRUE` 或 native 不存在时重建，默认
   `rebuild = FALSE`，所以风险主要来自显式重建或 native 缺失。
3. **预声明应与运行开始比较。** 协议 22:24:45.286 写入，第一轮进程 22:24:48.110 启动、计算 22:24:53.438 开始，预声明成立；原来写的
   “早于第一轮结束”已替换。
4. **“能通过编译说明没有其他使用者”不严谨。** 我补查了数据流：HEAD 中 structured lane 的 `matrices` 只出现在分配、作为 kernel 的写入
   目标与成员声明三处，没有被读取，也没有交给 view 或 plan；加上报告的 GPU 回归，删除成立。编译通过只作辅助证据。

范围：本复核覆盖共享主机这一轮及 ON 相对 OFF 的源码改动。[构建一致性](PHASE10_BUILD_CONSISTENCY_20261029.md) 与
[行和原型](PHASE10_STRUCTURED_ROWS_20261029.md) 两份报告只作背景阅读，它们的证据没有单独复算（第 1 节列出的源码 diff 除外）。

## 结论

```text
完整性    protocol（38935ddb…）、closure（7e6a4814…）、manifest（ad13ec17…）的 SHA-256 吻合；manifest 419 项全部一致。
          其中 full_cuda_ci_vertical.cu 按 off/on 分别封存：OFF 版等于 HEAD 的 blob（cd18b493…），ON 版等于当前工作区
          （13c44c26…）。audit.json 为 PASS_RECOMPUTED_SHARED_HOST
代码      ON 只改 full_cuda_ci_vertical.cu：新的 row-only kernel 是原 build_distance_rows_batch_kernel 去掉距离矩阵写入，
          fabs、每线程按列累加、共享内存归约树与输出位置都相同；其中没有乘法，FMA 策略不会改变结果。structured lane 的
          matrices 在 HEAD 中只被分配、写入与声明，没有读取或交给 view/plan，删除得到数据流与 GPU 回归的支持（编译通过
          只是辅助证据）；dense consumer 与 dense fallback 仍用原 kernel
协议      协议 22:24:45.286 写入，早于第一轮进程启动（22:24:48.110）与计算开始（22:24:53.438）；门直接绑定原 pending
          protocol（2f9c5cc4…），GPU1 独占、允许
          GPU0 背景任务；六轮按 OFF ON ON OFF OFF ON 运行。主机采样中，GPU0 每轮都有外部进程 2323839、2335739，GPU1
          每轮只有本轮自己的一个进程
计时      OFF 286.718/295.873/297.619 s，ON 293.149/289.468/291.056 s；中位 295.873 → 291.056 s（比值 0.983719，
          −1.6281%），配对比值 1.022430/0.978352/0.977948。不回退门（中位 ≤ 1.02、配对 ≤ 1.05）通过；速度门（≤ 0.98
          且 3/3）NOT_MET
归因      三对的 ON − OFF 为 +6.431/−6.405/−6.563 s，其中 L7+L8 optimizer 占 +5.784/−6.447/−6.799 s；再扣除 L4/L5 与
          Stage B，余数为 +0.179/+0.035/+0.057 s。差值主要集中在 L7/L8 optimizer，波动的原因本轮未验证；未识别出矩阵删除
          的稳定速度收益，与报告的判断一致
容量      七层 structured pool_device_bytes 均为 3,422,507,520 → 2,791,718,400 B（−630,789,120 B = 10 × 63,078,912 B），
          分配/释放 169 → 159；L3/L4/L6 最低空闲各多 650,117,120 B（620 MiB），L6 距 2 GiB 门 699.125 MiB（原 79.125）
数值      六轮各 240,498 个 CI/pMax 逐位一致；门 74/74（首轮）与 77/77（重复轮）；最大 authority 误差 3.05425129631942e-12
```

## 1. 源码与状态

- `git diff` 中 ON 的唯一 runtime 改动见上。工作区另有构建一致性的改动：`build_cuda_native.sh` 用 `METHOD_BATCH_NUMERICAL_FLAGS`
  让 `full_cuda_ci_method_batch.cu` 的全量与增量入口都带 `--fmad=false`（原先增量入口没有）；两个 R 测试把 `Sys.unsetenv` 的参数改为字符
  向量；两份 README 与路线图更新。这些都未提交。
- **工作区与安装的 native 不对应。** 工作区的 `full_cuda_ci_vertical.cu` 已是 ON 版本，而安装的仍是 `a1e16a9c…`（不含行和改动，也不含构建
  修复）。隔离脚本要校验工作区与已认证的 ON 源码一致，所以源码应保持原位；加载器只在 `rebuild = TRUE` 或 native 缺失时重建
  （`R/cuda_native.R:926`），风险来自显式重建或 native 缺失。验收前不要显式重建，交接中写明“原型源码在工作区、尚未安装”。

## 2. 复现与补充

- 六个 `report.json`：outer、各阶段中位（optimizer 108.090 → 103.092、Stage B 128.428 → 128.347、Stage A 14.539 → 14.565、native setup
  24.421 → 24.493、consumer dCov 14.354 → 14.350）、门数、容量检查点与 `pool_device_bytes`、分配计数，全部与报告一致。
- **620 MiB 与 601.57 MiB 的差。** 被删除的是 10 个各 63,078,912 B 的分配；每个跨 30.08 个 2 MiB 页、占 31 页即 62 MiB，10 个正好
  620 MiB，与实测的空闲改善完全吻合。这符合按 2 MiB 粒度分配的推断，没有单独验证。
- **共享主机的绝对秒数不可与 290.491 s 比较。** 两侧的 native setup 中位 24.42/24.49 s，比此前隔离的 postback runtime campaign（22.20/22.21 s）
  多约 2.2–2.3 s，Stage B 也多约 0.5 s。两侧同样受影响，原因（主机争用或重建的 native）未定。报告已把本轮限定为共享主机观察，我同意。

## 3. 对待执行的隔离对照的意见

同意报告的安排：两卡无外部 compute 时按原 pending protocol 补做六轮，不改门，不混入本轮。补充两点：

1. 这个改动省的是显存，Stage B 在本轮三对中只差 +0.43/−0.08/+0.03 s，预期耗时效果接近零；而 L7/L8 optimizer 时间的波动在以往 campaign 中
   单次可达约 +8.4 s（L8）与 +6.2 s（L7）。按协议，安装以数值、容量、资源与不回退门为准，速度门不是安装条件；隔离对照的速度门很可能仍不通过，
   这只是预期，不影响以不回退门判定。
2. 报告隔离对照时，除预声明的门外，也按结论中“归因”一栏的方式列出每对扣除 L7/L8（及 L4/L5、Stage B）后的余数，便于区分改动效果与
   L7/L8 的时间波动；这只用于描述差值的位置，不说明原因，也不改变门。

## 证据

`structured_rows_shared_host_20261029/{protocol,closure,manifest,audit}.json`、`{off,on}/{off,on}{1,2,3}/{report,telemetry-summary}.json`、
`*-host.jsonl`（GPU 进程列表）、`{off,on}/build.json`；源码 `git diff`（`full_cuda_ci_vertical.cu`、`build_cuda_native.sh`、两个 R 测试、
`goal-5.6.md`）。计算为只读一次性脚本；没有修改生产源码、native、默认配置或封存输入，没有 commit 或 push。
