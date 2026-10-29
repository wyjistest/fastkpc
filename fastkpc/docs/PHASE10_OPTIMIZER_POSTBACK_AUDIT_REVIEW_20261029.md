# Exact optimizer vector-postback 独立向量边界审计：复核

日期：2026-10-29。复核 [vector-postback 审计](PHASE10_OPTIMIZER_POSTBACK_AUDIT_20261029.md)（证据
`fastkpc/artifacts/full_cuda_ci/optimizer_postback_audit_20261029/attempt1/`）。只读复核：没有 GPU 运行或 CUDA 编译；在 `/tmp`
中重跑了审计工具，并做了一项 host 端交叉检查（第 2 节）。`STATIC_VECTOR_SLICE_AUDIT_COMPLETE` 成立；同意下一项只做隔离的 DLARFB
GPU 资格原型（GO），runtime 集成 NOT YET GO。accepted opt-in development median 仍为 311.440 s；默认关闭、L8 optimizer STOP、
Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。

## 修订记录（Codex 修改，我已复核）

同日 Codex 在本文与交叉检查工具中做了以下修改，我复核后全部接受：

1. 有限输入下输出没有 NaN，只是“未观察到哨兵污染”，不证明没有读取；guard 只检查相应区域的写入。
2. 敏感性对照中未检出的扰动，不再全归于舍入，也可能来自零结构或特殊值传播。
3. “L5 每块 k=32”改为 NB=32，并写明 Q/P 尾块为 14/13。
4. 工具原先对缺失或空输入以零用例、退出码 0 返回，这是缺陷；现已校验封存 inventory/TSV 的 SHA、逐行解析与完整计数，失败退出 2。

## 结论

```text
完整性    audit.json（f88aa37c…）、manifest.json（0695f219…）的 SHA-256 吻合；manifest 133 项（104 个生产源码、10 个输入、
          3 个工具、16 个文件）按当前文件重算一致；native 12acc3cd… 未变。输入之一是我修订后的 L4/L5 复核，此后我不再修改它
重跑      audit.py 写入 /tmp 的新目录，耗时 4.9 s：audit.json 逐字节相同（SHA 相同），symbolic.json、shapes.tsv、
          inventory.json 与提取的函数体也相同；protocol.json 只差输出目录路径
代码      DLARFB 两个实际分支的步骤、全部 DTRMM 为右乘、各块偏移与 IB、W/T 布局、P 侧 C 的列偏移 1+offset、切分整个 DORMBR
          会把 L4 的 NB 从 2 变成 32，均与 vendored 源码一致。符号工具的设计成立：表达式按 (op, left, right) 精确比较，
          V1 的 unit diagonal 与无效三角、T 的下三角不给有效符号，lane 逆序执行，依赖位集与两侧 guard 都检查
交叉检查  用 host 的 IEEE double 编译同一份提取的 reference 函数，对全部 1,553 个块形状 × 4 种输入 × 3 个种子 = 18,636 例，
          比较完整 DLARFB 与逐向量调用：C、W 逐位相同，V/T 未改，guard 完整，有限输入下未观察到有效 C 输出的 NaN 传播；输入含 ±0、
          T 对角为 0/1、C 中的 Inf/NaN。敏感性对照检出差异。这只是 host 等价性检查，不是 GPU 资格，也不是数值 authority
意见      同意原型范围与门；第 3 节列出几点供原型记录
```

## 1. 完整性与重跑

- 两个 SHA 与报告一致；manifest 的 133 项全部吻合，当前 native 仍为 `12acc3cd…`。
- 审计工具的输入之一是 `PHASE10_OPTIMIZER_L45_ATTRIBUTION_REVIEW_20261029.md`，封存的是我按 Codex 核对修订后的版本。为保持该
  manifest 有效，我不再修改那份复核。
- 在 `/tmp/postback_rerun` 重跑 `audit.py`：工具重新校验 native 与 104 个生产输入，提取函数、编译符号程序并执行，结果与封存
  证据一致。`protocol.json` 的差别只在 `-I` 与可执行文件的绝对路径。
- 读 `audit.py` 的形状枚举：块大小按 LWORK 推出（q=37 为 2，q=46 为 32），块按 DORMQR/DORMLQ 原倒序；Q 侧 C 的 leading
  dimension 在 direct 路径为 rows、QR 路径为 q，P 侧为 q；每个向量的 C 地址集合互不相交且并集为 m×n，W 地址集合互不相交
  且并集为 q×IB，T 区间不与 W 相交。符号形状取 direct 的两端与 QR 路径（QR 路径的形状与 rows 无关），中间的 direct 形状只做
  整数检查，报告已写明。
- 控制流的一处说明也核对了：延迟右向量完成既发生在接受之后，也会在 `trial_requires_complete_evaluation`（模糊的首个 Newton
  步）为真时于接受判断之前发生（`mgcv_multi_penalty_gcv.cu` 约 2694–2706、2800–2808 行）。

## 2. host 端交叉检查（方法与符号工具不同）

- 做法：把封存的 `symbolic_vendor.inc`（DCOPY、DGEMM、DTRMM、DLARFB 原样函数体）以 `double` 编译（`g++ -O2
  -ffp-contract=off`，x86-64 SSE2），对 `inventory.json` 中全部 1,553 个块调用，比较一次完整 DLARFB 与按 lane 逆序的逐向量调用
  （Q：m×1、C 偏移 r·ldc；P：1×n、C 偏移 r；W 偏移 r，ldwork = q）。
- 输入：四种模式（稠密随机；约 30% 的 ±0；T 对角取 0/1；C 中混入 ±Inf 与 NaN），各三个种子。V1 的 unit diagonal 与无效三角、
  T 的下三角、W 的初值都填 NaN 作哨兵。
- 结果：18,636 例全部逐位相同（C 与 W 的全部字节，含 NaN payload），V/T 不变，两侧 guard 不变；三种有限模式的输出中没有 NaN，
  这是未观察到哨兵污染的证据，不能证明没有发生读取；guard 也只检查相应区域的写入。共 709,404 次逐向量调用。
- 敏感性对照：在逐向量副本中把一个有限非零输入翻转最低位，18,636 例中 8,597 例被检出（其余可能受舍入、零结构或特殊值传播遮蔽，未逐例归因），说明比较不是
  平凡通过。
- 局限：host 编译器与 GPU 的指令选择不同；`--fmad=false` 与 `-ffp-contract=off` 都不做融合，但 GPU 的逐位一致仍须在 GPU 上
  证明。本检查不覆盖同步、DLARFT、DORMQR/DORMLQ 外层循环与 WORK(1) 写入。

## 3. 对隔离 GPU 原型的意见

同意范围：原 NB、串行 DLARFT、warp 0 各 lane 按向量调用原 DLARFB；保留 64 线程 block、全部 workspace 布局、`--fmad=false`、
q=37/46、原请求/cohort 顺序与 32 槽；先过形状、guard、info、资源闭环，再过全部 L4/L5 cohort 的逐位门，最后以完整层跨度与重复
outer wall-time 决定。补充几点，供原型记录：

1. **lane 控制流基本一致。** vendored DGEMM 只对 alpha/beta 判断（这里为常数 ±1 与 1），不对矩阵元素做零值判断；实际使用的
   右乘 DTRMM 只判断 V/T 的元素，而 V/T 对各向量相同。所以各 lane 走相同分支；差别只在第二个向量：L4 由 lane 0–4、L5 由
   lane 0–13 处理 r+32。
2. **访存形态两侧不同（待测的假设）。** Q 侧 lane 各占 C 的一列，同一条指令中各 lane 的地址相隔 ldc 个元素（至少 37×8 = 296 B），
   不能合并；P 侧 lane 各占一行，地址相邻，可以合并；两侧 W 的行地址相邻，V/T 为同址读取。Q、P 的收益可能不同，按报告分开计时
   即可看出。
3. **串行部分。** DLARFT 仍由 lane 0 执行；L5 的 NB=32，实际 Q 块为 14、32，P 块为 13、32，必须保留尾块。DLARFT 与 DLARFB 并行化之后的相对份额未知。报告计划分开记录
   DLARFT/DLARFB 与 Q/P，正好用于判断串行部分是否限制收益。
4. **同步与可见性。** `cooperative_sync(32)` 即 `__syncwarp()`，它对参与的线程提供内存顺序，T/W/C 位于 global workspace，可以
   沿用。需要的边界如报告所列：lane 0 写好 T 之后、各 lane 读 T 之前；全部向量完成之后、下一块覆盖 T/W 之前；DORMQR/DORMLQ
   对 WORK(1) 的写入在最后一块之后。compute-sanitizer 的 racecheck 只检查 shared memory，这里的冒险只能靠设计审查与逐位资格
   发现；synccheck 可以检查同步用法。

## 证据

后续入口加固（2026-10-29）：旧 `slice_check` 对不存在或空文件会以零用例、退出码 0 返回。
现已校验封存 inventory/TSV SHA、文件打开、逐行解析和完整计数；`--validate-only` 只执行输入资格，
不做浮点求解。该修复不追溯升级上面的历史 host 运行，也不改变原静态审计的封存证据。

后续 GPU 执行已完成，见 [隔离原型报告](PHASE10_OPTIMIZER_POSTBACK_GPU_20261029.md)：
完整 cohort 逐位门及三组 canonical 配对均通过，中位数 315.067 → 286.882 s。
该结果属于新的隔离研究证据，runtime 集成仍待单独资格验证。

`attempt1/{audit,manifest,protocol,symbolic,inventory}.json`、`attempt1/{shapes.tsv,symbolic_vendor.inc,symbolic}`、
`tools/research/optimizer_postback_audit_20261029/{audit.py,symbolic.cpp}`；源码 `src/cuda/third_party/lapack_3_12_{block_reflector,dormbr}_device.inc`、
`src/cuda/lapack_312_small_dgesdd.cuh`（`cooperative_sync`）、`src/cuda/mgcv_multi_penalty_gcv.cu`（延迟完成条件）。交叉检查工具与
命令见 [README](../tools/research/optimizer_postback_audit_review_20261029/README.md)（按 README 重新编译运行，结果同上）；`/tmp` 中的
重跑输出不属于封存证据，已删除。没有修改生产源码、native、默认配置或任何封存输入，没有 commit 或 push。
