# Structured component prefill：22 列结果复核

日期：2026-10-22。复核 [22 列完整 skeleton 资格](PHASE10_STRUCTURED_PREFILL_RESULT_20261022.md)
（native `4b92e48d…`）。本文不改代码、不改状态；新增一项 authority 抽查（CPU，R）。

## 结论

```text
资格结论            成立；源码/native 绑定、代码路径、回执数字与新增 authority 均复核无误
新增 authority      随机抽 6 个新增 CI（level 2–4），已安装 kpcalg 1.0.1 包逐位重现
收益构成（单序列）   structured − dense = −11.633 s = 阶段 B −13.311 s + optimizer +1.634 s + 其余 +0.04 s
optimizer 差异      随运行顺序单调上升（off 42.28 → dense 43.46 → structured 45.10 s，迭代数相同），
                    更像顺序漂移；结构性部分是阶段 B
canonical 显存      推算峰值约 21,990 MiB / 24,564 MiB，余量约 2.5 GiB，可以运行，但须看实测
提交前须处理        生产构建依赖未跟踪的研究脚本 generate_recorded_dbdsqr.py
下一步              同意 canonical opt-in；计时建议 dense-on / structured-on 交错
```

## 复核

- **绑定**：六份源码（one-call、vertical.{cu,hpp}、structured_eig.{cu,hpp}、
  `build_cuda_native.sh`）的 SHA 与 `closure.json` 一致；`build/` 中 native 为 `4b92e48d…`。
- **数值 kernel**：`full_cuda_ci_structured_eig.cu` 与研究实现的 kernel 主体逐行相同，
  只新增 plan API 和 `check_structured_boundary`（与适配器版本等价）。
  `#define lapack312 structured_lapack312` 把 vendored LAPACK 隔离到独立命名空间，
  避免与 optimizer TU（编译选项不同）的同名 device 函数发生 ODR 冲突。这一处理正确且必要。
- **构建**：structured TU 单独编译，带 `--fmad=false -arch=sm_89`，且有
  `#error` 守卫（未定义 `FASTKPC_STRUCTURED_FMAD_DISABLED` 时拒绝编译）；
  各增量模式共用同一个链接步骤，都会链接该对象。
- **`build_structured_level`**：
  - 先提交一个 wave 内的全部 lane，再按序读 status；
  - 槽位 `slots[l*width+i]` 与 commit 时的 `slots[i]` 对应一致；
  - structured 存储排在 lane stream 上，主 stream 等待各 lane 的完成事件后才做
    dense 回退及其存储；
  - wave 结束时整体 join，之后才 commit 并标记 pending；
  - 异常时置 poisoned，之后 lookup / build / resolve 全部拒绝；
  - lookup 与 commit 都校验 solver identity；
  - 析构时先清空 lanes（各自同步 stream），再同步主 stream。

  未见错误。
- **回执**：逐层阶段 B 与文档一致：L3 5.384 s（866/s）、L4 2.452 s（854/s）、
  L5 0.505 s（936/s）、L6 0.276 s（42 个分量，单批延迟约 0.28 s，与预期一致）；
  合计 7,270 / 8.618 s = 844/s。off 对 dense 为 −14.5 s，与此前 22 列重复中位数
  −13.8 s 相符。
- **新增 authority**：用已安装的 kpcalg 1.0.1 包（不经 `legacy_runner.R`）对
  4,518 个新增 key 中随机抽取的 6 个重算（level 2 两个、level 3 两个、level 4 两个），
  p 值全部逐位相同。这独立验证了 key 映射、22 列数据切片与 numCol = 35。

## 收益构成

| 项 | off | dense | structured | structured − dense |
|---|---:|---:|---:|---:|
| 完整耗时 | 108.932 | 94.408 | 82.775 | −11.633 |
| 阶段 B | — | 21.929 | 8.618 | −13.311 |
| optimizer host | 42.277 | 43.464 | 45.098 | +1.634 |
| consumer dCov host | 49.803 | 18.183 | 18.196 | +0.013 |
| prepared 构建 | 9.291 | 3.809 | 3.837 | +0.028 |

单位为秒。optimizer 的迭代总数三档都是 79,742，最大单 setup 时间也相同
（8.47–8.53 s）。增量分布在 L3–L5，并且随运行顺序单调上升。此前在 dense 上做的
ABBA 重复里，两种模式的 optimizer 中位数没有差异（43.76 / 43.94 s）。因此 −13.3 s
的阶段 B 是结构性收益，−11.6 s 的整体差值只代表这一次运行序列。

## canonical 前的两点

1. **显存**。canonical 的四次运行（off、dense-on 及两次诊断）都在接近结束时
   达到采样峰值，约 18,680 MiB，且 off 与 on 相同，说明峰值来自运行中累积的
   缓存，而不是 prefill。pool 从 L3 起常驻到结束，22 列实测增量为 +3,312 MiB，
   推算 canonical 峰值约 21,990 MiB，卡容量 24,564 MiB，余量约 2.5 GiB。
   若实测不足，可以在最后一个 prefill 层后释放 pool，或下调 11 GiB 的 cache 预算
   （canonical 单层最多 5.04 GiB，但 L3 reset 之前 L0–L2 的按需插入已占 4.13 GiB）。
2. **计时**。22 列的三档是单一顺序，optimizer 漂移约 1–2 s。canonical 每次约
   14 分钟，建议至少做 dense-on / structured-on / structured-on / dense-on 交错，
   阶段 B 与 optimizer 分开报告。按 22 列吞吐（约 850/s）换算，80,751 个分量约
   95 s，dense 为 233.4 s；这只是情景换算，不是承诺。

## 提交卫生

`build_cuda_native.sh` 每次构建都会运行
`fastkpc/tools/research/exact_speedup_20261020/generate_recorded_dbdsqr.py`。它的输入
`third_party/lapack_3_12_small_dgesdd_device.inc` 已被 git 跟踪，但脚本本身位于
未跟踪的 `tools/research/` 下。一旦提交生产改动而不带上该脚本，clean checkout
将无法构建。提交前应把生成器移到已跟踪的位置（例如 `fastkpc/tools/`），或直接
提交生成结果并在构建时校验其 SHA。

## 证据

抽查脚本与输出：`fastkpc/tools/research/structured_prefill_review_20261022/spot_authority.R`
（SHA `e78e3e11…`）与 `spot_authority.out`（SHA `8d54d659…`）。脚本读取 `authority.rds`、
canonical oracle 与数据集，调用 `kpcalg::kernelCItest`，固定种子抽样；从仓库根目录用
`Rscript` 运行即可复现，6 行 |diff| 全为 0。
