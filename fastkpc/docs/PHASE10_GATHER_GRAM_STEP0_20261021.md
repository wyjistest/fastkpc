# 路径 (i) 第 0 步：三字段预填充 → 按原 miss 分组 gather → 原 Gram 逐位

日期：2026-10-21。执行
[level 分量预填充方案](PHASE10_LEVEL_COMPONENT_PREFILL_PROTOTYPE_PLAN_20261021.md)
评审确定的窄验证："复用现有 2,592 个原 miss cohorts → 验证三字段缓存 →
gather → 原 Gram/reduction 的逐位一致"。它测试的是
[payload 门](PHASE10_COMPONENT_PAYLOAD_GATE_20261021.md)与
[singleton 差分重放](PHASE10_SINGLETON_GPU_PVALUE_REPLAY_20261021.md)都没有
覆盖的新数据流。

## 决策

```text
22 宽预填充的 vectors / selected_values / total 对 native 捕获      6,632 / 6,632 逐位一致
按原 miss 分组、原顺序 gather 后执行原 Gram 的 self_moment          6,632 / 6,632 逐位一致（2,592 个 cohort）
重复一次、cohort 处理顺序反转                                       均 6,632 / 6,632，三次输出逐位相同
对照：同一三字段在 22 宽分组下执行 Gram                             1,002 个 self_moment 不一致，全部来自原 miss_count = 1，
                                                                   最大 2.9103830456733704e-10，与 eager 方案的 STOP 完全一致
compute-sanitizer memcheck（前 40 个 cohort 冒烟）                   0 errors，0 bytes leaked
资源计数                                                           前后相同
第 0 步                                                            PASS：可进入默认关闭的 22 列原型实现
```

accepted native `fe8286f7…` 未修改；没有 CPU 回退；全部 eig、Gram、归约在
GPU 上执行。这是 source-shared 的隔离探针证据，不是 native 集成资格，
也不是性能证据。

## 方法

探针（`fastkpc/tools/research/gather_gram_20261021/`）直接包含生产
`full_cuda_ci_vertical.cu`，`build.py` 从源码逐字提取两段语句：

- `component_stage.inc`：完整生产分量阶段（距离 → eig → 选择 → Gram →
  self_moment → cache D2D），锚点与 SHA（`d2b648a0…`）和 payload 门探针相同；
- `gram_stage.inc`：只含 `cublasDgemmStridedBatched` Gram 调用与
  `reduce_legacy_self_moment_kernel` 启动，构建时断言它是前者的逐字子段。

`run_full` 与 payload 门探针相同；`run_gram` 把给定的 vectors 与
selected_values 上传到执行上下文的 miss 缓冲（`vectors + i × vector_stride`、
`selected_values + i × k`，与生产的布局、lda、stride 相同），以
`miss_count = count` 执行 Gram 语句并读回 self_moment。上下文使用 8 条
persistent eig lane、容量 22、11 GiB cache 预算、同一 handle 配置。

输入是两份已归档捕获，各自解压到独立新目录：冻结 default-Inf 残差捕获
（2,592 个 finalizer cohort，6,632 个分量）与 accepted-native 的实际 cache
payload 捕获（每个分量的四字段、残差字节与 miss cohort 元数据）。每个分量
的残差字节在两份捕获之间逐字节核对。

阶段：

1. 按 `ordered` 自然序以 22 宽做完整预填充，四字段逐一与 native payload 比较；
2. 对 2,592 个原 miss cohort（按 `component_position` 排序，断言长度等于
   `component_misses`），把第 1 阶段产出的 vectors/selected_values gather 后
   执行 Gram，self_moment 与 native 比较；
3. 对照：同一三字段按 22 宽分组执行 Gram；
4. 重复第 2 阶段；
5. cohort 处理顺序反转后再做一次。

## 结果

| 阶段 | 比较分量 | 不一致 | 最大绝对差 |
|---|---:|---:|---:|
| 1. 22 宽预填充：vectors / selected_values / total | 6,632 × 3 | 0 / 0 / 0 | 0 |
| 1. 22 宽预填充：self_moment | 6,632 | 1,002（原 miss_count = 1） | 2.91e-10 |
| 2. 原 miss 分组 gather → Gram：self_moment | 6,632 | 0 | 0 |
| 3. 对照 22 宽 Gram：self_moment | 6,632 | 1,002（原 miss_count = 1） | 2.91e-10 |
| 4. 重复第 2 阶段 | 6,632 | 0 | 0 |
| 5. 反转 cohort 顺序 | 6,632 | 0 | 0 |

原 miss 宽度分布：1 → 1,057，2 → 367，3 → 441，4 → 354，5 → 204，6 → 115，
7 → 46，8 → 4，9 → 2，10 → 2 个 cohort。第 3 阶段的 22 宽 Gram 输出与第 1
阶段预填充自带的 self_moment 逐位相同，说明 `run_gram` 与完整阶段中的 Gram
语句行为一致。完整运行 302 个预填充批、8,078 个 Gram 批，诊断耗时 24 s。

memcheck 冒烟（前 40 个 cohort、7 个预填充批、127 个 Gram 批）：同样的
PASS 结构（5 个 singleton 对照差异、gather 后 0 差异），`0 errors`、
`0 bytes leaked in 0 allocations`，退出码 0。

## 含义与边界

- 对捕获的全部分量，self_moment 只取决于 Gram 调用的 batchCount 类别
  （1 与 ≥ 2），不取决于 vectors 来自哪一批 eig、也不取决于 gather 的来源
  或 cohort 处理顺序。路径 (i) 的核心假设在真实数据流上成立。
- 这证明的是"按原 miss 分组 gather 后逐位复现"，前提是原型的物理分组与
  baseline 相同、pending 集合等于 baseline miss 集合；这两点仍由原型自己
  的分组序列摘要与 `gram_resolve_sequence_sha256` 门验证。
- 探针的 gather 是 H2D 上传，原型中是 cache 槽位到执行上下文的 D2D；字节
  相同，GEMM 不区分来源，但原型的 stream/event 顺序与状态机仍需在集成时
  验证。
- 未覆盖 canonical 的 48 宽上下文与 level 1–2；q 仅为 28/37/46/55。
- 本次没有新的 kpcalg 资格、没有完整 LSCP 运行、没有性能结论。

## 复现与证据

工具：[gather_gram_20261021/README.md](../tools/research/gather_gram_20261021/README.md)。
证据目录：`fastkpc/artifacts/full_cuda_ci/gather_gram_20261021/`（report.json、
memcheck 冒烟 report 与日志、build.json、两段提取语句、probe 二进制、
带逐文件 SHA 的 manifest.json）。

```text
native                 fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50
probe_gather.cu        75819c2752814cc17e8062ffccf15647651ff165ef14f26a975d4663ad60189e
build.py               efdc797cad795df7fff28b283656950902f400f6f5c13a077ac65b37be09064f
run.py                 c02f8ea037512bca8076749b97f997d948e3a08af628f104ae94468626d755ef
component_stage.inc    d2b648a05cb1c6dac130a2695812dcfc5df55aaffe28cbba4a83992e70e83d09
gram_stage.inc         04d868edc8aa50d3e77e40b2875491566941fc5fed23261a3be1e26f2bd5f53e
report.json            7a6fa005057a0b51909a624eb55fc89d942d16bd7ef04178521b0021ee8a85b2
memcheck.log           f2828c808f98499cb6be3c3697b1d3e40a81d1fdec8eddc558f56772dd215bc3
frozen capture tar     cbdc80c5f06c7d5a0b40ce031a3b950831a2b10854cd7ebd3ed57b0291cd4e72
native capture tar     e59e5f7bfb109349fed0a654a0487663105e2e3c0fec1d97811792a89f8ee900
manifest.json          197f004db8435d2f216700fd869ec1f65d916bb0c93d7bcca7bab6b8158dfe97
```

环境：`CUDA_VISIBLE_DEVICES=0`、`CUDA_DEVICE_MAX_CONNECTIONS=32`、数学库线程 1；
GPU1 无本实验负载。
