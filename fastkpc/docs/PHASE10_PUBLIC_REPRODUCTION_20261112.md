# 公开输入复现入口（2026-11-12）

状态：`PASS_LOCAL_PUBLIC_REPRODUCTION_AND_RELOCATION`。本轮交付独立构建、公开输入、
原版 R 参考、GPU A/C 正确性与重复计时、原始 receipt 重审及可迁移证据包。
44 对象独立全量构建、新原版 R 参考、调度重放、4 次 GPU 正确性运行、
12 次交错计时及 16 份原始 receipt 独立审计均通过。两套输入满足预声明
不回退门，速度均为 `NOT_PROVEN`。这是新公开输入复跑，不替代历史七输入成绩。

使用入口：[tools/reproduce_public/README.md](../tools/reproduce_public/README.md)。
机器可读身份见 [证据索引](PHASE10_PUBLIC_REPRODUCTION_ATTESTATION_20261112.json)。
该目录独立于历史七输入工具；历史报告、原始记录和 manifest 均保留。
生产数值源码和原版 `kpcalg/R` 未修改，主工作区 native 未替换。

## 本轮范围与身份

- 冻结生产提交：`2be6a816d3039a5f9340ba7018166d737ab11d9a`。
- 工具来自本轮工作区快照，`protocol.json` 分别记录原 Git HEAD、工具哈希
  和生产提交；不会把未提交工具描述成生产提交内已有的内容。
- 独立验证目录：`/data/wenyujianData/fastkpc public repro 20261112`。
  路径包含空格；独立 clone 不含本地主工作区未跟踪的真实数据目录。
- GPU1 RTX 4090，Intel Core i9-10900X（10 核／20 线程），CPU affinity
  0–19，数学库单线程，CUDA connections=32。
- 本轮完整复跑选择 `public500`（500×50，默认 rank 50）及 `blocks256`
  （256×12，默认 rank 25）。两套均使用完整矩阵和默认 Inf 自然停止。
- `dense500`、`dense1025` 是入口支持的可选公开输入；本轮只验证其生成
  身份，不将旧深层／dense GPU 资格改标成本轮完整复跑。

原版 R 参考 SHA-256：

```text
d0620dd9b06f006170b5e142b892c202619296797b2bf9ca842a7c2b5b3b715f
```

主工作区 native 保留：

```text
2727ef51899688b54da697d0008f99a4908922dbcacec1c30ed58cef09198301
```

新构建通过 44 次编译／44 个对象检查，未复用父对象，耗时 2,787.133 秒。
新 native SHA-256：

```text
63dde57dd8607a4846ee75efb5bf2a737cba674654fbdf6821a66c7ab472d6b2
```

`protocol.json` 绑定 217 份生产源码以及共 1,478 项源码／工具／环境文件。
最终 manifest 有 1,643 项证据文件，SHA-256：

```text
3abe2ef088140a4c2423d0b1d18b49aa30636db015fdc76aff47a1eb9cbe1ad6
```

`summary.json` SHA 为 `ded828f224a2fa013752ea78cafe41c84cc7c94a92c2834417de730db2f3d670`，
`receipt_audit.json` SHA 为 `f517eac60df706a5d53c702da3572f7802729213f20c98eaaba783830d45a5d2`。

## 入口保证与限制

从独立 detached clone 全量编译 44 个对象；源码直接对 Git blob 核对，
不复制旧对象。构建临时使用不含空格的目录别名，保留原构建脚本及其
逐条编译命令。新 native 使用自己的 SHA，不要求与历史 native 字节相同。

输入生成器只生成所选公开输入。环境检查记录 R/依赖、编译器、CUDA、
GPU UUID、CPU affinity；缺少依赖即失败，没有静默 SKIP。完整冻结构建
仍针对 sm_89，此入口当前只接受 compute capability 8.9；环境记录不是
容器或自动依赖安装器。

续跑要求源码、工具、输入、构建产物、依赖环境和已完成记录身份一致。
并发运行同一目录会被文件锁拒绝；不完整或失败任务保留证据，不自动覆盖。
改配置或失败后重新实验需要新目录。

逐输入执行原版 R reference、native host-order replay、OFF/ON 正确性及
`A1 C1 C2 A2 A3 C3` 六次 fresh-process 完整 WAN-PDAG 计时。CPU 参考
含取证开销，不能用作正式 CPU 性能 baseline。A/C 均为 GPU 配置，C
请求既有优化组合，实际启用仍由通用形状与容量条件决定。

保留 CI/pMax 绝对误差 `1e-10`、图／删边／sepset／计数／定向／PDAG／RNG、
GPU skeleton authority 和资源门。独立审计重读每套输入的八份原始 GPU
receipt，不加载 native。数值通过与速度判定分开：不回退门为中位数
`C/A <= 1.05`；速度门同时要求 `C/A <= 0.95` 及最慢 C 快于最快 A。
数值 PASS 或程序退出码零均不单独代表性能 PASS。

正确性要求所选 GPU 独占，计时要求两卡无外部计算；每约三秒记录计算
进程。此监测是采样证据，不能排除采样间隙的短进程，也不是全机 CPU
活动审计。发生检测到的干扰只终止本轮拥有的进程组并保留失败记录。

## 已完成的入口检查

七项 host-only 测试通过，覆盖证据搬迁、内容篡改／缺失、manifest 锚点、
绝对路径／父路径／symlink 越界、空 manifest、CPU affinity、缺失或非有限
计时、A/C 标识，以及速度／不回退门边界。Python 编译与三份 R 脚本解析
通过；实际同目录并发 resume 被锁拒绝。

四套公开输入重新生成后，RDS 文件及序列化矩阵哈希均与原七输入封存一致：

| 输入 | RDS SHA-256 |
|---|---|
| public500 | `efb81666e2cac91d14c5d0f951690cedf23b1626f4a06c281045e51ca5a38192` |
| blocks256 | `c04cf4b9c1f13be1d09bc8243fb23de5d5c9af994da59e1a64391cde831b7ed5` |
| dense500 | `d84df9b35c7e02a399964780cf21983d82a4f493f0fbc7ea0d0b25dc5151d6d2` |
| dense1025 | `49975f38a20d069d94469a6f7f99970cdb9174a558e17c521a568dc4dc3fcdb4` |

新 verifier 按历史 manifest 的 `evidence` 映射核对 505 项历史文件通过，
使用原锚点 `89306730984f8be645e14a2bbbbdca16598c31a530c956ff30411444763245a0`。
此项只检查身份，没有重新执行历史七套输入。

## 新运行与独立审计结果

两套新原版 R 参考均来自完整输入。public500 的原始 R 搜索包含 2,789 次
骨架 CI 和 89 次定向 CI；native host-order replay 另补算 2,792 个 R 任务，
图、pMax、sepset 与测试计数一致。blocks256 每次包含 98 次骨架 CI 和
32 次定向 CI。最终独立审计重新读取 4 份 correctness 和 12 份 timing
receipts，不加载 native，所有数值／行为／authority／资源门通过。

累计比较为 **23,096 次 skeleton CI、968 次定向 CI**，即两套输入的每次
CI 数各重复八次；不是 23,096 个不同测试案例。最大 CI 绝对误差为
`1.83623949379097e-13`，最大 pMax 误差为 `1.39110944985532e-13`。
不将数值容差内一致表述成所有位模式相同。

全部 12 份计时进程记录通过预声明的两卡采样隔离与运行顺序检查。
每次均为完整公共 WAN-PDAG 调用，两档各三次；以下 A/C 均为 GPU 配置：

| 输入 | A1 / A2 / A3（秒） | C1 / C2 / C3（秒） | A / C 中位数（秒） | C/A | 结论 |
|---|---|---|---|---:|---|
| public500 | 25.053 / 24.993 / 25.059 | 25.734 / 25.746 / 25.791 | 25.053 / 25.746 | 1.027661 | 不回退 PASS；速度 NOT_PROVEN |
| blocks256 | 1.515 / 1.518 / 1.521 | 1.511 / 1.521 / 1.514 | 1.518 / 1.514 | 0.997365 | 不回退 PASS；速度 NOT_PROVEN |

public500 ON 的中位数增加 **2.77%**，三次 ON 都慢于三次 OFF；其含义是
没有超过预声明 5% 回退容限，不能表述为加速。blocks256 的约 0.26% 下降
也没有达到速度门。本轮不据此默认开启优化组合。

阶段中位数单独保留，不作为可直接相加的某次完整运行账本：

| 输入 | A / C skeleton（秒） | A / C CPU orientation（秒） |
|---|---|---|
| public500 | 10.653 / 11.075 | 14.365 / 14.649 |
| blocks256 | 0.265 / 0.262 | 1.226 / 1.222 |

这些时间绑定新 native `63dde57d…472d6b2`，不与历史 `d8793173…10d7350`
的时间混算，也不提供 CPU 对 GPU 的速度判定。

## 交付边界

已生成并实际搬迁验证的本地公开证据包，包含源码／工具／环境、输入、native／对象、
原始参考和 GPU receipts、计时记录、重审输出及相对路径 manifest。
另行提供 manifest SHA 作为外部锚点；Python 文件校验不依赖 R/CUDA。
具备 R 依赖的接收者可在搬迁后的 source 目录重审 receipts，无需运行 GPU。
工具不会上传或发送文件。

本地包：`/data/wenyujianData/fastkpc-public-reproduction-20261112.tar.gz`，
233,512,223 bytes（约 222.7 MiB），SHA-256：

```text
66d49ffab55baa44b34e09988d34ef2891975ca715ef99caa3d5993ab2083fc7
```

实际解压目录为 `/data/wenyujianData/fastkpc public relocated 20261112/public-reproduction`。
从该目录内的封存 driver 执行校验，使用 `/usr/bin/python3` 并将 PATH
设为不存在的目录，仍通过全部 1,643 项哈希及外部 manifest 锚点校验，
没有调用 R/CUDA 工具。随后在该副本的 source 下、GPU 不可见的环境中
运行 raw-receipt audit，16 份 receipts 的全部原门通过，`native_loaded=false`。
输出 `/data/wenyujianData/fastkpc-public-reaudit-20261112.json` 与原审计输出
SHA 均为 `f517eac6…d45a5d2`，逐字节一致。

没有替换安装或修改原版 R；最终核对两者仍为上文 SHA。全部本轮构建、
参考、GPU、打包和审计进程已结束。本轮工具、报告及公开证据索引作为独立
Git 交接提交；运行时使用的未提交工具快照身份保持原记录，不追溯改写。

新公开包不含历史三套真实输入及其本地原始记录，不取代原七输入的独立
核验。一次本机干净复跑、搬迁校验和 raw-receipt 重审，也不等于第三方
在另一台机器完成复跑。Phase 10 ACTIVE、candidate NOT_FROZEN、holdout
SEALED、recommended=false 保持；本轮不启动 CPU 定向优化或 CPU/GPU
性能对照。
