# Phase 10 开发交接：q37/q46 exact vector postback

日期：2026-10-29。本次将已验收的默认关闭 runtime、研究工具及路线判断收口到 Git。
当前 accepted opt-in development median 为 **290.491 秒**，对应
`GO_DEFAULT_OFF_RUNTIME_DEVELOPMENT`。这是当前 canonical 配置的增量 source-closure
开发证据；Phase 10 **ACTIVE**、candidate **NOT_FROZEN**、holdout **SEALED**、
recommended **false**、L8 optimizer **STOP** 均不变。

## 提交身份与证据边界

Runtime 与工具提交为 `c469ac563dcaf7b3e7c704b302cfbdd8a6282e10`，包含八个生产/构建文件
和 54 个研究文件。后者覆盖 work-pair activity、recover-tile STOP、L4/L5 归因、
symbolic ownership audit、GPU 隔离原型、runtime 资格和 cubin 比较。

[Git source attestation](PHASE10_SOURCE_ATTESTATION_20261029.json) 将全部 **107 个生产输入**
及 **54 个新归档研究文件**与该提交的 Git blob 逐项核对；其中最新 runtime manifest
绑定的十个工具全部保持已测字节。提交前重算该 manifest 的 **638 项**
（107 个生产输入、十个工具、521 个产物），全部一致。另核对 baseline 的 104 个源码快照、
实际链接对象及安装 native。29 项关键证据索引包含历史 closure/manifest；历史 manifest
在本次只核对索引文件 SHA，不宣称重新扫描了它们的全部大型 payload。

已安装 native SHA-256：

```text
a1e16a9c0c8089ea90491426c56475394449efb03317be4a87f1925c7b73bd34
```

原运行 HEAD 仍为 `c38b04bad9874bc1eb17616f93124cd5ae601682`。本次没有 CUDA 重编、
GPU campaign 或默认配置修改。Git attestation 证明提交源码与已测源码闭包相同，
不把提交前 receipts 改称从该提交的 clean checkout 运行，也不升级为 promotion evidence。

各历史报告中的“未提交”“未 push”和“下一步”描述其执行当时状态；当前仓库身份及路线
以本交接为准。原始失败实验、已封存工具版本、receipts 和
[前次 attestation](PHASE10_SOURCE_ATTESTATION_20261028.json) 均保留。

## 实现与二进制身份

`FASTKPC_DCC_GAMMA_VECTOR_POSTBACK` 默认 OFF；显式为 `1` 时，capacity factory 仅对
Small64 的 q37/q46 选择独立 `postback64` 后端。one-call 的 ON 入口限定 `dcc.gamma`。
prepared 的后端身份在创建时固定，mixed-backend 请求在提交前拒绝；原 cohort、槽数、
请求顺序、数学容差、秩判断和 GPU 数值 authority 不变。

实际链接 **44 个对象**：重编两个 host 对象和一个独立 ON CUDA 对象，复用 **41 个**
认证父对象。安装目录的记录共有 45 个 `.o`，其中 `svd_transcript_bridge.o` 未参与本次
native 链接；这个目录清单与链接输入清单分别记录，不混用。

[入库的 cubin 比较 receipt](PHASE10_POSTBACK_CUBIN_COMPARISON_20261029.json) 与本地
封存结果逐字节相同，SHA-256 为
`a054d4d2564b5d6687643a36465cc98aafa78ecf4aabce26a4311c41ed3a6bb2`。
原 **17 个 cubin** 全部逐字节保留，只新增一个 postback cubin。比较工具的正确输入
复跑一致，错误 native SHA 与错误新增数量均以退出码 1 拒绝。

这只证明当前已安装二进制的身份。扩展头新增声明会改变 `r_api_cuda.cpp` 等复用 host
单元的预处理结果，不能据此证明未来全量重编的产物等价。完整重编资格仍待单独验证。
回乘 helper 的 static assertion 只固定本地 32/65/4160，vendored DORMQR/ilaenv
若修改，必须重新做完整 DORMBR/WORK 资格。

## 已接受性能和资格

以下数字来自同一 native 的预声明 OFF/ON campaign；历史 311.440 秒和隔离原型
286.882 秒均不替代本 campaign 的基线与结果。

| 配对 | OFF 秒 | ON 秒 |
| --- | ---: | ---: |
| 1 | 312.763 | 285.085 |
| 2 | 312.515 | 293.807 |
| 3 | 313.265 | 290.491 |
| 中位数 | **312.763** | **290.491** |

完整方法中位节省 **22.272 秒（7.1210%）**，三组均获益。L4/L5 同次 optimizer
之和中位数为 **59.057851 → 31.129370 秒**；L7/L8 波动全部保留，不从 outer
扣除或推断本次未采集的因果机制。Stage B 与 native setup 基本持平。

六次 canonical 各覆盖 **240,498 CI**，CI/pMax 位模式及 task/deletion/skeleton/
sepsets/n.edgetests/RNG、物理 cohort/Gram 顺序、authority、资源和容量门全部通过。
最大 authority 误差为 **3.05425129631942e-12**；[0.05,0.15] 内 1,337 个 authority
p 值无新位差异。没有新增真实数据或 final-PDAG campaign。

GPU 单元 20,628 例、sanitizer、runtime contracts 和完整 3,109 setups/24,583 targets
资格通过。ON canonical 的直接 fixed-evaluator targets 为零；该入口的覆盖来自独立
runtime contract 测试，不声称 canonical 验证了该入口。

OFF 原始容量检查点不变。ON 的 L3 与 L4 建池前后空闲仍与 OFF 相同，26 MiB 增量
出现在 L4 执行窗口及后续检查点。最紧 L6 检查点空闲 **2,230,452,224 B**，
比 2 GiB 门高 **79.125 MiB**。未独立分离模块加载与 stack 预留成本；离散检查点
不是连续显存上界或更大输入资格。

## 后续开发入口

| 主题 | 结论与入口 |
| --- | --- |
| Stage B work-pair activity | [诊断](PHASE10_STAGE_B_WORK_PAIR_TRACE_20261028.md)、[复核](PHASE10_STAGE_B_WORK_PAIR_TRACE_REVIEW_20261028.md)：重叠 activity 不能直接换成可节省秒数 |
| recover tile | [实验](PHASE10_STAGE_B_RECOVER_TILE_20261029.md)、[复核](PHASE10_STAGE_B_RECOVER_TILE_REVIEW_20261029.md)：tile16/8 均为 `STOP_NO_STAGE_GAIN`，未集成，不自动扩展 tile sweep |
| L4/L5 exact-primary | [归因](PHASE10_OPTIMIZER_L45_ATTRIBUTION_20261029.md)、[复核](PHASE10_OPTIMIZER_L45_ATTRIBUTION_REVIEW_20261029.md)：归因已导向 postback，跨 target 周期不能直接预测 outer 收益 |
| vector-postback ownership | [审计](PHASE10_OPTIMIZER_POSTBACK_AUDIT_20261029.md)、[复核](PHASE10_OPTIMIZER_POSTBACK_AUDIT_REVIEW_20261029.md)：保持 serial DLARFT、原 NB/LWORK 与 reference DLARFB 运算 |
| GPU 隔离原型 | [资格](PHASE10_OPTIMIZER_POSTBACK_GPU_20261029.md)、[复核](PHASE10_OPTIMIZER_POSTBACK_GPU_REVIEW_20261029.md)：隔离开发 GO，286.882 秒属于该独立 campaign |
| 默认关闭 runtime | [资格](PHASE10_OPTIMIZER_POSTBACK_RUNTIME_20261029.md)、[复核](PHASE10_OPTIMIZER_POSTBACK_RUNTIME_REVIEW_20261029.md)：当前 accepted opt-in median 290.491 秒 |

本次 runtime 与证据归档子任务已完成；整个 Phase 10 尚未完成，180/120 秒正式门仍未通过。
DLARFT 的约 57% 只属于隔离原型 L5 回乘周期份额，不是整体收益预算；没有据此开启新实验。
后续涉及额外显存的改动须以 ON 的容量状态为基线；冻结候选前仍需冻结 producer 上的
clean-checkout 重复性能与完整语义资格。

复现遵循[当前 runtime 工具说明](../tools/research/optimizer_postback_runtime_20261029/README.md)
及[二进制比较说明](../tools/research/optimizer_postback_runtime_review_20261029/README.md)。
本次入库检查通过 29 个 Python AST、一个 R parse、一个 shell syntax 及源码/工具
whitespace 检查；23 份 Markdown 中的 112 个本地链接均可解析。
数据集、native、captures、profiler traces 和大型 receipts 保留在本地；
源码归档不等于外部 fixtures 已发布。
