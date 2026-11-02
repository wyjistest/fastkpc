# Phase 10：fixed-VT 即时消费隔离原型 — STOP_NO_STAGE_GAIN

日期：2026-10-30。父源码 `1fb26aaff251a7f21d96ca62e819c6f750ee34bb`。

**结论：GPU 逐位与资源资格通过，完整 L2/L3 Stage B 没有净收益，按预注册门停止本候选。**
当前 5N 对照的各层中位数之和为 **101.642758 秒**，3N 候选为
**102.345465 秒**，观察回退 **0.702707 秒 / 0.69135%**。
三次 C 的总时长均高于三次 A。没有进入 L4–L8 晋级、生产集成或完整 canonical 实验。

生产源码、generated includes、安装 native 和默认开关均未改变。
当前接受的 canonical 隔离开发中位数仍为 **283.328 秒**，
`GO_CAPACITY / speed NOT_PROVEN`、Phase 10 **ACTIVE**、candidate **NOT_FROZEN**、
holdout **SEALED**、recommended **false** 保持不变。

## 范围、身份与协议

本轮只比较当前 fixed-VT work-pair **5N** 与按值即时消费 **3N**。
Original 7N 不作为性能分母。两档均由同一个当前 native 提供 consumer、cache、wave
及资源管理，recover tile36、64 threads、10 contexts、width64、11 GiB cache 和
8 dense eig lanes 保持相同。A/C 使用对称 interposer 和 host counters。

```text
accepted native:
b2b79fc6219841a7b9de193e312de27bbdb8bca77a9d3a9d4a626aa1e135b12b
baseline_timing.so:
30dd7230aed93b763091110af439e13f50e0053a40b4297e61a220b278559f88
candidate_timing.so:
f485b9acd3d985006b89f9789a15ca8351f71fe695e7b141af801118054c213a
```

[运行前协议](../tools/research/stage_b_immediate_20261030/README.md)与 `protocol.json`
在 GPU 工作开始前绑定当前 native、全部 **107 个生产输入**、两个 generated includes、
**10 个冻结工具**、**18 个依赖**和四份实验库。编译参数为
`-O3 -arch=sm_89 --fmad=false`，其他数值 helper 未改。

复用 `stage_b_attribution_20261028/attempt3/capture` 的原始 residuals、target 顺序与
payload hashes，保留其原 producer 身份；没有把旧 capture 重标为当前 producer 的新采集。
L2 **49,919**、L3 **54,878**，合计 **104,797** components，分别 **78/86 waves**。

使用物理 GPU1 `GPU-9c6883e3-fe8f-1a96-ebac-c8496069cba2`（RTX 4090），
CPU affinity 0–19、数学库单线程、CUDA connections32，每次 fresh process。
保持原时钟与 450 W power policy。每个进程的隔离监控均通过；200 ms 采样中 GPU0
始终为 50 MiB / 0% utilization，没有外部 compute 任务。采样记录不代表连续观测上界。

## 候选的唯一算法改动

[生成器](../tools/research/stage_b_immediate_20261030/build.py)从当前已认证的 fixed-VT
include 生成独立副本。四类 sweep 在原系数写入点调用
[按值消费 helper](../tools/research/stage_b_immediate_20261030/rotation.inc)：
identity-skip、尝试追加 log、读取旧 q 的 a/b、执行原两条 FP64 表达式。
日志满后继续更新 q。反向负号只作用于消费 payload。

lower→upper 仅删除无消费者的 work stores，保留 d/e 运算，不新增 q/log 更新。
候选删除固定配置下不可达的 NRU/NCC/dqds 分支和四段独立 `record_dlasr_` 消费循环，
保留入口拒绝及参数错误优先级。ABI 的 work 参数改名为 unused_work，删除其指针递减
及全部可执行 work 表达式；固定入口可以接收空的未使用参数。
2×2 直接 drot、sign/scal、swap 保留原路径和原 skip 行为。

没有加入 q 跨迭代寄存器缓存、restrict、算术变形、容差变化、recover 调整或额外并发。
原 Mode0 仍被对称编译，但实验 plan 拒绝该模式，所有正式提交均为 Mode1。

## GPU 逐字段资格

四份独立库为 `baseline_diag`、`candidate_diag`、`baseline_timing`、`candidate_timing`。
计时库没有检查点、canary、故障注入或设备诊断代码。

新增[检查点](../tools/research/stage_b_immediate_20261030/checkpoints.hpp)由 A 在设备上
保存、C 在相同语义位置逐字段比较，包括完整 d/e/q 的 FP64 位模式，以及活动区间、
方向、迭代、tol/threshold/shift、info、coverage、count/overflow 等字段。
键使用独立 ordinal、stage 和 sweep 序号，日志满后继续推进。
共同观察点位于 lower 完成、每个完整 sweep 的阈值处理与 q 消费之后、2×2 完成、
sign 完成、sort 完成、非收敛结束及入口返回。

| GPU 资格项 | Captured L2/L3 | 构造用例 |
| --- | ---: | ---: |
| targets / cases | 104,797 | 862 |
| 逐位比较的已存 rotations | 10,515,312,078 | 62,118,571 |
| 完整 sweep 检查点 | 68,297,253 | 404,891 |
| 全部检查点 | 71,714,360 | 413,483 |
| 直接比较的字段 | 75,283,090,946 | 380,010,599 |
| 单 target 最大检查点数 | 843 | 935 |
| mismatch | **0** | **0** |

每 target 的预声明上限为 **2,048**，未溢出、未截断。
字段比较使用整数/FP64 位比较，未用摘要 hash 替代。
另外逐位比较末态 d/e/q、完整已存 rotation 字段与顺序、root/recover 工作区、
payload、metadata 与 status。

[资格工具](../tools/research/stage_b_immediate_20261030/qualify.cu)覆盖 r=1…350、
signed zero/ties、不同量级/方向、0/1/count−1/count 日志容量、强制非收敛、
n=0/负 n/入口拒绝和参数错误优先级；10 个入口用例、10 个 helper 用例、
101 个 recover 边界用例包含尾批、跨批复用、列尾、overflow 拒绝和晚期 nonfinite 拒绝。
自然输入 coverage mask 为 **494**，构造用例补至全部 **2047**。

两项负控制分别破坏检查点 header 与数值位，比较器均按要求检出。
保留 default-stream 初始化 join、异常路径 RAII、shared/allocation guards。
缩小版 compute-sanitizer memcheck 为 **0 errors / 0 bytes leaked**。
最终 device allocations/bytes、streams、plans、cleanup errors 均为 **0**，CPU numerical
authority 为 false。诊断边界验证相同的失败状态；没有在本轮新增完整方法的混合 dense
fallback/失败零发布实验。

## 未插桩编译资源与实际访问

| 项目 | A：5N | C：3N |
| --- | ---: | ---: |
| REG | 82 | **80** |
| STACK / LOCAL | 0 / 0 | 0 / 0 |
| LDL / STL 静态位点 | 0 / 0 | 0 / 0 |
| dynamic shared，N=351 | 14,040 B | **8,424 B** |
| occupancy API blocks/SM 上界 | 6 | 10 |
| 固定 kernel 含可达 helper 的指令数 | 7,360 | 6,912 |
| 全函数 LDS / STS 静态位点 | 266 / 97 | 177 / 72 |

A 的归一化 PC/指令/操作数序列与当前已验收 native 的导出序列一致。
两份 timing cubin、完整 SASS、可达 helper 和 nvdisasm live ranges 均已导出。
没有将诊断库资源填入这张表，也没有设置候选 REG≤82 的事后门槛。

[固定地址核对工具](../tools/research/stage_b_immediate_20261030/audit_candidate.py)
绑定候选 SASS 与 cubin SHA。lower 转换只保留 d/e 访问；四个消费段均只有两次旧 q
load 和两次 q store，系数直接来自寄存器，两次 q 更新之间没有 c/s 重读。
消费与 QR 使用同一循环回边，原 work 中转与独立遍历消失。

| 消费段 | 候选 PC | 该段 live GPR 范围 |
| --- | --- | ---: |
| shifted backward | 0xa870–0xaa70 | 11–23 |
| shifted forward | 0xcfd0–0xd190 | 12–24 |
| zero backward | 0xf410–0xf630 | 13–22 |
| zero forward | 0x11940–0x11b40 | 14–24 |

这些是特定二进制的静态访问与 liveness 记录，不等于动态访存量、实测 occupancy 或
可节省秒数。shared 减少 **5,616 B/block**，仍不能与 101,088 B 的原 recover block
在 102,400 B/SM 内共驻。八次回放的 GPU1 采样显存峰值均为 **15,349 MiB**；
本轮未证明全局显存池容量收益。

## 完整 Stage B 交错对照

先执行 unchanged-native 与 A 认证回放，分别为 **101.617161 / 101.530314 秒**，
输出、Mode1、10 次 plan 创建/销毁、1,638 次提交、104,797 个 targets 与资源门均通过。
这两轮不进入正式中位数。

正式顺序严格保持 A1→C1→C2→A2→A3→C3：

| 运行 | L2 秒 | L3 秒 | 同轮合计秒 |
| --- | ---: | ---: | ---: |
| A1 | 48.445903 | 53.181339 | 101.627243 |
| C1 | 48.810955 | 53.523155 | 102.334110 |
| C2 | 48.764201 | 53.571572 | 102.335773 |
| A2 | 48.468803 | 53.196543 | 101.665346 |
| A3 | 48.446215 | 53.255407 | 101.701622 |
| C3 | 48.785832 | 53.559632 | 102.345465 |

| 统计口径 | A | C |
| --- | ---: | ---: |
| L2 中位数 | 48.446215 | 48.785832 |
| L3 中位数 | 53.196543 | 53.559632 |
| **各层中位数之和：正式判定口径** | **101.642758** | **102.345465** |
| 每轮总和的中位数 | 101.665346 | 102.335773 |

正式比值 **M_C/M_A=1.006913495424**。两个层的中位数都没有下降，未达到 ≤0.95；
最慢 C **102.345465** 也不小于最快 A **101.627243**。
实际上最快 C **102.334110** 仍高于最慢 A **101.701622**。
因此按原门判定 **STOP_NO_STAGE_GAIN**，没有降格修改成容量 GO 或 CONDITIONAL。

完整边界包括 roots、recover、status、wave joins 和 cache stores。
payload D2H/hash 在计时边界之外。两次认证加六次正式回放，合计 **838,376 次 component
payload hash 核对**全部匹配原 capture，dense fallback 均为 0，原 target/store 顺序与资源
闭环通过。这些 hash 核对与前述 GPU 逐字段比较分别报告。

## 证据闭环与复现

本地证据：`fastkpc/artifacts/full_cuda_ci/stage_b_immediate_20261030/attempt3/`。
`build.json`、`protocol.json`、`qualify.json`、`binary-resources.json`、`resources.json`、
`candidate-access-audit.json`、`timing.json`、`campaign.json`、`closure.json` 及原始输出均保留。
[只读核对工具](../tools/research/stage_b_immediate_20261030/verify.py)通过 **186 个 closure
文件**、107 个生产输入、10 个冻结工具和18个依赖的 SHA 核对，并重算正式判定。

`audit_candidate.py` 与 `verify.py` 是注册后新增的只读分析工具，其自身 SHA 分别记录；
它们没有修改冻结的构建工具、二进制、协议或计时程序，也不冒充运行前已有工具。

```text
build.json    ef2e7dc69fcd022aeb5f41cb8e59ac3bb7868647e1b13c713c0a8759932d6152
protocol.json e32a2218fd5cdfe0c8134358ea04aa3d9f91da6857c4cb57056600e260cc3676
qualify.json  34616cdf1fcb9c58cac96425c9873fad3e08427928ae920e734f916e7a88a551
timing.json   508306420e07e5644ddbc5355b697d749c57e9056d6bd440f5760a86135317a3
campaign.json 223fcfbb12c203284821d65fa84d157f9e3973ff6dd762b91054eb70f8e3ccb0
closure.json  f2829f073dc4b072a31803a003762cd374c7c114f30d0fdd94b604f4911f23fa
```

attempt1 因 no-work 词法检查误匹配继承注释而停；attempt2 因导出器假定 shared library
只有一个 cubin 而停。分别修正为检查可执行文本、按 fixed kernel symbol 选择 cubin，
完整保留失败记录。两次均未运行 GPU kernel，不涉及候选数值门放宽。

从仓库根目录、具备上述已认证本地输入时，可在一个不存在的新目录重建并运行：

```bash
python3 fastkpc/tools/research/stage_b_immediate_20261030/build.py /tmp/fastkpc-immediate-new
python3 fastkpc/tools/research/stage_b_immediate_20261030/campaign.py /tmp/fastkpc-immediate-new
python3 fastkpc/tools/research/stage_b_immediate_20261030/audit_candidate.py /tmp/fastkpc-immediate-new
python3 fastkpc/tools/research/stage_b_immediate_20261030/verify.py /tmp/fastkpc-immediate-new
```

原始 native、captures 和大型二进制证据仅保留本地；fresh clone 缺少这些输入会失败，
不会以 SKIP 作为 PASS。新构建的原始 cubin/库 SHA 可能因 NVCC 路径元数据变化而不同，
所有新输出重新绑定新 manifest；地址核对工具要求同一归一化指令序列。

这次完成了“源级可移动→机器码有目标→GPU 数值成立→完整阶段不获益”的隔离判定。
停止的是本次按值即时消费候选，不据此否定所有 Stage B 优化。没有新内部周期或硬件
stall 归因，不把观察到的微幅回退唯一归因于某种调度机制。
q55 继续后排，recover tile16/8 与 L8 optimizer 保持 STOP；180/120 秒门与完整方法
CI/pMax、图/RNG、fallback、容量及 clean-checkout 晋级要求均未改变。
