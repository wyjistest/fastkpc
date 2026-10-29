# q37/q46 vector postback 默认关闭 runtime 集成：复核

日期：2026-10-29。复核 [默认关闭的 runtime 集成资格](PHASE10_OPTIMIZER_POSTBACK_RUNTIME_20261029.md)（证据
`fastkpc/artifacts/full_cuda_ci/optimizer_postback_runtime_20261029/`）。只读复核：没有 GPU 运行；读了生产源码 diff，从新旧 native
中提取 cubin 比较，并用 `cuobjdump` 读资源。`GO_DEFAULT_OFF_RUNTIME_DEVELOPMENT` 成立；accepted opt-in development median 改为
290.491 s 符合以往做法（同一 native OFF/ON campaign 的 ON 中位数）。这是 incremental source-closure 开发证据，不是 clean producer 或
promotion；默认关闭、L8 optimizer STOP、Phase 10 ACTIVE、candidate NOT_FROZEN、holdout SEALED、recommended=false 均不变。
同日按 Codex 核对意见修订，见“修订记录”。

## 修订记录（Codex 核对后）

Codex 重新核对了 638 项文件哈希、已安装 native 与证据副本、17 个原 cubin 逐字节保留与 1 个新增 cubin、三组配对与中位数、240,498 个
CI/pMax 逐位一致和容量门，均成立；同意 runtime GO 与 290.491 s 开发基线。两处我复核后接受：

1. **“复用的 41 个对象若按当前源码重编，预处理后的代码不变”过强，已撤回。** `r_api_cuda.cpp` 直接包含
   `mgcv_multi_penalty_gcv_extended.hpp`（第 27 行），已链接的 `r_api_cuda.o` 是 baseline 的旧对象；新增的
   `FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND(postback64)` 会进入它的预处理结果（Codex 用相同参数、去除行号标记比较：旧版没有
   `postback64`，新版 32 处）。“只增加声明”不等于预处理结果不变；声明是否改变目标代码，没有重编就不知道。正确的说法是：当前安装版本
   复用了已认证的 41 个旧对象，原 17 个 cubin 逐字节不变；完整源码重编后的产物是否一致，仍需单独验证。第 1 节已改。
2. **DLARFT 的 57% 不是整体收益空间。** 它是隔离原型 profile 中 L5 回乘周期的占比，周期跨 target 重叠，不能换算成 outer 秒数；
   不建议据此立即开 DLARFT 优化。第 3 节已改。

另按 Codex 的建议，把 cubin 比较做成可复现、失败即报错的工具，并保存结果（见“证据”），以便封存。下一步（Codex 的建议，我同意）：
修正上述表述，封存 cubin 比较证据与提交后的 source-closure attestation，再按 runtime/工具与路线文档分开提交；无需重跑性能。

## 结论

```text
完整性    closure.json（835244b1…）、manifest.json（1c77c87d…）的 SHA-256 吻合；manifest 638 项（107 个生产输入、10 个工具、
          521 个文件）按当前文件重算一致；已安装的 native a1e16a9c… 与证据副本相同；baseline 保存原 native 12acc3cd… 与
          43 个对象，原 small64 对象（98bad23a…）在 build 目录中未变
代码      未发现缺陷。新 CUDA 单元定义 FASTKPC_LAPACK_VECTOR_POSTBACK、重命名全部 small64 符号后重新包含 small64 源码，
          得到独立的 postback64 后端；共享 LAPACK 头文件的改动都在该宏的条件编译内。工厂只在显式 ON、Small64 且
          q = 37/46 时选它，后端身份在 prepared 创建时固定，L4/L5 的 rebind/drain、fixed、finalize、单 batch 与多 setup
          路径都按身份分派，混合请求在提交前拒绝，L3/L6 槽拒绝 postback64；ON 只允许 dcc.gamma
二进制    原 native 的 17 个 cubin 在新 native 中逐字节相同，新 native 只多 1 个（postback 单元，14 个函数，optimizer
          REG 255 / STACK 7,232，evaluator 320）。已安装版本中 OFF 的 device 代码与原实现逐字节相同，比资源记录一致更强；
          这不证明完整源码重编会得到相同产物
资格      20,628 个单元用例零差异；1,719 例 memcheck/synccheck 零错误、零泄漏；L4/L5 runtime 合同测试及其 sanitizer 通过；
          完整 L4/L5 的 OFF 与 ON shadow 逐位通过；静态门 OFF [7120, 288]、ON [7232, 320]
计时      协议 15:08:58 写入（门：ON/OFF 中位比 ≤ 0.98、3/3），六次按 OFF ON ON OFF OFF ON 运行。重算：OFF
          312.763/312.515/313.265 s，ON 285.085/293.807/290.491 s，中位 312.763 → 290.491 s（−22.272 s，−7.1210%），
          每次 ON 都快于每次 OFF；同次 L4+L5 optimizer 之和中位 59.058 → 31.129 s，三对节省 27.869/27.973/27.977 s；
          扣除 L4/L5、L7、L8 后其余为 −0.182/−0.073/+0.153 s；Stage B 与 native setup 不变
容量      OFF 的全部原始检查点与原实现相同。ON 的 L3 检查点不变；从 L4 起空闲少 27,262,976 B（26 MiB），L4 池增量相应
          多 26 MiB；最紧的 L6 为 2.077271 GiB，比 2 GiB 门高 79.125 MiB。来源未独立区分，报告已写明
计数      OFF 的 vector 计数全为零；ON 的 vector 后端创建/销毁 64 个、3,109 个 batch、24,583 个 target（q37 17,767、
          q46 6,816、其他 0），finalization 24,583，fixed 0；原后端承担其余 56,168 个 target
```

## 1. 代码复核

- **开关**（`mgcv_multi_penalty_gcv_capacity.cpp`）：`FASTKPC_DCC_GAMMA_VECTOR_POSTBACK` 未设置、为空或为 `0` 时 OFF，`1` 时 ON，
  其他值抛出。工厂在每次创建 prepared 时读取；ON 且为 Small64、q = 37/46 时创建 postback64，其余照旧。`MultiPenaltyGcvCapacityPrepared`
  的 bucket、handle、q 与后端标志都是 `const`，构造时再次检查形状。
- **分派**：`prepared_info`、L4/L5 的 rebind/drain、`optimize_batch_with_residual`、`evaluate_fixed_batch`、
  `finalize_selected_coefficients` 都先看后端标志；多 setup 路径按首个请求选后端，`run_multi_backend` 要求每个请求持有同一类 handle，
  混合时抛出；L3/L6 的 rebind/drain 遇到 postback64 抛出。one-call 在 ON 时要求 `ci_method == "dcc.gamma"`，并输出
  `vector_postback_requested` 与按后端的执行计数（原子计数，前后差分）。
- **独立后端**（`mgcv_multi_penalty_gcv_postback.cu`）：定义宏并把 small64 的命名空间（经 `lapack312_multi_small64` 宏链）、类与
  全部入口改名，再包含 `mgcv_multi_penalty_gcv_small64.cu`。原 small64 对象不重编，调用图不含新路径。
- **回乘实现**（`lapack_312_vector_postback.cuh`）：与隔离原型的 `postback_apply` 逻辑相同（去掉了 profile），LWORK 由
  `dgesdd_bdspac(q)` 给出；`static_assert` 把本地的 32/65/4160 钉住。它不读取 vendored DORMQR 与 `ilaenv` 的实际值，两者是否一致仍
  由完整 DORMBR/workspace 逐位用例保证，vendored 代码若改动须重跑。
- **增量 closure**：当前安装版本复用了已认证的 41 个旧对象，原 17 个 cubin 逐字节不变；完整源码重编后的产物是否一致，仍需单独验证。
  共享头文件 `lapack_312_small_dgesdd.cuh` 的改动都在 `#if defined(FASTKPC_LAPACK_VECTOR_POSTBACK)` 内，对不定义该宏的单元只改变
  行号；但 `mgcv_multi_penalty_gcv_extended.hpp` 新增的 postback64 声明会进入包含它的单元的预处理结果，例如复用的 `r_api_cuda.o`。

## 2. 复现与补充

- 六个 `report.json`：outer、各层 optimizer、Stage B、native setup、门数（76/74、79/77、79/77）、配对与扣除后的余数，全部与报告一致。
- 容量：与原实现的参考值逐项比较，OFF 三次 9 个检查点全部相同；ON 的 L3 建池前后、optimizer 后、增长与总增量都相同，L4 最低空闲、
  L5 创建后、L6 最低空闲各少 26 MiB，L4 池增量多 26 MiB（809,500,672 → 836,763,648 B，仍低于 1 GiB 门）。增量落在 L4 窗口内，
  与新后端在 L4 首次 launch 时 stack 预留增长（约 21 MiB 的估计）并加载独立模块相符；未独立区分，同报告。
- ON 的 L4 optimizer 比隔离原型的候选慢一点：三次 15.34–15.40 s，原型 15.20–15.22 s（中位差 0.151 s）；L5 中位差 0.041 s。
  分属两次 campaign，原因未查；独立模块在首次使用时加载是可能之一。对 22.272 s 的收益影响很小。

## 3. 意见

1. GO 与基线更新成立。290.491 s 只代表当前 canonical 配置下的 ON；L7/L8 的窗口波动（本轮 ON2 的 L7 多 6.17 s）仍在，后续小改动
   的门要考虑。
2. 容量余量收窄：ON 时 L6 距 2 GiB 门 79.125 MiB（OFF 105.125 MiB），L4 池增量 798 MiB/1,024 MiB。以后增加显存的改动须以 ON
   为准重新测量。
3. 状态：生产源码改动尚未 commit，新 native 已安装在 `fastkpc/build/`；路线图的新条目与证据一致。是否像 10-28 那样提交并做 Git
   attestation，由用户决定。
4. 隔离原型的 profile 中，串行 DLARFT 占 L5 回乘周期约 57%。这只是该阶段内、跨 target 重叠的周期份额，不能换算成 outer 秒数，
   也不是整体收益空间；不建议据此立即开 DLARFT 优化。

## 证据

`attempt1/{closure,manifest,build,device-build,installation,analysis}.json`、`attempt1/canonical/{protocol.json,*/report.json}`、
`attempt1/qualification2/{unit-*,on-*,off-*,contracts*,static-gates}.json` 与日志、`baseline/{fastkpc_cuda.so,objects/}`、`fastkpc/build/fastkpc_cuda.so`；
生产源码 diff（`git diff` 五个文件与三个新文件）。cubin 比较已做成工具
[`cubin_compare.py`](../tools/research/optimizer_postback_runtime_review_20261029/README.md)（SHA `dbc945d5…`）：核对两个 native 的身份，
提取全部 cubin 按 SHA-256 比较，原 cubin 缺失、提取为空或新增数不符时失败，不覆盖已有输出。结果
`fastkpc/artifacts/full_cuda_ci/optimizer_postback_runtime_review_20261029/cubin-compare.json`（SHA `a054d4d2…`）：PASS，共同 17、
旧版独有 0、新版独有 1；错误身份与错误预期两种负例都以退出码 1 失败。没有修改生产源码、native、默认配置或封存输入，没有 commit 或 push。
