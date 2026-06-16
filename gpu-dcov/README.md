# gpu-dcov — GPU 版 dcov.gamma(kpcalg 距离协方差独立性检验)

精确(满秩)统计量的 CUDA 实现,替代 kpcalg::dcov.gamma 的 RSpectra 截断近似。
零 n×n 存储:距离动态重算、双中心化折进归约、全程 FP64 累加,显存 O(n)。

## 文件

- `dcov_gpu.cu` — CUDA kernels + R .Call 接口(返回 5 个标量,p 值在 R 侧算)
- `dcov_gamma_gpu.R` — R 包装,`dcov.gamma.gpu(x, y, index=1)` 返回与原版相同结构的 htest
- `build.sh` — nvcc 编译(sm_89,RTX 4090)
- `validate.R` — 数值验证(GPU vs 精确 CPU 参考 vs 原实现)
- `bench.R` — 基准测试

## 用法

```sh
sh build.sh
```

```r
source("dcov_gamma_gpu.R")
dcov.gpu.warmup()          # 可选,首次调用约 0.26s 的 CUDA 上下文初始化
dcov.gamma.gpu(x, y)       # x, y 向量或矩阵(支持多元,原版只支持向量)
```

## 实测(2026-06-13,RTX 4090,CUDA 12.5,R 4.4.1)

| n | 原版 (numCol=100) | 精确 CPU 直接算法 | GPU | GPU vs 原版 | GPU vs CPU |
|---|---|---|---|---|---|
| 1000 | 1.20 s | 0.066 s | <0.1 ms | >10000× | >600× |
| 3000 | 11.2 s | 0.54 s | 1 ms | ~11000× | 535× |
| 10000 | 117 s | 6.2 s | 4 ms | ~29000× | 1557× |
| 30000 | — | —(内存不可行) | 35 ms | — | — |
| 100000 | — | — | 352 ms | — | — |

数值验证:GPU 与精确 CPU 参考一致到 ~1e-15;多元输入与 index≠1 均验证通过。

## 与原版的有意差异

1. 无 `numCol` 参数——特征分解截断在数学上多余(nV² = sum(A∘B)/n 恒等),
   原版默认 numCol=n/10 在 H0 下有 ~4% 统计量误差(实测 p 值偏差 2–20%)。
2. `index` 按文档生效(原版静默忽略)。
3. p 值用 `pgamma(lower.tail=FALSE)`,不会像原版 `1-pgamma()` 在 p<1e-16 时下溢为 0。

即:GPU 版是"修对了的"精确版本,p 值与原版有截断近似量级的差异,临界边(p≈α)
的判定可能与原版翻转——这是行为修正,不是误差。
