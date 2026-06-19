source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP mgcvExtractGPU single-penalty GCV: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcvExtractGPU single-penalty GCV: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
if (!fastkpc_cuda_available()) {
  cat("SKIP mgcvExtractGPU single-penalty GCV: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

check_case <- function(formula, data, S, k, label) {
  legacy <- mgcv::gam(formula = formula, data = data, method = "GCV.Cp")
  legacy_sp <- as.numeric(legacy$sp)
  sp_grid <- exp(seq(log(legacy_sp / 5), log(legacy_sp * 5), length.out = 15L))

  gpu <- fastkpc_mgcv_extract_gpu_gcv(
    formula = formula,
    data = data,
    setup_sp = 1,
    sp_grid = sp_grid,
    target = 1L,
    S = S,
    k = k,
    bs = "tp",
    device = "cuda",
    allow_cpu_fallback = FALSE,
    gcv_strategy = "spectral"
  )

  assert_true(identical(gpu$backend_family, "mgcvExtractGPU"),
              paste(label, "backend family"))
  assert_true(identical(gpu$mode, "single-penalty-gpu-gcv"),
              paste(label, "mode"))
  assert_true(identical(gpu$sp_source, "fastkpc-r-cpu-spectral"),
              paste(label, "sp source"))
  assert_true(identical(gpu$sp_selection_backend_executed,
                        "r-cpu-spectral"),
              paste(label, "sp selection backend"))
  assert_true(identical(gpu$gcv_source, "fastkpc-r-cpu-spectral"),
              paste(label, "spectral GCV source"))
  assert_true(identical(gpu$gcv_score_backend_executed, "r-cpu-spectral"),
              paste(label, "GCV score backend"))
  assert_true(identical(gpu$selected_solve_backend_executed, "cuda"),
              paste(label, "selected solve backend"))
  assert_true(identical(gpu$solve_source, "mgcvExtractGPU"),
              paste(label, "solve source"))
  assert_true(isTRUE(gpu$is_self_contained_gcv),
              paste(label, "self-contained GCV flag"))
  assert_true(identical(gpu$used_device, "cuda"),
              paste(label, "used device"))
  assert_true(isTRUE(gpu$native_gpu_solve_used),
              paste(label, "native GPU solve used"))
  assert_true(!isTRUE(gpu$fallback_used),
              paste(label, "no fallback"))
  assert_true(length(gpu$sp) == 1L && gpu$sp %in% sp_grid,
              paste(label, "selected sp should come from grid"))
  assert_true(is.data.frame(gpu$grid) && nrow(gpu$grid) == length(sp_grid),
              paste(label, "grid diagnostics"))
  assert_true(all(c("sp", "rss", "edf", "gcv", "valid") %in% names(gpu$grid)),
              paste(label, "grid diagnostic columns"))
  assert_true(which.min(gpu$grid$gcv) == gpu$selected_grid_index,
              paste(label, "selected grid index should minimize GCV"))
  assert_true(abs(log(gpu$sp / legacy_sp)) <= log(5) + 1e-8,
              paste(label, "selected smoothness should remain near legacy on this grid"))
  assert_true(all(is.finite(gpu$residuals)) && length(gpu$residuals) == nrow(data),
              paste(label, "finite residuals"))
  assert_true(is.finite(gpu$edf) && gpu$edf > 0 && gpu$edf < nrow(data),
              paste(label, "finite edf"))
  assert_true(is.finite(gpu$score) && gpu$score == min(gpu$grid$gcv),
              paste(label, "score should be selected GCV"))
  assert_true(isTRUE(gpu$capabilities$supported$single_penalty_gpu_gcv),
              paste(label, "capability should report single-penalty GPU GCV"))
  assert_true(isTRUE(gpu$capabilities$supported$self_contained_gcv),
              paste(label, "capability should report self-contained GCV"))

  fixed <- fastkpc_mgcv_extract_gpu_fixed_sp(
    formula = formula,
    data = data,
    sp = gpu$sp,
    target = 1L,
    S = S,
    k = k,
    bs = "tp",
    device = "cuda",
    allow_cpu_fallback = FALSE,
    solve_strategy = "handle"
  )
  assert_true(max(abs(gpu$residuals - fixed$residuals)) < 1e-7,
              paste(label, "GCV result should equal fixed-sp CUDA solve at selected sp"))
}

set.seed(246)
n <- 72
s1 <- stats::runif(n, -2, 2)
y1 <- sin(s1) + stats::rnorm(n, sd = 0.06)
check_case(
  formula = y ~ s(s1, k = 10, bs = "tp"),
  data = data.frame(y = y1, s1 = s1),
  S = 2L,
  k = 10L,
  label = "|S|=1"
)

s2 <- stats::runif(n, -2, 2)
y2 <- sin(s1) + cos(s2) + stats::rnorm(n, sd = 0.06)
check_case(
  formula = y ~ s(s1, s2, k = 12, bs = "tp"),
  data = data.frame(y = y2, s1 = s1, s2 = s2),
  S = c(2L, 3L),
  k = 12L,
  label = "|S|=2"
)

cat("PASS mgcvExtractGPU single-penalty GCV\n")
