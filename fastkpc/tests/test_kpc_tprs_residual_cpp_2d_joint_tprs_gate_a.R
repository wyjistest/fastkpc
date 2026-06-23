source("fastkpc/R/kpc_tprs_residual_cpp.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_has_names <- function(x, required, message) {
  missing <- setdiff(required, names(x))
  if (length(missing) > 0L) {
    fail(paste0(message, ": missing ", paste(missing, collapse = ", ")))
  }
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP kpcTprsResidualCPP 2D Gate A: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(62312)
n <- 90L
theta <- seq(0, 2 * pi, length.out = n + 1L)[seq_len(n)]
r <- 0.5 + stats::runif(n)
s1 <- r * cos(theta) + stats::rnorm(n, sd = 0.03)
s2 <- r * sin(theta) + stats::rnorm(n, sd = 0.03)
S <- cbind(s1 = s1, s2 = s2)
y <- sin(s1) + cos(1.4 * s2) + stats::rnorm(n, sd = 0.03)

parity <- fastkpc_kpc_tprs_geometry_parity(y = y, S = S)
assert_has_names(
  parity,
  c("projector_distance", "penalty_spectrum_distance",
    "transform_metrics", "conditioning_size"),
  "2D geometry parity"
)
assert_true(identical(parity$conditioning_size, 2L), "2D conditioning size")
assert_true(parity$projector_distance <= 1e-6,
            "2D raw function-space projector should match mgcv")

candidate <- kpc_tprs_residual_cpp_setup(S)
oracle <- fastkpc_kpc_tprs_mgcv_oracle_setup(y, S, sp = 1)
absorbed_projector_distance <- fastkpc_kpc_tprs_projector_distance(
  candidate$absorbed$X,
  fastkpc_kpc_tprs_oracle_absorbed_basis(oracle)
)
assert_true(absorbed_projector_distance <= 1e-6,
            "2D absorbed function-space projector should match mgcv")

rotated <- parity$transform_metrics[
  parity$transform_metrics$transform == "rotation", , drop = FALSE]
assert_true(nrow(rotated) == 1L, "2D rotation metric is present")
assert_true(rotated$candidate_projector_distance <= 1e-6,
            "2D joint isotropic TPRS should be rotation invariant")

cat("PASS kpcTprsResidualCPP 2D joint TPRS Gate A\n")
