# Offline analysis of the cohort bit-identity cases: which cohort shapes change
# the fitted/residual bytes, whether the change depends on width or position,
# and how large the numeric differences are.
args <- commandArgs(TRUE)
stopifnot(length(args) == 1L)
out <- args[[1L]]
files <- sort(list.files(file.path(out, "cases"), pattern = "\\.rds$", full.names = TRUE))
cases <- lapply(files, readRDS)
s <- do.call(rbind, lapply(cases, function(d) d$summary))
cat(sprintf("cases %d, calls %d, native %s\n", nrow(s), sum(s$calls), cases[[1]]$native_sha256))
cat("\n== per level ==\n")
agg <- aggregate(cbind(cases = 1, targets, coef_multi = coefficient_targets_with_multiple_digests,
  logsp_multi = log_sp_targets_with_multiple_digests,
  fitted_multi = fitted_targets_with_multiple_digests,
  resid_multi = residual_targets_with_multiple_digests,
  fixed_multi = fixed_residual_targets_with_multiple_digests) ~ level + q, data = s, FUN = sum)
agg$max_fitted_diff <- aggregate(max_fitted_abs_diff ~ level + q, data = s, FUN = max)$max_fitted_abs_diff
agg$max_resid_diff <- aggregate(max_residual_abs_diff ~ level + q, data = s, FUN = max)$max_residual_abs_diff
print(agg)
cat("\nanchor identical (archived production-dispatcher pair outputs):", table(s$anchor_all_identical, useNA = "ifany"), "\n")
cat("optimizer-route vs exact-fixed residual identical columns:", sum(s$optimizer_vs_fixed_residual_identical_columns), "/", sum(s$optimizer_vs_fixed_residual_columns), "\n")

# Which cohorts mismatch the pair reference, by cohort name (fitted values)?
dig <- do.call(rbind, lapply(cases, function(d) d$digests))
fit <- dig[dig$quantity == "optimizer_fitted", ]
ref <- fit[fit$cohort == "pair", c("case", "target", "sha256")]
names(ref)[3] <- "ref_sha256"
fit <- merge(fit, ref, by = c("case", "target"))
fit$mismatch <- fit$sha256 != fit$ref_sha256
cat("\n== fitted mismatch rate vs pair cohort, by cohort shape ==\n")
by_shape <- aggregate(cbind(columns = 1, mismatch) ~ cohort, data = fit, FUN = sum)
by_shape$rate <- round(by_shape$mismatch / by_shape$columns, 3)
print(by_shape[order(by_shape$cohort), ], row.names = FALSE)
cat("\n== fitted mismatch rate by cohort width ==\n")
by_width <- aggregate(cbind(columns = 1, mismatch) ~ width, data = fit, FUN = sum)
by_width$rate <- round(by_width$mismatch / by_width$columns, 3)
print(by_width, row.names = FALSE)

# Position dependence: same targets, same width, different order.
pos_pairs <- list(c("prefix_03", "prefix_3_shifted"), c("pair", "pair_swapped"),
                  c("all_natural", "all_reversed"), c("all_natural", "all_permuted"),
                  c("pair", "pair_repeat"))
cat("\n== same width, different column order: per-target fitted digest identical? ==\n")
for (pp in pos_pairs) {
  a <- fit[fit$cohort == pp[1], c("case", "target", "sha256")]
  b <- fit[fit$cohort == pp[2], c("case", "target", "sha256")]
  m <- merge(a, b, by = c("case", "target"))
  cat(sprintf("%-18s vs %-18s: %d / %d columns identical\n", pp[1], pp[2], sum(m$sha256.x == m$sha256.y), nrow(m)))
}
# Width classes: do all cohorts of the same width agree with each other per target?
cat("\n== distinct fitted digests per (case,target) among cohorts of equal width ==\n")
w <- aggregate(sha256 ~ case + target + width, data = fit, FUN = function(x) length(unique(x)))
print(table(width = w$width, distinct = w$sha256))
# Numeric magnitude: ULP-scale of mismatches relative to |fitted|
pt <- do.call(rbind, lapply(cases, function(d) d$per_target))
fp <- pt[pt$quantity == "optimizer_fitted" & pt$distinct > 1, ]
cat("\nfitted mismatching targets:", nrow(fp), " max abs diff:", format(max(fp$max_abs_diff), digits = 3),
    " quantiles:", format(quantile(fp$max_abs_diff, c(.5, .9, 1)), digits = 3), "\n")
rp <- pt[pt$quantity == "optimizer_residuals" & pt$distinct > 1, ]
cat("residual mismatching targets:", nrow(rp), " max abs diff:", format(max(rp$max_abs_diff), digits = 3), "\n")
cp <- pt[pt$quantity == "optimizer_coefficients", ]
cat("coefficient targets with >1 digest:", sum(cp$distinct > 1), "/", nrow(cp), "\n")
lp <- pt[pt$quantity == "selected_log_sp", ]
cat("selected log-SP targets with >1 digest:", sum(lp$distinct > 1), "/", nrow(lp), "\n")
el <- unlist(lapply(cases, function(d) d$elapsed))
cat("\ncall elapsed seconds: median", round(median(el), 3), "max", round(max(el), 3), "total", round(sum(el), 1), "\n")
res <- list(summary = s, by_shape = by_shape, by_width = by_width, width_classes = as.data.frame.matrix(table(w$width, w$sha256)))
saveRDS(res, file.path(out, "analysis.rds"))
write.csv(by_shape, file.path(out, "fitted_mismatch_by_cohort.csv"), row.names = FALSE)
write.csv(by_width, file.path(out, "fitted_mismatch_by_width.csv"), row.names = FALSE)
