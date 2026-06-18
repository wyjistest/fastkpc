source("fastkpc/R/mgcv_compat_contract.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}
assert_equal <- function(actual, expected, message) {
  if (!identical(actual, expected)) {
    fail(paste0(message, ": expected ", paste(expected, collapse = ","),
                " got ", paste(actual, collapse = ",")))
  }
}

assert_equal(
  fastkpc_regrxons_formula_class(integer()),
  "direct-ci",
  "|S| == 0 must be direct CI, not residualization"
)

assert_equal(
  fastkpc_regrxons_formula_class(3L),
  "full-smooth",
  "|S| == 1 must use joint/full smooth"
)

assert_equal(
  fastkpc_regrxons_formula_class(c(3L, 4L)),
  "full-smooth",
  "|S| == 2 must use joint/full smooth"
)

assert_equal(
  fastkpc_regrxons_formula_class(c(3L, 4L, 5L)),
  "additive-smooth",
  "|S| > 2 must use additive smooth"
)

sem <- fastkpc_regrxons_semantics(c(4L, 2L), target = 1L, n = 20L, p = 5L)
assert_equal(sem$formula_class, "full-smooth", "formula class")
assert_equal(sem$conditioning_variable_order_used_in_formula, c(4L, 2L),
             "formula order must preserve caller order")
assert_equal(sem$conditioning_set_as_set, c(2L, 4L),
             "set key must be sorted independently")
assert_true(grepl("kpcalg_regrXonS_v1", sem$compatibility_mode),
            "compatibility mode must be explicit")

setup <- fastkpc_setup_fingerprint(sem, R_version = "4.6.0",
                                   mgcv_version = "1.9-4",
                                   model_matrix_hash = "XHASH",
                                   penalty_hashes = c("S1", "S2"),
                                   constraint_hash = "CHASH",
                                   rank_metadata = "rank=7")
target_a <- fastkpc_target_fingerprint(target = 1L, y_hash = "YA",
                                       selected_sp = c(0.1, 1.0),
                                       score = 12.5, edf = 3.2)
target_b <- fastkpc_target_fingerprint(target = 2L, y_hash = "YB",
                                       selected_sp = c(0.2, 2.0),
                                       score = 13.5, edf = 4.2)

setup_values <- as.character(unlist(setup$fields, use.names = FALSE))
target_specific_values <- c("YA", "YB", "0.1", "0.2")
assert_true(!any(setup_values %in% target_specific_values),
            "setup fields must not contain target-specific values")
assert_true(!identical(target_a$fingerprint, target_b$fingerprint),
            "target fingerprint must differ across targets")

cat("PASS mgcv compatibility contract\n")
