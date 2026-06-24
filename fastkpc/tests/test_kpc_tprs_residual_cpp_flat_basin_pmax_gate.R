source("fastkpc/R/kpc_tprs_residual_cpp_qualification.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_result <- function(pmax_value) {
  list(
    skeleton = list(
      adjacency = matrix(c(FALSE, TRUE, TRUE, FALSE), 2L, 2L),
      pMax = matrix(c(1, pmax_value, pmax_value, 1), 2L, 2L),
      sepsets = replicate(
        2L, replicate(2L, integer(), simplify = FALSE), simplify = FALSE),
      n.edgetests = 1L
    )
  )
}

bounded <- fastkpc_kpc_tprs_graph_agreement_row(
  reference = make_result(0.4),
  candidate = make_result(0.4 + 0.000197021546988863),
  scenario_id = "flat-gcv-basin",
  repeat_id = 1L
)
assert_true(isTRUE(bounded$passed[[1L]]),
            "flat-basin pMax drift should pass the qualification gate")
assert_true(identical(
  as.numeric(bounded$pmax_abs_tol[[1L]]),
  fastkpc_kpc_tprs_pmax_abs_tol()
), "graph agreement should report the pMax tolerance")

large <- fastkpc_kpc_tprs_graph_agreement_row(
  reference = make_result(0.4),
  candidate = make_result(0.4 + fastkpc_kpc_tprs_pmax_abs_tol() * 2),
  scenario_id = "large-drift",
  repeat_id = 1L
)
assert_true(isFALSE(large$passed[[1L]]),
            "larger pMax drift should still fail the qualification gate")

cat("PASS kpcTprsResidualCPP flat-basin pMax gate\n")
