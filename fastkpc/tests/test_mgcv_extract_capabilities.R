source("fastkpc/R/mgcv_extract_oracle.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

cap <- fastkpc_mgcv_extract_capabilities()

assert_true(identical(cap$backend, "mgcvExtract"), "backend name")
assert_true(identical(cap$role, "version-pinned oracle"), "backend role")
assert_true(isTRUE(cap$supported$family == "gaussian_identity"),
            "gaussian identity support")
assert_true(isTRUE(cap$supported$residual_output_only),
            "residual-only support")
assert_true(isTRUE(cap$supported$fixed_sp_self_solve),
            "fixed-sp self-solve support")
assert_true(isTRUE(cap$supported$gcv_bridge),
            "GCV bridge support")
assert_true(isTRUE(cap$unsupported$self_contained_gcv),
            "self-contained GCV remains unsupported")
assert_true(isTRUE(cap$unsupported$cuda_mgcv_subset),
            "CUDA mgcv subset remains unsupported")
assert_true(nzchar(cap$version_pins$R_version), "R version pin")
assert_true("mgcv_version" %in% names(cap$version_pins),
            "mgcv version pin field")
assert_true(identical(cap$baseline$tag, "mgcv-gate-b-v1"),
            "baseline tag")
assert_true(identical(cap$baseline$commit, "5da2313"),
            "baseline commit")

cat("PASS mgcv extract capabilities\n")
