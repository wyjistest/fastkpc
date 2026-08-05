source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

normalize_sepsets <- function(value) {
  p <- length(value)
  output <- vector("list", p * (p - 1L) / 2L)
  position <- 0L
  for (left in seq_len(p - 1L)) {
    for (right in seq.int(left + 1L, p)) {
      position <- position + 1L
      output[[position]] <- sort(unique(c(
        as.integer(value[[left]][[right]]),
        as.integer(value[[right]][[left]])
      )))
    }
  }
  output
}

trace_frame <- function(value) {
  if (!length(value)) {
    return(data.frame(
      target = integer(), other = integer(), subset_key = character(),
      residual_conditioning_key = character(), p_value = numeric(),
      rejected = logical()
    ))
  }
  output <- do.call(rbind, lapply(value, function(entry) {
    data.frame(
      target = as.integer(entry$target),
      other = as.integer(entry$other),
      subset_key = paste(as.integer(entry$subset), collapse = "|"),
      residual_conditioning_key = paste(
        as.integer(entry$residual_conditioning_set), collapse = "|"
      ),
      p_value = as.numeric(entry$p.value),
      rejected = isTRUE(entry$rejected),
      stringsAsFactors = FALSE
    )
  }))
  rownames(output) <- NULL
  output
}

run_kpcalg_oracle <- function(data, method, seed) {
  env <- fastkpc_legacy_env()
  suff_stat <- list(
    data = data, ic.method = method, index = 1, numCol = 35L,
    sig = 1, p = 100L
  )
  skeleton_calls <- list()
  traced_skeleton_test <- function(x, y, S, suffStat) {
    p_value <- env$kernelCItest(x, y, S, suffStat)
    skeleton_calls[[length(skeleton_calls) + 1L]] <<- data.frame(
      level = length(S), x = as.integer(x), y = as.integer(y),
      S_key = paste(sort(as.integer(S)), collapse = "|"),
      p_value = as.numeric(p_value),
      deleted = as.numeric(p_value) >= 0.1,
      stringsAsFactors = FALSE
    )
    p_value
  }
  set.seed(seed)
  skeleton <- pcalg::skeleton(
    suffStat = suff_stat,
    indepTest = traced_skeleton_test,
    alpha = 0.1,
    labels = colnames(data),
    m.max = 2L,
    method = "stable"
  )

  orientation_calls <- list()
  original_regrvonps <- env$regrVonPS
  env$regrVonPS <- function(G, V, S, suffStat,
                            indepTest = env$kernelCItest, alpha = 0.2) {
    subset <- as.integer(S)
    parents <- which(G[, V] == 1 & G[V, ] == 0, arr.ind = TRUE)
    conditioning <- sort(unique(c(subset, as.integer(parents))))
    position <- 0L
    traced_orientation_test <- function(x, y, S = NULL, suffStat) {
      position <<- position + 1L
      p_value <- indepTest(x = x, y = y, S = S, suffStat = suffStat)
      orientation_calls[[length(orientation_calls) + 1L]] <<- list(
        target = as.integer(V), other = subset[[position]], subset = subset,
        residual_conditioning_set = conditioning,
        p.value = as.numeric(p_value),
        rejected = as.numeric(p_value) < alpha
      )
      p_value
    }
    original_regrvonps(
      G = G, V = V, S = subset, suffStat = suffStat,
      indepTest = traced_orientation_test, alpha = alpha
    )
  }
  orientation <- env$udag2wanpdag(
    gInput = skeleton,
    suffStat = suff_stat,
    indepTest = env$kernelCItest,
    alpha = 0.1,
    verbose = FALSE,
    solve.confl = FALSE,
    orientCollider = TRUE,
    rules = c(TRUE, TRUE, TRUE)
  )
  list(
    skeleton = skeleton,
    skeleton_trace = do.call(rbind, skeleton_calls),
    orientation_trace = trace_frame(orientation_calls),
    pdag = methods::as(orientation@graph, "matrix"),
    rng_end_state = .Random.seed
  )
}

candidate_skeleton_trace <- function(value) {
  value <- value[!value$native_edge_ignored, , drop = FALSE]
  output <- data.frame(
    level = as.integer(value$level),
    x = as.integer(value$x),
    y = as.integer(value$y),
    S_key = as.character(value$S_key),
    p_value = as.numeric(value$p_used),
    deleted = as.logical(value$native_edge_deleted),
    stringsAsFactors = FALSE
  )
  output$key <- paste(output$level, output$x, output$y, output$S_key,
                      sep = "\037")
  output
}

set.seed(9127)
n <- 56L
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.08),
  x2 = cos(z) + stats::rnorm(n, sd = 0.08),
  x3 = z + stats::rnorm(n, sd = 0.08),
  x4 = z^2 + stats::rnorm(n, sd = 0.08),
  x5 = z^3 + stats::rnorm(n, sd = 0.08)
)

if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP compatible CUDA kpcalg-authority WAN-PDAG: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

missing_seed_error <- tryCatch({
  fastkpc_orient_wanpdag_kpcalg_authority(
    list(), data, ci_method = "dcc.perm",
    permutation_params = list(replicates = 100L, seed = NULL,
                              include_observed = TRUE)
  )
  ""
}, error = conditionMessage)
assert_true(grepl("explicit non-negative seed", missing_seed_error,
                  fixed = TRUE),
            "permutation authority must fail closed without a seed")

methods <- c("dcc.perm", "hsic.gamma", "hsic.perm")
for (method in methods) {
  oracle <- run_kpcalg_oracle(data, method, seed = 707L)
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  set.seed(707L)
  candidate <- fastkpc_compatible_cuda_wanpdag(
    data,
    alpha = 0.1,
    options = list(
      max_conditioning_size = 2L,
      numCol = 35L,
      trace_level = "logical",
      ci_method = method,
      hsic_params = list(sig = 1),
      permutation_params = list(
        replicates = 100L, seed = 707L, include_observed = TRUE
      )
    )
  )
  candidate_rng_end <- .Random.seed
  candidate_trace <- candidate_skeleton_trace(candidate$skeleton$tasks)
  oracle_trace <- oracle$skeleton_trace
  oracle_trace$key <- paste(
    oracle_trace$level, oracle_trace$x, oracle_trace$y, oracle_trace$S_key,
    sep = "\037"
  )
  assert_true(!anyDuplicated(candidate_trace$key) &&
                !anyDuplicated(oracle_trace$key) &&
                setequal(candidate_trace$key, oracle_trace$key),
              paste(method, "skeleton CI identities changed"))
  oracle_trace <- oracle_trace[
    match(candidate_trace$key, oracle_trace$key), , drop = FALSE
  ]
  tolerance <- if (method == "hsic.gamma") 1e-10 else 0
  assert_true(all(abs(candidate_trace$p_value - oracle_trace$p_value) <=
                    tolerance),
              paste(method, "skeleton p-values changed"))
  assert_true(identical(candidate_trace$deleted,
                        as.logical(oracle_trace$deleted)),
              paste(method, "skeleton decision trace changed"))
  assert_true(identical(
    unname(candidate$skeleton$adjacency != 0),
    unname(methods::as(oracle$skeleton@graph, "matrix") != 0)
  ), paste(method, "skeleton adjacency changed"))
  assert_true(identical(
    normalize_sepsets(candidate$skeleton$sepsets),
    normalize_sepsets(oracle$skeleton@sepset)
  ), paste(method, "skeleton sepsets changed"))
  assert_true(identical(
    as.integer(candidate$skeleton$n.edgetests),
    as.integer(oracle$skeleton@n.edgetests)
  ), paste(method, "skeleton n.edgetests changed"))

  candidate_orientation_trace <- trace_frame(candidate$orientation$ci_trace)
  assert_true(identical(
    candidate_orientation_trace[, c(
      "target", "other", "subset_key", "residual_conditioning_key"
    )],
    oracle$orientation_trace[, c(
      "target", "other", "subset_key", "residual_conditioning_key"
    )]
  ), paste(method, "orientation CI identities changed"))
  assert_true(identical(candidate_orientation_trace$p_value,
                        oracle$orientation_trace$p_value),
              paste(method, "orientation p-values changed"))
  assert_true(identical(candidate_orientation_trace$rejected,
                        oracle$orientation_trace$rejected),
              paste(method, "orientation decisions changed"))
  assert_true(identical(as.integer(unname(candidate$orientation$pdag)),
                        as.integer(unname(oracle$pdag))),
              paste(method, "WAN-PDAG changed"))
  assert_true(identical(candidate$orientation$ci_backend,
                        "kpcalg-cpu-authority") &&
                !isTRUE(candidate$orientation$diagnostics$native_orientation_executed),
              paste(method, "did not use kpcalg orientation authority"))
  assert_true(
    identical(candidate$config$skeleton_authority, "full_cuda") &&
      identical(candidate$config$orientation_authority,
                "kpcalg_cpu_wanpdag") &&
      identical(candidate$config$native_cuda_orientation_status,
                "experimental") &&
      identical(candidate$authority_receipt$skeleton_authority,
                "full_cuda") &&
      identical(candidate$authority_receipt$orientation_authority,
                "kpcalg_cpu_wanpdag") &&
      identical(candidate$authority_receipt$ci_method, method),
    paste(method, "authority receipt changed")
  )
  expected_p_contract <- if (method == "hsic.gamma") {
    "absolute-tolerance-1e-10"
  } else {
    "bitwise-exact"
  }
  assert_true(identical(
    candidate$authority_receipt$numerical_contract$skeleton_p_value,
    expected_p_contract
  ), paste(method, "numerical contract changed"))
  if (method %in% c("dcc.perm", "hsic.perm")) {
    assert_true(identical(candidate_rng_end, oracle$rng_end_state),
                paste(method, "final R RNG state changed"))
    assert_true(identical(
      candidate$authority_receipt$permutation_rng_contract$contract,
      "legacy-r-global-stream-exact"
    ), paste(method, "permutation RNG contract changed"))
  } else {
    assert_true(identical(
      candidate$authority_receipt$permutation_rng_contract$contract,
      "not-applicable"
    ), paste(method, "non-permutation RNG contract changed"))
  }
}

cat("test_compatible_cuda_wanpdag_authority.R: PASS\n")
