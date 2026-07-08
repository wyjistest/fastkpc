source("fastkpc/R/mgcv_residual_oracle_trace.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_entry <- function(x, y, S) {
  list(x = x, y = y, S_xy = as.integer(S), S_yx = integer())
}

pMax <- matrix(0.5, 8L, 8L)
pairs <- list(
  c(1L, 2L, 0.10001), c(1L, 3L, 0.10004),
  c(2L, 4L, 0.10008), c(3L, 5L, 0.101),
  c(4L, 6L, 0.12), c(5L, 7L, 0.14),
  c(6L, 8L, 0.18), c(2L, 7L, 0.21),
  c(3L, 8L, 0.3), c(4L, 8L, 0.4)
)
for (pair in pairs) {
  pMax[pair[[1L]], pair[[2L]]] <- pair[[3L]]
  pMax[pair[[2L]], pair[[1L]]] <- pair[[3L]]
}

result <- list(
  skeleton = list(
    pMax = pMax,
    per.level.log = list(
      list(
        make_entry(1L, 2L, 3L),
        make_entry(1L, 3L, 4L),
        make_entry(2L, 4L, 5L)
      ),
      list(
        make_entry(3L, 5L, c(1L, 2L)),
        make_entry(4L, 6L, c(1L, 3L)),
        make_entry(5L, 7L, c(2L, 4L))
      ),
      list(
        make_entry(6L, 8L, c(1L, 2L, 3L)),
        make_entry(2L, 7L, c(1L, 4L, 5L)),
        make_entry(3L, 8L, c(2L, 4L, 6L))
      ),
      list(
        make_entry(4L, 8L, c(1L, 2L, 3L, 5L))
      )
    )
  )
)

assert_true(exists("fastkpc_mgcv_oracle_expanded_cases_from_skeleton_result"),
            "expanded oracle case selector must exist")
cases <- fastkpc_mgcv_oracle_expanded_cases_from_skeleton_result(
  result,
  alpha = 0.1,
  near_alpha_count = 4L,
  per_s_size_count = 2L,
  per_level_count = 2L,
  max_cases = 12L
)
tiny_cases <- fastkpc_mgcv_oracle_cases_from_skeleton_result(
  result,
  alpha = 0.1,
  near_alpha_count = 2L
)

required <- c(
  "case_id", "x", "y", "S", "role", "source", "source_level",
  "source_pmax", "source_distance_to_alpha", "source_case_rank"
)
missing <- setdiff(required, names(cases))
assert_true(length(missing) == 0L,
            paste("expanded cases missing field", missing[[1L]]))
assert_true(nrow(cases) > nrow(tiny_cases),
            "expanded selector should exceed tiny oracle")
assert_true(nrow(cases) <= 12L, "expanded selector should respect max_cases")
assert_true(all(c(1L, 2L, 3L) %in% cases$source_level),
            "expanded cases should cover multiple skeleton levels")
assert_true(all(c(1L, 2L, 3L) %in%
                  vapply(cases$S, function(x) {
                    length(fastkpc_mgcv_oracle_parse_s(x))
                  }, integer(1))),
            "expanded cases should cover |S|=1,2,3")
assert_true(any(grepl("near-alpha", cases$role, fixed = TRUE)),
            "expanded cases should retain near-alpha roles")
assert_true(!any(duplicated(paste(cases$x, cases$y, cases$S, sep = "|"))),
            "expanded cases should be unique by target pair and S")

cat("PASS mgcv residual expanded oracle cases\n")
