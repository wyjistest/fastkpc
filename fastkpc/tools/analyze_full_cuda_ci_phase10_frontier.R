args <- commandArgs(trailingOnly = TRUE)
trace_path <- if (length(args) >= 1L) args[[1L]] else
  "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1/logical_ci_trace.rds"
capacity <- if (length(args) >= 2L) as.integer(args[[2L]]) else 64L

stopifnot(length(capacity) == 1L, !is.na(capacity), capacity >= 1L)
trace <- readRDS(trace_path)
required <- c("level", "x", "y", "S_key", "deletes_edge")
stopifnot(is.data.frame(trace), all(required %in% names(trace)))

edge_key <- function(x, y) paste0(pmin(x, y), "|", pmax(x, y))
bucket_key <- function(value) ifelse(nzchar(value), paste0("S:", value), "DIRECT")
trace$edge_key <- edge_key(trace$x, trace$y)

touch_lru <- function(lru, key, capacity) {
  found <- match(key, lru, nomatch = 0L)
  if (found > 0L) lru <- lru[-found]
  lru <- c(key, lru)
  if (length(lru) > capacity) lru <- head(lru, capacity)
  list(lru = lru, hit = found > 0L)
}

choose_key <- function(buckets, active_keys, lru, policy) {
  keys <- active_keys
  stopifnot(length(keys) > 0L)
  sizes <- vapply(keys, function(key) {
    length(get(key, envir = buckets, inherits = FALSE))
  }, integer(1L))
  if (startsWith(policy, "cache-aware")) {
    cached <- keys %in% lru
    if (any(cached)) {
      factor <- switch(
        policy,
        "cache-aware" = Inf,
        "cache-aware-2x" = 2,
        "cache-aware-4x" = 4,
        stop("unknown cache-aware policy")
      )
      cached_max <- max(sizes[cached])
      if (cached_max * factor >= max(sizes)) {
        keys <- keys[cached]
        sizes <- sizes[cached]
      }
    }
  }
  keys[order(-sizes, keys)][[1L]]
}

simulate_level <- function(rows, policy, capacity) {
  edge_rows <- split(seq_len(nrow(rows)), rows$edge_key)
  positions <- rep.int(1L, length(edge_rows))
  active <- rep.int(TRUE, length(edge_rows))
  names(positions) <- names(edge_rows)
  names(active) <- names(edge_rows)
  buckets <- new.env(parent = emptyenv(), hash = TRUE)
  active_keys <- character()
  enqueue <- function(edge_index) {
    row_index <- edge_rows[[edge_index]][[positions[[edge_index]]]]
    key <- bucket_key(rows$S_key[[row_index]])
    if (exists(key, envir = buckets, inherits = FALSE)) {
      bucket <- get(key, envir = buckets, inherits = FALSE)
    } else {
      bucket <- integer()
      active_keys <<- c(active_keys, key)
    }
    assign(key, c(bucket, edge_index), envir = buckets)
  }
  for (edge_index in seq_along(edge_rows)) enqueue(edge_index)

  lru <- character()
  setup_requests <- 0L
  setup_hits <- 0L
  setup_misses <- 0L
  batches <- 0L
  residual_fits <- 0L
  optimized <- new.env(parent = emptyenv(), hash = TRUE)
  optimized_targets <- 0L
  evaluated <- 0L

  active_count <- length(edge_rows)
  while (active_count > 0L) {
    key <- choose_key(buckets, active_keys, lru, policy)
    ready_edges <- get(key, envir = buckets, inherits = FALSE)
    rm(list = key, envir = buckets)
    active_keys <- active_keys[active_keys != key]
    row_indices <- vapply(ready_edges, function(edge_index) {
      edge_rows[[edge_index]][[positions[[edge_index]]]]
    }, integer(1L))
    stopifnot(all(bucket_key(rows$S_key[row_indices]) == key))
    conditioning_key <- rows$S_key[[row_indices[[1L]]]]

    batches <- batches + 1L
    evaluated <- evaluated + length(row_indices)
    targets <- sort(unique(c(rows$x[row_indices], rows$y[row_indices])))
    residual_fits <- residual_fits + length(targets)
    if (nzchar(conditioning_key)) {
      setup_requests <- setup_requests + 1L
      touched <- touch_lru(lru, key, capacity)
      lru <- touched$lru
      if (touched$hit) setup_hits <- setup_hits + 1L else
        setup_misses <- setup_misses + 1L
      target_keys <- paste0(conditioning_key, ":", targets)
      missing <- !vapply(target_keys, exists, logical(1L), envir = optimized,
                         inherits = FALSE)
      for (target_key in target_keys[missing]) {
        assign(target_key, TRUE, envir = optimized)
      }
      optimized_targets <- optimized_targets + sum(missing)
    }

    for (index in seq_along(ready_edges)) {
      edge_index <- ready_edges[[index]]
      row_index <- row_indices[[index]]
      if (isTRUE(rows$deletes_edge[[row_index]])) {
        active[[edge_index]] <- FALSE
        active_count <- active_count - 1L
        next
      }
      positions[[edge_index]] <- positions[[edge_index]] + 1L
      if (positions[[edge_index]] > length(edge_rows[[edge_index]])) {
        active[[edge_index]] <- FALSE
        active_count <- active_count - 1L
      } else {
        enqueue(edge_index)
      }
    }
  }

  data.frame(
    level = rows$level[[1L]],
    policy = policy,
    evaluated = evaluated,
    batches = batches,
    setup_requests = setup_requests,
    setup_hits = setup_hits,
    setup_misses = setup_misses,
    optimizer_targets = optimized_targets,
    residual_fits = residual_fits,
    stringsAsFactors = FALSE
  )
}

policies <- if (length(args) >= 3L) strsplit(args[[3L]], ",", fixed = TRUE)[[1L]] else
  c("largest-frontier", "cache-aware")
stopifnot(length(policies) > 0L,
          all(policies %in% c(
            "largest-frontier", "cache-aware", "cache-aware-2x",
            "cache-aware-4x"
          )))
results <- do.call(rbind, lapply(sort(unique(trace$level)), function(level) {
  rows <- trace[trace$level == level, , drop = FALSE]
  result <- do.call(rbind, lapply(policies, function(policy) {
    value <- simulate_level(rows, policy, capacity)
    message("level=", level, " policy=", policy,
            " batches=", value$batches,
            " setup_misses=", value$setup_misses)
    value
  }))
  result
}))

print(results, row.names = FALSE)
totals <- aggregate(
  results[c("evaluated", "batches", "setup_requests", "setup_hits",
            "setup_misses", "optimizer_targets", "residual_fits")],
  list(policy = results$policy), sum
)
print(totals, row.names = FALSE)
