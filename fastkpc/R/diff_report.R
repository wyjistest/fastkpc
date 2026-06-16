undirected_edges <- function(adjacency) {
  adjacency <- as.matrix(adjacency)
  positions <- which(adjacency & upper.tri(adjacency), arr.ind = TRUE)
  lapply(seq_len(nrow(positions)), function(i) as.integer(positions[i, ]))
}

edge_set_keys <- function(edges) {
  vapply(edges, function(edge) paste(edge, collapse = "-"), character(1))
}

compare_adjacency <- function(old, new) {
  old_edges <- undirected_edges(old)
  new_edges <- undirected_edges(new)
  old_keys <- edge_set_keys(old_edges)
  new_keys <- edge_set_keys(new_edges)

  added_keys <- setdiff(new_keys, old_keys)
  removed_keys <- setdiff(old_keys, new_keys)
  unchanged_keys <- intersect(old_keys, new_keys)

  parse_edge <- function(key) as.integer(strsplit(key, "-", fixed = TRUE)[[1]])
  list(
    added_edges = lapply(added_keys, parse_edge),
    removed_edges = lapply(removed_keys, parse_edge),
    unchanged_edges = lapply(unchanged_keys, parse_edge)
  )
}

compare_pmax <- function(old, new) {
  old <- as.matrix(old)
  new <- as.matrix(new)
  if (!identical(dim(old), dim(new))) stop("pMax dimensions differ")
  diff <- abs(new - old)
  positions <- which(upper.tri(diff), arr.ind = TRUE)
  entries <- lapply(seq_len(nrow(positions)), function(i) {
    row <- positions[i, 1]
    col <- positions[i, 2]
    list(
      pair = as.integer(c(row, col)),
      old = old[row, col],
      new = new[row, col],
      abs_diff = diff[row, col]
    )
  })
  entries <- entries[order(vapply(entries, function(x) x$abs_diff, numeric(1)),
                           decreasing = TRUE)]
  list(
    max_abs_diff = if (length(entries) == 0) 0 else entries[[1]]$abs_diff,
    mean_abs_diff = if (length(entries) == 0) 0 else mean(vapply(entries, function(x) x$abs_diff, numeric(1))),
    top_20_diffs = head(entries, 20)
  )
}

normalize_sepset <- function(value) {
  if (is.null(value)) return(integer(0))
  sort(as.integer(value))
}

compare_sepsets <- function(old, new) {
  if (length(old) != length(new)) stop("sepset dimensions differ")
  p <- length(old)
  matching_count <- 0L
  differing_pairs <- list()
  for (i in seq_len(p - 1L)) {
    for (j in (i + 1L):p) {
      old_value <- normalize_sepset(old[[i]][[j]])
      new_value <- normalize_sepset(new[[i]][[j]])
      if (identical(old_value, new_value)) {
        matching_count <- matching_count + 1L
      } else {
        differing_pairs[[length(differing_pairs) + 1L]] <- list(
          pair = as.integer(c(i, j)),
          old = old_value,
          new = new_value
        )
      }
    }
  }
  list(
    matching_count = matching_count,
    differing_count = length(differing_pairs),
    differing_pairs = differing_pairs
  )
}

summarize_graph_diff <- function(old_result, new_result) {
  list(
    adjacency = compare_adjacency(old_result$adjacency, new_result$adjacency),
    pMax = compare_pmax(old_result$pMax, new_result$pMax),
    sepsets = compare_sepsets(old_result$sepsets, new_result$sepsets),
    n_edgetests = list(
      old = old_result$n.edgetests,
      new = new_result$n.edgetests,
      diff = new_result$n.edgetests - old_result$n.edgetests
    )
  )
}
