source("fastkpc/R/diff_report.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

edge_key <- function(edge) paste(as.integer(edge), collapse = "-")

old_adj <- matrix(FALSE, 4, 4)
old_adj[1, 2] <- old_adj[2, 1] <- TRUE
old_adj[1, 3] <- old_adj[3, 1] <- TRUE
new_adj <- old_adj
new_adj[1, 3] <- new_adj[3, 1] <- FALSE
new_adj[2, 4] <- new_adj[4, 2] <- TRUE

adj_diff <- compare_adjacency(old_adj, new_adj)
assert_true(length(adj_diff$added_edges) == 1, "one undirected edge should be added")
assert_true(length(adj_diff$removed_edges) == 1, "one undirected edge should be removed")
assert_true(edge_key(adj_diff$added_edges[[1]]) == "2-4", "added edge should be 2-4")
assert_true(edge_key(adj_diff$removed_edges[[1]]) == "1-3", "removed edge should be 1-3")

old_pmax <- matrix(0, 3, 3)
new_pmax <- old_pmax
new_pmax[1, 2] <- new_pmax[2, 1] <- 0.2
new_pmax[1, 3] <- new_pmax[3, 1] <- 0.8
pmax_diff <- compare_pmax(old_pmax, new_pmax)
assert_true(abs(pmax_diff$max_abs_diff - 0.8) < 1e-12,
            "max pMax difference should be 0.8")
assert_true(pmax_diff$top_20_diffs[[1]]$pair[1] == 1 &&
              pmax_diff$top_20_diffs[[1]]$pair[2] == 3,
            "largest pMax diff should be first")

old_sep <- list(vector("list", 3), vector("list", 3), vector("list", 3))
new_sep <- list(vector("list", 3), vector("list", 3), vector("list", 3))
old_sep[[1]][[2]] <- c(3L)
new_sep[[1]][[2]] <- c(2L)
old_sep[[2]][[1]] <- c(3L)
new_sep[[2]][[1]] <- c(2L)
sep_diff <- compare_sepsets(old_sep, new_sep)
assert_true(sep_diff$differing_count == 1, "one undirected sepset should differ")
assert_true(edge_key(sep_diff$differing_pairs[[1]]$pair) == "1-2",
            "differing sepset pair should be 1-2")

summary <- summarize_graph_diff(
  list(adjacency = old_adj, pMax = old_pmax, sepsets = old_sep, n.edgetests = c(2L, 3L)),
  list(adjacency = new_adj, pMax = new_pmax, sepsets = new_sep, n.edgetests = c(2L, 4L))
)
assert_true(summary$n_edgetests$old[2] == 3L && summary$n_edgetests$new[2] == 4L,
            "summary should include n.edgetests comparison")

cat("test_diff_report.R: PASS\n")
