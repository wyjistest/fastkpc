fastkpc_scenario_names <- function() {
  c("chain", "fork", "collider", "independent", "additive")
}

fastkpc_truth_matrix <- function(p, edges) {
  truth <- matrix(FALSE, nrow = p, ncol = p)
  if (length(edges) > 0L) {
    for (edge in edges) {
      truth[edge[[1L]], edge[[2L]]] <- TRUE
    }
  }
  truth
}

fastkpc_truth_pdag <- function(adjacency) {
  pdag <- matrix(0L, nrow = nrow(adjacency), ncol = ncol(adjacency))
  pdag[adjacency] <- 1L
  pdag
}

generate_fastkpc_scenario <- function(scenario, seed, n) {
  if (length(scenario) != 1L || !(scenario %in% fastkpc_scenario_names())) {
    stop("Unknown fastkpc validation scenario: ", scenario, call. = FALSE)
  }
  if (length(seed) != 1L || is.na(seed)) stop("seed must be a scalar", call. = FALSE)
  if (length(n) != 1L || is.na(n) || n < 4L) {
    stop("n must be a scalar integer >= 4", call. = FALSE)
  }
  set.seed(as.integer(seed))
  n <- as.integer(n)
  noise <- function(sd = 0.12) stats::rnorm(n, sd = sd)

  if (scenario == "chain") {
    z <- stats::runif(n, -pi, pi)
    x1 <- z + noise(0.08)
    x2 <- sin(x1) + noise(0.12)
    x3 <- x2^2 + noise(0.12)
    x4 <- noise(1)
    edges <- list(c(1L, 2L), c(2L, 3L))
    description <- "Nonlinear chain with one independent variable."
  } else if (scenario == "fork") {
    z <- stats::runif(n, -pi, pi)
    x1 <- z + noise(0.08)
    x2 <- sin(x1) + noise(0.12)
    x3 <- cos(x1) + noise(0.12)
    x4 <- noise(1)
    edges <- list(c(1L, 2L), c(1L, 3L))
    description <- "Nonlinear fork with one independent variable."
  } else if (scenario == "collider") {
    x1 <- stats::runif(n, -pi, pi)
    x2 <- stats::runif(n, -pi, pi)
    x3 <- sin(x1) + cos(x2) + noise(0.12)
    x4 <- noise(1)
    edges <- list(c(1L, 3L), c(2L, 3L))
    description <- "Two independent causes with a nonlinear collider."
  } else if (scenario == "independent") {
    x1 <- noise(1)
    x2 <- noise(1)
    x3 <- noise(1)
    x4 <- noise(1)
    edges <- list()
    description <- "Four mutually independent variables."
  } else if (scenario == "additive") {
    z1 <- stats::runif(n, -pi, pi)
    z2 <- stats::runif(n, -pi, pi)
    x1 <- z1 + noise(0.08)
    x2 <- z2 + noise(0.08)
    x3 <- sin(x1) + cos(x2) + noise(0.12)
    x4 <- x3 + 0.2 * noise(1)
    edges <- list(c(1L, 3L), c(2L, 3L), c(3L, 4L))
    description <- "Additive nonlinear parents followed by one child."
  } else {
    stop("Unknown fastkpc validation scenario: ", scenario, call. = FALSE)
  }

  data <- cbind(x1 = x1, x2 = x2, x3 = x3, x4 = x4)
  storage.mode(data) <- "double"
  truth <- fastkpc_truth_matrix(ncol(data), edges)
  list(
    name = scenario,
    seed = as.integer(seed),
    n = n,
    data = data,
    truth = list(adjacency = truth, pdag = fastkpc_truth_pdag(truth)),
    description = description
  )
}
