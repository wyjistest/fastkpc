source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/validation_campaign.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

data <- cbind(
  x1 = seq(-2, 2, length.out = 100),
  x2 = sin(seq(-2, 2, length.out = 100)),
  x3 = cos(seq(-2, 2, length.out = 100)),
  x4 = seq(-2, 2, length.out = 100)^2
)

a <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
              engine = "cpu", residual_backend = "fastSpline", seed = 9)
b <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
              engine = "cpu", residual_backend = "fastSpline", seed = 9)

assert_true(identical(a$skeleton$adjacency, b$skeleton$adjacency),
            "skeleton adjacency should repeat exactly")
assert_true(identical(a$skeleton$sepsets, b$skeleton$sepsets),
            "sepsets should repeat exactly")
assert_true(max(abs(a$skeleton$pMax - b$skeleton$pMax)) == 0,
            "pMax should repeat exactly for same engine")
assert_true(identical(a$orientation$pdag, b$orientation$pdag),
            "pdag should repeat exactly")
assert_true(identical(a$orientation$counts, b$orientation$counts),
            "orientation counts should repeat exactly")

campaign_a <- run_fastkpc_validation_campaign(
  seeds = c(31),
  n_values = c(70),
  scenarios = c("chain"),
  engines = c("cpu"),
  residual_backends = c("fastSpline"),
  legacy = FALSE
)
campaign_b <- run_fastkpc_validation_campaign(
  seeds = c(31),
  n_values = c(70),
  scenarios = c("chain"),
  engines = c("cpu"),
  residual_backends = c("fastSpline"),
  legacy = FALSE
)

strip_elapsed <- function(campaign) {
  campaign$runs$elapsed_total_sec <- 0
  campaign$timings$elapsed_sec <- 0
  campaign
}
assert_true(identical(strip_elapsed(campaign_a)$runs, strip_elapsed(campaign_b)$runs),
            "campaign run rows should repeat except timing")

hsic_a <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
                   engine = "cpu", residual_backend = "linear",
                   graph_stage = "skeleton", ci_method = "hsic.gamma",
                   seed = 10)
hsic_b <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
                   engine = "cpu", residual_backend = "linear",
                   graph_stage = "skeleton", ci_method = "hsic.gamma",
                   seed = 10)
assert_true(identical(hsic_a$skeleton$adjacency, hsic_b$skeleton$adjacency),
            "HSIC gamma adjacency should repeat exactly")
assert_true(max(abs(hsic_a$skeleton$pMax - hsic_b$skeleton$pMax)) == 0,
            "HSIC gamma pMax should repeat exactly")

perm_a <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
                   engine = "cpu", residual_backend = "linear",
                   graph_stage = "skeleton", ci_method = "hsic.perm",
                   permutation_params = list(replicates = 20L, seed = 77L,
                                             include_observed = TRUE))
perm_b <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
                   engine = "cpu", residual_backend = "linear",
                   graph_stage = "skeleton", ci_method = "hsic.perm",
                   permutation_params = list(replicates = 20L, seed = 77L,
                                             include_observed = TRUE))
assert_true(identical(perm_a$skeleton$adjacency, perm_b$skeleton$adjacency),
            "HSIC permutation fixed-seed adjacency should repeat exactly")
assert_true(max(abs(perm_a$skeleton$pMax - perm_b$skeleton$pMax)) == 0,
            "HSIC permutation fixed-seed pMax should repeat exactly")

campaign_hsic_a <- run_fastkpc_validation_campaign(
  seeds = c(32), n_values = c(50), scenarios = c("chain"),
  engines = c("cpu"), residual_backends = c("linear"),
  ci_methods = c("hsic.gamma"), legacy = FALSE, benchmark = FALSE
)
campaign_hsic_b <- run_fastkpc_validation_campaign(
  seeds = c(32), n_values = c(50), scenarios = c("chain"),
  engines = c("cpu"), residual_backends = c("linear"),
  ci_methods = c("hsic.gamma"), legacy = FALSE, benchmark = FALSE
)
assert_true(identical(strip_elapsed(campaign_hsic_a)$runs,
                      strip_elapsed(campaign_hsic_b)$runs),
            "HSIC validation campaign run rows should repeat except timing")

cat("test_fastkpc_reproducibility.R: PASS\n")
