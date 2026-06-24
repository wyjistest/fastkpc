args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  file.path("fastkpc", "artifacts", "kpc_tprs_residual_cpp_qualification")
}

source("fastkpc/R/kpc_tprs_residual_cpp_qualification.R")

fastkpc_env_int <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (!nzchar(value)) return(default)
  as.integer(value)
}

fastkpc_env_bool <- function(name, default) {
  value <- tolower(Sys.getenv(name, ""))
  if (!nzchar(value)) return(default)
  value %in% c("1", "true", "yes", "y")
}

result <- fastkpc_run_kpc_tprs_residual_cpp_qualification(
  output_dir = output_dir,
  repeats = fastkpc_env_int("FASTKPC_KPC_TPRS_REPEATS", 3L),
  real_data_path = Sys.getenv("FASTKPC_KPC_TPRS_REAL_DATA", ""),
  no_oracle_check = fastkpc_env_bool("FASTKPC_KPC_TPRS_NO_ORACLE", TRUE)
)

cat("wrote kpcTprsResidualCPP qualification artifacts:", output_dir, "\n")
print(result$qualification_summary)
if (!isTRUE(result$summary$passed)) {
  quit(save = "no", status = 1)
}
