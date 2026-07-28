source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = function(error) error)
  assert_true(inherits(error, "error"), message)
  assert_true(
    grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, ": unexpected error: ", conditionMessage(error))
  )
}

load_fastkpc_cuda_native(rebuild = FALSE)
before <- .Call(
  "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
)

contracts <- fastkpc_full_cuda_phase35_load_contract_set()
for (name in names(contracts)) {
  contract <- contracts[[name]]
  native <- fastkpc_full_cuda_phase35_native_contract_identity(
    contract$path, expected_contract_name = name
  )
  assert_true(
    identical(
      names(native),
      c(
        "contract_name", "contract_schema_version", "semantic_major",
        "semantic_minor", "semantic_patch", "canonical_json", "sha256"
      )
    ) &&
      identical(native$contract_name, name) &&
      identical(native$contract_schema_version,
                "full-cuda-ci-tracked-contract-v1") &&
      identical(native$semantic_major, 1L) &&
      identical(native$semantic_minor, 0L) &&
      identical(native$semantic_patch, 0L) &&
      identical(native$canonical_json, contract$canonical_json) &&
      identical(native$sha256, contract$sha256),
    paste0("native contract identity must match R for ", name)
  )
}

assert_true(
  identical(
    fastkpc_full_cuda_phase35_native_sha256_utf8("abc"),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
  ),
  "native SHA-256 must pass the standard abc vector"
)

fixture <- tempfile(fileext = ".json")
on.exit(unlink(fixture), add = TRUE)
writeLines(
  c(
    "{",
    "  \"semantic_version\": {\"patch\": 0, \"minor\": 0, \"major\": 1},",
    "  \"payload\": {\"z\": 2, \"a\": \"\\u0061\"},",
    "  \"contract_schema_version\": \"full-cuda-ci-tracked-contract-v1\",",
    "  \"contract_name\": \"native_fixture_v1\",",
    "  \"campaign\": \"full-cuda-legacy-compatible-ci\",",
    "  \"phase_introduced\": \"3.5\"",
    "}"
  ),
  fixture, useBytes = TRUE
)
native_fixture <- fastkpc_full_cuda_phase35_native_contract_identity(
  fixture, "native_fixture_v1"
)
r_fixture <- fastkpc_full_cuda_phase35_contract_identity_from_path(
  fixture, "native_fixture_v1", validate_known_contract = FALSE
)
assert_true(
  identical(native_fixture$canonical_json, r_fixture$canonical_json) &&
    identical(native_fixture$sha256, r_fixture$sha256),
  "native and R canonicalization must agree for reordered escaped JSON"
)

writeLines('{"a":1,"a":2}', fixture, useBytes = TRUE)
assert_error(
  fastkpc_full_cuda_phase35_native_contract_identity(fixture, "duplicate"),
  "duplicate object key",
  "native parser must reject duplicate keys"
)
writeLines('{"a":1.5}', fixture, useBytes = TRUE)
assert_error(
  fastkpc_full_cuda_phase35_native_contract_identity(fixture, "fractional"),
  "integer JSON numbers",
  "native parser must reject fractional JSON numbers"
)
writeLines('{"contract_name":"actual"}', fixture, useBytes = TRUE)
assert_error(
  fastkpc_full_cuda_phase35_native_contract_identity(fixture, "expected"),
  "contract name mismatch",
  "native parser must bind the expected contract name"
)

after <- .Call(
  "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
)
assert_true(
  identical(before, after),
  "native tracked-contract validation must create no CUDA resources"
)

cat("PASS full CUDA CI Phase 3.5 native contract parser and hash parity\n")
