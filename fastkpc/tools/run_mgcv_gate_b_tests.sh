#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

Rscript fastkpc/tests/test_mgcv_compat_contract.R
Rscript fastkpc/tests/test_mgcv_extract_setup_contract.R
Rscript fastkpc/tests/test_mgcv_penalty_assembly.R
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
Rscript fastkpc/tests/test_mgcv_extract_gcv_bridge.R
Rscript fastkpc/tests/test_mgcv_extract_batch_cpu.R
