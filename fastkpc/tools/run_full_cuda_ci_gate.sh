#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_full_cuda_ci_one_call_cache.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_hardening.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_phase10_stream_determinism.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_campaign_helpers.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_holdout_helpers.R
bash fastkpc/tools/run_full_cuda_ci_phase10_campaign.sh
Rscript fastkpc/tests/test_full_cuda_ci_phase10_campaign_artifact.R

if [[ ! -f fastkpc/artifacts/full_cuda_ci/sealed_promotion_holdout_v1/manifest.json ]]; then
  if [[ -z "${FASTKPC_PROMOTION_HOLDOUT_RELEASE_DIR:-}" || \
        -z "${FASTKPC_PROMOTION_HOLDOUT_RELEASE_TOKEN:-}" ]]; then
    echo "Phase 10 gate: sealed holdout release envelope/token is missing" >&2
    exit 1
  fi
  taskset -c "${FASTKPC_PHASE10_CPU_AFFINITY:-0-19}" \
    Rscript fastkpc/tools/run_full_cuda_ci_phase10_holdout.R release
fi

Rscript fastkpc/tools/run_full_cuda_ci_phase10_holdout.R validate
Rscript fastkpc/tests/test_full_cuda_ci_phase10_holdout_artifact.R
