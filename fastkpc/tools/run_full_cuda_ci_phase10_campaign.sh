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

staging_dir=${FASTKPC_PHASE10_CAMPAIGN_STAGING_DIR:-fastkpc/artifacts/full_cuda_ci/phase10_promotion_staging_default_inf_v2}
artifact_dir=${FASTKPC_PHASE10_CAMPAIGN_ARTIFACT_DIR:-fastkpc/artifacts/full_cuda_ci/promotion_351x48_default_inf_v2}
affinity=${FASTKPC_PHASE10_CPU_AFFINITY:-0-19}

run_r() {
  taskset -c "${affinity}" Rscript "$@"
}

run_r fastkpc/tools/run_full_cuda_ci_phase10_campaign.R \
  prepare "${staging_dir}" "${artifact_dir}"

run_specs=(
  "candidate_cold 1"
  "candidate_cold 2"
  "candidate_cold 3"
  "candidate_cold 4"
  "candidate_cold 5"
  "candidate_warm 1"
  "correct_baseline 1"
  "correct_baseline 2"
  "candidate_warm 2"
  "candidate_warm 3"
  "correct_baseline 3"
  "correct_baseline 4"
  "candidate_warm 4"
  "candidate_warm 5"
  "correct_baseline 5"
)

for spec in "${run_specs[@]}"; do
  read -r mode repetition <<<"${spec}"
  run_r fastkpc/tools/run_full_cuda_ci_phase10_worker.R \
    "${mode}" "${repetition}" "${staging_dir}"
done

run_r fastkpc/tools/run_full_cuda_ci_phase10_campaign.R \
  publish "${staging_dir}" "${artifact_dir}"
