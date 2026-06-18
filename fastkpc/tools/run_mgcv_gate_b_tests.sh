#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

Rscript fastkpc/tests/test_cuda_build_lock_contract.R
Rscript fastkpc/tests/test_mgcv_compat_contract.R
Rscript fastkpc/tests/test_mgcv_extract_setup_contract.R
Rscript fastkpc/tests/test_mgcv_penalty_assembly.R
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
Rscript fastkpc/tests/test_mgcv_extract_gcv_bridge.R
Rscript fastkpc/tests/test_mgcv_extract_batch_cpu.R
Rscript fastkpc/tests/test_mgcv_extract_capabilities.R
Rscript fastkpc/tests/test_mgcv_extract_gpu_fixed_sp_api.R
Rscript fastkpc/tests/test_mgcv_extract_gpu_setup_handle.R
Rscript fastkpc/tests/test_mgcv_extract_gpu_handle_solve.R
Rscript fastkpc/tests/test_mgcv_extract_gpu_handle_batch_solve.R
Rscript fastkpc/tests/test_mgcv_extract_gpu_native_fixed_sp.R
Rscript fastkpc/tests/test_mgcv_extract_gpu_native_batch_bridge.R
Rscript fastkpc/tests/test_mgcv_extract_gpu_same_setup_batch.R
Rscript fastkpc/tests/test_mgcv_extract_gpu_gcv_cpu_fallback.R
Rscript fastkpc/tests/test_mgcv_extract_gpu_gcv_single_penalty.R
Rscript fastkpc/tests/test_mgcv_self_solve_purity.R
Rscript fastkpc/tests/test_mgcv_gate_b_campaign.R
Rscript fastkpc/tests/test_hybrid_canonical_replay.R
Rscript fastkpc/tests/test_hybrid_graph_replay_policy.R
Rscript fastkpc/tests/test_hybrid_golden_snapshots.R
Rscript fastkpc/tests/test_hybrid_calibration_campaign.R
Rscript fastkpc/tests/test_mgcv_extract_gpu_graph_campaign.R
Rscript fastkpc/tests/test_tprs_approx_decision_memo.R
