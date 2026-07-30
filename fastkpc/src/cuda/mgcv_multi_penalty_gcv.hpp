#ifndef FASTKPC_MGCV_MULTI_PENALTY_GCV_HPP
#define FASTKPC_MGCV_MULTI_PENALTY_GCV_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fastkpc {

constexpr int kMultiPenaltyGcvMaximumCoefficientDim = 64;
constexpr int kMultiPenaltyGcvMaximumPenaltyCount = 7;
constexpr int kMultiPenaltyGcvMaximumConcurrentSetups = 32;

class MultiPenaltyGcvCudaPrepared;
class MultiPenaltyGcvCudaResidualBatch;

struct MultiPenaltyGcvCudaDiagnostics {
  std::string schema_version;
  std::string execution_strategy;
  int device_id = -1;
  std::string gpu_name;
  int prepared_setup_upload_count = 0;
  int target_batch_upload_count = 0;
  int cuda_qt_y_kernel_launch_count = 0;
  int cuda_objective_kernel_launch_count = 0;
  int cuda_objective_target_count = 0;
  int cuda_optimizer_kernel_launch_count = 0;
  int cuda_optimizer_target_count = 0;
  int cuda_optimizer_objective_count = 0;
  std::uint64_t cuda_penalty_factor_augmentation_cycles = 0;
  std::uint64_t cuda_qr_svd_cycles = 0;
  std::uint64_t cuda_qr_bidiagonal_reduction_cycles = 0;
  std::uint64_t cuda_bidiagonal_svd_cycles = 0;
  std::uint64_t cuda_svd_vector_postback_cycles = 0;
  std::uint64_t cuda_left_vector_product_cycles = 0;
  std::uint64_t cuda_score_construction_cycles = 0;
  std::uint64_t cuda_derivative_hessian_cycles = 0;
  int cuda_complete_evaluation_count = 0;
  int cuda_score_only_evaluation_count = 0;
  int cuda_guarded_qr_evaluation_count = 0;
  int cuda_stable_svd_evaluation_count = 0;
  int cuda_selected_evaluation_reuse_count = 0;
  int cuda_stability_replay_kernel_launch_count = 0;
  int cuda_stability_merge_kernel_launch_count = 0;
  int cuda_stability_replay_target_count = 0;
  int cuda_stability_replay_selected_count = 0;
  int cuda_stability_replay_error_count = 0;
  int cuda_stability_replay_discarded_complete_evaluation_count = 0;
  int cuda_stability_replay_discarded_score_only_evaluation_count = 0;
  int cuda_stability_replay_discarded_guarded_qr_evaluation_count = 0;
  int cuda_stability_replay_discarded_stable_svd_evaluation_count = 0;
  std::uint64_t cuda_stability_replay_discarded_cycles = 0;
  double cuda_stability_replay_max_log_sp_spread = 0.0;
  int cuda_terminal_boundary_confirmation_count = 0;
  int cuda_terminal_boundary_confirmation_accepted_count = 0;
  int cuda_terminal_boundary_confirmation_rejected_count = 0;
  int cuda_terminal_boundary_confirmation_strong_delta_accepted_count = 0;
  int cuda_terminal_boundary_confirmation_identity_tie_accepted_count = 0;
  int cuda_terminal_boundary_confirmation_complete_evaluation_count = 0;
  int cuda_terminal_boundary_confirmation_stable_svd_evaluation_count = 0;
  std::uint64_t cuda_terminal_boundary_confirmation_cycles = 0;
  double cuda_terminal_boundary_confirmation_max_identity_disagreement = 0.0;
  double cuda_terminal_boundary_confirmation_max_identity_ratio = 0.0;
  double cuda_terminal_boundary_confirmation_max_delta_disagreement = 0.0;
  double cuda_terminal_boundary_confirmation_max_delta_ratio = 0.0;
  int cuda_hessian_eigensolver_count = 0;
  int cuda_selected_fit_count = 0;
  int cpu_objective_count = 0;
  int cpu_optimizer_count = 0;
  int cpu_multi_penalty_solve_count = 0;
  int fallback_count = 0;
  int cuda_error_count = 0;
  int svd_nonconverged_count = 0;
  int aggregate_rank_failure_count = 0;
  int device_allocation_count = 0;
  int h2d_copy_count = 0;
  int d2h_copy_count = 0;
  bool target_specific_log_sp = false;
  bool true_batched_kernel = false;
  bool independent_target_states = false;
  bool normal_equations_used = false;
  bool double_precision = true;
  double total_host_ms = 0.0;
};

struct MultiPenaltyGcvCudaEvaluation {
  std::string schema_version;
  std::string rank_path;
  int n = 0;
  int coefficient_dim = 0;
  int penalty_count = 0;
  int target_count = 0;
  std::vector<double> rss;
  std::vector<double> edf;
  std::vector<double> score;
  std::vector<double> condition;
  std::vector<int> aggregate_penalty_rank;
  std::vector<int> numerical_rank;
  std::vector<int> solver_info;
  std::vector<double> gradient;
  std::vector<double> hessian;
  std::vector<double> coefficients;
  MultiPenaltyGcvCudaDiagnostics diagnostics;
};

struct MultiPenaltyGcvCudaOptimizerControl {
  double convergence_tolerance = 1e-7;
  int max_step_halving = 25;
  int max_iterations = 400;
  double max_newton_step = 5.0;
  double boundary_probe_step = 2.0;
  int max_boundary_probes = 5;
  double rank_tolerance = 1.4901161193847656e-8;
};

struct MultiPenaltyGcvCudaOptimization {
  std::string schema_version;
  std::string rank_path;
  std::string optimizer_path;
  int n = 0;
  int coefficient_dim = 0;
  int penalty_count = 0;
  int target_count = 0;
  std::vector<double> initial_log_sp;
  std::vector<double> selected_log_sp;
  std::vector<double> rss;
  std::vector<double> edf;
  std::vector<double> score;
  std::vector<double> condition;
  std::vector<double> gradient;
  std::vector<double> hessian;
  std::vector<double> coefficients;
  std::vector<double> rms_gradient;
  std::vector<double> hessian_eigenvalues;
  std::vector<int> aggregate_penalty_rank;
  std::vector<int> numerical_rank;
  std::vector<int> solver_info;
  std::vector<int> optimizer_iterations;
  std::vector<int> score_calls;
  std::vector<int> objective_calls;
  std::vector<int> step_halving_count;
  std::vector<int> newton_trial_count;
  std::vector<int> steepest_descent_trial_count;
  std::vector<int> boundary_probe_count;
  std::vector<int> boundary_accepted_count;
  std::vector<int> boundary_status;
  std::vector<int> fully_converged;
  std::vector<int> hessian_positive_definite;
  std::vector<int> step_failed;
  std::vector<int> optimizer_status;
  MultiPenaltyGcvCudaDiagnostics diagnostics;
};

struct MultiPenaltyGcvCudaPreparedInfo {
  int device_id = -1;
  std::string gpu_name;
  int n = 0;
  int coefficient_dim = 0;
  int penalty_count = 0;
  int target_capacity = 0;
  int setup_upload_count = 0;
  std::size_t setup_h2d_bytes = 0;
  int device_allocation_count = 0;
  int workspace_grow_count = 0;
  int solve_count = 0;
  int cublas_gemm_count = 0;
  int residual_kernel_count = 0;
  int residual_shadow_d2h_count = 0;
  std::size_t residual_shadow_d2h_bytes = 0;
  bool residual_slot_leased = false;
  std::uint64_t generation = 0;
};

struct MultiPenaltyGcvCudaResidualInfo {
  int n = 0;
  int coefficient_dim = 0;
  int target_count = 0;
  int device_id = -1;
  std::vector<std::string> target_keys;
  std::vector<int> optimizer_status;
  bool device_resident = false;
  bool released = false;
  std::uint64_t generation = 0;
};

struct MultiPenaltyGcvCudaResidualConsumerView {
  const double* residuals = nullptr;
  int n = 0;
  int target_count = 0;
  int device_id = -1;
  cudaStream_t producer_stream = nullptr;
  cudaEvent_t producer_completion_event = nullptr;
  std::vector<std::string> target_keys;
};

struct MultiPenaltyGcvCudaResidualShadow {
  int n = 0;
  int coefficient_dim = 0;
  int target_count = 0;
  std::vector<double> coefficients;
  std::vector<double> fitted;
  std::vector<double> residuals;
};

struct MultiPenaltyGcvCudaBatchResult {
  MultiPenaltyGcvCudaOptimization optimization;
  std::shared_ptr<MultiPenaltyGcvCudaResidualBatch> residual;
};

struct MultiPenaltyGcvCudaMultiRequest {
  std::shared_ptr<MultiPenaltyGcvCudaPrepared> prepared;
  std::vector<double> Y;
  int n = 0;
  int target_count = 0;
  std::vector<std::string> target_keys;
  MultiPenaltyGcvCudaOptimizerControl control;
};

struct MultiPenaltyGcvCudaMultiDiagnostics {
  std::string schema_version;
  std::string execution_strategy;
  int setup_count = 0;
  int target_count = 0;
  int requested_concurrency = 0;
  int worker_count = 0;
  int max_host_calls_in_flight = 0;
  int setup_stream_count = 0;
  double summed_setup_host_ms = 0.0;
  double wall_host_ms = 0.0;
  double host_overlap_factor = 0.0;
};

struct MultiPenaltyGcvCudaMultiResult {
  std::string schema_version;
  std::vector<MultiPenaltyGcvCudaBatchResult> setups;
  MultiPenaltyGcvCudaMultiDiagnostics diagnostics;
};

MultiPenaltyGcvCudaEvaluation multi_penalty_gcv_evaluate_cuda(
    const double* X,
    const double* Y,
    const double* magic_qr_packed,
    const double* magic_tau,
    const double* magic_r,
    const int* magic_pivot_zero_based,
    const std::vector<std::vector<double>>& penalty_roots,
    const std::vector<std::vector<double>>& penalty_matrices,
    const std::vector<int>& penalty_ranks,
    const double* log_sp,
    int n,
    int coefficient_dim,
    int penalty_count,
    int target_count,
    double rank_tolerance);

MultiPenaltyGcvCudaOptimization multi_penalty_gcv_optimize_cuda(
    const double* X,
    const double* Y,
    const double* magic_qr_packed,
    const double* magic_tau,
    const double* magic_r,
    const int* magic_pivot_zero_based,
    const std::vector<std::vector<double>>& penalty_roots,
    const std::vector<std::vector<double>>& penalty_matrices,
    const std::vector<int>& penalty_ranks,
    const double* initial_log_sp,
    int n,
    int coefficient_dim,
    int penalty_count,
    int target_count,
    const MultiPenaltyGcvCudaOptimizerControl& control);

std::shared_ptr<MultiPenaltyGcvCudaPrepared>
create_multi_penalty_gcv_cuda_prepared(
    const double* X,
    const double* magic_qr_packed,
    const double* magic_tau,
    const double* magic_r,
    const int* magic_pivot_zero_based,
    const std::vector<std::vector<double>>& penalty_roots,
    const std::vector<std::vector<double>>& penalty_matrices,
    const std::vector<int>& penalty_ranks,
    const double* initial_log_sp,
    int n,
    int coefficient_dim,
    int penalty_count,
    int target_capacity,
    int device_id);

MultiPenaltyGcvCudaPreparedInfo multi_penalty_gcv_cuda_prepared_info(
    const std::shared_ptr<MultiPenaltyGcvCudaPrepared>& prepared);

MultiPenaltyGcvCudaBatchResult multi_penalty_gcv_cuda_optimize_batch(
    const std::shared_ptr<MultiPenaltyGcvCudaPrepared>& prepared,
    const double* Y,
    int n,
    int target_count,
    const std::vector<std::string>& target_keys,
    const MultiPenaltyGcvCudaOptimizerControl& control);

MultiPenaltyGcvCudaMultiResult multi_penalty_gcv_cuda_optimize_multi(
    std::vector<MultiPenaltyGcvCudaMultiRequest> requests,
    int requested_concurrency);

MultiPenaltyGcvCudaResidualInfo multi_penalty_gcv_cuda_residual_info(
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>& residual);

MultiPenaltyGcvCudaResidualConsumerView
acquire_multi_penalty_gcv_cuda_residual_consumer_view(
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>& residual);

MultiPenaltyGcvCudaResidualShadow
materialize_multi_penalty_gcv_cuda_residual_shadow(
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>& residual);

void release_multi_penalty_gcv_cuda_residual(
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>& residual);

}  // namespace fastkpc

#endif
