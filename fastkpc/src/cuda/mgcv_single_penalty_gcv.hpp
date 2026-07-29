#ifndef FASTKPC_MGCV_SINGLE_PENALTY_GCV_HPP
#define FASTKPC_MGCV_SINGLE_PENALTY_GCV_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace fastkpc {

constexpr char kSinglePenaltyGcvCudaSchemaVersion[] =
  "full-cuda-ci-single-penalty-gcv-cuda-v1";
constexpr char kSinglePenaltyGcvCudaMultiSchemaVersion[] =
  "full-cuda-ci-single-penalty-gcv-cuda-multi-v1";
constexpr int kSinglePenaltyGcvTranscriptCapacity = 512;
constexpr int kSinglePenaltyGcvMaximumConcurrentSetups = 16;

enum class SinglePenaltyGcvTermination : int {
  ScoreAndGradient = 0,
  StepHalvingExhausted = 1,
  FlatObjective = 2,
  IterationLimit = 3,
  NonfiniteObjective = 4,
  TranscriptOverflow = 5
};

enum class SinglePenaltyGcvTranscriptStage : int {
  Refinement = 0,
  IterationState = 1,
  BoundaryProbe = 2
};

enum class SinglePenaltyGcvStepSource : int {
  Newton = 0,
  SteepestDescent = 1,
  SteepestDescentAfterNewtonRejection = 2,
  InfinityProbe = 3
};

struct SinglePenaltyGcvGridCell {
  double rss = 0.0;
  double edf = 0.0;
  double score = 0.0;
  int valid = 0;
};

struct SinglePenaltyGcvOptimizerResult {
  double sp = 0.0;
  double log_sp = 0.0;
  double rss = 0.0;
  double edf = 0.0;
  double score = 0.0;
  double gradient = 0.0;
  double hessian = 0.0;
  double reported_rms_gradient = 0.0;
  double pre_boundary_log_sp = 0.0;
  int iteration_count = 0;
  int score_call_count = 0;
  int actual_objective_call_count = 0;
  int fully_converged = 0;
  int hessian_positive_definite = 0;
  int boundary_probe_count = 0;
  int boundary_accepted_count = 0;
  int termination = 0;
};

struct SinglePenaltyGcvTranscriptEntry {
  double current_log_sp = 0.0;
  double proposed_step = 0.0;
  double trial_log_sp = 0.0;
  double objective = 0.0;
  double gradient = 0.0;
  double hessian = 0.0;
  int stage = 0;
  int iteration = 0;
  int evaluation = 0;
  int accepted = 0;
  int step_source = 0;
};

struct SinglePenaltyGcvCudaDiagnostics {
  std::string schema_version;
  std::string sp_selection_backend_executed;
  std::string gcv_score_backend_executed;
  std::string optimizer_backend_executed;
  std::string exact_replay_backend_executed;
  int device_id = -1;
  std::string gpu_name;
  int n = 0;
  int coefficient_dim = 0;
  int target_count = 0;
  int candidate_count = 0;
  int penalty_rank = 0;
  int penalty_nullity = 0;
  int cuda_gcv_batches = 0;
  int cuda_gcv_targets = 0;
  int cuda_gcv_iterations = 0;
  int cuda_gcv_nonconverged = 0;
  int cuda_gcv_boundary_targets = 0;
  int legacy_mgcv_target_calls = 0;
  int cpu_score_count = 0;
  int cpu_optimizer_count = 0;
  int fallback_count = 0;
  int projection_gemm_count = 0;
  int mgcv_qt_y_kernel_launch_count = 0;
  int grid_kernel_launch_count = 0;
  int exact_mroot_kernel_launch_count = 0;
  int exact_svd_call_count = 0;
  int exact_svd_nonconverged_count = 0;
  int exact_objective_kernel_launch_count = 0;
  int exact_endpoint_kernel_launch_count = 0;
  int exact_endpoint_comparison_count = 0;
  int exact_endpoint_svd_call_count = 0;
  int exact_endpoint_failure_count = 0;
  int exact_endpoint_trial_accepted_count = 0;
  int exact_derivative_refresh_count = 0;
  int exact_derivative_svd_call_count = 0;
  int exact_derivative_failure_count = 0;
  int spectral_optimizer_target_count = 0;
  int spectral_only_target_count = 0;
  int exact_replay_target_count = 0;
  int exact_replay_endpoint_risk_count = 0;
  int exact_replay_convergence_risk_count = 0;
  int exact_replay_boundary_risk_count = 0;
  int exact_replay_numerical_risk_count = 0;
  int selective_replay_target_count = 0;
  int selective_replay_deferred_count = 0;
  int selective_replay_steepest_count = 0;
  int selective_replay_residual_risk_count = 0;
  int selective_replay_other_count = 0;
  int optimizer_kernel_launch_count = 0;
  int augmented_eigensolver_call_count = 0;
  int augmented_objective_kernel_launch_count = 0;
  int augmented_optimizer_control_sync_count = 0;
  int h2d_copy_count = 0;
  std::size_t h2d_bytes = 0;
  int compact_d2h_count = 0;
  std::size_t compact_d2h_bytes = 0;
  int grid_d2h_count = 0;
  std::size_t grid_d2h_bytes = 0;
  int transcript_d2h_count = 0;
  std::size_t transcript_d2h_bytes = 0;
  int device_allocation_count = 0;
  int stream_ordered_allocation_count = 0;
  int synchronous_allocation_count = 0;
  std::size_t device_allocation_bytes = 0;
  double upload_cuda_ms = 0.0;
  double projection_cuda_ms = 0.0;
  double grid_cuda_ms = 0.0;
  double optimizer_cuda_ms = 0.0;
  double d2h_cuda_ms = 0.0;
  double total_host_ms = 0.0;
  bool cublas_pedantic_math = false;
  bool cublas_atomics_not_allowed = false;
  bool cublas_user_workspace_installed = false;
  bool score_matrix_materialized = false;
  bool transcript_materialized = false;
  bool target_rhs_projected_on_cuda = false;
  bool target_selection_on_cuda = false;
  bool optimizer_target_coverage_complete = false;
};

struct SinglePenaltyGcvCudaResult {
  std::string schema_version;
  int n = 0;
  int coefficient_dim = 0;
  int target_count = 0;
  int candidate_count = 0;
  std::vector<SinglePenaltyGcvOptimizerResult> targets;
  std::vector<SinglePenaltyGcvGridCell> grid;
  std::vector<SinglePenaltyGcvTranscriptEntry> transcript;
  std::vector<int> transcript_counts;
  std::vector<int> transcript_overflow;
  std::vector<int> deferred_exact_replay_flags;
  SinglePenaltyGcvCudaDiagnostics diagnostics;
};

struct SinglePenaltyMrootCudaResult {
  int coefficient_dim = 0;
  int requested_rank = 0;
  int candidate_count = 0;
  std::vector<double> roots;
  std::vector<int> ranks;
  std::vector<int> pivots;
};

struct SinglePenaltyGcvCudaOwnedInput {
  std::vector<double> X;
  std::vector<double> Y;
  std::vector<double> rhs_transform;
  std::vector<double> eigenvalues;
  std::vector<double> magic_qr_packed;
  std::vector<double> magic_tau;
  std::vector<double> magic_r;
  std::vector<double> magic_penalty_root;
  std::vector<double> magic_penalty_matrix;
  std::vector<int> target_ids;
  int n = 0;
  int coefficient_dim = 0;
  int target_count = 0;
  int penalty_rank = 0;
  double initial_sp = 0.0;
  std::vector<double> sp_grid;
  bool materialize_grid = false;
  bool keep_transcript = false;
};

struct SinglePenaltyGcvCudaMultiDiagnostics {
  std::string schema_version;
  std::string execution_strategy;
  int device_id = -1;
  int setup_count = 0;
  int requested_concurrency = 0;
  int worker_count = 0;
  int worker_device_bind_count = 0;
  int max_host_calls_in_flight = 0;
  int fused_exact_replay_target_count = 0;
  int fused_exact_replay_kernel_launch_count = 0;
  std::size_t fused_exact_replay_device_bytes = 0;
  double setup_host_ms_sum = 0.0;
  double spectral_setup_wall_ms = 0.0;
  double fused_exact_replay_cuda_ms = 0.0;
  double fused_exact_replay_host_ms = 0.0;
  double wall_host_ms = 0.0;
  double host_overlap_factor = 0.0;
  bool fused_exact_replay_executed = false;
};

struct SinglePenaltyGcvCudaMultiResult {
  std::string schema_version;
  std::vector<SinglePenaltyGcvCudaResult> setups;
  SinglePenaltyGcvCudaMultiDiagnostics diagnostics;
};

SinglePenaltyMrootCudaResult single_penalty_mroot_cuda(
  const double* penalty_matrix,
  int coefficient_dim,
  int penalty_rank,
  const std::vector<double>& log_sp);

SinglePenaltyGcvCudaResult single_penalty_gcv_cuda(
  const double* X,
  const double* Y,
  const double* rhs_transform,
  const double* eigenvalues,
  const double* magic_qr_packed,
  const double* magic_tau,
  const double* magic_r,
  const double* magic_penalty_root,
  const double* magic_penalty_matrix,
  const int* target_ids,
  int n,
  int coefficient_dim,
  int target_count,
  int penalty_rank,
  double initial_sp,
  const std::vector<double>& sp_grid,
  bool materialize_grid,
  bool keep_transcript,
  bool defer_exact_replay = false);

SinglePenaltyGcvCudaMultiResult single_penalty_gcv_cuda_multi(
  const std::vector<SinglePenaltyGcvCudaOwnedInput>& inputs,
  int requested_concurrency);

const char* single_penalty_gcv_termination_name(int value);
const char* single_penalty_gcv_transcript_stage_name(int value);
const char* single_penalty_gcv_step_source_name(int value);

}  // namespace fastkpc

#endif
