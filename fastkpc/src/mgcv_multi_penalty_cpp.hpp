#ifndef FASTKPC_MGCV_MULTI_PENALTY_CPP_HPP
#define FASTKPC_MGCV_MULTI_PENALTY_CPP_HPP

#include <string>
#include <vector>

namespace fastkpc {

struct MultiPenaltyGcvEvaluation {
  int n = 0;
  int coefficient_dim = 0;
  int free_dim = 0;
  int penalty_count = 0;
  int constraint_rank = 0;
  int augmented_penalty_rank = 0;
  int numerical_rank = 0;
  double rss = 0.0;
  double edf = 0.0;
  double score = 0.0;
  double condition = 0.0;
  std::string condition_bucket;
  std::vector<double> log_sp;
  std::vector<double> gradient;
  std::vector<double> hessian;
  std::vector<double> coefficients;
  std::vector<double> fitted;
  std::vector<double> residuals;
};

struct MultiPenaltyOptimizerControl {
  double convergence_tolerance = 1e-7;
  int max_step_halving = 25;
  int max_iterations = 400;
  double max_newton_step = 5.0;
  double boundary_probe_step = 2.0;
  int max_boundary_probes = 5;
  double rank_tolerance = 1.4901161193847656e-8;
  bool keep_transcript = false;
};

struct MultiPenaltyOptimizerTranscriptEntry {
  std::string stage;
  int iteration = 0;
  int evaluation = 0;
  int coordinate = -1;
  std::vector<double> current_log_sp;
  std::vector<double> proposed_step;
  std::vector<double> trial_log_sp;
  double score = 0.0;
  std::vector<double> gradient;
  std::vector<double> hessian;
  std::vector<double> hessian_eigenvalues;
  bool accepted = false;
  std::string step_source;
  int numerical_rank = 0;
  double condition = 0.0;
};

struct MultiPenaltyGcvOptimization {
  MultiPenaltyGcvEvaluation selected;
  std::vector<double> initial_log_sp;
  int optimizer_iterations = 0;
  int score_calls = 0;
  int objective_calls = 0;
  int step_halving_count = 0;
  int newton_trial_count = 0;
  int steepest_descent_trial_count = 0;
  int boundary_probe_count = 0;
  int boundary_accepted_count = 0;
  bool fully_converged = false;
  bool hessian_positive_definite = false;
  bool step_failed = false;
  double rms_gradient = 0.0;
  std::string convergence_code;
  std::vector<std::string> boundary_status;
  std::vector<MultiPenaltyOptimizerTranscriptEntry> transcript;
};

MultiPenaltyGcvEvaluation multi_penalty_gcv_evaluate_cpp(
    const double* X,
    const double* y,
    int n,
    int coefficient_dim,
    const std::vector<std::vector<double>>& penalty_blocks,
    const std::vector<int>& penalty_dimensions,
    const std::vector<int>& penalty_offsets,
    const std::vector<int>& penalty_ranks,
    const std::vector<double>& log_sp,
    const double* H,
    bool has_H,
    const double* constraint,
    int constraint_rows,
    double rank_tolerance);

MultiPenaltyGcvOptimization multi_penalty_gcv_optimize_cpp(
    const double* X,
    const double* y,
    int n,
    int coefficient_dim,
    const std::vector<std::vector<double>>& penalty_blocks,
    const std::vector<int>& penalty_dimensions,
    const std::vector<int>& penalty_offsets,
    const std::vector<int>& penalty_ranks,
    const double* H,
    bool has_H,
    const double* constraint,
    int constraint_rows,
    const MultiPenaltyOptimizerControl& control);

}  // namespace fastkpc

#endif
