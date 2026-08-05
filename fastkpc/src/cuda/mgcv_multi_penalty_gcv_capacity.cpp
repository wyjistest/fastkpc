#include "mgcv_multi_penalty_gcv_capacity.hpp"

#include "mgcv_multi_penalty_gcv_extended.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <variant>

namespace fastkpc {

static_assert(1 + 9 * 21 <= 192,
              "the first extended bucket does not cover level 21");
static_assert(1 + 9 * 42 <= 384,
              "the second extended bucket does not cover level 42");
static_assert(1 + 9 * 62 ==
                kMultiPenaltyGcvDefaultKpcMaximumCoefficientDim,
              "the default-kpcalg coefficient envelope changed");

using CapacityHandle = std::variant<
  std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>,
  std::shared_ptr<MultiPenaltyGcvCudaPrepared>,
  std::shared_ptr<MultiPenaltyGcvCudaPreparedext192>,
  std::shared_ptr<MultiPenaltyGcvCudaPreparedext384>,
  std::shared_ptr<MultiPenaltyGcvCudaPreparedext559>>;

class MultiPenaltyGcvCapacityPrepared {
 public:
  MultiPenaltyGcvCapacityPrepared(
      MultiPenaltyGcvCapacityBucket bucket_value,
      CapacityHandle handle_value)
      : bucket(bucket_value), handle(std::move(handle_value)) {}

  MultiPenaltyGcvCapacityBucket bucket;
  CapacityHandle handle;
};

namespace {

int maximum_concurrency_for_bucket(MultiPenaltyGcvCapacityBucket bucket) {
  switch (bucket) {
    case MultiPenaltyGcvCapacityBucket::Small64:
    case MultiPenaltyGcvCapacityBucket::Base80:
      return 64;
    case MultiPenaltyGcvCapacityBucket::Extended192:
      return 8;
    case MultiPenaltyGcvCapacityBucket::Extended384:
      return 2;
    case MultiPenaltyGcvCapacityBucket::Extended559:
      return 1;
  }
  throw std::runtime_error("unknown multi-penalty CUDA capacity bucket");
}

template <typename BackendResult, typename Release>
MultiPenaltyGcvCudaOptimization consume_batch_result(
    BackendResult result, Release release) {
  if (!result.residual) {
    throw std::runtime_error(
      "capacity-dispatched multi-penalty optimizer returned no residual token");
  }
  release(result.residual);
  result.residual.reset();
  return std::move(result.optimization);
}

template <typename BackendRequest, typename Prepared, typename Optimize,
          typename Release>
MultiPenaltyGcvCapacityMultiResult run_multi_backend(
    std::vector<MultiPenaltyGcvCapacityRequest> requests,
    MultiPenaltyGcvCapacityBucket expected_bucket,
    int requested_concurrency,
    Optimize optimize,
    Release release) {
  std::vector<BackendRequest> backend_requests;
  backend_requests.reserve(requests.size());
  for (MultiPenaltyGcvCapacityRequest& request : requests) {
    if (!request.prepared || request.prepared->bucket != expected_bucket) {
      throw std::runtime_error(
        "capacity-dispatched multi-penalty request mixes backend buckets");
    }
    BackendRequest backend;
    backend.prepared = std::get<std::shared_ptr<Prepared>>(
      request.prepared->handle);
    backend.Y = std::move(request.Y);
    backend.n = request.n;
    backend.target_count = request.target_count;
    backend.target_keys = std::move(request.target_keys);
    backend.control = request.control;
    backend_requests.push_back(std::move(backend));
  }

  auto backend_result = optimize(
    std::move(backend_requests), requested_concurrency);
  MultiPenaltyGcvCapacityMultiResult output;
  output.diagnostics = std::move(backend_result.diagnostics);
  output.setups.reserve(backend_result.setups.size());
  for (auto& setup : backend_result.setups) {
    if (!setup.residual) {
      throw std::runtime_error(
        "capacity-dispatched multi-penalty setup has no residual token");
    }
    release(setup.residual);
    setup.residual.reset();
    output.setups.push_back(std::move(setup.optimization));
  }
  return output;
}

}  // namespace

MultiPenaltyGcvCapacityBucket multi_penalty_gcv_capacity_bucket(
    int coefficient_dim, int penalty_count) {
  if (coefficient_dim <= 0 || penalty_count <= 1) {
    throw std::runtime_error(
      "multi-penalty CUDA capacity request is invalid");
  }
  if (coefficient_dim <= 64 && penalty_count <= 7) {
    return MultiPenaltyGcvCapacityBucket::Small64;
  }
  if (coefficient_dim <= 80 && penalty_count <= 8) {
    return MultiPenaltyGcvCapacityBucket::Base80;
  }
  if (coefficient_dim <= 192 && penalty_count <= 21) {
    return MultiPenaltyGcvCapacityBucket::Extended192;
  }
  if (coefficient_dim <= 384 && penalty_count <= 42) {
    return MultiPenaltyGcvCapacityBucket::Extended384;
  }
  if (coefficient_dim <= kMultiPenaltyGcvDefaultKpcMaximumCoefficientDim &&
      penalty_count <= kMultiPenaltyGcvDefaultKpcMaximumPenaltyCount) {
    return MultiPenaltyGcvCapacityBucket::Extended559;
  }
  throw std::runtime_error(
    "multi-penalty CUDA shape exceeds the default-kpcalg envelope");
}

int multi_penalty_gcv_capacity_max_concurrent_setups(
    int coefficient_dim, int penalty_count) {
  return maximum_concurrency_for_bucket(
    multi_penalty_gcv_capacity_bucket(coefficient_dim, penalty_count));
}

std::shared_ptr<MultiPenaltyGcvCapacityPrepared>
create_multi_penalty_gcv_capacity_prepared(
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
    int device_id) {
  const MultiPenaltyGcvCapacityBucket bucket =
    multi_penalty_gcv_capacity_bucket(coefficient_dim, penalty_count);
  switch (bucket) {
    case MultiPenaltyGcvCapacityBucket::Small64:
      return std::make_shared<MultiPenaltyGcvCapacityPrepared>(
        bucket, create_multi_penalty_gcv_cuda_prepared_small64(
          X, magic_qr_packed, magic_tau, magic_r,
          magic_pivot_zero_based, penalty_roots, penalty_matrices,
          penalty_ranks, initial_log_sp, n, coefficient_dim,
          penalty_count, target_capacity, device_id));
    case MultiPenaltyGcvCapacityBucket::Base80:
      return std::make_shared<MultiPenaltyGcvCapacityPrepared>(
        bucket, create_multi_penalty_gcv_cuda_prepared(
          X, magic_qr_packed, magic_tau, magic_r,
          magic_pivot_zero_based, penalty_roots, penalty_matrices,
          penalty_ranks, initial_log_sp, n, coefficient_dim,
          penalty_count, target_capacity, device_id));
    case MultiPenaltyGcvCapacityBucket::Extended192:
      return std::make_shared<MultiPenaltyGcvCapacityPrepared>(
        bucket, create_multi_penalty_gcv_cuda_prepared_ext192(
          X, magic_qr_packed, magic_tau, magic_r,
          magic_pivot_zero_based, penalty_roots, penalty_matrices,
          penalty_ranks, initial_log_sp, n, coefficient_dim,
          penalty_count, target_capacity, device_id));
    case MultiPenaltyGcvCapacityBucket::Extended384:
      return std::make_shared<MultiPenaltyGcvCapacityPrepared>(
        bucket, create_multi_penalty_gcv_cuda_prepared_ext384(
          X, magic_qr_packed, magic_tau, magic_r,
          magic_pivot_zero_based, penalty_roots, penalty_matrices,
          penalty_ranks, initial_log_sp, n, coefficient_dim,
          penalty_count, target_capacity, device_id));
    case MultiPenaltyGcvCapacityBucket::Extended559:
      return std::make_shared<MultiPenaltyGcvCapacityPrepared>(
        bucket, create_multi_penalty_gcv_cuda_prepared_ext559(
          X, magic_qr_packed, magic_tau, magic_r,
          magic_pivot_zero_based, penalty_roots, penalty_matrices,
          penalty_ranks, initial_log_sp, n, coefficient_dim,
          penalty_count, target_capacity, device_id));
  }
  throw std::runtime_error("unknown multi-penalty CUDA capacity bucket");
}

MultiPenaltyGcvCudaPreparedInfo multi_penalty_gcv_capacity_prepared_info(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared) {
  if (!prepared) {
    throw std::runtime_error(
      "capacity-dispatched multi-penalty setup has been freed");
  }
  switch (prepared->bucket) {
    case MultiPenaltyGcvCapacityBucket::Small64:
      return multi_penalty_gcv_cuda_prepared_info_small64(
        std::get<std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>>(
          prepared->handle));
    case MultiPenaltyGcvCapacityBucket::Base80:
      return multi_penalty_gcv_cuda_prepared_info(
        std::get<std::shared_ptr<MultiPenaltyGcvCudaPrepared>>(
          prepared->handle));
    case MultiPenaltyGcvCapacityBucket::Extended192:
      return multi_penalty_gcv_cuda_prepared_info_ext192(
        std::get<std::shared_ptr<MultiPenaltyGcvCudaPreparedext192>>(
          prepared->handle));
    case MultiPenaltyGcvCapacityBucket::Extended384:
      return multi_penalty_gcv_cuda_prepared_info_ext384(
        std::get<std::shared_ptr<MultiPenaltyGcvCudaPreparedext384>>(
          prepared->handle));
    case MultiPenaltyGcvCapacityBucket::Extended559:
      return multi_penalty_gcv_cuda_prepared_info_ext559(
        std::get<std::shared_ptr<MultiPenaltyGcvCudaPreparedext559>>(
          prepared->handle));
  }
  throw std::runtime_error("unknown multi-penalty CUDA capacity bucket");
}

MultiPenaltyGcvCudaOptimization multi_penalty_gcv_capacity_optimize_batch(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared,
    const double* Y,
    int n,
    int target_count,
    const std::vector<std::string>& target_keys,
    const MultiPenaltyGcvCudaOptimizerControl& control) {
  if (!prepared) {
    throw std::runtime_error(
      "capacity-dispatched multi-penalty setup has been freed");
  }
  switch (prepared->bucket) {
    case MultiPenaltyGcvCapacityBucket::Small64:
      return consume_batch_result(
        multi_penalty_gcv_cuda_optimize_batch_small64(
          std::get<std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>>(
            prepared->handle),
          Y, n, target_count, target_keys, control),
        release_multi_penalty_gcv_cuda_residual_small64);
    case MultiPenaltyGcvCapacityBucket::Base80:
      return consume_batch_result(
        multi_penalty_gcv_cuda_optimize_batch(
          std::get<std::shared_ptr<MultiPenaltyGcvCudaPrepared>>(
            prepared->handle),
          Y, n, target_count, target_keys, control),
        release_multi_penalty_gcv_cuda_residual);
    case MultiPenaltyGcvCapacityBucket::Extended192:
      return consume_batch_result(
        multi_penalty_gcv_cuda_optimize_batch_ext192(
          std::get<std::shared_ptr<MultiPenaltyGcvCudaPreparedext192>>(
            prepared->handle),
          Y, n, target_count, target_keys, control),
        release_multi_penalty_gcv_cuda_residual_ext192);
    case MultiPenaltyGcvCapacityBucket::Extended384:
      return consume_batch_result(
        multi_penalty_gcv_cuda_optimize_batch_ext384(
          std::get<std::shared_ptr<MultiPenaltyGcvCudaPreparedext384>>(
            prepared->handle),
          Y, n, target_count, target_keys, control),
        release_multi_penalty_gcv_cuda_residual_ext384);
    case MultiPenaltyGcvCapacityBucket::Extended559:
      return consume_batch_result(
        multi_penalty_gcv_cuda_optimize_batch_ext559(
          std::get<std::shared_ptr<MultiPenaltyGcvCudaPreparedext559>>(
            prepared->handle),
          Y, n, target_count, target_keys, control),
        release_multi_penalty_gcv_cuda_residual_ext559);
  }
  throw std::runtime_error("unknown multi-penalty CUDA capacity bucket");
}

MultiPenaltyGcvCapacityMultiResult multi_penalty_gcv_capacity_optimize_multi(
    std::vector<MultiPenaltyGcvCapacityRequest> requests,
    int requested_concurrency) {
  if (requests.empty() || !requests.front().prepared) {
    throw std::runtime_error(
      "capacity-dispatched multi-penalty request batch is empty");
  }
  const MultiPenaltyGcvCapacityBucket bucket =
    requests.front().prepared->bucket;
  if (requested_concurrency <= 0) {
    throw std::runtime_error(
      "capacity-dispatched multi-penalty concurrency must be positive");
  }
  requested_concurrency = std::min(
    requested_concurrency, maximum_concurrency_for_bucket(bucket));
  switch (bucket) {
    case MultiPenaltyGcvCapacityBucket::Small64:
      return run_multi_backend<
        MultiPenaltyGcvCudaMultiRequestsmall64,
        MultiPenaltyGcvCudaPreparedsmall64>(
          std::move(requests), bucket, requested_concurrency,
          multi_penalty_gcv_cuda_optimize_multi_small64,
          release_multi_penalty_gcv_cuda_residual_small64);
    case MultiPenaltyGcvCapacityBucket::Base80:
      return run_multi_backend<
        MultiPenaltyGcvCudaMultiRequest, MultiPenaltyGcvCudaPrepared>(
          std::move(requests), bucket, requested_concurrency,
          multi_penalty_gcv_cuda_optimize_multi,
          release_multi_penalty_gcv_cuda_residual);
    case MultiPenaltyGcvCapacityBucket::Extended192:
      return run_multi_backend<
        MultiPenaltyGcvCudaMultiRequestext192,
        MultiPenaltyGcvCudaPreparedext192>(
          std::move(requests), bucket, requested_concurrency,
          multi_penalty_gcv_cuda_optimize_multi_ext192,
          release_multi_penalty_gcv_cuda_residual_ext192);
    case MultiPenaltyGcvCapacityBucket::Extended384:
      return run_multi_backend<
        MultiPenaltyGcvCudaMultiRequestext384,
        MultiPenaltyGcvCudaPreparedext384>(
          std::move(requests), bucket, requested_concurrency,
          multi_penalty_gcv_cuda_optimize_multi_ext384,
          release_multi_penalty_gcv_cuda_residual_ext384);
    case MultiPenaltyGcvCapacityBucket::Extended559:
      return run_multi_backend<
        MultiPenaltyGcvCudaMultiRequestext559,
        MultiPenaltyGcvCudaPreparedext559>(
          std::move(requests), bucket, requested_concurrency,
          multi_penalty_gcv_cuda_optimize_multi_ext559,
          release_multi_penalty_gcv_cuda_residual_ext559);
  }
  throw std::runtime_error("unknown multi-penalty CUDA capacity bucket");
}

}  // namespace fastkpc
