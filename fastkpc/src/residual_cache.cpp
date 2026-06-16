#include "residual_cache.hpp"

#include <algorithm>
#include <tuple>

bool ResidualCacheKey::operator<(const ResidualCacheKey& other) const {
  return std::tie(target, conditioning_set, n_rows, n_cols, backend_name,
                  backend_params) <
    std::tie(other.target, other.conditioning_set, other.n_rows, other.n_cols,
             other.backend_name, other.backend_params);
}

ResidualCacheKey make_residual_cache_key(
  int target,
  const std::vector<int>& conditioning_set,
  int n_rows,
  int n_cols,
  const std::string& backend_name,
  const std::string& backend_params) {
  ResidualCacheKey key;
  key.target = target;
  key.conditioning_set = conditioning_set;
  std::sort(key.conditioning_set.begin(), key.conditioning_set.end());
  key.n_rows = n_rows;
  key.n_cols = n_cols;
  key.backend_name = backend_name;
  key.backend_params = backend_params;
  return key;
}

ResidualCacheOptions linear_residual_cache_options(bool enabled) {
  ResidualCacheOptions options;
  options.enabled = enabled;
  options.backend = make_residual_backend_config("linear", default_fastspline_params());
  return options;
}

ResidualCacheOptions backend_residual_cache_options(const std::string& name,
                                                    const FastSplineParams& fastspline_params,
                                                    bool enabled) {
  ResidualCacheOptions options;
  options.enabled = enabled;
  options.backend = make_residual_backend_config(name, fastspline_params);
  return options;
}

ResidualCache::ResidualCache(ResidualCacheOptions options)
    : options_(options) {
  stats_.enabled = options.enabled;
  stats_.requests = 0;
  stats_.hits = 0;
  stats_.misses = 0;
  stats_.computations = 0;
  stats_.stored_vectors = 0;
  stats_.stored_values = 0;
  stats_.backend_name = options.backend.name;
}

const std::vector<double>& ResidualCache::get(
  const Rcpp::NumericMatrix& data,
  int target,
  const std::vector<int>& conditioning_set) {
  ++stats_.requests;
  const ResidualCacheKey key = make_residual_cache_key(
    target, conditioning_set, data.nrow(), data.ncol(), options_.backend.name,
    options_.backend.params);

  if (!options_.enabled) {
    ++stats_.computations;
    scratch_ = compute_residuals_with_backend(data, target, key.conditioning_set,
                                              options_.backend);
    return scratch_;
  }

  std::map<ResidualCacheKey, std::vector<double> >::iterator it = values_.find(key);
  if (it != values_.end()) {
    ++stats_.hits;
    return it->second;
  }

  ++stats_.misses;
  ++stats_.computations;
  std::vector<double> residuals =
    compute_residuals_with_backend(data, target, key.conditioning_set, options_.backend);
  std::pair<std::map<ResidualCacheKey, std::vector<double> >::iterator, bool> inserted =
    values_.insert(std::make_pair(key, residuals));
  stats_.stored_vectors = static_cast<int>(values_.size());
  stats_.stored_values = stats_.stored_vectors * data.nrow();
  return inserted.first->second;
}

ResidualCacheStats ResidualCache::stats() const {
  ResidualCacheStats out = stats_;
  out.stored_vectors = static_cast<int>(values_.size());
  out.stored_values = out.stored_vectors * (values_.empty() ? 0 :
    static_cast<int>(values_.begin()->second.size()));
  return out;
}
