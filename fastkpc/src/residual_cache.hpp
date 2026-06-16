#ifndef FASTKPC_RESIDUAL_CACHE_HPP
#define FASTKPC_RESIDUAL_CACHE_HPP

#include "residual_backend_registry.hpp"

#include <Rcpp.h>
#include <map>
#include <string>
#include <vector>

struct ResidualCacheOptions {
  bool enabled;
  ResidualBackendConfig backend;
};

struct ResidualCacheStats {
  bool enabled;
  int requests;
  int hits;
  int misses;
  int computations;
  int stored_vectors;
  int stored_values;
  std::string backend_name;
};

struct ResidualCacheKey {
  int target;
  std::vector<int> conditioning_set;
  int n_rows;
  int n_cols;
  std::string backend_name;
  std::string backend_params;

  bool operator<(const ResidualCacheKey& other) const;
};

ResidualCacheKey make_residual_cache_key(
  int target,
  const std::vector<int>& conditioning_set,
  int n_rows,
  int n_cols,
  const std::string& backend_name,
  const std::string& backend_params);

class ResidualCache {
 public:
  explicit ResidualCache(ResidualCacheOptions options);

  const std::vector<double>& get(const Rcpp::NumericMatrix& data,
                                 int target,
                                 const std::vector<int>& conditioning_set);

  ResidualCacheStats stats() const;

 private:
  ResidualCacheOptions options_;
  ResidualCacheStats stats_;
  std::map<ResidualCacheKey, std::vector<double> > values_;
  std::vector<double> scratch_;
};

ResidualCacheOptions linear_residual_cache_options(bool enabled);
ResidualCacheOptions backend_residual_cache_options(const std::string& name,
                                                    const FastSplineParams& fastspline_params,
                                                    bool enabled);

#endif
