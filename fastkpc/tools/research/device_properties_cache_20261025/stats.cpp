// Read-only diagnostic bridge, called only outside the full-method timer.
#include "cuda/cuda_device_properties.hpp"
#include <Rcpp.h>
extern "C" SEXP device_properties_cache_stats() {
  BEGIN_RCPP
  const auto s=fastkpc_cuda_device_properties_cache_stats();
  return Rcpp::List::create(Rcpp::Named("enabled")=s.enabled,
    Rcpp::Named("requests")=double(s.requests),Rcpp::Named("queries")=double(s.queries),
    Rcpp::Named("failures")=double(s.failures),Rcpp::Named("hits")=double(s.hits),
    Rcpp::Named("entries")=double(s.entries));
  END_RCPP
}
