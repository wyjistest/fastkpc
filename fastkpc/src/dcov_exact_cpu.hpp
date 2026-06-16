#ifndef FASTKPC_DCOV_EXACT_CPU_HPP
#define FASTKPC_DCOV_EXACT_CPU_HPP

#include <Rcpp.h>
#include <vector>

double dcov_exact_pvalue(const std::vector<double>& x,
                         const std::vector<double>& y,
                         double index,
                         bool legacy_index);

std::vector<double> residualize_lm(const Rcpp::NumericMatrix& data,
                                   int target,
                                   const std::vector<int>& conditioning_set);

#endif
