#ifndef FASTKPC_FASTSPLINE_BASIS_HPP
#define FASTKPC_FASTSPLINE_BASIS_HPP

#include <Rcpp.h>
#include <string>
#include <vector>

struct FastSplineParams {
  int degree;
  int knots;
  double lambda_min;
  double lambda_max;
  int lambda_count;
  double ridge;
  std::string mode;
};

struct FastSplineDesign {
  std::vector<double> X;
  std::vector<double> P;
  int n;
  int p;
};

FastSplineParams default_fastspline_params();
std::string serialize_fastspline_params(const FastSplineParams& params);

std::vector<double> quantile_knots(const std::vector<double>& x, int knots);
std::vector<double> cubic_bspline_basis(const std::vector<double>& x,
                                        const FastSplineParams& params,
                                        int* n_basis);
std::vector<double> second_difference_penalty(int n_basis);

FastSplineDesign make_fastspline_design(const Rcpp::NumericMatrix& data,
                                        const std::vector<int>& conditioning_set,
                                        const FastSplineParams& params);

#endif
