/*
 * Native thin-plate regression-spline setup for the Phase 7 compatible path.
 *
 * The Lanczos and thin-plate Householder arithmetic below is an independent
 * C++ adaptation of the corresponding routines in mgcv 1.9-1 (tprs.c,
 * matrix.c, and mat.c), Copyright (C) 2000-2023 Simon N. Wood. mgcv and this
 * project are distributed under the GNU General Public License, version 2 or
 * later. The operation order is intentionally retained because it is part of
 * the version-pinned numerical semantics qualified by Phase 7.
 */

#include "full_cuda_ci_native_setup.hpp"

#include "full_cuda_ci_contract.hpp"

#include <R_ext/Applic.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include <R_ext/RS.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fastkpc {
namespace {

constexpr const char* kSchemaVersion = "full-cuda-ci-native-setup-v1";
constexpr const char* kSemanticVersion =
  "mgcv-1.9-1-tprs-native-setup-v1";

double profile_elapsed_ms(
    const std::chrono::steady_clock::time_point& started) {
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
}

class RowMatrix {
 public:
  RowMatrix() = default;
  RowMatrix(int rows, int cols)
      : rows_(rows), cols_(cols), values_(
          static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
          0.0) {
    if (rows < 0 || cols < 0) throw std::invalid_argument("negative matrix size");
  }

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  double& operator()(int row, int col) {
    return values_[static_cast<std::size_t>(row) * cols_ + col];
  }
  double operator()(int row, int col) const {
    return values_[static_cast<std::size_t>(row) * cols_ + col];
  }
  double* row_data(int row) {
    return values_.data() + static_cast<std::size_t>(row) * cols_;
  }
  const double* row_data(int row) const {
    return values_.data() + static_cast<std::size_t>(row) * cols_;
  }

 private:
  int rows_ = 0;
  int cols_ = 0;
  std::vector<double> values_;
};

struct UniqueLocations {
  RowMatrix values;
  std::vector<int> original_to_unique;
};

struct SmoothSetup {
  RowMatrix X;
  RowMatrix penalty;
  std::vector<double> shift;
  int basis_dimension = 0;
  int penalty_rank = 0;
  int null_space_dimension = 0;
  int unique_location_count = 0;
  int lanczos_iterations = 0;
  double penalty_scale = 0.0;
};

double r_real_mean(const Rcpp::NumericMatrix& values, int column) {
  const int n = values.nrow();
  long double mean = 0.0L;
  for (int row = 0; row < n; ++row) mean += values(row, column);
  mean /= static_cast<long double>(n);
  long double correction = 0.0L;
  for (int row = 0; row < n; ++row) {
    correction += static_cast<long double>(values(row, column)) - mean;
  }
  mean += correction / static_cast<long double>(n);
  return static_cast<double>(mean);
}

double r_col_mean(const RowMatrix& values, int column) {
  long double total = 0.0L;
  for (int row = 0; row < values.rows(); ++row) total += values(row, column);
  return static_cast<double>(total / static_cast<long double>(values.rows()));
}

double eta_constant(int m, int d) {
  const double pi = std::asin(1.0) * 2.0;
  const double gamma_half = std::sqrt(pi);
  const int d2 = d / 2;
  const int m2 = 2 * m;
  if (m2 <= d) throw std::runtime_error("thin plate spline requires 2m > d");
  double value = 1.0;
  if (d % 2 == 0) {
    value = ((m + 1 + d2) % 2) ? -1.0 : 1.0;
    for (int i = 0; i < m2 - 1; ++i) value /= 2.0;
    for (int i = 0; i < d2; ++i) value /= pi;
    for (int i = 2; i < m; ++i) value /= static_cast<double>(i);
    for (int i = 2; i <= m - d2; ++i) value /= static_cast<double>(i);
  } else {
    value = gamma_half;
    const int k = m - (d - 1) / 2;
    for (int i = 0; i < k; ++i) value /= -0.5 - static_cast<double>(i);
    for (int i = 0; i < m; ++i) value /= 4.0;
    for (int i = 0; i < d2; ++i) value /= pi;
    value /= gamma_half;
    for (int i = 2; i < m; ++i) value /= static_cast<double>(i);
  }
  return value;
}

double eta_squared_distance(int m, int d, double squared_distance,
                            double eta0) {
  if (squared_distance <= 0.0) return 0.0;
  const int d2 = d / 2;
  double value = eta0;
  if (d % 2 == 0) {
    value *= std::log(squared_distance) * 0.5;
    for (int i = 0; i < m - d2; ++i) value *= squared_distance;
  } else {
    for (int i = 0; i < m - d2 - 1; ++i) value *= squared_distance;
    value *= std::sqrt(squared_distance);
  }
  return value;
}

int null_space_dimension(int d, int m) {
  if (2 * m <= d) {
    m = 1;
    while (2 * m < d + 2) ++m;
  }
  int dimension = 1;
  for (int i = 0; i < d; ++i) dimension *= d + m - 1 - i;
  for (int i = 2; i <= d; ++i) dimension /= i;
  return dimension;
}

std::vector<int> polynomial_powers(int dimension, int order,
                                   int null_dimension) {
  std::vector<int> powers(
    static_cast<std::size_t>(dimension) * null_dimension, 0);
  std::vector<int> index(static_cast<std::size_t>(dimension), 0);
  for (int term = 0; term < null_dimension; ++term) {
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
      powers[static_cast<std::size_t>(term) +
             static_cast<std::size_t>(null_dimension) * coordinate] =
        index[static_cast<std::size_t>(coordinate)];
    }
    int sum = 0;
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
      sum += index[static_cast<std::size_t>(coordinate)];
    }
    if (sum < order - 1) {
      ++index[0];
    } else {
      sum -= index[0];
      index[0] = 0;
      for (int coordinate = 1; coordinate < dimension; ++coordinate) {
        ++index[static_cast<std::size_t>(coordinate)];
        ++sum;
        if (sum == order) {
          sum -= index[static_cast<std::size_t>(coordinate)];
          index[static_cast<std::size_t>(coordinate)] = 0;
        } else {
          break;
        }
      }
    }
  }
  return powers;
}

UniqueLocations sorted_unique_locations(const RowMatrix& shifted) {
  std::vector<int> order(static_cast<std::size_t>(shifted.rows()));
  for (int row = 0; row < shifted.rows(); ++row) {
    order[static_cast<std::size_t>(row)] = row;
  }
  std::sort(order.begin(), order.end(), [&](int left, int right) {
    for (int column = 0; column < shifted.cols(); ++column) {
      if (shifted(left, column) < shifted(right, column)) return true;
      if (shifted(left, column) > shifted(right, column)) return false;
    }
    return left < right;
  });

  std::vector<int> unique_order;
  std::vector<int> mapping(static_cast<std::size_t>(shifted.rows()), -1);
  for (int index : order) {
    bool is_new = unique_order.empty();
    if (!is_new) {
      const int prior = unique_order.back();
      for (int column = 0; column < shifted.cols(); ++column) {
        if (shifted(index, column) != shifted(prior, column)) {
          is_new = true;
          break;
        }
      }
    }
    if (is_new) unique_order.push_back(index);
    mapping[static_cast<std::size_t>(index)] =
      static_cast<int>(unique_order.size()) - 1;
  }

  RowMatrix unique(static_cast<int>(unique_order.size()), shifted.cols());
  for (int row = 0; row < unique.rows(); ++row) {
    const int source = unique_order[static_cast<std::size_t>(row)];
    for (int column = 0; column < unique.cols(); ++column) {
      unique(row, column) = shifted(source, column);
    }
  }
  return UniqueLocations{std::move(unique), std::move(mapping)};
}

std::vector<double> to_column_major(const RowMatrix& matrix) {
  std::vector<double> result(
    static_cast<std::size_t>(matrix.rows()) * matrix.cols(), 0.0);
  for (int column = 0; column < matrix.cols(); ++column) {
    for (int row = 0; row < matrix.rows(); ++row) {
      result[static_cast<std::size_t>(row) +
             static_cast<std::size_t>(matrix.rows()) * column] =
        matrix(row, column);
    }
  }
  return result;
}

RowMatrix from_column_major(const std::vector<double>& values,
                            int rows, int cols) {
  RowMatrix result(rows, cols);
  for (int column = 0; column < cols; ++column) {
    for (int row = 0; row < rows; ++row) {
      result(row, column) = values[static_cast<std::size_t>(row) +
                                   static_cast<std::size_t>(rows) * column];
    }
  }
  return result;
}

void tridiagonal_eigen(std::vector<double>* diagonal,
                       std::vector<double>* off_diagonal,
                       std::vector<double>* vectors, int n) {
  char compz = 'I';
  int ldz = n;
  int info = 0;
  int lwork = -1;
  int liwork = -1;
  double work_query = 0.0;
  int iwork_query = 0;
  F77_CALL(dstedc)(&compz, &n, diagonal->data(), off_diagonal->data(),
                   vectors->data(), &ldz, &work_query, &lwork,
                   &iwork_query, &liwork, &info FCONE);
  if (info != 0) throw std::runtime_error("tridiagonal workspace query failed");
  lwork = static_cast<int>(std::floor(work_query));
  if (work_query - lwork > 0.5) ++lwork;
  liwork = iwork_query;
  std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
  std::vector<int> iwork(static_cast<std::size_t>(liwork), 0);
  F77_CALL(dstedc)(&compz, &n, diagonal->data(), off_diagonal->data(),
                   vectors->data(), &ldz, work.data(), &lwork,
                   iwork.data(), &liwork, &info FCONE);
  if (info != 0) throw std::runtime_error("tridiagonal eigen solve failed");

  for (int left = 0; left < n / 2; ++left) {
    const int right = n - left - 1;
    std::swap((*diagonal)[static_cast<std::size_t>(left)],
              (*diagonal)[static_cast<std::size_t>(right)]);
    for (int row = 0; row < n; ++row) {
      std::swap((*vectors)[static_cast<std::size_t>(row) +
                           static_cast<std::size_t>(n) * left],
                (*vectors)[static_cast<std::size_t>(row) +
                           static_cast<std::size_t>(n) * right]);
    }
  }
}

int lanczos_largest_magnitude(const std::vector<double>& matrix,
                              int n, int requested,
                              std::vector<double>* vectors,
                              std::vector<double>* values) {
  const double tolerance = std::pow(std::numeric_limits<double>::epsilon(), 0.7);
  int upper_requested = requested;
  int lower_requested = 0;
  int check_frequency = requested / 2;
  if (check_frequency < 10) check_frequency = 10;
  int tenth = static_cast<int>(std::floor(n / 10.0));
  if (tenth < 1) tenth = 1;
  if (tenth < check_frequency) check_frequency = tenth;

  std::vector<std::vector<double> > q(static_cast<std::size_t>(n + 1));
  q[0].assign(static_cast<std::size_t>(n), 0.0);
  unsigned long random_state = 1;
  constexpr unsigned long multiplier = 106;
  constexpr unsigned long increment = 1283;
  constexpr unsigned long modulus = 6075;
  double norm = 0.0;
  for (int row = 0; row < n; ++row) {
    random_state = (random_state * multiplier + increment) % modulus;
    double value = static_cast<double>(random_state) /
      static_cast<double>(modulus) - 0.5;
    q[0][static_cast<std::size_t>(row)] = value;
    value = -value;
    norm += q[0][static_cast<std::size_t>(row)] *
      q[0][static_cast<std::size_t>(row)];
  }
  norm = std::sqrt(norm);
  for (double& value : q[0]) value /= norm;

  std::vector<double> diagonal(static_cast<std::size_t>(n), 0.0);
  std::vector<double> off_diagonal(static_cast<std::size_t>(n), 0.0);
  std::vector<double> tri_off(static_cast<std::size_t>(n), 0.0);
  std::vector<double> tri_values(static_cast<std::size_t>(n), 0.0);
  std::vector<double> z(static_cast<std::size_t>(n), 0.0);
  std::vector<double> errors(static_cast<std::size_t>(n), 1e300);
  std::vector<double> tri_vectors;
  int tri_length = 0;
  int completed = n;
  const int increment_one = 1;
  const double alpha = 1.0;
  const double beta = 0.0;
  const char upper = 'U';

  for (int iteration = 0; iteration < n; ++iteration) {
    F77_CALL(dsymv)(&upper, &n, &alpha, matrix.data(), &n,
                    q[static_cast<std::size_t>(iteration)].data(),
                    &increment_one, &beta, z.data(), &increment_one FCONE);
    double projection = 0.0;
    for (int row = 0; row < n; ++row) {
      projection += q[static_cast<std::size_t>(iteration)]
        [static_cast<std::size_t>(row)] * z[static_cast<std::size_t>(row)];
    }
    diagonal[static_cast<std::size_t>(iteration)] = projection;
    if (iteration == 0) {
      for (int row = 0; row < n; ++row) {
        z[static_cast<std::size_t>(row)] -= projection *
          q[0][static_cast<std::size_t>(row)];
      }
    } else {
      const double prior = off_diagonal[static_cast<std::size_t>(iteration - 1)];
      for (int row = 0; row < n; ++row) {
        z[static_cast<std::size_t>(row)] -= projection *
            q[static_cast<std::size_t>(iteration)][static_cast<std::size_t>(row)] +
          prior * q[static_cast<std::size_t>(iteration - 1)]
            [static_cast<std::size_t>(row)];
      }
      for (int pass = 0; pass < 2; ++pass) {
        for (int basis = 0; basis <= iteration; ++basis) {
          double coefficient = -F77_CALL(ddot)(
            &n, z.data(), &increment_one,
            q[static_cast<std::size_t>(basis)].data(), &increment_one);
          F77_CALL(daxpy)(&n, &coefficient,
                          q[static_cast<std::size_t>(basis)].data(),
                          &increment_one, z.data(), &increment_one);
        }
      }
    }
    double residual_norm = 0.0;
    for (double value : z) residual_norm += value * value;
    residual_norm = std::sqrt(residual_norm);
    off_diagonal[static_cast<std::size_t>(iteration)] = residual_norm;
    if (iteration < n - 1) {
      q[static_cast<std::size_t>(iteration + 1)].assign(
        static_cast<std::size_t>(n), 0.0);
      for (int row = 0; row < n; ++row) {
        q[static_cast<std::size_t>(iteration + 1)]
          [static_cast<std::size_t>(row)] =
            z[static_cast<std::size_t>(row)] / residual_norm;
      }
    }

    if ((iteration >= requested && iteration % check_frequency == 0) ||
        iteration == n - 1) {
      const int size = iteration + 1;
      for (int index = 0; index < size; ++index) {
        tri_values[static_cast<std::size_t>(index)] =
          diagonal[static_cast<std::size_t>(index)];
      }
      for (int index = 0; index < iteration; ++index) {
        tri_off[static_cast<std::size_t>(index)] =
          off_diagonal[static_cast<std::size_t>(index)];
      }
      tri_length = size;
      tri_vectors.assign(static_cast<std::size_t>(size) * size, 0.0);
      std::vector<double> local_values(tri_values.begin(),
                                       tri_values.begin() + size);
      std::vector<double> local_off(tri_off.begin(), tri_off.begin() + size);
      tridiagonal_eigen(&local_values, &local_off, &tri_vectors, size);
      std::copy(local_values.begin(), local_values.end(), tri_values.begin());

      double tri_norm = std::abs(tri_values[0]);
      if (std::abs(tri_values[static_cast<std::size_t>(iteration)]) > tri_norm) {
        tri_norm = std::abs(tri_values[static_cast<std::size_t>(iteration)]);
      }
      for (int eigen = 0; eigen < size; ++eigen) {
        errors[static_cast<std::size_t>(eigen)] = std::abs(
          off_diagonal[static_cast<std::size_t>(iteration)] *
          tri_vectors[static_cast<std::size_t>(eigen) * size + iteration]);
      }
      if (iteration >= requested) {
        const double maximum_error = tri_norm * tolerance;
        int positive = 0;
        int negative = 0;
        bool converged = true;
        while (positive + negative < requested) {
          if (std::abs(tri_values[static_cast<std::size_t>(positive)]) >=
              std::abs(tri_values[static_cast<std::size_t>(iteration - negative)])) {
            if (errors[static_cast<std::size_t>(positive)] > maximum_error) {
              converged = false;
              break;
            }
            ++positive;
          } else {
            if (errors[static_cast<std::size_t>(negative)] > maximum_error) {
              converged = false;
              break;
            }
            ++negative;
          }
        }
        if (converged) {
          upper_requested = positive;
          lower_requested = negative;
          completed = iteration + 1;
          break;
        }
      }
    }
  }

  if (tri_length <= 0 || completed <= 0) {
    throw std::runtime_error("Lanczos decomposition did not produce Ritz vectors");
  }
  vectors->assign(static_cast<std::size_t>(n) * requested, 0.0);
  values->assign(static_cast<std::size_t>(requested), 0.0);
  for (int eigen = 0; eigen < upper_requested; ++eigen) {
    (*values)[static_cast<std::size_t>(eigen)] =
      tri_values[static_cast<std::size_t>(eigen)];
    for (int basis = 0; basis < completed; ++basis) {
      const double coefficient = tri_vectors[
        static_cast<std::size_t>(basis) +
        static_cast<std::size_t>(eigen) * tri_length];
      for (int row = 0; row < n; ++row) {
        (*vectors)[static_cast<std::size_t>(row) +
                   static_cast<std::size_t>(n) * eigen] +=
          q[static_cast<std::size_t>(basis)][static_cast<std::size_t>(row)] *
          coefficient;
      }
    }
  }
  for (int eigen = upper_requested;
       eigen < upper_requested + lower_requested; ++eigen) {
    const int source = completed - (upper_requested + lower_requested - eigen);
    (*values)[static_cast<std::size_t>(eigen)] =
      tri_values[static_cast<std::size_t>(source)];
    for (int basis = 0; basis < completed; ++basis) {
      const double coefficient = tri_vectors[
        static_cast<std::size_t>(basis) +
        static_cast<std::size_t>(source) * tri_length];
      for (int row = 0; row < n; ++row) {
        (*vectors)[static_cast<std::size_t>(row) +
                   static_cast<std::size_t>(n) * eigen] +=
          q[static_cast<std::size_t>(basis)][static_cast<std::size_t>(row)] *
          coefficient;
      }
    }
  }
  return completed;
}

void qt_householders(RowMatrix* storage, RowMatrix* input) {
  const int rows = input->rows();
  const int cols = input->cols();
  for (int step = 0; step < rows; ++step) {
    double scale = 0.0;
    for (int column = 0; column < cols - step; ++column) {
      scale = std::max(scale, std::abs((*input)(step, column)));
    }
    if (scale != 0.0) {
      for (int column = 0; column < cols - step; ++column) {
        (*input)(step, column) /= scale;
      }
    }
    double length = 0.0;
    for (int column = 0; column < cols - step; ++column) {
      const double value = (*input)(step, column);
      length += value * value;
    }
    length = std::sqrt(length);
    if ((*input)(step, cols - step - 1) < 0.0) length = -length;
    (*input)(step, cols - step - 1) += length;
    const double multiplier = length != 0.0
      ? 1.0 / (length * (*input)(step, cols - step - 1)) : 0.0;
    const double terminal = length * scale;
    for (int row = step + 1; row < rows; ++row) {
      double projection = 0.0;
      for (int column = 0; column < cols - step; ++column) {
        projection += (*input)(step, column) * (*input)(row, column);
      }
      projection *= multiplier;
      for (int column = 0; column < cols - step; ++column) {
        (*input)(row, column) -= projection * (*input)(step, column);
      }
    }
    const double root_multiplier = std::sqrt(multiplier);
    for (int column = 0; column < cols - step; ++column) {
      (*storage)(step, column) =
        (*input)(step, column) * root_multiplier;
    }
    for (int column = cols - step; column < cols; ++column) {
      (*storage)(step, column) = 0.0;
    }
    (*input)(step, cols - step - 1) = -terminal;
    for (int column = 0; column < cols - step - 1; ++column) {
      (*input)(step, column) = 0.0;
    }
  }
}

void hq_post(RowMatrix* matrix, const RowMatrix& householders) {
  std::vector<double> product(static_cast<std::size_t>(matrix->rows()), 0.0);
  for (int transform = 0; transform < householders.rows(); ++transform) {
    for (int row = 0; row < matrix->rows(); ++row) {
      double value = 0.0;
      for (int column = 0; column < matrix->cols(); ++column) {
        value += (*matrix)(row, column) * householders(transform, column);
      }
      product[static_cast<std::size_t>(row)] = value;
    }
    for (int row = 0; row < matrix->rows(); ++row) {
      for (int column = 0; column < matrix->cols(); ++column) {
        (*matrix)(row, column) -= product[static_cast<std::size_t>(row)] *
          householders(transform, column);
      }
    }
  }
}

void hq_pre_transpose(RowMatrix* matrix, const RowMatrix& householders) {
  std::vector<double> product(static_cast<std::size_t>(matrix->cols()), 0.0);
  for (int transform = 0; transform < householders.rows(); ++transform) {
    for (int column = 0; column < matrix->cols(); ++column) {
      double value = 0.0;
      for (int row = 0; row < matrix->rows(); ++row) {
        value += (*matrix)(row, column) * householders(transform, row);
      }
      product[static_cast<std::size_t>(column)] = value;
    }
    for (int row = 0; row < matrix->rows(); ++row) {
      for (int column = 0; column < matrix->cols(); ++column) {
        (*matrix)(row, column) -= product[static_cast<std::size_t>(column)] *
          householders(transform, row);
      }
    }
  }
}

RowMatrix t_transpose_times_u(const RowMatrix& t, const RowMatrix& u) {
  RowMatrix result(t.cols(), u.cols());
  for (int source_row = 0; source_row < t.rows(); ++source_row) {
    for (int output_row = 0; output_row < t.cols(); ++output_row) {
      const double multiplier = t(source_row, output_row);
      for (int column = 0; column < u.cols(); ++column) {
        result(output_row, column) += multiplier * u(source_row, column);
      }
    }
  }
  return result;
}

double lapack_norm(RowMatrix matrix, char kind) {
  std::vector<double> packed = to_column_major(matrix);
  int rows = matrix.rows();
  int cols = matrix.cols();
  int leading = std::max(1, rows);
  std::vector<double> work(static_cast<std::size_t>(std::max(1, rows)), 0.0);
  return F77_CALL(dlange)(&kind, &rows, &cols, packed.data(), &leading,
                          work.data() FCONE);
}

std::vector<double> apply_linpack_qt(const std::vector<double>& qr,
                                     const std::vector<double>& qraux,
                                     int dimension,
                                     const std::vector<double>& values,
                                     int value_columns) {
  std::vector<double> result(values.size(), 0.0);
  int rank = 1;
  int columns = value_columns;
  F77_CALL(dqrqty)(const_cast<double*>(qr.data()), &dimension, &rank,
                   const_cast<double*>(qraux.data()),
                   const_cast<double*>(values.data()), &columns,
                   result.data());
  return result;
}

std::pair<RowMatrix, RowMatrix> absorb_centering_constraint(
    const RowMatrix& X, const RowMatrix& penalty) {
  const int dimension = X.cols();
  std::vector<double> qr(static_cast<std::size_t>(dimension), 0.0);
  for (int column = 0; column < dimension; ++column) {
    qr[static_cast<std::size_t>(column)] = r_col_mean(X, column);
  }
  int leading = dimension;
  int rows = dimension;
  int columns = 1;
  double tolerance = 1e-7;
  int rank = 0;
  std::vector<double> qraux(1, 0.0);
  std::vector<int> pivot(1, 1);
  std::vector<double> work(2, 0.0);
  F77_CALL(dqrdc2)(qr.data(), &leading, &rows, &columns, &tolerance,
                   &rank, qraux.data(), pivot.data(), work.data());
  if (rank != 1 || pivot[0] != 1) {
    throw std::runtime_error("native centering constraint is rank deficient");
  }

  std::vector<double> transposed_X(
    static_cast<std::size_t>(dimension) * X.rows(), 0.0);
  for (int observation = 0; observation < X.rows(); ++observation) {
    for (int coefficient = 0; coefficient < dimension; ++coefficient) {
      transposed_X[static_cast<std::size_t>(coefficient) +
                    static_cast<std::size_t>(dimension) * observation] =
        X(observation, coefficient);
    }
  }
  std::vector<double> transformed_X = apply_linpack_qt(
    qr, qraux, dimension, transposed_X, X.rows());
  RowMatrix absorbed_X(X.rows(), dimension - 1);
  for (int observation = 0; observation < X.rows(); ++observation) {
    for (int coefficient = 0; coefficient < dimension - 1; ++coefficient) {
      absorbed_X(observation, coefficient) = transformed_X[
        static_cast<std::size_t>(coefficient + 1) +
        static_cast<std::size_t>(dimension) * observation];
    }
  }

  std::vector<double> packed_penalty = to_column_major(penalty);
  std::vector<double> left = apply_linpack_qt(
    qr, qraux, dimension, packed_penalty, dimension);
  std::vector<double> transposed_left(
    static_cast<std::size_t>(dimension) * (dimension - 1), 0.0);
  for (int free_row = 0; free_row < dimension - 1; ++free_row) {
    for (int original_column = 0; original_column < dimension;
         ++original_column) {
      transposed_left[static_cast<std::size_t>(original_column) +
                       static_cast<std::size_t>(dimension) * free_row] =
        left[static_cast<std::size_t>(free_row + 1) +
             static_cast<std::size_t>(dimension) * original_column];
    }
  }
  std::vector<double> right = apply_linpack_qt(
    qr, qraux, dimension, transposed_left, dimension - 1);
  RowMatrix absorbed_penalty(dimension - 1, dimension - 1);
  for (int row = 0; row < dimension - 1; ++row) {
    for (int column = 0; column < dimension - 1; ++column) {
      absorbed_penalty(row, column) = right[
        static_cast<std::size_t>(column + 1) +
        static_cast<std::size_t>(dimension) * row];
    }
  }
  return std::make_pair(std::move(absorbed_X),
                        std::move(absorbed_penalty));
}

SmoothSetup build_smooth(const Rcpp::NumericMatrix& conditioning,
                         const std::vector<int>& columns,
                         int basis_dimension) {
  const int n = conditioning.nrow();
  const int dimension = static_cast<int>(columns.size());
  int order = 0;
  if (2 * order <= dimension) {
    order = 0;
    while (2 * order < dimension + 2) ++order;
  }
  const int null_dimension = null_space_dimension(dimension, order);
  if (basis_dimension < null_dimension + 1) {
    throw std::runtime_error("native basis dimension is too small");
  }

  std::vector<double> shifts(static_cast<std::size_t>(dimension), 0.0);
  RowMatrix shifted(n, dimension);
  for (int local = 0; local < dimension; ++local) {
    const int source = columns[static_cast<std::size_t>(local)];
    shifts[static_cast<std::size_t>(local)] =
      r_real_mean(conditioning, source);
    for (int row = 0; row < n; ++row) {
      shifted(row, local) = conditioning(row, source) -
        shifts[static_cast<std::size_t>(local)];
    }
  }
  UniqueLocations unique = sorted_unique_locations(shifted);
  if (unique.values.rows() > 2000) {
    throw std::runtime_error("native TPRS supports at most 2000 unique locations");
  }
  if (unique.values.rows() < basis_dimension) {
    throw std::runtime_error(
      "native TPRS has fewer unique locations than its basis dimension");
  }

  const double eta0 = eta_constant(order, dimension);
  RowMatrix E(unique.values.rows(), unique.values.rows());
  for (int row = 0; row < E.rows(); ++row) {
    for (int prior = 0; prior < row; ++prior) {
      double squared_distance = 0.0;
      for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        const double difference = unique.values(row, coordinate) -
          unique.values(prior, coordinate);
        squared_distance += difference * difference;
      }
      const double value = eta_squared_distance(
        order, dimension, squared_distance, eta0);
      E(row, prior) = value;
      E(prior, row) = value;
    }
  }

  const std::vector<int> powers = polynomial_powers(
    dimension, order, null_dimension);
  RowMatrix T(unique.values.rows(), null_dimension);
  for (int row = 0; row < T.rows(); ++row) {
    for (int term = 0; term < null_dimension; ++term) {
      double value = 1.0;
      for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        const int power = powers[static_cast<std::size_t>(term) +
          static_cast<std::size_t>(null_dimension) * coordinate];
        for (int exponent = 0; exponent < power; ++exponent) {
          value *= unique.values(row, coordinate);
        }
      }
      T(row, term) = value;
    }
  }

  std::vector<double> eigenvectors;
  std::vector<double> eigenvalues;
  const int lanczos_iterations = lanczos_largest_magnitude(
    to_column_major(E), E.rows(), basis_dimension,
    &eigenvectors, &eigenvalues);
  RowMatrix U = from_column_major(
    eigenvectors, E.rows(), basis_dimension);
  RowMatrix TU = t_transpose_times_u(T, U);
  RowMatrix Z(null_dimension, basis_dimension);
  qt_householders(&Z, &TU);

  RowMatrix X_unique = U;
  for (int row = 0; row < X_unique.rows(); ++row) {
    for (int column = 0; column < X_unique.cols(); ++column) {
      X_unique(row, column) *= eigenvalues[static_cast<std::size_t>(column)];
    }
  }
  hq_post(&X_unique, Z);
  for (int row = 0; row < X_unique.rows(); ++row) {
    for (int column = basis_dimension - null_dimension;
         column < basis_dimension; ++column) {
      X_unique(row, column) = 0.0;
    }
    for (int term = 0; term < null_dimension; ++term) {
      X_unique(row, basis_dimension - null_dimension + term) = T(row, term);
    }
  }
  RowMatrix X(n, basis_dimension);
  for (int row = 0; row < n; ++row) {
    const int source = unique.original_to_unique[static_cast<std::size_t>(row)];
    for (int column = 0; column < basis_dimension; ++column) {
      X(row, column) = X_unique(source, column);
    }
  }

  RowMatrix penalty(basis_dimension, basis_dimension);
  for (int index = 0; index < basis_dimension; ++index) {
    penalty(index, index) = eigenvalues[static_cast<std::size_t>(index)];
  }
  hq_post(&penalty, Z);
  hq_pre_transpose(&penalty, Z);
  for (int row = 0; row < basis_dimension; ++row) {
    for (int column = basis_dimension - null_dimension;
         column < basis_dimension; ++column) {
      penalty(row, column) = 0.0;
      penalty(column, row) = 0.0;
    }
  }

  for (int column = 0; column < basis_dimension; ++column) {
    double squared_norm = 0.0;
    for (int row = 0; row < n; ++row) {
      squared_norm += X(row, column) * X(row, column);
    }
    const double rms = std::sqrt(squared_norm / static_cast<double>(n));
    if (!std::isfinite(rms) || rms == 0.0) {
      throw std::runtime_error("native TPRS RMS scaling encountered a zero column");
    }
    for (int row = 0; row < n; ++row) X(row, column) /= rms;
    for (int index = 0; index < basis_dimension; ++index) {
      penalty(column, index) /= rms;
    }
    for (int index = 0; index < basis_dimension; ++index) {
      penalty(index, column) /= rms;
    }
  }

  RowMatrix symmetric_penalty(basis_dimension, basis_dimension);
  for (int row = 0; row < basis_dimension; ++row) {
    for (int column = 0; column < basis_dimension; ++column) {
      symmetric_penalty(row, column) =
        (penalty(row, column) + penalty(column, row)) / 2.0;
    }
  }
  const double x_norm = lapack_norm(X, 'I');
  const double penalty_norm = lapack_norm(symmetric_penalty, 'O');
  const double penalty_scale = penalty_norm / (x_norm * x_norm);
  if (!std::isfinite(penalty_scale) || penalty_scale <= 0.0) {
    throw std::runtime_error("native TPRS penalty scaling failed");
  }
  for (int row = 0; row < basis_dimension; ++row) {
    for (int column = 0; column < basis_dimension; ++column) {
      symmetric_penalty(row, column) /= penalty_scale;
    }
  }
  std::pair<RowMatrix, RowMatrix> absorbed =
    absorb_centering_constraint(X, symmetric_penalty);

  return SmoothSetup{
    std::move(absorbed.first),
    std::move(absorbed.second),
    std::move(shifts),
    basis_dimension,
    basis_dimension - null_dimension,
    null_dimension - 1,
    unique.values.rows(),
    lanczos_iterations,
    penalty_scale
  };
}

double r_mean_vector(const std::vector<double>& values) {
  if (values.empty()) throw std::runtime_error("mean of empty vector");
  long double mean = 0.0L;
  for (double value : values) mean += value;
  mean /= static_cast<long double>(values.size());
  long double correction = 0.0L;
  for (double value : values) {
    correction += static_cast<long double>(value) - mean;
  }
  mean += correction / static_cast<long double>(values.size());
  return static_cast<double>(mean);
}

struct PivotedQr {
  RowMatrix packed;
  RowMatrix Q;
  RowMatrix R;
  std::vector<double> tau;
  std::vector<int> pivot;
};

PivotedQr pivoted_qr(const RowMatrix& input) {
  const int rows = input.rows();
  const int columns = input.cols();
  if (rows < columns || columns <= 0) {
    throw std::runtime_error("native pivoted QR requires n >= p > 0");
  }
  std::vector<double> packed = to_column_major(input);
  std::vector<int> pivot(static_cast<std::size_t>(columns), 0);
  std::vector<double> tau(static_cast<std::size_t>(columns), 0.0);
  int info = 0;
  int lwork = -1;
  double work_query = 0.0;
  F77_CALL(dgeqp3)(&rows, &columns, packed.data(), &rows, pivot.data(),
                   tau.data(), &work_query, &lwork, &info);
  if (info < 0) throw std::runtime_error("native dgeqp3 workspace query failed");
  lwork = static_cast<int>(work_query);
  std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
  F77_CALL(dgeqp3)(&rows, &columns, packed.data(), &rows, pivot.data(),
                   tau.data(), work.data(), &lwork, &info);
  if (info < 0) throw std::runtime_error("native dgeqp3 failed");

  std::vector<double> q_values(
    static_cast<std::size_t>(rows) * columns, 0.0);
  for (int column = 0; column < columns; ++column) {
    q_values[static_cast<std::size_t>(column) * rows + column] = 1.0;
  }
  const char side = 'L';
  const char transpose = 'N';
  int q_columns = columns;
  int reflectors = columns;
  lwork = -1;
  work_query = 0.0;
  F77_CALL(dormqr)(&side, &transpose, &rows, &q_columns, &reflectors,
                   packed.data(), &rows, tau.data(), q_values.data(), &rows,
                   &work_query, &lwork, &info FCONE FCONE);
  if (info != 0) throw std::runtime_error("native dormqr workspace query failed");
  lwork = static_cast<int>(work_query);
  work.assign(static_cast<std::size_t>(lwork), 0.0);
  F77_CALL(dormqr)(&side, &transpose, &rows, &q_columns, &reflectors,
                   packed.data(), &rows, tau.data(), q_values.data(), &rows,
                   work.data(), &lwork, &info FCONE FCONE);
  if (info != 0) throw std::runtime_error("native dormqr failed");

  RowMatrix r_matrix(columns, columns);
  for (int column = 0; column < columns; ++column) {
    for (int row = 0; row <= column; ++row) {
      r_matrix(row, column) = packed[
        static_cast<std::size_t>(row) +
        static_cast<std::size_t>(rows) * column];
    }
  }
  return PivotedQr{
    from_column_major(packed, rows, columns),
    from_column_major(q_values, rows, columns),
    std::move(r_matrix),
    std::move(tau),
    std::move(pivot)
  };
}

RowMatrix pivoted_cholesky_root(const RowMatrix& penalty,
                                int requested_rank) {
  const int dimension = penalty.rows();
  if (dimension <= 0 || penalty.cols() != dimension ||
      requested_rank <= 0 || requested_rank > dimension) {
    throw std::runtime_error("native mroot dimensions are invalid");
  }
  std::vector<double> factor = to_column_major(penalty);
  for (int column = 0; column < dimension; ++column) {
    for (int row = column + 1; row < dimension; ++row) {
      factor[static_cast<std::size_t>(row) +
             static_cast<std::size_t>(dimension) * column] = 0.0;
    }
  }
  std::vector<int> pivot(static_cast<std::size_t>(dimension), 0);
  std::vector<double> work(static_cast<std::size_t>(2 * dimension), 0.0);
  const char upper = 'U';
  double tolerance = 0.0;
  int actual_rank = 0;
  int info = 0;
  F77_CALL(dpstrf)(&upper, &dimension, factor.data(), &dimension,
                   pivot.data(), &actual_rank, &tolerance, work.data(),
                   &info FCONE);
  if (info < 0 || actual_rank < requested_rank) {
    throw std::runtime_error("native mroot pivoted Cholesky rank mismatch");
  }
  if (actual_rank < dimension) {
    for (int column = actual_rank; column < dimension; ++column) {
      for (int row = actual_rank; row < dimension; ++row) {
        factor[static_cast<std::size_t>(row) +
               static_cast<std::size_t>(dimension) * column] = 0.0;
      }
    }
  }
  std::vector<int> inverse_order(static_cast<std::size_t>(dimension), -1);
  for (int position = 0; position < dimension; ++position) {
    const int original = pivot[static_cast<std::size_t>(position)] - 1;
    if (original < 0 || original >= dimension) {
      throw std::runtime_error("native mroot pivot is malformed");
    }
    inverse_order[static_cast<std::size_t>(original)] = position;
  }
  RowMatrix root(dimension, requested_rank);
  for (int original = 0; original < dimension; ++original) {
    const int reordered_column =
      inverse_order[static_cast<std::size_t>(original)];
    for (int root_column = 0; root_column < requested_rank; ++root_column) {
      root(original, root_column) = factor[
        static_cast<std::size_t>(root_column) +
        static_cast<std::size_t>(dimension) * reordered_column];
    }
  }
  return root;
}

RowMatrix symmetric_tcrossprod(const RowMatrix& root) {
  const int rows = root.rows();
  const int columns = root.cols();
  std::vector<double> packed = to_column_major(root);
  std::vector<double> result(static_cast<std::size_t>(rows) * rows, 0.0);
  const char upper = 'U';
  const char no_transpose = 'N';
  const double one = 1.0;
  const double zero = 0.0;
  F77_CALL(dsyrk)(&upper, &no_transpose, &rows, &columns, &one,
                  packed.data(), &rows, &zero, result.data(), &rows
                  FCONE FCONE);
  for (int row = 1; row < rows; ++row) {
    for (int column = 0; column < row; ++column) {
      result[static_cast<std::size_t>(row) +
             static_cast<std::size_t>(rows) * column] =
        result[static_cast<std::size_t>(column) +
               static_cast<std::size_t>(rows) * row];
    }
  }
  return from_column_major(result, rows, rows);
}

std::vector<double> initial_smoothing_parameters(
    const RowMatrix& X,
    const std::vector<RowMatrix>& penalties,
    const std::vector<int>& offsets) {
  const int coefficient_count = X.cols();
  std::vector<double> x_diagonal(
    static_cast<std::size_t>(coefficient_count), 0.0);
  for (int column = 0; column < coefficient_count; ++column) {
    long double total = 0.0L;
    for (int row = 0; row < X.rows(); ++row) {
      const double square = X(row, column) * X(row, column);
      total += square;
    }
    x_diagonal[static_cast<std::size_t>(column)] =
      static_cast<double>(total);
  }
  std::vector<double> penalty_diagonal(
    static_cast<std::size_t>(coefficient_count), 0.0);
  std::vector<unsigned char> penalized(
    static_cast<std::size_t>(coefficient_count), 0);
  std::vector<double> result(penalties.size(), 0.0);
  const double epsilon_power = std::pow(
    std::numeric_limits<double>::epsilon(), 0.8);

  for (std::size_t index = 0; index < penalties.size(); ++index) {
    const RowMatrix& penalty = penalties[index];
    const int dimension = penalty.rows();
    double maximum = 0.0;
    for (int row = 0; row < dimension; ++row) {
      for (int column = 0; column < dimension; ++column) {
        maximum = std::max(maximum, std::abs(penalty(row, column)));
      }
    }
    const double threshold = epsilon_power * maximum;
    std::vector<unsigned char> active(static_cast<std::size_t>(dimension), 0);
    std::vector<double> active_x;
    std::vector<double> active_penalty;
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
      long double row_total = 0.0L;
      long double column_total = 0.0L;
      for (int other = 0; other < dimension; ++other) {
        row_total += std::abs(penalty(coordinate, other));
        column_total += std::abs(penalty(other, coordinate));
      }
      const double row_mean = static_cast<double>(
        row_total / static_cast<long double>(dimension));
      const double column_mean = static_cast<double>(
        column_total / static_cast<long double>(dimension));
      const double diagonal = std::abs(penalty(coordinate, coordinate));
      if (row_mean > threshold && column_mean > threshold &&
          diagonal > threshold) {
        active[static_cast<std::size_t>(coordinate)] = 1;
        const int global = offsets[index] + coordinate;
        active_x.push_back(x_diagonal[static_cast<std::size_t>(global)]);
        active_penalty.push_back(penalty(coordinate, coordinate));
      }
    }
    if (active_x.empty()) {
      throw std::runtime_error("native initial.sp found no active coordinates");
    }
    const double size_x = r_mean_vector(active_x);
    const double size_penalty = r_mean_vector(active_penalty);
    if (!std::isfinite(size_penalty) || size_penalty <= 0.0) {
      throw std::runtime_error("native initial.sp penalty is not positive");
    }
    result[index] = size_x / size_penalty;
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
      const int global = offsets[index] + coordinate;
      if (active[static_cast<std::size_t>(coordinate)] != 0) {
        penalized[static_cast<std::size_t>(global)] = 1;
      }
      penalty_diagonal[static_cast<std::size_t>(global)] +=
        result[index] * penalty(coordinate, coordinate);
    }
  }

  std::vector<int> used;
  for (int coordinate = 0; coordinate < coefficient_count; ++coordinate) {
    if (penalty_diagonal[static_cast<std::size_t>(coordinate)] > 0.0 &&
        penalized[static_cast<std::size_t>(coordinate)] != 0 &&
        x_diagonal[static_cast<std::size_t>(coordinate)] > 0.0) {
      used.push_back(coordinate);
    }
  }
  if (used.empty()) throw std::runtime_error("native initial.sp has no used coordinates");
  auto mean_hat = [&]() {
    std::vector<double> values;
    values.reserve(used.size());
    for (int coordinate : used) {
      const double x = x_diagonal[static_cast<std::size_t>(coordinate)];
      const double penalty =
        penalty_diagonal[static_cast<std::size_t>(coordinate)];
      values.push_back(x / (x + penalty));
    }
    return r_mean_vector(values);
  };
  int guard = 0;
  while (mean_hat() > 0.4) {
    for (double& value : result) value *= 10.0;
    for (double& value : penalty_diagonal) value *= 10.0;
    if (++guard > 1000) throw std::runtime_error("native initial.sp did not bracket 0.4");
  }
  while (mean_hat() < 0.4) {
    for (double& value : result) value /= 10.0;
    for (double& value : penalty_diagonal) value /= 10.0;
    if (++guard > 2000) throw std::runtime_error("native initial.sp did not bracket 0.4");
  }
  return result;
}

Rcpp::NumericMatrix to_rcpp(const RowMatrix& values) {
  Rcpp::NumericMatrix result(values.rows(), values.cols());
  for (int column = 0; column < values.cols(); ++column) {
    for (int row = 0; row < values.rows(); ++row) {
      result(row, column) = values(row, column);
    }
  }
  return result;
}

std::string input_fingerprint_payload(const Rcpp::NumericMatrix& conditioning) {
  std::ostringstream payload;
  payload << "schema_version=" << kSchemaVersion << '\n'
          << "semantic_version=" << kSemanticVersion << '\n'
          << "n=" << conditioning.nrow() << '\n'
          << "S_size=" << conditioning.ncol() << '\n'
          << "column_major_ieee754=";
  payload << std::hex << std::setfill('0');
  for (int column = 0; column < conditioning.ncol(); ++column) {
    for (int row = 0; row < conditioning.nrow(); ++row) {
      std::uint64_t bits = 0;
      const double value = conditioning(row, column);
      std::memcpy(&bits, &value, sizeof(bits));
      payload << std::setw(16) << bits;
    }
  }
  payload << '\n';
  return payload.str();
}

}  // namespace

class NativeUnivariateSmooth {
 public:
  NativeUnivariateSmooth(int source_column, SmoothSetup smooth)
      : source_column(source_column), smooth(std::move(smooth)) {}

  int source_column;
  SmoothSetup smooth;
};

namespace {

void validate_native_conditioning(
    const Rcpp::NumericMatrix& conditioning) {
  const int n = conditioning.nrow();
  const int conditioning_size = conditioning.ncol();
  if (n <= 0 || conditioning_size < 1 || conditioning_size > 7) {
    Rcpp::stop("native setup requires n > 0 and 1 <= |S| <= 7");
  }
  for (int column = 0; column < conditioning_size; ++column) {
    for (int row = 0; row < n; ++row) {
      if (!std::isfinite(conditioning(row, column))) {
        Rcpp::stop("native setup conditioning data must be finite");
      }
    }
  }
}

Rcpp::List assemble_native_setup(
    const Rcpp::NumericMatrix& conditioning,
    const std::vector<const SmoothSetup*>& smooths,
    NativeSetupProfile* profile) {
  const int n = conditioning.nrow();
  const int conditioning_size = conditioning.ncol();
  const std::size_t expected_smooth_count = conditioning_size <= 2 ?
    1U : static_cast<std::size_t>(conditioning_size);
  if (smooths.size() != expected_smooth_count ||
      std::any_of(smooths.begin(), smooths.end(), [](const SmoothSetup* value) {
        return value == nullptr;
      })) {
    Rcpp::stop("native setup smooth primitives are malformed");
  }

  auto stage_started = std::chrono::steady_clock::now();
  int coefficient_count = 1;
  for (const SmoothSetup* smooth : smooths) {
    coefficient_count += smooth->X.cols();
  }
  Rcpp::NumericMatrix X(n, coefficient_count);
  for (int row = 0; row < n; ++row) X(row, 0) = 1.0;
  Rcpp::List penalty_blocks(smooths.size());
  Rcpp::IntegerVector offsets(smooths.size());
  Rcpp::IntegerVector ranks(smooths.size());
  Rcpp::IntegerVector basis_dimensions(smooths.size());
  Rcpp::IntegerVector null_dimensions(smooths.size());
  Rcpp::NumericVector penalty_scales(smooths.size());
  Rcpp::IntegerVector unique_counts(smooths.size());
  Rcpp::IntegerVector lanczos_iterations(smooths.size());
  Rcpp::List shifts(smooths.size());
  Rcpp::CharacterVector penalty_names(smooths.size());

  int destination = 1;
  for (std::size_t index = 0; index < smooths.size(); ++index) {
    const SmoothSetup& smooth = *smooths[index];
    if (smooth.X.rows() != n) {
      Rcpp::stop("native setup smooth primitive row count changed");
    }
    for (int column = 0; column < smooth.X.cols(); ++column) {
      for (int row = 0; row < n; ++row) {
        X(row, destination + column) = smooth.X(row, column);
      }
    }
    penalty_blocks[static_cast<int>(index)] = to_rcpp(smooth.penalty);
    offsets[static_cast<int>(index)] = destination + 1;
    ranks[static_cast<int>(index)] = smooth.penalty_rank;
    basis_dimensions[static_cast<int>(index)] = smooth.basis_dimension;
    null_dimensions[static_cast<int>(index)] = smooth.null_space_dimension;
    penalty_scales[static_cast<int>(index)] = smooth.penalty_scale;
    unique_counts[static_cast<int>(index)] = smooth.unique_location_count;
    lanczos_iterations[static_cast<int>(index)] = smooth.lanczos_iterations;
    shifts[static_cast<int>(index)] = Rcpp::wrap(smooth.shift);
    penalty_names[static_cast<int>(index)] =
      "penalty_" + std::to_string(index + 1);
    destination += smooth.X.cols();
  }
  penalty_blocks.attr("names") = penalty_names;
  shifts.attr("names") = Rcpp::CharacterVector(
    penalty_names.begin(), penalty_names.end());
  if (profile != nullptr) {
    profile->block_assembly_ms += profile_elapsed_ms(stage_started);
  }

  stage_started = std::chrono::steady_clock::now();
  Rcpp::NumericMatrix gram_matrix(coefficient_count, coefficient_count);
  const char upper = 'U';
  const char transpose = 'T';
  const double one = 1.0;
  const double zero = 0.0;
  F77_CALL(dsyrk)(&upper, &transpose, &coefficient_count, &n, &one,
                  X.begin(), &n, &zero, gram_matrix.begin(),
                  &coefficient_count FCONE FCONE);
  for (int row = 1; row < coefficient_count; ++row) {
    for (int column = 0; column < row; ++column) {
      gram_matrix(row, column) = gram_matrix(column, row);
    }
  }
  if (profile != nullptr) {
    profile->gram_ms += profile_elapsed_ms(stage_started);
  }

  stage_started = std::chrono::steady_clock::now();
  const std::string fingerprint_payload =
    input_fingerprint_payload(conditioning);
  const std::string semantic_fingerprint =
    full_cuda_ci_sha256_utf8(fingerprint_payload);
  const std::string formula_class = conditioning_size <= 2
    ? "full-smooth" : "additive-smooth";
  if (profile != nullptr) {
    profile->fingerprint_ms += profile_elapsed_ms(stage_started);
  }

  stage_started = std::chrono::steady_clock::now();
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("schema_version") = kSchemaVersion,
    Rcpp::Named("semantic_version") = kSemanticVersion,
    Rcpp::Named("semantic_fingerprint") = semantic_fingerprint,
    Rcpp::Named("semantic_fingerprint_payload") = fingerprint_payload,
    Rcpp::Named("formula_class") = formula_class,
    Rcpp::Named("n") = n,
    Rcpp::Named("S_size") = conditioning_size,
    Rcpp::Named("X") = X,
    Rcpp::Named("gram_matrix") = gram_matrix,
    Rcpp::Named("penalty_blocks") = penalty_blocks,
    Rcpp::Named("penalty_offsets") = offsets,
    Rcpp::Named("penalty_ranks") = ranks,
    Rcpp::Named("basis_dimensions") = basis_dimensions,
    Rcpp::Named("null_space_dimensions") = null_dimensions,
    Rcpp::Named("shifts") = shifts,
    Rcpp::Named("smooth_S_scale") = penalty_scales,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("legacy_mgcv_setup_count") = 0,
      Rcpp::Named("r_callback_count") = 0,
      Rcpp::Named("unsupported_count") = 0,
      Rcpp::Named("unique_location_counts") = unique_counts,
      Rcpp::Named("lanczos_iterations") = lanczos_iterations,
      Rcpp::Named("coordinate_semantics") =
        "mgcv-1.9-1-Rlanczos-QT-LINPACK-constraint-v1"
    )
  );
  if (profile != nullptr) {
    profile->setup_packaging_ms += profile_elapsed_ms(stage_started);
  }
  return result;
}

}  // namespace

Rcpp::List full_cuda_ci_native_setup(
    const Rcpp::NumericMatrix& conditioning,
    NativeSetupProfile* profile) {
  auto stage_started = std::chrono::steady_clock::now();
  const int conditioning_size = conditioning.ncol();
  validate_native_conditioning(conditioning);
  if (profile != nullptr) {
    profile->input_validation_ms += profile_elapsed_ms(stage_started);
  }

  stage_started = std::chrono::steady_clock::now();
  std::vector<SmoothSetup> smooths;
  if (conditioning_size <= 2) {
    std::vector<int> columns(static_cast<std::size_t>(conditioning_size));
    for (int column = 0; column < conditioning_size; ++column) {
      columns[static_cast<std::size_t>(column)] = column;
    }
    const int basis_dimension = conditioning_size == 1 ? 10 : 30;
    smooths.push_back(build_smooth(conditioning, columns, basis_dimension));
  } else {
    for (int column = 0; column < conditioning_size; ++column) {
      smooths.push_back(build_smooth(conditioning, {column}, 10));
    }
  }
  if (profile != nullptr) {
    profile->smooth_build_ms += profile_elapsed_ms(stage_started);
  }

  std::vector<const SmoothSetup*> smooth_pointers;
  smooth_pointers.reserve(smooths.size());
  for (const SmoothSetup& smooth : smooths) {
    smooth_pointers.push_back(&smooth);
  }
  return assemble_native_setup(conditioning, smooth_pointers, profile);
}

std::shared_ptr<const NativeUnivariateSmooth>
full_cuda_ci_native_univariate_smooth(
    const Rcpp::NumericMatrix& data,
    int source_column,
    NativeSetupProfile* profile) {
  auto stage_started = std::chrono::steady_clock::now();
  if (data.nrow() <= 0 || source_column < 0 || source_column >= data.ncol()) {
    Rcpp::stop("native univariate smooth source is invalid");
  }
  for (int row = 0; row < data.nrow(); ++row) {
    if (!std::isfinite(data(row, source_column))) {
      Rcpp::stop("native univariate smooth data must be finite");
    }
  }
  if (profile != nullptr) {
    profile->input_validation_ms += profile_elapsed_ms(stage_started);
  }

  stage_started = std::chrono::steady_clock::now();
  SmoothSetup smooth = build_smooth(data, {source_column}, 10);
  if (profile != nullptr) {
    profile->smooth_build_ms += profile_elapsed_ms(stage_started);
  }
  return std::make_shared<const NativeUnivariateSmooth>(
    source_column, std::move(smooth));
}

Rcpp::List full_cuda_ci_native_additive_setup(
    const Rcpp::NumericMatrix& conditioning,
    const std::vector<int>& source_columns,
    const std::vector<std::shared_ptr<const NativeUnivariateSmooth>>& smooths,
    NativeSetupProfile* profile) {
  auto stage_started = std::chrono::steady_clock::now();
  validate_native_conditioning(conditioning);
  const int conditioning_size = conditioning.ncol();
  if (conditioning_size <= 2 ||
      source_columns.size() != static_cast<std::size_t>(conditioning_size) ||
      smooths.size() != static_cast<std::size_t>(conditioning_size)) {
    Rcpp::stop("native additive smooth primitive set is malformed");
  }
  for (int index = 0; index < conditioning_size; ++index) {
    const std::shared_ptr<const NativeUnivariateSmooth>& smooth =
      smooths[static_cast<std::size_t>(index)];
    if (!smooth || smooth->source_column !=
          source_columns[static_cast<std::size_t>(index)] ||
        smooth->smooth.X.rows() != conditioning.nrow()) {
      Rcpp::stop("native additive smooth primitive identity changed");
    }
  }
  if (profile != nullptr) {
    profile->input_validation_ms += profile_elapsed_ms(stage_started);
  }

  std::vector<const SmoothSetup*> smooth_pointers;
  smooth_pointers.reserve(smooths.size());
  for (const std::shared_ptr<const NativeUnivariateSmooth>& smooth : smooths) {
    smooth_pointers.push_back(&smooth->smooth);
  }
  return assemble_native_setup(conditioning, smooth_pointers, profile);
}

Rcpp::List full_cuda_ci_native_geometry_prepare(
    const Rcpp::NumericMatrix& X_input,
    const Rcpp::List& penalty_blocks,
    const Rcpp::IntegerVector& penalty_offsets,
    const Rcpp::IntegerVector& penalty_ranks,
    NativeSetupProfile* profile) {
  auto stage_started = std::chrono::steady_clock::now();
  const int n = X_input.nrow();
  const int coefficient_count = X_input.ncol();
  const int penalty_count = penalty_blocks.size();
  if (n <= coefficient_count || coefficient_count <= 0 ||
      penalty_count <= 0 || penalty_offsets.size() != penalty_count ||
      penalty_ranks.size() != penalty_count) {
    Rcpp::stop("native geometry dimensions are invalid");
  }
  RowMatrix X(n, coefficient_count);
  for (int column = 0; column < coefficient_count; ++column) {
    for (int row = 0; row < n; ++row) {
      const double value = X_input(row, column);
      if (!std::isfinite(value)) Rcpp::stop("native geometry X must be finite");
      X(row, column) = value;
    }
  }

  std::vector<RowMatrix> penalties;
  std::vector<int> offsets;
  std::vector<int> ranks;
  penalties.reserve(static_cast<std::size_t>(penalty_count));
  offsets.reserve(static_cast<std::size_t>(penalty_count));
  ranks.reserve(static_cast<std::size_t>(penalty_count));
  for (int index = 0; index < penalty_count; ++index) {
    if (!Rf_isReal(penalty_blocks[index]) ||
        !Rf_isMatrix(penalty_blocks[index])) {
      Rcpp::stop("native geometry penalties must be numeric matrices");
    }
    Rcpp::NumericMatrix block(penalty_blocks[index]);
    const int dimension = block.nrow();
    const int offset = penalty_offsets[index] - 1;
    const int rank = penalty_ranks[index];
    if (dimension <= 0 || block.ncol() != dimension || offset < 0 ||
        offset + dimension > coefficient_count || rank <= 0 ||
        rank > dimension) {
      Rcpp::stop("native geometry penalty metadata is invalid");
    }
    RowMatrix penalty(dimension, dimension);
    for (int column = 0; column < dimension; ++column) {
      for (int row = 0; row < dimension; ++row) {
        const double value = block(row, column);
        if (!std::isfinite(value)) {
          Rcpp::stop("native geometry penalties must be finite");
        }
        penalty(row, column) = value;
      }
    }
    penalties.push_back(std::move(penalty));
    offsets.push_back(offset);
    ranks.push_back(rank);
  }
  if (profile != nullptr) {
    profile->geometry_input_ms += profile_elapsed_ms(stage_started);
  }

  stage_started = std::chrono::steady_clock::now();
  PivotedQr qr = pivoted_qr(X);
  if (profile != nullptr) {
    profile->geometry_qr_ms += profile_elapsed_ms(stage_started);
  }

  stage_started = std::chrono::steady_clock::now();
  Rcpp::List roots(penalty_count);
  Rcpp::List matrices(penalty_count);
  Rcpp::CharacterVector names(penalty_count);
  for (int index = 0; index < penalty_count; ++index) {
    RowMatrix local_root = pivoted_cholesky_root(
      penalties[static_cast<std::size_t>(index)],
      ranks[static_cast<std::size_t>(index)]);
    RowMatrix full_root(coefficient_count,
                        ranks[static_cast<std::size_t>(index)]);
    for (int local = 0; local < local_root.rows(); ++local) {
      for (int column = 0; column < local_root.cols(); ++column) {
        full_root(offsets[static_cast<std::size_t>(index)] + local, column) =
          local_root(local, column);
      }
    }
    RowMatrix pivoted_root(coefficient_count, full_root.cols());
    for (int row = 0; row < coefficient_count; ++row) {
      const int source = qr.pivot[static_cast<std::size_t>(row)] - 1;
      for (int column = 0; column < full_root.cols(); ++column) {
        pivoted_root(row, column) = full_root(source, column);
      }
    }
    roots[index] = to_rcpp(pivoted_root);
    matrices[index] = to_rcpp(symmetric_tcrossprod(pivoted_root));
    names[index] = "penalty_" + std::to_string(index + 1);
  }
  roots.attr("names") = names;
  matrices.attr("names") = names;
  if (profile != nullptr) {
    profile->geometry_penalty_ms += profile_elapsed_ms(stage_started);
  }

  stage_started = std::chrono::steady_clock::now();
  const std::vector<double> initial_sp = initial_smoothing_parameters(
    X, penalties, offsets);
  std::vector<double> initial_log_sp(initial_sp.size(), 0.0);
  std::transform(initial_sp.begin(), initial_sp.end(),
                 initial_log_sp.begin(), [](double value) {
                   return std::log(value);
                 });
  if (profile != nullptr) {
    profile->geometry_initial_sp_ms += profile_elapsed_ms(stage_started);
  }

  stage_started = std::chrono::steady_clock::now();
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("schema_version") = "full-cuda-ci-native-geometry-v1",
    Rcpp::Named("magic_q") = to_rcpp(qr.Q),
    Rcpp::Named("magic_qr_packed") = to_rcpp(qr.packed),
    Rcpp::Named("magic_tau") = Rcpp::wrap(qr.tau),
    Rcpp::Named("magic_r") = to_rcpp(qr.R),
    Rcpp::Named("magic_pivot") = Rcpp::wrap(qr.pivot),
    Rcpp::Named("penalty_roots") = roots,
    Rcpp::Named("penalty_matrices") = matrices,
    Rcpp::Named("initial_sp") = Rcpp::wrap(initial_sp),
    Rcpp::Named("initial_log_sp") = Rcpp::wrap(initial_log_sp),
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("legacy_mgcv_mroot_count") = 0,
      Rcpp::Named("legacy_mgcv_initial_sp_count") = 0,
      Rcpp::Named("r_qr_count") = 0,
      Rcpp::Named("native_qr_count") = 1,
      Rcpp::Named("native_mroot_count") = penalty_count,
      Rcpp::Named("native_initial_sp_count") = 1
    )
  );
  if (profile != nullptr) {
    profile->geometry_packaging_ms += profile_elapsed_ms(stage_started);
  }
  return result;
}

}  // namespace fastkpc
