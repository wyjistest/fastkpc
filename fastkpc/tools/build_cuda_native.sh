#!/bin/sh
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD="$ROOT/build"
NVCC=/usr/local/cuda/bin/nvcc
CXX=$(R CMD config CXX17)
CXXSTD=$(R CMD config CXX17STD)
CXXFLAGS=$(R CMD config CXX17FLAGS)
R_CPPFLAGS=$(R CMD config --cppflags)
RCPP_FLAGS=$(Rscript -e 'cat(Rcpp:::CxxFlags())')
RCPP_ARMADILLO_INCLUDE=$(Rscript -e 'cat(system.file("include", package="RcppArmadillo"))')
BLAS_LIBS=$(R CMD config BLAS_LIBS)
LAPACK_LIBS=$(R CMD config LAPACK_LIBS)
FLIBS=$(R CMD config FLIBS)

mkdir -p "$BUILD"

COMMON_INC="$R_CPPFLAGS $RCPP_FLAGS -I$RCPP_ARMADILLO_INCLUDE -I/usr/local/cuda/include -I$ROOT/src -I$ROOT/src/cuda"
COMMON_CXX="$CXXSTD $CXXFLAGS -fPIC $COMMON_INC"

"$CXX" $COMMON_CXX -c "$ROOT/src/dcov_exact_cpu.cpp" -o "$BUILD/dcov_exact_cpu.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/hsic_cpu.cpp" -o "$BUILD/hsic_cpu.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/ci_method.cpp" -o "$BUILD/ci_method.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/fastspline_basis.cpp" -o "$BUILD/fastspline_basis.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/fastspline_solver.cpp" -o "$BUILD/fastspline_solver.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/residual_backend.cpp" -o "$BUILD/residual_backend.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/residual_backend_registry.cpp" -o "$BUILD/residual_backend_registry.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/residual_cache.cpp" -o "$BUILD/residual_cache.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/orientation_matrix.cpp" -o "$BUILD/orientation_matrix.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/orientation_rules.cpp" -o "$BUILD/orientation_rules.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/regrvonps_native.cpp" -o "$BUILD/regrvonps_native.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/regrvonps_device.cpp" -o "$BUILD/regrvonps_device.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/wanpdag_engine.cpp" -o "$BUILD/wanpdag_engine.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/skeleton_engine.cpp" -o "$BUILD/skeleton_engine.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/skeleton_task_scheduler.cpp" -o "$BUILD/skeleton_task_scheduler.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/skeleton_engine_cuda.cpp" -o "$BUILD/skeleton_engine_cuda.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/cuda/cuda_status.cpp" -o "$BUILD/cuda_status.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/r_api_cuda.cpp" -o "$BUILD/r_api_cuda.o"

"$NVCC" -O3 -arch=sm_89 -Xcompiler -fPIC -std=c++17 \
  $COMMON_INC -c "$ROOT/src/cuda/dcov_batch_cuda.cu" \
  -o "$BUILD/dcov_batch_cuda.o"

"$NVCC" -O3 -arch=sm_89 -Xcompiler -fPIC -std=c++17 \
  $COMMON_INC -c "$ROOT/src/cuda/hsic_batch_cuda.cu" \
  -o "$BUILD/hsic_batch_cuda.o"

"$NVCC" -O3 -arch=sm_89 -Xcompiler -fPIC -std=c++17 \
  $COMMON_INC -c "$ROOT/src/cuda/fastspline_batched_solver.cu" \
  -o "$BUILD/fastspline_batched_solver.o"

"$NVCC" -O3 -arch=sm_89 -Xcompiler -fPIC -std=c++17 \
  $COMMON_INC -c "$ROOT/src/cuda/fastspline_residual_cuda.cu" \
  -o "$BUILD/fastspline_residual_cuda.o"

"$CXX" -shared -o "$BUILD/fastkpc_cuda.so" \
  "$BUILD/dcov_exact_cpu.o" \
  "$BUILD/hsic_cpu.o" \
  "$BUILD/ci_method.o" \
  "$BUILD/fastspline_basis.o" \
  "$BUILD/fastspline_solver.o" \
  "$BUILD/residual_backend.o" \
  "$BUILD/residual_backend_registry.o" \
  "$BUILD/residual_cache.o" \
  "$BUILD/orientation_matrix.o" \
  "$BUILD/orientation_rules.o" \
  "$BUILD/regrvonps_native.o" \
  "$BUILD/regrvonps_device.o" \
  "$BUILD/wanpdag_engine.o" \
  "$BUILD/skeleton_engine.o" \
  "$BUILD/skeleton_task_scheduler.o" \
  "$BUILD/skeleton_engine_cuda.o" \
  "$BUILD/cuda_status.o" \
  "$BUILD/r_api_cuda.o" \
  "$BUILD/dcov_batch_cuda.o" \
  "$BUILD/hsic_batch_cuda.o" \
  "$BUILD/fastspline_batched_solver.o" \
  "$BUILD/fastspline_residual_cuda.o" \
  $LAPACK_LIBS $BLAS_LIBS $FLIBS \
  -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver \
  -L"$(R RHOME)/lib" -lR

echo "built: $BUILD/fastkpc_cuda.so"
